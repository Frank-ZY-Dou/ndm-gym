"""Speed of the fused RigidFormer kernels (ndm_gym/kernels/rf_block.py) against the reference in HDMI's scene model at
size RF-S (HOIRigidFormerNet: width 128, 3 blocks, 4 heads, 2 registers): one block, the whole network's forward, and
the world model's step fed back on itself (HOIWorldModel.step, as wmbench/speed_hoi.py runs it), as GPU time per call
(CUDA events; every variant timed once per round, rounds interleaved, the median) and env-steps/s.

Variants ``<kind>:<mode>``, compiled with torch.compile ``mode`` (or ``eager``), under bf16 autocast: kind ``ref`` the
model itself; ``fused`` the whole network in one kernel (``fused_copy(whole_network=True)``; at the block level a
FusedBlock); ``blocks`` only the blocks fused (``fused_copy``). The model's inference copies
(``HOIWorldModel.inference_copy``: token attention, merged read-out heads, the step outside the network as two Triton
kernels): ``copy-bf16`` the network in bf16 end to end (without autocast), ``copy-amp`` its fp32 parameters under
autocast, ``copy-whole`` and ``copy-blocks`` that copy with the whole network or its blocks fused (network and step
levels). At the block and network levels, the CUDA graphs of reduce-overhead copy the inputs into their own buffers
before every replay (in every variant; about 0.14 ms per call at N=4096 for a block's inputs; marking them static does
not stop it here); in the step they are made inside the graph. The kernels' knobs are ``--knobs``, or the fastest of a
sweep over them (``--sweep`` on one block, or the whole network with ``--sweep_on net``: block_m, num_warps, stages,
hidden_chunk, merge, fast_exp, head_chunk; every setting compared with its reference in fp32 first, a miscompiled one
listed as failed). --diagnose runs the whole-network kernel's stages one by one, each in its own process (an illegal
memory access ends a process). Speed does not depend on the weights or the
states: by default random weights and synthetic states; --ckpt loads a trained world model (fit_hoi.py's _wm.pt) and
--data takes the start states from a collect_hoi.py dataset (the first 1024 environments, tiled to N, as speed_hoi.py).
Numbers from a shared GPU are not comparable with benchmarks/ndm_speed/results.md, measured on a GPU that ran nothing else.

  python tools/bench_rf_kernel.py --level block net step --num_envs 1024 4096 16384 \
      --variants ref:reduce-overhead ref:max-autotune blocks:reduce-overhead fused:reduce-overhead --sweep
"""

import argparse
import itertools
import os
import statistics
import sys

import torch
import triton
from triton.errors import TritonError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ndm_gym.core.nets as nets
from ndm_gym.core.rotations import so3_exp, yaw_mat
from ndm_gym.kernels.rf_block import FusedBlock, FusedHOIRigidFormerNet, fused_copy, self_check
from ndm_gym.robots.g1 import G1_JOINT_NAMES, NUM_J
from ndm_gym.tasks.hdmi import DECIMATION, JOINT_NAMES, NUM_ACTIONS
from ndm_gym.worldmodels.hoi import ContextLayout, HOIWorldModel, rigidformer_args

MOVE_SUITCASE = ContextLayout(55, False, 0)  # HDMISim.context_layout of move_suitcase (33 tokens)
ERRORS = (torch.OutOfMemoryError, RuntimeError, ValueError, TritonError)  # out of GPU memory or of the kernel's resources, refused settings


def constants(task: str | None) -> dict:
    """The task's ndm_constants (needs Isaac Lab's packages), or stand-ins of the same shapes with ``task`` None."""
    if task is not None:
        from ndm_gym.sim.rigid.hdmi_sim import ndm_constants
        from ndm_gym.tasks.hdmi_tasks import TASKS

        return ndm_constants(TASKS[task])
    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    return {"eef_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"], "q0": [0.0] * NUM_J, "act_scale": [0.4] * NUM_ACTIONS,
            "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip


def world_model(a, dev) -> HOIWorldModel:
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu")
        args, layout = ck["args"], ContextLayout(*ck["layout"])
        if args["backbone"] != "rigidformer":
            raise ValueError(f"{a.ckpt} is not a RigidFormer world model")
        rf = rigidformer_args(constants(args["task"]), args["rf_dim"], args["rf_layers"], args["rf_heads"], args["rf_registers"])
        wm = HOIWorldModel(args["kind"], layout.base, len(rf["eef_bodies"]), layout, rigidformer=rf)  # a hinged object's base is in ctx
        wm.load_state_dict(ck["wm"])
        return wm.to(dev)
    torch.manual_seed(0)
    rf = rigidformer_args(constants(a.task), 128, 3, 4, 2)
    wm = HOIWorldModel("accel", False, 2, MOVE_SUITCASE, rigidformer=rf)
    with torch.no_grad():  # the heads, FiLM and the relation bias start at zero; give every path a signal
        for m in [wm.net.root_head, wm.net.joint_head, wm.net.obj_head, wm.net.eef_head, wm.net.feet_head] + [b.film for b in wm.net.blocks]:
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
        wm.net.rel_bias.normal_(0.0, 0.3)
    return wm.to(dev)


def scene(wm: HOIWorldModel, n: int, data: str | None, dev, seed: int = 0):
    """Start states (S, So), actions u and contexts ctx of n environments."""
    if data:
        from ndm_gym.state import unflatten_object, unflatten_state

        d = torch.load(data, map_location="cpu", mmap=True)["data"]
        m = min(1024, d["s"].shape[1])
        idx = torch.arange(n) % m
        t = d["s"].shape[0] // 4
        S = unflatten_state(d["s"][t, :m].clone().to(dev)[idx])
        o = d["o"][t, :m].clone().to(dev)[idx]
        So = (o[..., :1], o[..., 1:2]) if wm.hinged else unflatten_object(o)
        return S, So, d["u"][t, :m].clone().to(dev)[idx], d["ctx"][t, :m].clone().to(dev)[idx]
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    R = yaw_mat(so3_exp(torch.cat((0.05 * r(n, 2), 3.0 * r(n, 1)), -1)))
    S = (torch.cat((r(n, 2), 0.78 + 0.02 * r(n, 1)), -1), R @ so3_exp(0.05 * r(n, 3)), 0.3 * r(n, 3), 0.3 * r(n, 3), 0.1 * r(n, NUM_J),
         0.5 * r(n, NUM_J))  # fmt: skip
    So = (S[0] + torch.tensor([0.4, 0.0, 0.1]) + 0.05 * r(n, 3), so3_exp(r(n, 3)), 0.1 * r(n, 3), 0.1 * r(n, 3))
    lay = wm.layout
    poses = [torch.cat((S[0] + r(n, 3), so3_exp(r(n, 3)).flatten(1)), -1) for _ in range(int(lay.base) + lay.props)]
    ctx = torch.cat([1.0 + 0.1 * r(n, lay.scalars)] + poses, -1)
    to = lambda x: x.to(dev)  # noqa: E731
    return tuple(map(to, S)), tuple(map(to, So)), to(0.2 * r(n, DECIMATION, NUM_ACTIONS)), to(ctx)


def interleaved(fns: dict, iters: int, rounds: int, warmup: int = 3) -> dict:
    """Median and min GPU time per call (ms) of every callable that runs; the others' errors as strings."""
    failed = {}
    for k, fn in list(fns.items()):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
        except ERRORS as e:
            failed[k] = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
            del fns[k]
            torch.cuda.empty_cache()
    res = {k: [] for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                fn()
            e1.record()
            torch.cuda.synchronize()
            res[k].append(e0.elapsed_time(e1) / iters)
    return {k: (statistics.median(v), min(v)) for k, v in res.items()} | failed


class _Captured(Exception):
    pass


def first_block_args(net, x, amp: bool = True) -> tuple:
    """The first block's inputs as the network builds them under bf16 autocast (or without; the forward stops there)."""
    cap = {}

    def hook(module, args):
        cap["args"] = args
        raise _Captured

    handle = net.blocks[0].register_forward_pre_hook(hook)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            net(x)
    except _Captured:
        pass
    finally:
        handle.remove()
    return tuple(t.clone() if t is not None else None for t in cap["args"])




KNOBS = ("block_m", "num_warps", "stages", "hidden_chunk", "merge", "fast_exp", "prefetch", "head_chunk")


def sweep(a, wm: HOIWorldModel, n: int, dev) -> dict:
    """The fastest setting of the kernels' knobs at n environments, on the first block (--sweep_on block: FusedBlock)
    or the whole network (net: FusedHOIRigidFormerNet, also head_chunk). Every setting is first compared with its
    reference in fp32 (``self_check``); one that is miscompiled or does not fit this GPU is listed as failed."""
    net = wm.net
    with torch.no_grad():
        x = wm.in_norm(wm.inputs(*scene(wm, n, a.data, dev)))
        args = first_block_args(net, x) if a.sweep_on == "block" else (x,)
    T, rope_pairs = net.n_body + 1 + net.n_props + net.n_reg, 3 * net.rope.freqs.numel()
    heads = a.sweep_head_chunk if a.sweep_on == "net" else [None]
    fns = {}
    grid = itertools.product(a.sweep_block_m, a.sweep_num_warps, a.sweep_stages, a.sweep_hidden_chunk, a.sweep_merge, a.sweep_fast_exp,
                             a.sweep_prefetch, heads)  # fmt: skip
    for k in (k for k in grid if not (k[6] and k[4] != 1)):  # prefetch needs merge 1
        kw = {name: v for name, v in zip(KNOBS, k) if v is not None}
        name = " ".join(f"{k}={v}" for k, v in kw.items())
        try:
            if a.sweep_on == "block":
                f = FusedBlock(net.blocks[0], **kw)
                self_check(net.blocks[0], f, T, rope_pairs, True)
            else:
                f = FusedHOIRigidFormerNet.from_net(net, **kw)
                self_check(net, f, 0, 0, True)
            torch.cuda.synchronize()
        except ERRORS as e:
            print(f"sweep  {n:6d} {name:76s} failed: {type(e).__name__}: {str(e).splitlines()[0][:100]}", flush=True)
            continue
        fns[name] = (lambda f: lambda: f(*args))(f)
    with torch.no_grad():
        res = interleaved(fns, a.iters, a.rounds)
    timed = {k: v for k, v in res.items() if isinstance(v, tuple)}
    for k, v in sorted(timed.items(), key=lambda kv: kv[1][0]):
        print(f"sweep  {n:6d} {k:76s} {v[0]:8.3f} ms (min {v[1]:.3f})", flush=True)
    best = min(timed, key=lambda k: timed[k][0])
    return {k: int(v) for k, v in (kv.split("=") for kv in best.split())}


def compiled(f, mode: str):
    return f if mode == "eager" else torch.compile(f, mode=None if mode == "default" else mode)


ABLATIONS = {"full block": 0, "no attention": 1, "no SwiGLU": 2, "no softmax": 4, "no QK-RMSNorm and rotation": 8,
             "neither attention nor SwiGLU": 3}  # fmt: skip


def ablate(a, wm: HOIWorldModel, n: int, knobs: dict, dev):
    """The fused block timed with parts left out (FusedBlock.debug ABL; outputs wrong) at n environments, each call a
    replayed CUDA graph (no launch overhead), interleaved: what each part of the kernel costs on this GPU."""
    with torch.no_grad():
        args = first_block_args(wm.net, wm.in_norm(wm.inputs(*scene(wm, n, a.data, dev))))
    fns = {}
    for name, code in ABLATIONS.items():
        fb = FusedBlock(wm.net.blocks[0], **{k: v for k, v in knobs.items() if k != "head_chunk"})
        fb.debug["ABL"] = code
        with torch.no_grad():
            fb(*args)  # compiles
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fb(*args)
        fns[name] = graph.replay
    print("knobs: " + " ".join(f"{k}={getattr(fb, k)}" for k in KNOBS[:-1]), flush=True)
    with torch.no_grad():
        res = interleaved(fns, a.iters, a.rounds)
    full = res["full block"][0]
    for name, (med, mn) in res.items():
        print(f"ablate {n:6d} {name:30s} {med:8.3f} ms (min {mn:.3f}) {100 * med / full:5.0f} % of the full block", flush=True)
    kernel_report()


def kernel_report():
    """Registers and spills per thread and shared memory of every fused kernel compiled so far (eagerly launched)."""
    from ndm_gym.kernels.rf_block import _block_kernel, _net_kernel

    for kern in (_block_kernel, _net_kernel):
        for cache in kern.device_caches.values():
            for k in cache[0].values():
                md = k.metadata
                print(f"kernel {kern.fn.__name__}: {k.n_regs} registers, {k.n_spills} spilled, {md.shared} B shared, num_warps {md.num_warps}, "
                      f"num_stages {md.num_stages}", flush=True)  # fmt: skip


KINDS = ("ref", "fused", "blocks", "copy-bf16", "copy-amp", "copy-whole", "copy-blocks")


def build_models(wm: HOIWorldModel, kinds: set, knobs: dict) -> dict:
    """The model of every variant kind (module docstring)."""
    block_kw = {k: v for k, v in knobs.items() if k != "head_chunk"}
    models = {"ref": wm}
    if "fused" in kinds:
        models["fused"] = fused_copy(wm, whole_network=True, **knobs)
    if "blocks" in kinds:
        models["blocks"] = fused_copy(wm, **block_kw)
    if "copy-bf16" in kinds:
        models["copy-bf16"] = wm.inference_copy(torch.bfloat16, "token", fused=True)
    if kinds & {"copy-amp", "copy-whole", "copy-blocks"}:
        amp = wm.inference_copy(None, "token", fused=True)
        models["copy-amp"] = amp
        if "copy-whole" in kinds:
            models["copy-whole"] = fused_copy(amp, whole_network=True, **knobs)
        if "copy-blocks" in kinds:
            models["copy-blocks"] = fused_copy(amp, **block_kw)
    return models


def variants(a, models: dict, knobs: dict, level: str, n: int, dev) -> dict:
    """Callables of one level per variant, each one call (block, network) or one control step (step)."""
    wm = models["ref"]
    S, So, u, ctx = scene(wm, n, a.data, dev)
    with torch.no_grad():
        xin = wm.in_norm(wm.inputs(S, So, u, ctx))
        bargs = first_block_args(wm.net, xin) if level == "block" else None
    block_kw = {k: v for k, v in knobs.items() if k != "head_chunk"}
    fns = {}
    for v in a.variants:
        kind, mode = v.split(":", 1)
        graphs = mode in ("reduce-overhead", "max-autotune")
        amp = kind != "copy-bf16"  # the bf16 copy runs without autocast, as its step does
        if level in ("block", "net"):
            if level == "block" and kind in ("copy-whole", "copy-blocks"):
                print(f"{level:6s} {n:6d} {v:24s} left out: at the block level the same kernel as fused", flush=True)
                continue
            args = (xin,)
            if level == "net":
                f = models[kind].net
            elif kind in ("ref", "fused", "blocks"):
                f, args = wm.net.blocks[0] if kind == "ref" else FusedBlock(wm.net.blocks[0], **block_kw), bargs
            else:  # the copy's own first block, on the inputs its network gives it
                with torch.no_grad():
                    f, args = models[kind].net.blocks[0], first_block_args(models[kind].net, xin, amp)
            f = compiled(f, mode)

            def fn(f=f, args=args, graphs=graphs, amp=amp):
                if graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    return f(*args)
        else:
            step, state = compiled(models[kind].step, mode), {"S": S, "So": So}

            def fn(step=step, state=state, graphs=graphs):  # speed_hoi.py's loop: the step fed back on itself
                if graphs:  # CUDA graphs reuse their output buffers: mark the step and keep copies of the state
                    torch.compiler.cudagraph_mark_step_begin()
                S1, So1, _ = step(state["S"], state["So"], u, ctx)
                state["S"], state["So"] = (tuple(t.clone() for t in S1), tuple(t.clone() for t in So1)) if graphs else (S1, So1)

            def check(step=step, graphs=graphs):  # one step from the start states
                if graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                return torch.cat([t.flatten(1).clone() for part in step(S, So, u, ctx) for t in part], 1)

        fns[v] = fn
        if level != "block":
            validate(level, n, v, check if level == "step" else fn, wm, xin, (S, So, u, ctx))
    return fns


REF_CHUNK = 8192  # environments per call of the eager references in validate (at 65536 they do not fit beside the graphs)


def validate(level: str, n: int, name: str, fn, wm: HOIWorldModel, xin, scene_args):
    """Print a variant's largest difference from the model run eagerly in fp32 (network output, or one step's
    states and forces), beside the eager bf16-autocast model's: a variant far beyond it computes something else. The
    eager models run on REF_CHUNK environments at a time (every environment is computed on its own)."""
    flat = lambda out: torch.cat([t.flatten(1) for part in out for t in part], 1)  # noqa: E731
    S, So, u, ctx = scene_args

    def eager(s):  # the eager model on the environments s
        return wm.net(xin[s]).float() if level == "net" else flat(wm.step(tuple(t[s] for t in S), tuple(t[s] for t in So), u[s], ctx[s]))

    chunks = [slice(i, min(i + REF_CHUNK, n)) for i in range(0, n, REF_CHUNK)]
    amp, tf32 = nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32 = None, False
            ref32 = torch.cat([eager(s) for s in chunks])
            nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32 = torch.bfloat16, tf32
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=level == "net"):  # the step enters autocast itself
                ref16 = torch.cat([eager(s) for s in chunks])
            y = fn().float()
    finally:
        nets.AMP_DTYPE, torch.backends.cuda.matmul.allow_tf32 = amp, tf32
    err, tol = float((y - ref32).abs().max()), float((ref16 - ref32).abs().max())
    print(f"check  {level:6s} {n:6d} {name:24s} max |difference| from the eager fp32 model {err:.3g} (eager bf16 autocast's {tol:.3g})"
          f"{'  WRONG' if not err <= 3 * tol + 1e-6 else ''}", flush=True)  # fmt: skip


# the whole-network kernel's diagnosis: a configuration per process (an illegal memory access ends the process);
# the outer-loop forms compute the network (checked against it), the others skip a stage (only whether they run)
DIAGNOSIS = {"blocks only (FusedBlock)": None, "whole network": {}, "rotation from the positions": {"ROPE": 2},
             "launch num_stages 3": {"LAUNCH_STAGES": 3},
             "loop over blocks unpipelined": {"OUTER": 1}, "loop over blocks unrolled": {"OUTER": 2}, "no rotation": {"ROPE": 0},
             "no embedding product": {"SKIP_EMBED": True},
             "no blocks": {"SKIP_BLOCKS": True}, "no read-out SwiGLU": {"SKIP_HEAD": True}, "no output store": {"SKIP_STORE": True},
             "embedding and store only": {"SKIP_BLOCKS": True, "SKIP_HEAD": True}, "blocks only, no store": {"SKIP_EMBED": True, "SKIP_HEAD": True,
                                                                                                         "SKIP_STORE": True}}  # fmt: skip


def diagnose_one(a, name: str, n: int, knobs: dict, dev):
    """One configuration of DIAGNOSIS on n environments: prints OK / WRONG (error against the fp32 network) or RAN."""
    wm = world_model(a, dev).eval()
    with torch.no_grad():
        x = wm.in_norm(wm.inputs(*scene(wm, n, a.data, dev)))
        cfg = DIAGNOSIS[name]
        if cfg is None:
            fused = fused_copy(wm.net, check=False, **{k: v for k, v in knobs.items() if k != "head_chunk"})
        else:
            fused = FusedHOIRigidFormerNet.from_net(wm.net, force=True, **knobs)
            fused.debug.update(cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fused(x).float()
        torch.cuda.synchronize()
        if cfg is not None and set(cfg) - {"OUTER", "LAUNCH_STAGES"} and cfg != {"ROPE": 2}:
            print(f"RAN {tuple(y.shape)}", flush=True)
            return
        torch.backends.cuda.matmul.allow_tf32 = False
        ref32 = wm.net(x).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16 = wm.net(x).float()
        err, tol = float((y - ref32).abs().max()), float((ref16 - ref32).abs().max())
        print(f"{'OK' if err <= 2 * tol else 'WRONG'}: max error {err:.3g} vs fp32 (autocast's {tol:.3g})", flush=True)


def diagnose(a, knobs: dict):
    import json
    import subprocess

    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}; knobs {knobs or 'default'}", flush=True)
    for n in (1, 1000):
        for name in DIAGNOSIS:
            cmd = [sys.executable, os.path.abspath(__file__), "--diagnose_one", name, "--num_envs", str(n), "--knobs", *a.knobs]
            cmd += [f"--{k}={v}" for k, v in (("task", a.task), ("ckpt", a.ckpt), ("data", a.data)) if v]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith(("OK", "WRONG", "RAN"))]
            errs = [ln for ln in r.stderr.splitlines() if "Error" in ln or "error" in ln]
            res = lines[-1] if r.returncode == 0 and lines else "FAULT: " + (errs[-1].strip()[:150] if errs else f"exit code {r.returncode}")
            print(f"diagnose N={n:5d} {name:32s} {res}", flush=True)
    print(json.dumps({"done": True}))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--level", nargs="+", default=["block", "net", "step"], choices=["block", "net", "step"])
    p.add_argument("--num_envs", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--variants", nargs="+", default=["ref:reduce-overhead", "ref:max-autotune", "blocks:reduce-overhead", "fused:reduce-overhead"])
    p.add_argument("--iters", type=int, default=20, help="calls per variant and round")
    p.add_argument("--rounds", type=int, default=9, help="interleaved rounds; the median is reported")
    p.add_argument("--task", type=str, default=None, help="the task's ndm_constants (Isaac Lab's packages) instead of stand-ins")
    p.add_argument("--ckpt", type=str, default=None, help="a trained world model (fit_hoi.py's _wm.pt)")
    p.add_argument("--data", type=str, default=None, help="start states from a collect_hoi.py dataset")
    p.add_argument("--mem_gb", type=float, default=None, help="cap this process's GPU memory (set_per_process_memory_fraction)")
    p.add_argument("--knobs", nargs="*", default=[], help="the kernels' knobs, e.g. block_m=128 num_warps=8 stages=2 hidden_chunk=64 head_chunk=32")
    p.add_argument("--sweep", action="store_true", help="choose the knobs by a sweep at the largest N")
    p.add_argument("--sweep_on", type=str, default="block", choices=["block", "net"], help="the sweep times one FusedBlock or the whole network")
    p.add_argument("--sweep_block_m", type=int, nargs="+", default=[128], help="256 needs more shared memory than an A100 or H100 has")
    p.add_argument("--sweep_num_warps", type=int, nargs="+", default=[8], help="4 and 16 were 2x slower on the A100 and H100")
    p.add_argument("--sweep_stages", type=int, nargs="+", default=[2, 3], help="1 is refused (miscompiled on an H100)")
    p.add_argument("--sweep_hidden_chunk", type=int, nargs="+", default=[32, 64])
    p.add_argument("--sweep_head_chunk", type=int, nargs="+", default=[32, 64], help="--sweep_on net")
    p.add_argument("--sweep_merge", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sweep_fast_exp", type=int, nargs="+", default=[0])
    p.add_argument("--sweep_prefetch", type=int, nargs="+", default=[0, 1])
    p.add_argument("--diagnose", action="store_true", help="the whole-network kernel's stages one by one, each in its own process")
    p.add_argument("--diagnose_one", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ablate", action="store_true", help="time the fused block with parts left out, at the largest N")
    a = p.parse_args()
    dev = "cuda"
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    nets.AMP_DTYPE = torch.bfloat16
    knobs = {k: int(v) for k, v in (kv.split("=") for kv in a.knobs)}
    if a.diagnose_one:
        return diagnose_one(a, a.diagnose_one, a.num_envs[0], knobs, dev)
    if a.ablate:
        return ablate(a, world_model(a, dev).eval(), max(a.num_envs), knobs, dev)
    if a.diagnose:
        return diagnose(a, knobs)
    wm = world_model(a, dev).eval()
    n_par, n_tok = sum(t.numel() for t in wm.net.parameters()) / 1e6, wm.net.n_body + 1 + wm.net.n_props + wm.net.n_reg
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}; HOIRigidFormerNet {n_par:.2f} M parameters, "
          f"{n_tok} tokens; {'trained ' + a.ckpt if a.ckpt else 'random weights'}", flush=True)
    if a.sweep:
        knobs = sweep(a, wm, max(a.num_envs), dev)
    kinds = {v.split(":", 1)[0] for v in a.variants}
    if not kinds <= set(KINDS):
        raise ValueError(f"variant kinds are {', '.join(KINDS)}; got {sorted(kinds)}")
    models = build_models(wm, kinds, knobs)
    if "blocks" in models:  # the knobs as resolved for this GPU
        fb = models["blocks"].net.blocks[0]
        print("knobs (blocks): " + " ".join(f"{k}={getattr(fb, k)}" for k in KNOBS[:-1]), flush=True)
    if "fused" in models:
        print("knobs (whole network): " + " ".join(f"{k}={v}" for k, v in models["fused"].net.knobs.items()), flush=True)
    kernel_report()  # as compiled for the self-checks
    print(f"{'level':6s} {'N':>6s} {'variant':24s} {'ms / call':>10s} {'min':>8s} {'env-steps/s':>12s} {'vs first':>8s}", flush=True)
    for level in a.level:
        for n in a.num_envs:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                res = interleaved(variants(a, models, knobs, level, n, dev), a.iters, a.rounds)
            first = next((r[0] for r in res.values() if isinstance(r, tuple)), None)
            for v, r in res.items():
                if isinstance(r, str):
                    print(f"{level:6s} {n:6d} {v:24s} {r}", flush=True)
                    continue
                med, mn = r
                print(f"{level:6s} {n:6d} {v:24s} {med:10.3f} {mn:8.3f} {n / med * 1e3:12,.0f} {first / med:7.2f}x", flush=True)
            print(f"{level:6s} {n:6d} peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            torch._dynamo.reset()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
