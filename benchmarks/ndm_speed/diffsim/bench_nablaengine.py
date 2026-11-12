#!/usr/bin/env python
"""NablaEngine (nablaengine.jax_ffi under jax.jit) on the fixed workload of common.py.

Run in the NablaEngine environment (README.md), one process per measurement and world count:
  source ../env.sh; cd ../nablaengine/NablaEngine; source .dk_env/activate.sh
  CUDA_VISIBLE_DEVICES=<gpu> python $DIFFSIM/bench/bench_nablaengine.py --tuned --nworld 4096 --measure forward --csv out.csv
(--measure both starts one process per measurement, so every line has its own peak memory.)

Modes
  --as_published  the authors' own usage and defaults: iterations 10, substeps 4, the interface's gradient
                  convention (clamp_kv 0, normal 3), capacity 64 contacts / 192 constraint rows per world (fixed in
                  the engine). (a): one jitted jax_ffi.step call per control step, dispatched back to back from
                  Python, as scripts/paper_bench/ours.py times it. (b): jax.value_and_grad of a lax.scan rollout with
                  the step tapes kept (no rematerialization), as NablaMimic's configs/ours.args (--no-remat).
  --tuned         as above, with the knobs below.

Speed knobs of --tuned (and what it leaves alone, and why)
  fused rollout       (a) runs each 50-step segment as one XLA program (lax.scan over jax_ffi.step) instead of one
                      dispatch per control step: no per-call Python / XLA dispatch around a step of 1-2 ms.
  gradient program    (b) is one XLA program in both modes; the engine documents no faster setting for an A100
                      (sm_80): DK_SM90_REG_CAPS and DK_SM120_* are for other targets (docs/configuration.md).
  tape memory         kept (23.6 kB per world and control step): rematerialization (jax.checkpoint with
                      jax_ffi.remat_policy()) saves no time, and this rollout has nothing else to recompute.
  solver iterations   Newton cap 5 instead of the engine's default 10: the value the model file stores, and enough
                      for this workload (common.TUNED_ITERATIONS: MuJoCo C gives the same rollout at 5 and 10; caps
                      below 5 change it). The engine leaves its solver loop on convergence, so this only shortens
                      the worlds that would use more. --iterations K sets it; the CSV records K and the world-steps
                      that ended at the cap.
  capacity            NOT lowered: --dk_maxcon 32 sets the engine's experiment switch DK_MAXCON (32 contacts per
                      world instead of 64; the library is rebuilt by nvcc, and the engine warns); worlds beyond the
                      capacity are counted either way (overflow_worlds).
  compiler            the library is built by the nvcc that $NVCC names (here CUDA 12.0); the cache is keyed by the
                      nvcc version, so another toolkit builds its own library (README.md).
  memory pool         XLA_PYTHON_CLIENT_PREALLOCATE=false in both modes (the authors' kit and trainer set it), so
                      the process memory is what the run needs, not JAX's 75 % preallocation.
The engine has no line-search option (--ls_iterations is refused) and no CPU path: --smoke_cpu replaces the engine by
a trivial stand-in to exercise this script's own code on the CPU; it says nothing about the engine.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SIM = "nablaengine"
DIAG = ("fwd_contact_overflow_worlds", "fwd_row_overflow_worlds", "fwd_iteration_limit_worlds", "fwd_worlds",
        "nonfinite_rows")


def main():
    ap = C.parser(__doc__)
    ap.add_argument("--dk_maxcon", type=int, default=None, help="experiment: DK_MAXCON (contact capacity per world)")
    ap.add_argument("--exact_grad", action="store_true", help="the exact derivative (clamp_kv 1, normal 0) instead of "
                                                              "the interface's convention")
    a = C.check_args(ap.parse_args())
    if a.ls_iterations is not None:
        raise SystemExit("bench_nablaengine: the engine has no line-search option (--ls_iterations)")
    if a.dk_maxcon is not None and not a.tuned:
        raise SystemExit("bench_nablaengine: --dk_maxcon is an experiment switch of --tuned")
    if a.measure == "both":
        sys.exit(C.respawn(__file__, sys.argv[1:], [["--measure", "forward"], ["--measure", "grad"]], drop=("--measure",)))
    if a.dk_maxcon is not None:
        os.environ["DK_MAXCON"] = str(a.dk_maxcon)
    if a.smoke_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import numpy as np

    mjm, meta = C.load_model()
    IT = int(a.iterations) if a.iterations is not None else (C.TUNED_ITERATIONS if a.tuned else 10)
    KV, NM = (1, 0) if a.exact_grad else (0, 3)
    N, H, SEG = a.nworld, a.horizon, a.segment_steps
    nseg = a.steps // SEG
    dev = jax.devices()[0]

    if a.smoke_cpu:  # a stand-in with the engine's call signatures (development only)
        def step(q, v, w, u, fr, com):
            return q.at[:, 7:].add(0.3 * (u - q[:, 7:])), 0.9 * v + 0.01 * jnp.pad(u - q[:, 7:], ((0, 0), (6, 0))), w
        read_diag = lambda: dict.fromkeys(DIAG, 0)  # noqa: E731
        last_forward = lambda n: dict(contacts_max=0, rows_max=0)  # noqa: E731
        first_iters = lambda: np.zeros(N)  # noqa: E731
        info = dict(maxcon=64, nefc=192, arch="cpu-stub", key="stub", built=False, commit="")
        version, tape_mib, nvcc, init_s = "stub", 0.0, "", 0.0
    else:
        import hashlib

        import nablaengine
        from nablaengine import cuda as dk_cuda
        from nablaengine import fixtures
        from nablaengine import jax_ffi as F

        if dev.platform != "gpu":
            raise RuntimeError(f"JAX runs on {dev.platform}: bench_nablaengine needs a CUDA device")
        fx = fixtures.fixtures_dir() / "g1_train.mjb"
        h = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (fx, C.MODEL_FILE)]
        if h[0] != h[1]:
            raise RuntimeError(f"the engine's fixture {fx} is not the benchmark model {C.MODEL_FILE}")
        t0 = time.perf_counter()
        info = F.init()
        init_s = time.perf_counter() - t0
        step = F.make_step(iterations=IT, substeps=C.SUBSTEPS, clamp_kv=KV, normal=NM)
        read_diag = lambda: F.read_diagnostics(reset=True)  # noqa: E731
        last_forward = F.last_forward
        version, nvcc = nablaengine.__version__, dk_cuda.nvcc()
        tape_mib = F.tape_bytes(N, C.SUBSTEPS, H) / 2 ** 20

    wl = C.workload(mjm, N, SEG, a.seed)
    put = lambda x: jax.device_put(jnp.asarray(x), dev)  # noqa: E731
    q0, v0, w0, U = put(wl["qpos0"]), put(wl["qvel0"]), put(wl["warm0"]), put(wl["targets"])
    qref = put(wl["q_ref"])
    fr = put(np.full((N,), C.foot_friction(mjm, meta), np.float32))
    com = put(np.zeros((N, 3), np.float32))
    if not a.smoke_cpu:
        _info = jax.jit(lambda q, v, w, u, fr, com: F.forward(q, v, w, u, fr, com, iterations=IT, substeps=C.SUBSTEPS)[3])
        first_iters = lambda: np.asarray(_info(q0, v0, w0, U[0], fr, com))[:, 0]  # noqa: E731

    def segment(q, v, w, U, fr, com):
        return jax.lax.scan(lambda s, u: (step(*s, u, fr, com), None), (q, v, w), U)[0]

    def loss_fn(U, q, v, w, fr, com):
        def body(s, u):
            s = step(*s, u, fr, com)
            return s, 0.5 * jnp.sum((s[0][:, 7:] - qref) ** 2) + 0.5 * C.W_V * jnp.sum(s[1] ** 2)
        return jnp.sum(jax.lax.scan(body, (q, v, w), U)[1]) / N

    def aot(fn, *args):
        t = time.perf_counter()
        c = jax.jit(fn).lower(*args).compile()
        return c, time.perf_counter() - t

    def peak():
        st = dev.memory_stats() or {}
        return round(st["peak_bytes_in_use"] / 2 ** 20, 1) if "peak_bytes_in_use" in st else ""

    knobs = [f"jax_ffi;jit;clamp_kv={KV};normal={NM};no_remat;XLA_PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']};"
             f"DK_MAXCON={os.environ.get('DK_MAXCON', 'default')}"]
    base = C.base_record(a, SIM, mjm)
    base.update(sim_version=version, sim_commit=info.get("commit") or "", iterations=IT, ls_iterations="engine",
                tolerance="engine", capacity_contacts=f"{info['maxcon']} per world", capacity_rows=f"{info['nefc']} per world",
                capacity_note="fixed in the engine (DK_MAXCON)",
                packages=C.packages(("jax", "jaxlib", "jax-cuda12-plugin", "mujoco", "numpy")),
                notes=f"device {dev}; arch {info['arch']}; library key {info['key']} built_now={info['built']} init {init_s:.1f}s; "
                      f"nvcc {nvcc}; keyframe {wl['keyframe']}"
                      + ("; CPU SMOKE TEST WITH A STAND-IN FOR THE ENGINE: meaningless numbers" if a.smoke_cpu else ""))
    out = C.Csv(a.csv, base)

    def settings(final_q, final_v, d, last):
        """The (c) columns: the engine's counters d of the timed repeats, the last forward call's high-water marks,
        the Newton iterations of the first control step's last substep, and the final state."""
        fq, fv = np.asarray(final_q), np.asarray(final_v)
        return dict(overflow_worlds=f"{d['fwd_contact_overflow_worlds'] + d['fwd_row_overflow_worlds']} world-steps",
                    iter_limit_worlds=f"{d['fwd_iteration_limit_worlds']}/{d['fwd_worlds']} world-steps at the cap",
                    solver_iter_mean=f"{float(first_iters().mean()):.3f} (first control step, last substep)",
                    contacts_max=last.get("contacts_max", ""), rows_max=last.get("rows_max", ""),
                    fallen_worlds=int((fq[:, 2] < C.FALL_Z).sum()),
                    finite=bool(np.isfinite(fq).all() and np.isfinite(fv).all()))

    if a.measure == "forward":
        try:
            if a.tuned:
                prog, comp = aot(segment, q0, v0, w0, U, fr, com)

                def repeat():
                    t = time.perf_counter()
                    outs = [prog(q0, v0, w0, U, fr, com) for _ in range(nseg)]
                    jax.block_until_ready(outs)
                    return time.perf_counter() - t, outs[-1]
            else:
                prog, comp = aot(step, q0, v0, w0, U[0], fr, com)
                us = [U[k] for k in range(SEG)]

                def repeat():
                    t = time.perf_counter()
                    for _ in range(nseg):
                        s = (q0, v0, w0)
                        for k in range(SEG):
                            s = prog(*s, us[k], fr, com)
                    jax.block_until_ready(s)
                    return time.perf_counter() - t, s
            repeat()  # warm-up
            read_diag()
            times, final = [], None
            for _ in range(a.repeats):
                dt, final = repeat()
                times.append(dt)
            mem = dict(peak_mem_framework_mib=peak(), peak_mem_process_mib=C.process_gpu_mem_mib())
            d, last = read_diag(), last_forward(N)
            sps = [N * a.steps / t for t in times]
            out.row(measure="forward", status="ok", env_steps_per_s=round(C.median(sps), 1),
                    ms_per_control_step=round(1e3 * C.median(times) / a.steps, 5), repeat_values=C.fmt(sps, 1),
                    repeat_unit="env_steps_per_s", compile_s=round(comp, 2),
                    knobs=";".join(knobs + ["fused_segment_scan" if a.tuned else "one_dispatch_per_control_step"]),
                    **mem, **settings(final[0], final[1], d, last))
        except Exception as e:  # noqa: BLE001
            if not C.is_oom(e):
                raise
            out.row(measure="forward", status="oom", notes=str(e).split("\n")[0][:300])
        return

    try:
        Uh = U[:H]
        prog, comp = aot(jax.value_and_grad(loss_fn), Uh, q0, v0, w0, fr, com)
        jax.block_until_ready(prog(Uh, q0, v0, w0, fr, com))  # warm-up
        read_diag()
        times, res = [], None
        for _ in range(a.repeats):
            t = time.perf_counter()
            res = prog(Uh, q0, v0, w0, fr, com)
            jax.block_until_ready(res)
            times.append(time.perf_counter() - t)
        mem = dict(peak_mem_framework_mib=peak(), peak_mem_process_mib=C.process_gpu_mem_mib())
        d, last = read_diag(), last_forward(N)
        g = np.asarray(res[1])
        ms = [1e3 * t / H for t in times]
        fin = jax.jit(segment)(q0, v0, w0, Uh, fr, com)
        st = settings(fin[0], fin[1], d, last)
        st["finite"] = bool(st["finite"] and np.isfinite(g).all() and d["nonfinite_rows"] == 0)
        out.row(measure="grad", status="ok", env_steps_per_s=round(N / (1e-3 * C.median(ms)), 1),
                ms_per_control_step=round(C.median(ms), 5), repeat_values=C.fmt(ms, 5), repeat_unit="ms_per_control_step",
                compile_s=round(comp, 2), loss=float(res[0]), grad_abs_mean=float(np.abs(g).mean()),
                knobs=";".join(knobs + [f"value_and_grad(scan);tape_mib={tape_mib:.0f}"]), **mem, **st)
        if a.smoke_cpu:  # development check of this script's loss and gradient plumbing on the stand-in
            f = jax.jit(loss_fn)
            Un, eps = np.asarray(Uh), 1e-2
            for (t, w, j) in ((0, 0, 16), (H - 1, N - 1, 20)):
                up, dn = Un.copy(), Un.copy()
                up[t, w, j] += eps
                dn[t, w, j] -= eps
                fd = (float(f(put(up), q0, v0, w0, fr, com)) - float(f(put(dn), q0, v0, w0, fr, com))) / (2 * eps)
                C.say(f"FD check target[{t},{w},{j}]: reverse mode {g[t, w, j]:+.6e}  finite difference {fd:+.6e}")
    except Exception as e:  # noqa: BLE001
        if not C.is_oom(e):
            raise
        out.row(measure="grad", status="oom", notes=str(e).split("\n")[0][:300])


if __name__ == "__main__":
    main()
