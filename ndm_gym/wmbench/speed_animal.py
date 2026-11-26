"""An animal tracking world model as the simulator (ndm_gym.train_animal's checkpoint): its throughput beside the training's
MuJoCo-Warp body, and its open-loop error.

Throughput in world-steps per second, one step being one control step: ``physics``, AnimalSim's control step (the body's
physics steps at MIMIC-MJX's settings, as the training runs them, one CUDA graph); ``ndm_fp32``, the checkpoint's world model
as the algorithm steps it in imagination (``world_model``, ``wm_step``: the network's next qpos and qvel, the rat's
activations in closed form), its step one CUDA graph; ``ndm_bf16``, the same with the network's inference copy in bf16
(worldmodels.mjcf.inference_copy, checked there against the fp32 network), compiled (``--compile``) and one CUDA graph. Every
world starts at a frame of the run's clips (drawn per world), at the reference's qpos and qvel with the activations at rest,
and holds that frame's controls of action 0 (the reference's joint targets for servos, the middle of the range otherwise).
Each number is the median of ``--repeats`` runs of ``--steps`` steps after ``--warmup`` steps and ``--warm_seconds`` of calls
(wmbench.speed_mjcf.timed). A count of worlds that does not fit is written with its error.

Open-loop error (``--open_loop`` worlds, 0: none): the checkpoint's policy in the simulator for ``--horizon`` control steps from
frames of the clips (no exploration noise, no resets), its controls recorded, and the world model (fp32) rolled out from the
same starts under the same controls; the tracked bodies' mean position error (mm) after 1 step, half the horizon and the
horizon, over the worlds whose episode did not terminate within the horizon.

Run: ``python -m ndm_gym.wmbench.speed_animal --ckpt RUN/ckpt_004000.pt --out speed.json [--nworld 1024 4096 16384]``
"""

from __future__ import annotations

import argparse
import json
import statistics


def world_model(run: argparse.Namespace, task, wm_lay, pos_lead: float):
    """The world model of a run (``run``: its checkpoint's args) as the algorithm builds it (algos.animal_algos)."""
    from ndm_gym.algos.animal_algos import ANIMAL_ALGOS
    from ndm_gym.worldmodels.animal import AnimalWorldModel

    kind = ANIMAL_ALGOS[run.algo].kind
    lead = {"pos_lead": pos_lead} if kind == "accel" else {}
    return AnimalWorldModel(wm_lay, n_act=task.na, hidden=tuple(run.wm_hidden), act=run.wm_act, layer_norm=not run.wm_no_ln, kind=kind, **lead)


def wm_step(wm, task, qpos, qvel, act, u):
    """One imagined control step (algos.animal_algos.AnimalSuperMimic.wm_step): the network's qpos and qvel, the activations
    in closed form."""
    q1, v1 = wm.step(qpos, qvel, task.wm_control(act, u))
    return q1, v1, (task.activation_step(act, u)[0] if task.na else act)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_dir", default=None, help="the clips (default: the checkpoint's)")
    p.add_argument("--nworld", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--what", nargs="+", default=["physics", "ndm_fp32", "ndm_bf16"], choices=["physics", "ndm_fp32", "ndm_bf16"])
    p.add_argument("--compile", default="max-autotune-no-cudagraphs", help="ndm_bf16: torch.compile's mode, or none")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--warm_seconds", type=float, default=5.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--open_loop", type=int, default=256, help="worlds of the open-loop error (0: not measured)")
    p.add_argument("--horizon", type=int, default=16, help="the open-loop rollout's control steps (training's --horizon)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import mujoco_warp
    import torch
    import warp as wp

    from ndm_gym.policies.hand_policy import HandPolicy
    from ndm_gym.robots.animals import ANIMALS
    from ndm_gym.sim.rigid.animal_sim import AnimalSim
    from ndm_gym.train_animal import build, parse_clips, world_model_lead
    from ndm_gym.wmbench.speed_mjcf import timed
    from ndm_gym.worldmodels.mjcf import fused_compile, inference_copy

    dev = "cuda:0"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    run = argparse.Namespace(**ck["args"])
    spec = ANIMALS[run.animal]
    model, wm_lay, pos_lead, task = build(run.animal, parse_clips(run.clip_ids), a.data_dir or run.data_dir, dev, run.reward_std_scale,
                                          run.action_scale, run.goal_stride, fused_fk=run.fused_fk,
                                          ref_qvel=getattr(run, "ref_qvel", "stored"))  # fmt: skip (a run of before --ref_qvel: stored)
    wm = world_model(run, task, wm_lay, world_model_lead(run, pos_lead)).to(dev)  # the run's --wm_pos_lead, if it set one
    wm.load_state_dict(ck["wm"])
    wm = wm.eval().requires_grad_(False)
    ref = task.ref
    starts = torch.nonzero(torch.arange(ref.frames, device=dev) < ref.end).squeeze(-1)  # every frame but a clip's last
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    out = {"ckpt": a.ckpt, "animal": run.animal, "algo": run.algo, "iteration": ck["iteration"], "gpu": torch.cuda.get_device_name(0),
           "wm_kind": wm.kind, "wm_hidden": list(run.wm_hidden), "substeps": round(spec.ctrl_dt / model.opt.timestep), "ctrl_dt": spec.ctrl_dt,
           "nconmax": spec.nconmax, "njmax": spec.njmax, "torch": torch.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", None),
           "warp": wp.config.version, "rows": []}  # fmt: skip
    for n in a.nworld:
        f = starts[torch.randint(0, starts.shape[0], (n,), device=dev, generator=gen)]
        q0, v0 = ref.qpos[f].clone(), ref.qvel[f].clone()
        act0 = torch.zeros(n, task.na, device=dev)
        u = task.setpoint(torch.zeros(n, task.nu, device=dev), f)  # the frame's controls of action 0
        for what in a.what:
            row = {"what": what, "nworld": n}
            try:
                if what == "physics":
                    sim = AnimalSim(model, task, n, dev, spec.ctrl_dt, joint_noise=0.0, nconmax=spec.nconmax, njmax=spec.njmax)
                    sim.qpos.copy_(q0), sim.qvel.copy_(v0), sim.act.zero_(), sim.ctrl.copy_(u), sim.warmstart.zero_()
                    with wp.ScopedDevice(dev):
                        runs = timed(lambda: wp.capture_launch(sim.graph), wp.synchronize_device, a.warmup, a.steps, a.repeats,  # noqa: B023
                                     warm_seconds=a.warm_seconds)  # fmt: skip
                    row["detail"] = f"AnimalSim, {out['substeps']} physics steps one CUDA graph, nconmax {spec.nconmax}, njmax {spec.njmax} per world"
                    del sim
                else:
                    m = wm if what == "ndm_fp32" else inference_copy(wm, torch.bfloat16)
                    q, v, act = q0.clone(), v0.clone(), act0.clone()

                    def one(q, v, act, m=m):
                        return wm_step(m, task, q, v, act, u)  # noqa: B023

                    fn = one if what == "ndm_fp32" or a.compile == "none" else torch.compile(one, mode=a.compile, dynamic=False)

                    def body():
                        q1, v1, act1 = fn(q, v, act)  # noqa: B023
                        q.copy_(q1), v.copy_(v1), act.copy_(act1)  # noqa: B023

                    with torch.no_grad():
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s), fused_compile(m):
                            for _ in range(3):  # compilation and the allocator's warm-up, outside the capture
                                body()
                        torch.cuda.current_stream().wait_stream(s)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            body()
                        q.copy_(q0), v.copy_(v0), act.copy_(act0)
                        runs = timed(g.replay, torch.cuda.synchronize, a.warmup, a.steps, a.repeats, warm_seconds=a.warm_seconds)
                    row["detail"] = ("fp32, its step one CUDA graph" if what == "ndm_fp32" else
                                     f"bf16 inference copy, compiled ({a.compile}), one CUDA graph")  # fmt: skip
                sec = statistics.median(runs)
                row.update(world_steps_per_s=a.steps * n / sec, ms_per_step=1e3 * sec / a.steps, runs_s=runs)
            except Exception as error:  # a count that does not fit, for instance: written, not hidden
                row["error"] = f"{type(error).__name__}: {error}"
                torch.cuda.empty_cache()
            out["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "runs_s"}), flush=True)
    if a.open_loop:
        n, H = a.open_loop, a.horizon
        f0 = starts[torch.randint(0, starts.shape[0], (n,), device=dev, generator=gen)]
        sim = AnimalSim(model, task, n, dev, spec.ctrl_dt, episode=10**9, joint_noise=0.0, nconmax=spec.nconmax, njmax=spec.njmax)
        sim.reset(torch.ones(n, dtype=torch.bool, device=dev), frame=f0)
        pi = HandPolicy(task.obs_dim, task.goal_frame_dim * run.goal_frames, task.nu).to(dev)
        pi.load_state_dict(ck["pi"])
        pi.eval()
        S = task.unflatten(sim.state())
        W = tuple(x.clone() for x in S)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        a_prev, errors = torch.zeros(n, task.nu, device=dev), {}
        with torch.no_grad():
            for t in range(1, H + 1):
                fr = sim.f.clone()
                act = pi(task.obs(S, task.kin(S[0]), a_prev), task.goal(S, fr, run.goal_frames))
                u_t = task.setpoint(act, fr)
                s1, term, _ = sim.step(u_t, reset=False)
                alive &= ~term
                S, W = task.unflatten(s1), wm_step(wm, task, *W, u_t)
                if t in (1, H // 2, H):
                    errors[t] = torch.linalg.vector_norm(task.kin(W[0]) - task.kin(S[0]), dim=-1).mean(-1)
                a_prev = act
        mm = {"rat": 1000.0, "stick": 1000.0, "fly": 10.0}[run.animal]
        out["open_loop"] = {"worlds": n, "alive": int(alive.sum()), "horizon": H, "starts": "frames of the run's clips",
                            "bodies_mm": {str(t): float(e[alive].mean()) * mm for t, e in errors.items()}}  # fmt: skip
        print(json.dumps({"open_loop": out["open_loop"]}), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
