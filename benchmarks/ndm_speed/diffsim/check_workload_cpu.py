#!/usr/bin/env python
"""CPU check of the benchmark workload (common.py) with MuJoCo C in float64: how long the worlds stay up without a
balance controller, and the contacts, constraint rows and Newton iterations per substep. No GPU is used.

  python check_workload_cpu.py [nworld=64] [control steps=50] [iterations=10]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    import mujoco
    import numpy as np
    nworld = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    seg = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    mjm, meta = C.load_model()
    mjm.opt.iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print("model", C.MODEL_FILE.name, "| stored options before override:", C.model_options(C.load_model()[0]))
    print("keyframes:", [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(mjm.nkey)],
          "| ngeom", mjm.ngeom, "npair", mjm.npair, "| pair condims", dict(zip(*np.unique(mjm.pair_dim, return_counts=True))),
          "| foot friction", C.foot_friction(mjm, meta))
    wl = C.workload(mjm, nworld, seg, 0)
    print("keyframe used:", wl["keyframe"], "| root height", float(wl["qpos0"][0, 2]), "| target range",
          float(wl["targets"].min()), float(wl["targets"].max()), "| q_ref[:6]", np.round(wl["q_ref"][:6], 3))
    d = mujoco.MjData(mjm)
    ncon, nefc, nit, z = [], [], [], np.zeros((nworld, seg))
    t0 = time.perf_counter()
    for w in range(nworld):
        mujoco.mj_resetData(mjm, d)
        d.qpos[:], d.qvel[:] = wl["qpos0"][w], wl["qvel0"][w]
        for t in range(seg):
            d.ctrl[:] = wl["targets"][t, w]
            for _ in range(C.SUBSTEPS):
                mujoco.mj_step(mjm, d)
                ncon.append(d.ncon)
                nefc.append(d.nefc)
                nit.append(int(d.solver_niter[0]))
            z[w, t] = d.qpos[2]
    dt = time.perf_counter() - t0
    ncon, nefc, nit = np.asarray(ncon), np.asarray(nefc), np.asarray(nit)
    print(f"{nworld} worlds x {seg} control steps in {dt:.1f} s on one core ({1e6 * dt / (nworld * seg * C.SUBSTEPS):.0f} us per substep)")
    print(f"contacts per substep: mean {ncon.mean():.1f} max {ncon.max()} | constraint rows: mean {nefc.mean():.1f} max {nefc.max()} | "
          f"Newton iterations: mean {nit.mean():.2f} max {nit.max()} ({100 * (nit >= mjm.opt.iterations).mean():.2f} % of substeps at the cap)")
    for t in sorted({0, 9, 19, 31, seg - 1}):
        if t < seg:
            print(f"after control step {t + 1:3d}: root height min {z[:, t].min():.3f} mean {z[:, t].mean():.3f}; "
                  f"worlds below {C.FALL_Z} m: {int((z[:, t] < C.FALL_Z).sum())}/{nworld}")


if __name__ == "__main__":
    main()
