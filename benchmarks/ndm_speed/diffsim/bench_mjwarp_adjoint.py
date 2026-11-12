#!/usr/bin/env python
"""MuJoCo Warp with the analytic adjoint of the open pull request google-deepmind/mujoco_warp#1535 (the fork
etaoxing/mujoco_warp at 4af093f = the PR's head 357a75d + the two fixes of etaoxing/mujoco_warp#1) on the fixed
workload of common.py.

Run in the mjwarp_adjoint environment (README.md), one process per measurement and world count:
  source ../env.sh; source ../mjwarp_adjoint/activate.sh
  CUDA_VISIBLE_DEVICES=<gpu> python bench_mjwarp_adjoint.py --tuned --nworld 4096 --measure grad --csv out.csv
(--measure both starts the two measurements as two processes: mjw.enable_grad() changes the kernels, so the forward
is measured without it.) The forward measurement also runs on a stock MuJoCo Warp release (same API).

The differentiated rollout uses one-step checkpointing, the pull request's own pattern (contrib/diffsim/_rollout.py):
the forward keeps only the step inputs (qpos, qvel, warm start) of every 5 ms physics step; the backward restores
them step by step, recomputes that step under a wp.Tape (mjw.step(m, d, d_out) records the analytic adjoint) and runs
tape.backward(). Two Data objects serve a rollout of any length.

Modes
  --as_published  the pull request author's own G1 settings (contrib/diffsim/sysid_g1.py) and upstream's G1 benchmark
                  (benchmarks/unitree_g1): iterations 10, ls_iterations 20, nconmax 48, njmax 192, the ITERATIONS /
                  LS_ITERATIONS warnings masked; one CUDA graph per physics step (forward) and per backward physics
                  step, the bookkeeping kernels launched from Python; the backward recomputes each step with the full
                  solver from the stored input warm start.
  --tuned         the knobs below.

Speed knobs of --tuned (and why)
  CUDA graphs         one graph per control step (target load + 4 x mjw.step + bookkeeping) and one per backward
                      physics step including its restore / accumulate kernels: fewer host launches per step.
  capacities          contacts and constraint rows sized to the workload by the rule NablaEngine's harness documents
                      for MuJoCo Warp (benchmarks/bench/adapters/mjw_common.py): a calibration segment at 64 contacts /
                      192 rows per world measures mu = max(broadphase candidates, contacts) per substep, c_max and
                      r_max; naconmax = ceil(1.25 mu_total) + c_max, nconmax = ceil(1.25 mu), njmax = the smallest of
                      MuJoCo Warp's sizes >= 1.1 r_max (the 10 % because GPU runs do not repeat exactly). Smaller
                      buffers, less memory traffic. The overflow bits are read back after every measurement
                      (overflow_worlds); --nconmax / --njmax set the capacities by hand.
  backward recompute  each backward step restarts its solver at the forward's converged acceleration and runs one
                      Newton iteration instead of repeating the whole solve (the adjoint is taken at the converged
                      solution; NablaEngine's reference wrapper does the same, _Sim.backward_ws). --no_warm_recompute
                      turns it off.
  overflow printing   m.opt.warn_overflow = 0: no GPU printf in the kernels (the bits are still recorded).
  line search         ls_iterations as the model stores them (8), not the demo's 20; --ls_iterations sets it.
  solver iterations   Newton cap 5 instead of 10: the value the model file stores, and enough for this workload
                      (common.TUNED_ITERATIONS: MuJoCo C gives the same rollout at 5 and 10; caps below 5 change
                      it). MuJoCo Warp leaves the solver loop as soon as every world has converged, so the cap only
                      shortens the worlds that would use more. --iterations K sets it; the CSV records K and the
                      worlds that reached it.
  checkpointing       always one-step (above): it is what makes a 32-step rollout fit; no knob.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SIM = "mujoco_warp_adjoint"
PUBLISHED = dict(iterations=10, ls_iterations=20, nconmax=48, njmax=192)
GENEROUS = dict(nconmax=64, njmax=192)


def kernels(wp):
    @wp.kernel(enable_backward=False)
    def load3(idx: wp.array(dtype=int), div: int, src: wp.array3d(dtype=float), dst: wp.array2d(dtype=float)):
        w, i = wp.tid()
        t = int(wp.floor(float(idx[0]) / float(div)))  # idx // div, written without integer-division semantics
        dst[w, i] = src[t, w, i]

    @wp.kernel(enable_backward=False)
    def store3(idx: wp.array(dtype=int), src: wp.array2d(dtype=float), dst: wp.array3d(dtype=float)):
        w, i = wp.tid()
        dst[idx[0], w, i] = src[w, i]

    @wp.kernel(enable_backward=False)
    def add3(idx: wp.array(dtype=int), div: int, src: wp.array2d(dtype=float), dst: wp.array3d(dtype=float)):
        w, i = wp.tid()
        t = int(wp.floor(float(idx[0]) / float(div)))
        dst[t, w, i] = dst[t, w, i] + src[w, i]

    @wp.kernel(enable_backward=False)
    def bump(idx: wp.array(dtype=int), delta: int):
        idx[0] = idx[0] + delta

    @wp.kernel(enable_backward=False)
    def loss_grad(idx: wp.array(dtype=int), substeps: int, scale: float, w_v: float, s_q: wp.array3d(dtype=float),
                  s_v: wp.array3d(dtype=float), q_ref: wp.array(dtype=float), adj_q: wp.array2d(dtype=float),
                  adj_v: wp.array2d(dtype=float)):
        # the direct term dL/dstate of state idx[0] when it ends a control step (dim = (nworld, nv))
        w, i = wp.tid()
        k = idx[0]
        if k > 0 and k % substeps == 0:
            adj_v[w, i] = adj_v[w, i] + scale * w_v * s_v[k, w, i]
            if i >= 6:
                adj_q[w, i + 1] = adj_q[w, i + 1] + scale * (s_q[k, w, i + 1] - q_ref[i - 6])

    return load3, store3, add3, bump, loss_grad


def main():
    ap = C.parser(__doc__)
    ap.add_argument("--nconmax", type=int, default=None, help="contacts per world (default: per mode)")
    ap.add_argument("--njmax", type=int, default=None, help="constraint rows per world (default: per mode)")
    ap.add_argument("--no_warm_recompute", action="store_true", help="--tuned: recompute backward steps with the full solver")
    a = C.check_args(ap.parse_args())
    if a.measure == "both":
        sys.exit(C.respawn(__file__, sys.argv[1:], [["--measure", "forward"], ["--measure", "grad"]], drop=("--measure",)))

    import dataclasses

    import mujoco
    import numpy as np
    import warp as wp

    wp.config.quiet = True
    import mujoco_warp as mjw
    from mujoco_warp._src.types import OverflowType

    grad = a.measure == "grad"
    mjm, meta = C.load_model()
    stored_ls = int(mjm.opt.ls_iterations)
    IT = int(a.iterations) if a.iterations is not None else (C.TUNED_ITERATIONS if a.tuned else PUBLISHED["iterations"])
    LS = int(a.ls_iterations) if a.ls_iterations is not None else (PUBLISHED["ls_iterations"] if a.as_published else stored_ls)
    mjm.opt.iterations, mjm.opt.ls_iterations = IT, LS
    N, H, SEG, S = a.nworld, a.horizon, a.segment_steps, C.SUBSTEPS
    nseg = a.steps // SEG
    wl = C.workload(mjm, N, SEG, a.seed)
    device = "cpu" if a.smoke_cpu else "cuda:0"
    use_graph = not a.smoke_cpu
    warm_recompute = a.tuned and not a.no_warm_recompute
    ITER_BITS = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)

    with wp.ScopedDevice(device):
        dev = wp.get_device()
        if grad:
            mjw.enable_grad()
        load3, store3, add3, bump, loss_grad = kernels(wp)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        m = mjw.put_model(mjm)
        m.opt.warn_overflow = 0 if a.tuned else int(m.opt.warn_overflow) & ~ITER_BITS
        fa = lambda x: wp.array(np.ascontiguousarray(x), dtype=float)  # noqa: E731
        q0, v0, w0, U, q_ref = fa(wl["qpos0"]), fa(wl["qvel0"]), fa(wl["warm0"]), fa(wl["targets"]), fa(wl["q_ref"])
        t_idx, k_idx = wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)

        def reset(d):
            wp.copy(d.qpos, q0)
            wp.copy(d.qvel, v0)
            wp.copy(d.qacc_warmstart, w0)
            t_idx.zero_()
            k_idx.zero_()

        def ctrl_load(d, idx, div):
            wp.launch(load3, dim=(N, C.NU), inputs=[idx, div, U], outputs=[d.ctrl])

        def capture(fn):
            """fn as a CUDA graph (after one eager run: kernel builds and allocations happen outside the capture)."""
            fn()
            wp.synchronize()
            if not use_graph:
                return fn
            with wp.ScopedCapture() as cap:
                fn()
            g = cap.graph
            return lambda: wp.capture_launch(g)

        # ---- capacities
        if a.nconmax is not None or a.njmax is not None:
            if a.nconmax is None or a.njmax is None:
                raise SystemExit("give both --nconmax and --njmax")
            cap = dict(nconmax=a.nconmax, njmax=a.njmax, naconmax=a.nconmax * N, note="given on the command line")
        elif a.as_published:
            cap = dict(nconmax=PUBLISHED["nconmax"], njmax=PUBLISHED["njmax"], naconmax=PUBLISHED["nconmax"] * N,
                       note="the PR author's G1 demo and upstream's G1 benchmark")
        else:
            dc = mjw.put_data(mjm, mjd, nworld=N, **GENEROUS)
            reset(dc)
            dc.overflow.zero_()
            pool = cmax = rmax = 0
            for _ in range(SEG):
                ctrl_load(dc, t_idx, 1)
                for _ in range(S):
                    mjw.step(m, dc)
                    nac = int(dc.nacon.numpy()[0])
                    pool = max(pool, nac, int(dc.ncollision.numpy()[0]))
                    rmax = max(rmax, int(dc.nefc.numpy().max()))
                    cmax = max(cmax, int(np.bincount(dc.contact.worldid.numpy()[:nac], minlength=N).max()) if nac else 0)
                wp.launch(bump, dim=1, inputs=[t_idx, 1])
            bad = int((dc.overflow.numpy() & ~ITER_BITS != 0).sum())
            if bad:
                raise RuntimeError(f"capacity calibration: {bad} worlds overflow {GENEROUS} on this workload")
            cap = dict(nconmax=int(np.ceil(1.25 * pool / N)), njmax=C.valid_size(int(np.ceil(1.1 * rmax))),
                       naconmax=int(np.ceil(1.25 * pool)) + cmax,
                       note=f"calibrated on this workload: pool {pool} entries over {N} worlds, c_max {cmax}, r_max {rmax}")
            del dc
        mk = lambda: mjw.put_data(mjm, mjd, nworld=N, nconmax=cap["nconmax"], njmax=cap["njmax"], naconmax=cap["naconmax"])  # noqa: E731

        base = C.base_record(a, SIM, mjm)
        commit = ""
        try:
            commit = subprocess.run(["git", "-C", str(Path(mjw.__file__).resolve().parent), "rev-parse", "HEAD"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True).stdout.strip()
        except OSError:
            pass
        import importlib.metadata as md
        try:
            version = md.version("mujoco-warp")
        except md.PackageNotFoundError:
            version = "source tree"
        tol = float(np.asarray(m.opt.tolerance.numpy()).reshape(-1)[0])
        base.update(sim_version=version, sim_commit=commit, iterations=IT, ls_iterations=LS,
                    tolerance=f"{tol:.3g} (device model; stored {mjm.opt.tolerance:.3g})",
                    capacity_contacts=f"{cap['nconmax']}/world, {cap['naconmax']} total", capacity_rows=cap["njmax"],
                    capacity_note=cap["note"], packages=C.packages(("warp-lang", "mujoco", "numpy")),
                    notes=f"device {dev}; warp {wp.config.version}; enable_grad={int(grad)}; keyframe {wl['keyframe']}"
                          + ("; CPU SMOKE TEST: timings meaningless" if a.smoke_cpu else ""))
        out = C.Csv(a.csv, base)
        knobs = [f"graph={'control_step' if a.tuned else 'physics_step'}" if use_graph else "no_graph(cpu)",
                 f"warn_overflow={int(m.opt.warn_overflow)}"]

        def settings(d, others, final_q):
            """The (c) columns after the timed repeats. d: the Data of the forward steps (its sticky overflow bits give
            the worlds at the iteration caps); others: further Data objects whose capacity bits count as well (the
            backward recompute's d_out, whose one-iteration solver always ends at its cap)."""
            ov = d.overflow.numpy().astype(np.int64).reshape(-1)
            capbits = ov & ~ITER_BITS
            for o in others:
                capbits = capbits | (o.overflow.numpy().astype(np.int64).reshape(-1) & ~ITER_BITS)
            fq = final_q.numpy() if hasattr(final_q, "numpy") else np.asarray(final_q)
            it = d.solver_niter.numpy().reshape(-1)
            return dict(overflow_worlds=int((capbits != 0).sum()),
                        iter_limit_worlds=f"{int((ov & int(OverflowType.ITERATIONS) != 0).sum())}/{N} worlds reached the Newton "
                                          f"cap in some substep, {int((ov & int(OverflowType.LS_ITERATIONS) != 0).sum())} the "
                                          f"line-search cap",
                        solver_iter_mean=f"{float(it.mean()):.3f} (last substep)",
                        contacts_max=f"{int(d.nacon.numpy()[0])} in all worlds (last substep)",
                        rows_max=f"{int(d.nefc.numpy().max())} (last substep)", fallen_worlds=int((fq[:, 2] < C.FALL_Z).sum()),
                        finite=bool(np.isfinite(fq).all()))

        def mem():
            fw = round(wp.get_mempool_used_mem_high(dev) / 2 ** 20, 1) if dev.is_cuda else ""
            return dict(peak_mem_framework_mib=fw, peak_mem_process_mib=C.process_gpu_mem_mib() if dev.is_cuda else "")

        if not grad:
            try:
                d = mk()
                reset(d)

                def phys():
                    mjw.step(m, d)

                def control_step():
                    ctrl_load(d, t_idx, 1)
                    for _ in range(S):
                        mjw.step(m, d)
                    wp.launch(bump, dim=1, inputs=[t_idx, 1])
                t0 = time.perf_counter()
                g = capture(control_step if a.tuned else phys)
                comp = time.perf_counter() - t0

                def repeat():
                    t = time.perf_counter()
                    for _ in range(nseg):
                        reset(d)
                        for _ in range(SEG):
                            if a.tuned:
                                g()
                            else:
                                ctrl_load(d, t_idx, 1)
                                for _ in range(S):
                                    g()
                                wp.launch(bump, dim=1, inputs=[t_idx, 1])
                    wp.synchronize()
                    return time.perf_counter() - t
                repeat()  # warm-up
                d.overflow.zero_()
                times = [repeat() for _ in range(a.repeats)]
                sps = [N * a.steps / t for t in times]
                out.row(measure="forward", status="ok", env_steps_per_s=round(C.median(sps), 1),
                        ms_per_control_step=round(1e3 * C.median(times) / a.steps, 5), repeat_values=C.fmt(sps, 1),
                        repeat_unit="env_steps_per_s", compile_s=round(comp, 2), knobs=";".join(knobs), **mem(),
                        **settings(d, [], d.qpos))
            except Exception as e:  # noqa: BLE001
                if not C.is_oom(e):
                    raise
                out.row(measure="forward", status="oom", notes=str(e)[:300])
            return

        # ---- (b) forward + reverse-mode gradient of the H-step rollout loss with respect to the H target vectors
        try:
            K = H * S
            d, d_out = mk(), mk()
            for arr in (d.qpos, d.qvel, d.ctrl):
                arr.requires_grad = True
            bc = mjw.create_backward_context(m, d)
            m_bw = dataclasses.replace(m, opt=dataclasses.replace(m.opt, iterations=1)) if warm_recompute else m
            z3 = lambda n, k: wp.zeros((n, N, k), dtype=float)  # noqa: E731
            s_q, s_v, s_w = z3(K + 1, C.NQ), z3(K + 1, C.NV), z3(K, C.NV)
            s_a = z3(K, C.NV) if warm_recompute else None
            adj_q, adj_v, G = wp.zeros((N, C.NQ), dtype=float), wp.zeros((N, C.NV), dtype=float), z3(H, C.NU)
            scale = 1.0 / N
            holder = {}

            def save_state():
                wp.launch(store3, dim=(N, C.NQ), inputs=[k_idx, d.qpos], outputs=[s_q])
                wp.launch(store3, dim=(N, C.NV), inputs=[k_idx, d.qvel], outputs=[s_v])

            def f_pre():
                save_state()
                wp.launch(store3, dim=(N, C.NV), inputs=[k_idx, d.qacc_warmstart], outputs=[s_w])

            def f_phys():
                mjw.step(m, d)

            def f_post():
                if warm_recompute:  # after a step qacc_warmstart holds the step's converged acceleration
                    wp.launch(store3, dim=(N, C.NV), inputs=[k_idx, d.qacc_warmstart], outputs=[s_a])
                wp.launch(bump, dim=1, inputs=[k_idx, 1])

            def f_control_step():
                ctrl_load(d, t_idx, 1)
                for _ in range(S):
                    f_pre()
                    f_phys()
                    f_post()
                wp.launch(bump, dim=1, inputs=[t_idx, 1])

            def add_loss_grad():
                wp.launch(loss_grad, dim=(N, C.NV), inputs=[k_idx, S, scale, C.W_V, s_q, s_v, q_ref], outputs=[adj_q, adj_v])

            def b_restore():
                wp.launch(load3, dim=(N, C.NQ), inputs=[k_idx, 1, s_q], outputs=[d.qpos])
                wp.launch(load3, dim=(N, C.NV), inputs=[k_idx, 1, s_v], outputs=[d.qvel])
                wp.launch(load3, dim=(N, C.NV), inputs=[k_idx, 1, s_a if warm_recompute else s_w], outputs=[d.qacc_warmstart])
                ctrl_load(d, k_idx, S)

            def b_core():  # the pull request's backward step: taped recompute d -> d_out, then its analytic adjoint
                tape = wp.Tape()
                with mjw.backward_context(bc):
                    with tape:
                        mjw.step(m_bw, d, d_out)
                    tape.zero()
                    wp.copy(d_out.qpos.grad, adj_q)
                    wp.copy(d_out.qvel.grad, adj_v)
                    tape.backward()
                holder["tape"] = tape

            def b_post():
                wp.copy(adj_q, d.qpos.grad)
                wp.copy(adj_v, d.qvel.grad)
                wp.launch(add3, dim=(N, C.NU), inputs=[k_idx, S, d.ctrl.grad], outputs=[G])
                add_loss_grad()  # the direct term of state k (zero unless k ends a control step)
                wp.launch(bump, dim=1, inputs=[k_idx, -1])

            def b_step():
                b_restore()
                b_core()
                b_post()

            def rollout(fwd, bwd_core_or_step):
                """One differentiated rollout; fwd / bwd are graphs or the functions themselves."""
                reset(d)
                for _ in range(H):
                    if a.tuned:
                        fwd()
                    else:
                        ctrl_load(d, t_idx, 1)
                        for _ in range(S):
                            f_pre()
                            fwd()
                            f_post()
                        wp.launch(bump, dim=1, inputs=[t_idx, 1])
                save_state()        # state K
                adj_q.zero_()
                adj_v.zero_()
                G.zero_()
                add_loss_grad()     # the direct term of state K
                wp.launch(bump, dim=1, inputs=[k_idx, -1])
                for _ in range(K):
                    if a.tuned:
                        bwd_core_or_step()
                    else:
                        b_restore()
                        bwd_core_or_step()
                        b_post()

            t0 = time.perf_counter()
            reset(d)
            g_f = capture(f_control_step if a.tuned else f_phys)
            # a valid state for the first backward capture: one eager rollout
            rollout(f_control_step if a.tuned else f_phys, b_step if a.tuned else b_core)
            wp.synchronize()
            k_idx.fill_(1)  # a valid step index for the eager run inside capture()
            g_b = capture(b_step if a.tuned else b_core)
            comp = time.perf_counter() - t0
            rollout(g_f, g_b)  # warm-up with the graphs
            wp.synchronize()
            d.overflow.zero_()
            d_out.overflow.zero_()
            times = []
            for _ in range(a.repeats):
                t = time.perf_counter()
                rollout(g_f, g_b)
                wp.synchronize()
                times.append(time.perf_counter() - t)
            ms = [1e3 * t / H for t in times]
            gn = G.numpy()
            sq, sv = s_q.numpy()[S::S], s_v.numpy()[S::S]
            loss = float((0.5 * ((sq[:, :, 7:] - wl["q_ref"]) ** 2).sum() + 0.5 * C.W_V * (sv ** 2).sum()) / N)
            st = settings(d, [d_out], sq[-1])
            st["finite"] = bool(st["finite"] and np.isfinite(gn).all())
            kn = knobs + [f"one_step_checkpointing;backward_recompute={'warm_1_iteration' if warm_recompute else 'full_solve'}"]
            out.row(measure="grad", status="ok", env_steps_per_s=round(N / (1e-3 * C.median(ms)), 1),
                    ms_per_control_step=round(C.median(ms), 5), repeat_values=C.fmt(ms, 5), repeat_unit="ms_per_control_step",
                    compile_s=round(comp, 2), loss=loss, grad_abs_mean=float(np.abs(gn).mean()), knobs=";".join(kn), **mem(), **st)
            if a.smoke_cpu:  # development check of the chain: central differences of the loss in a few target entries
                def loss_of(Unp):
                    U.assign(Unp)
                    reset(d)
                    for _ in range(H):
                        f_control_step()
                    save_state()
                    q, v = s_q.numpy()[S::S], s_v.numpy()[S::S]
                    return float((0.5 * ((q[:, :, 7:] - wl["q_ref"]) ** 2).sum() + 0.5 * C.W_V * (v ** 2).sum()) / N)
                Un, eps = wl["targets"].copy(), 2e-3
                for (t, w, j) in ((0, 0, 16), (H - 1, N - 1, 20), (H // 2, 0, 3)):
                    up, dn = Un.copy(), Un.copy()
                    up[t, w, j] += eps
                    dn[t, w, j] -= eps
                    fd = (loss_of(up) - loss_of(dn)) / (2 * eps)
                    C.say(f"FD check target[{t},{w},{j}]: adjoint {gn[t, w, j]:+.6e}  finite difference {fd:+.6e}")
        except Exception as e:  # noqa: BLE001
            if not C.is_oom(e):
                raise
            out.row(measure="grad", status="oom", notes=str(e)[:300])


if __name__ == "__main__":
    main()
