#!/usr/bin/env python
"""MJX-JAX (mujoco-mjx, impl='jax') on the fixed workload of common.py.

Run in the mjx environment (README.md), one process per measurement and world count:
  source ../mjx/activate.sh
  CUDA_VISIBLE_DEVICES=<gpu> python bench_mjx.py --tuned --nworld 4096 --measure forward --csv out.csv
(--measure both and --grad_variant both start one process per measurement, so every line has its own peak memory.)

What stock MJX cannot do at the matched settings, and the alternatives
  Reverse-mode differentiation of mjx.step works only with opt.iterations == 1: for more iterations the Newton
  solver's outer loop is a jax.lax.while_loop, which JAX cannot differentiate in reverse mode
  (https://github.com/google-deepmind/mujoco/issues/2259). Measurement (b) therefore has two variants, named in
  every CSV line (columns iterations, knobs and notes):
    --grad_variant iter1  stock MJX with opt.iterations = 1: ONE Newton iteration per substep instead of the cap of
                          the forward measurement (5). On this model that is other physics, and unstable: with
                          one iteration per substep MuJoCo C reaches joint speeds above 180 rad/s after two control
                          steps and aborts with "huge value in QACC" within 0.2-0.6 s (common.TUNED_ITERATIONS).
                          The line therefore times a rollout that is not the workload's; its notes also give the
                          loss of the same rollout at the cap of the forward measurement, so the difference is in
                          the output (CPU smoke test, 2 worlds x 2 steps: 34.3 against 0.114).
    --grad_variant scan   the iteration cap of the forward measurement, with a small patch applied in this process
                          only (PATCH A below): the outer loop becomes the module's own fixed-length scan. Every
                          world then runs every iteration (finished ones as no-ops), so the cap is a cost factor.

Modes
  --as_published  stock mujoco-mjx as its documentation, MuJoCo Playground and NablaEngine's timing kit run it:
                  mjx.put_model(m, impl='jax'); jit(vmap(control step)), the control step a lax.scan of 4 x mjx.step
                  that carries the mjx.Data; one dispatch per control step; every one of the model's 502 contact
                  pairs padded statically (MJX-JAX has no broadphase); XLA's default flags; iterations (5),
                  ls_iterations (8) and tolerance as the model stores them: the options MJX's authors ship for the
                  G1 (MuJoCo Menagerie unitree_g1/g1_mjx.xml). NablaEngine's kit timed MJX at 10: --iterations 10.
                  (b): jax.value_and_grad of that rollout, residuals of all 128 substeps kept (no
                  rematerialization), --grad_variant iter1 only. At the benchmark's world counts this is expected to
                  end as status=oom (reported as such); the official gradient tutorial uses 64 worlds
                  (mjx/training_apg.ipynb), which --nworld 64 reproduces.
  --tuned         the knobs below; (b) runs both variants unless --grad_variant picks one.

Speed knobs of --tuned (and why)
  fused rollout       (a): each 50-step segment is one XLA program (lax.scan over control steps) instead of one
                      dispatch per control step.
  state-only carry    the scans carry (qpos, qvel, qacc_warmstart) and build the mjx.Data inside every substep from
                      a constant template: a small loop state, and what makes checkpointing cheap.
  contact caps        MJX's documented substitute for a broadphase (doc "Performance Tuning": the custom numerics
                      max_geom_pairs and max_contact_points): per geom-type group only the G pairs with the nearest
                      bounding spheres reach the collision function, and per condim only the P nearest contacts
                      reach the solver. The static sizes shrink from every pair (about 1500 constraint rows) to
                      what the workload needs. G and P are calibrated at start-up: MuJoCo C (float64, CPU) runs the
                      first --calib_worlds worlds of the workload and counts, per substep, the pairs whose bounding
                      spheres overlap (per group) and the contacts (per condim); G and P = 1.25 x the maxima + 2.
                      With G >= overlapping pairs and P > contacts the capped model computes the same physics as
                      the full one. overflow_worlds counts the worlds where, in some substep of the measured
                      rollout, a group had more than G overlapping pairs or a condim had P active contacts (all
                      slots in use: possibly more dropped). The binary model has no <custom> block, so the two
                      numerics are supplied by PATCH B below. --max_geom_pairs / --max_contact_points set them by
                      hand; --no_caps leaves the full contact set.
  XLA GEMM flag       XLA_FLAGS=--xla_gpu_triton_gemm_any=true, which the MJX documentation says should be set on
                      NVIDIA GPUs ("can yield a 30% speedup"; NablaEngine's kit measured no effect for this model).
                      --xla_triton default leaves XLA's default.
  tolerance           1e-6, MuJoCo Warp's effective value (it clamps the stored 1e-8 to 1e-6): the solver loop ends
                      at the same criterion as there, after fewer iterations than with the stored 1e-8. Under vmap
                      the loop runs until the slowest world is done. --tolerance sets it; the as_published mode keeps
                      the stored value.
  checkpointing       (b): jax.checkpoint around every substep: the backward pass keeps only the 128 step inputs and
                      recomputes one substep at a time (one extra forward), instead of the residuals of all 128.
                      --no_checkpoint turns it off.
  solver iterations   the stored cap of 5 in both modes: MJX documents "as low as stays stable (often just one)",
                      but for this model 5 is the lowest cap that leaves the rollout unchanged
                      (common.TUNED_ITERATIONS). --iterations K / --ls_iterations K set them; the CSV records the
                      cap, the mean iterations and the worlds that reached it.
  dense / sparse      nothing to turn: opt.jacobian "auto" is dense for nv < 60 on a GPU (35 dofs here).
  precision           float32 by the workload's definition (the official gradient tutorial enables x64).
  memory pool         XLA_PYTHON_CLIENT_PREALLOCATE=false in both modes, so the process memory is what the run needs.

PATCH A (only --grad_variant scan; this process only, no file is changed): while the gradient program is traced,
  the name `jax` inside mujoco.mjx._src.solver points to a proxy whose lax.while_loop(cond, body, init) calls the
  module's own _while_loop_scan(cond, body, init, max_iter=opt.iterations) ("jit ok, reverse-mode autodiff ok", which
  MJX already uses for its line search). Nothing else of the solver changes.
PATCH B (contact caps; this process only): mujoco.mjx._src.collision_driver._numeric returns G and P for the names
  max_geom_pairs and max_contact_points, as if the model declared <custom><numeric name="max_geom_pairs" .../>.
OBSERVER (statistics only, never in a timed program): MJX-JAX does not store the solver's iteration count in its Data,
  so while the untimed statistics rollout is traced the same proxy passes lax.while_loop through unchanged and keeps
  the returned context's solver_niter.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SIM = "mjx_jax"
MATCHED_ITERATIONS = C.TUNED_ITERATIONS  # the cap of (a) and of --grad_variant scan in both modes: the stored 5
TUNED_TOLERANCE = 1e-6
ISSUE = "https://github.com/google-deepmind/mujoco/issues/2259"


def main():
    ap = C.parser(__doc__)
    ap.add_argument("--grad_variant", default=None, choices=["iter1", "scan", "both"],
                    help="(b): see the header (default: iter1 for --as_published, both for --tuned)")
    ap.add_argument("--max_geom_pairs", type=int, default=None, help="--tuned: G by hand (default: calibrated)")
    ap.add_argument("--max_contact_points", type=int, default=None, help="--tuned: P by hand (default: calibrated)")
    ap.add_argument("--no_caps", action="store_true", help="--tuned: keep the full contact set")
    ap.add_argument("--calib_worlds", type=int, default=256, help="--tuned: worlds of the MuJoCo C calibration")
    ap.add_argument("--xla_triton", default=None, choices=["default", "on"], help="default: on for --tuned")
    ap.add_argument("--tolerance", type=float, default=None, help="solver tolerance (default: per mode)")
    ap.add_argument("--no_checkpoint", action="store_true", help="--tuned (b): keep every substep's residuals")
    a = C.check_args(ap.parse_args())
    variant = a.grad_variant or ("both" if a.tuned else "iter1")
    if a.as_published and (variant != "iter1" or a.no_caps or a.max_geom_pairs is not None
                           or a.max_contact_points is not None or a.no_checkpoint):
        raise SystemExit("bench_mjx: --as_published is stock MJX: --grad_variant iter1 only, no caps, no checkpoint options")
    jobs = []
    if a.measure in ("forward", "both"):
        jobs.append(["--measure", "forward"])
    if a.measure in ("grad", "both"):
        jobs += [["--measure", "grad", "--grad_variant", v] for v in (("iter1", "scan") if variant == "both" else (variant,))]
    if len(jobs) > 1:
        sys.exit(C.respawn(__file__, sys.argv[1:], jobs, drop=("--measure", "--grad_variant")))
    grad = a.measure == "grad"

    triton = a.xla_triton or ("on" if a.tuned else "default")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if triton == "on" and "xla_gpu_triton_gemm_any" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=true").strip()
    if a.smoke_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx
    from mujoco.mjx._src import collision_driver as mjx_cd
    from mujoco.mjx._src import solver as mjx_solver

    real_jax = mjx_solver.jax
    dev = jax.devices()[0]
    if dev.platform != "gpu" and not a.smoke_cpu:
        raise RuntimeError(f"JAX runs on {dev.platform}: bench_mjx needs a CUDA device (or --smoke_cpu)")
    mjm, meta = C.load_model()
    stored = C.model_options(mjm)
    REF_IT = int(a.iterations) if a.iterations is not None else MATCHED_ITERATIONS  # the cap of (a) and of `scan`
    IT = 1 if (grad and variant == "iter1") else REF_IT
    LS = int(a.ls_iterations) if a.ls_iterations is not None else stored["ls_iterations"]
    TOL = float(a.tolerance) if a.tolerance is not None else (TUNED_TOLERANCE if a.tuned else stored["tolerance"])
    mjm.opt.iterations, mjm.opt.ls_iterations, mjm.opt.tolerance = IT, LS, TOL
    N, H, SEG, S = a.nworld, a.horizon, a.segment_steps, C.SUBSTEPS
    nseg = a.steps // SEG
    wl = C.workload(mjm, N, SEG, a.seed)

    # ---- the pair groups MJX collides (its own grouping), for the calibration and the overflow count
    groups = {k: np.asarray(v) for k, v in mjx_cd._geom_groups(mjm).items()}
    cullable = [(k, v[:, 0], v[:, 1]) for k, v in groups.items() if not (set(k.types) & mjx_cd._GEOM_NO_BROADPHASE)]
    gdesc = ";".join(f"{mujoco.mjtGeom(k.types[0]).name[7:]}-{mujoco.mjtGeom(k.types[1]).name[7:]}/condim{k.condim}:{len(v)}"
                     for k, v in groups.items())

    # ---- contact caps (PATCH B)
    caps, calib_note = None, "every pair padded statically: cannot overflow"
    if a.tuned and not a.no_caps:
        if a.max_geom_pairs is not None or a.max_contact_points is not None:
            if a.max_geom_pairs is None or a.max_contact_points is None:
                raise SystemExit("give both --max_geom_pairs and --max_contact_points")
            caps = (a.max_geom_pairs, a.max_contact_points)
            calib_note = "max_geom_pairs / max_contact_points given on the command line"
        else:
            t0 = time.perf_counter()
            mjm.opt.iterations = REF_IT  # the calibration always solves with the cap of the forward measurement
            mc = _calibrate(mujoco, np, mjm, wl, cullable, min(N, a.calib_worlds), SEG, S)
            mjm.opt.iterations = IT
            caps = (int(np.ceil(1.25 * mc["pairs"])) + 2, int(np.ceil(1.25 * max(mc["contacts"].values()))) + 2)
            calib_note = (f"calibrated with MuJoCo C on {min(N, a.calib_worlds)} worlds in {time.perf_counter() - t0:.0f} s: at "
                          f"most {mc['pairs']} overlapping pairs per group, contacts per condim {mc['contacts']}")
        orig_numeric = mjx_cd._numeric
        table = {"max_geom_pairs": caps[0], "max_contact_points": caps[1]}
        mjx_cd._numeric = lambda m, name: table[name] if name in table else orig_numeric(m, name)  # PATCH B

    mx = mjx.put_model(mjm, impl="jax")
    d0 = mjx.make_data(mx)
    ncon_pad, nefc_pad = int(d0._impl.contact.dist.shape[0]), int(d0._impl.efc_J.shape[0])
    cdim = np.asarray(d0._impl.contact.dim)
    put = lambda x: jax.device_put(jnp.asarray(x), dev)  # noqa: E731
    q0, v0, w0, U, qref = put(wl["qpos0"]), put(wl["qvel0"]), put(wl["warm0"]), put(wl["targets"]), put(wl["q_ref"])
    rbound = jnp.asarray(np.asarray(mjm.geom_rbound), jnp.float32)

    # ---- one control step of one world
    def substep_state(q, v, w, u):
        d = mjx.step(mx, d0.replace(qpos=q, qvel=v, qacc_warmstart=w, ctrl=u))
        return d.qpos, d.qvel, d.qacc_warmstart

    def control_step_published(q, v, w, u):  # the Data is the loop state (MuJoCo Playground's pattern)
        d = d0.replace(qpos=q, qvel=v, qacc_warmstart=w, ctrl=u)
        d = jax.lax.scan(lambda d, _: (mjx.step(mx, d), None), d, (), S)[0]
        return d.qpos, d.qvel, d.qacc_warmstart

    def control_step_state(q, v, w, u):      # state-only loop state
        return jax.lax.scan(lambda s, _: (substep_state(*s, u), None), (q, v, w), (), S)[0]

    step_pub, step_state = jax.vmap(control_step_published), jax.vmap(control_step_state)

    def segment(q, v, w, U):
        return jax.lax.scan(lambda s, u: (step_state(*s, u), None), (q, v, w), U)[0]

    # ---- (c): statistics of a rollout with the model of the measurement (not timed)
    def stats_rollout(q, v, w, U):
        """Per world over the rollout: the summed and the largest solver iterations of a substep, the most active
        contacts per condim (1, 3), the most overlapping pairs of any cullable group; and the final state."""
        def sub(q, v, w, u):
            niter.clear()
            d = mjx.step(mx, d0.replace(qpos=q, qvel=v, qacc_warmstart=w, ctrl=u))
            if IT == 1:  # no loop in the solver: one iteration
                it = jnp.ones((), jnp.int32)
            else:
                assert len(niter) == 1, f"expected one solver loop per step, saw {len(niter)}"
                it = niter[0].astype(jnp.int32)
            act = d._impl.contact.dist < 0
            a1, a3 = jnp.sum(act & (cdim == 1)), jnp.sum(act & (cdim == 3))
            x = d.geom_xpos
            cand = jnp.zeros((), jnp.int32)
            for _k, g1, g2 in cullable:
                dist = jnp.linalg.norm(x[g2] - x[g1], axis=-1) - (rbound[g1] + rbound[g2])
                cand = jnp.maximum(cand, jnp.sum(dist < 0).astype(jnp.int32))
            return (d.qpos, d.qvel, d.qacc_warmstart), (it, a1.astype(jnp.int32), a3.astype(jnp.int32), cand)

        def ctrl(carry, u):
            s, acc = carry

            def body(c, _):
                s, acc = c
                s, (it, a1, a3, cand) = jax.vmap(sub)(*s, u)
                acc = (acc[0] + it, jnp.maximum(acc[1], it), jnp.maximum(acc[2], a1), jnp.maximum(acc[3], a3),
                       jnp.maximum(acc[4], cand))
                return (s, acc), None
            return jax.lax.scan(body, (s, acc), (), S)[0], None
        z = jnp.zeros((q.shape[0],), jnp.int32)
        (s, acc), _ = jax.lax.scan(ctrl, ((q, v, w), (z, z, z, z, z)), U)
        return s, acc

    niter = []  # OBSERVER: the solver contexts' iteration counts seen while stats_rollout is traced

    def _observe(cond_fun, body_fun, init_val):
        out = real_jax.lax.while_loop(cond_fun, body_fun, init_val)
        niter.append(out.solver_niter)
        return out

    def settings(Useg, final=None):
        """The (c) columns from an untimed statistics rollout over the targets Useg; final: the (qpos, qvel) a timed
        program ended with (used for fallen_worlds / finite when the statistics rollout does not fit in memory)."""
        mjx_solver.jax = _proxy(real_jax, _observe)
        try:
            s, acc = jax.jit(stats_rollout)(q0, v0, w0, Useg)
        except Exception as e:  # noqa: BLE001
            if not C.is_oom(e) or final is None:
                raise
            fq, fv = np.asarray(final[0]), np.asarray(final[1])
            return dict(overflow_worlds="", iter_limit_worlds="the statistics rollout ran out of memory", solver_iter_mean="",
                        contacts_max="", rows_max="", fallen_worlds=int((fq[:, 2] < C.FALL_Z).sum()),
                        finite=bool(np.isfinite(fq).all() and np.isfinite(fv).all()))
        finally:
            mjx_solver.jax = real_jax
        fq, fv = np.asarray(s[0]), np.asarray(s[1])
        it_sum, it_max, a1, a3, cand = (np.asarray(x) for x in acc)
        nsub = int(Useg.shape[0]) * S
        over = 0
        if caps is not None:
            over = int(((cand > caps[0]) | (a1 >= caps[1]) | (a3 >= caps[1])).sum())
        return dict(overflow_worlds=over,
                    iter_limit_worlds=f"{int((it_max >= IT).sum())}/{N} worlds reached the cap in some substep",
                    solver_iter_mean=round(float(it_sum.mean()) / nsub, 3),
                    contacts_max=f"{int((a1 + a3).max())} active (condim1 {int(a1.max())}, condim3 {int(a3.max())}); "
                                 f"overlapping pairs per group max {int(cand.max())}",
                    rows_max=f"{int((a1 + 4 * a3).max())} active contact rows",
                    fallen_worlds=int((fq[:, 2] < C.FALL_Z).sum()),
                    finite=bool(np.isfinite(fq).all() and np.isfinite(fv).all()))

    def aot(fn, *args):
        t = time.perf_counter()
        c = jax.jit(fn).lower(*args).compile()
        return c, time.perf_counter() - t

    def peak():
        st = dev.memory_stats() or {}
        return round(st["peak_bytes_in_use"] / 2 ** 20, 1) if "peak_bytes_in_use" in st else ""

    import importlib.metadata as md
    base = C.base_record(a, SIM, mjm)
    knobs = [f"impl=jax;XLA_PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']};XLA_FLAGS={os.environ.get('XLA_FLAGS', '') or 'default'}",
             f"contact_caps={'G%d,P%d' % caps if caps else 'none'}"]
    base.update(sim_version=md.version("mujoco-mjx"), sim_commit="", iterations=IT, ls_iterations=LS, tolerance=f"{TOL:.3g}",
                capacity_contacts=f"{ncon_pad} padded per world", capacity_rows=f"{nefc_pad} padded per world",
                capacity_note=calib_note, packages=C.packages(("jax", "jaxlib", "jax-cuda12-plugin", "mujoco", "numpy")),
                notes=f"device {dev}; stored options {stored['iterations']}/{stored['ls_iterations']}/{stored['tolerance']:.0e}; "
                      f"pair groups {gdesc}; keyframe {wl['keyframe']}"
                      + ("; CPU SMOKE TEST: timings meaningless" if a.smoke_cpu else ""))
    out = C.Csv(a.csv, base)

    if not grad:
        try:
            if a.tuned:
                prog, comp = aot(segment, q0, v0, w0, U)

                def repeat():
                    t = time.perf_counter()
                    outs = [prog(q0, v0, w0, U) for _ in range(nseg)]
                    jax.block_until_ready(outs)
                    return time.perf_counter() - t, outs[-1]
            else:
                prog, comp = aot(step_pub, q0, v0, w0, U[0])
                us = [U[k] for k in range(SEG)]

                def repeat():
                    t = time.perf_counter()
                    for _ in range(nseg):
                        s = (q0, v0, w0)
                        for k in range(SEG):
                            s = prog(*s, us[k])
                    jax.block_until_ready(s)
                    return time.perf_counter() - t, s
            repeat()  # warm-up
            times, final = [], None
            for _ in range(a.repeats):
                dt, final = repeat()
                times.append(dt)
            mem = dict(peak_mem_framework_mib=peak(), peak_mem_process_mib=C.process_gpu_mem_mib())
            sps = [N * a.steps / t for t in times]
            out.row(measure="forward", status="ok", env_steps_per_s=round(C.median(sps), 1),
                    ms_per_control_step=round(1e3 * C.median(times) / a.steps, 5), repeat_values=C.fmt(sps, 1),
                    repeat_unit="env_steps_per_s", compile_s=round(comp, 2),
                    knobs=";".join(knobs + ["fused_segment_scan;state_only_carry" if a.tuned else
                                            "one_dispatch_per_control_step;data_carry"]), **mem, **settings(U, final))
        except Exception as e:  # noqa: BLE001
            if not C.is_oom(e):
                raise
            out.row(measure="forward", status="oom", notes=str(e).split("\n")[0][:300])
        return

    # ---- (b)
    if variant == "iter1":
        alternative = (f"ALTERNATIVE iter1: stock MJX, opt.iterations = 1 instead of {REF_IT} (reverse mode needs it: "
                    f"{ISSUE}); other physics, unstable on this model")
    else:
        alternative = (f"ALTERNATIVE scan: PATCH A, the solver's outer while_loop replaced by its own fixed-length scan "
                    f"of {IT} iterations ({ISSUE})")
    ckpt = a.tuned and not a.no_checkpoint

    def loss_tuned(U, q, v, w):
        sub = jax.vmap(substep_state)
        sub = jax.checkpoint(sub) if ckpt else sub

        def ctrl(s, u):
            s = jax.lax.scan(lambda s, _: (sub(*s, u), None), s, (), S)[0]
            return s, 0.5 * jnp.sum((s[0][:, 7:] - qref) ** 2) + 0.5 * C.W_V * jnp.sum(s[1] ** 2)
        s, terms = jax.lax.scan(ctrl, (q, v, w), U)
        return jnp.sum(terms) / N, (s[0], s[1])  # the loss; the final state (auxiliary output, not differentiated)

    def loss_published(U, q, v, w):
        def ctrl(s, u):
            s = step_pub(*s, u)
            return s, 0.5 * jnp.sum((s[0][:, 7:] - qref) ** 2) + 0.5 * C.W_V * jnp.sum(s[1] ** 2)
        s, terms = jax.lax.scan(ctrl, (q, v, w), U)
        return jnp.sum(terms) / N, (s[0], s[1])

    loss_fn = loss_tuned if a.tuned else loss_published
    Uh = U[:H]
    try:
        if variant == "scan":  # PATCH A, while the gradient program is traced
            mjx_solver.jax = _proxy(real_jax, lambda c, b, i: mjx_solver._while_loop_scan(c, b, i, IT))
        try:
            prog, comp = aot(jax.value_and_grad(loss_fn, has_aux=True), Uh, q0, v0, w0)
        finally:
            if variant == "scan":
                mjx_solver.jax = real_jax
        jax.block_until_ready(prog(Uh, q0, v0, w0))  # warm-up
        times, res = [], None
        for _ in range(a.repeats):
            t = time.perf_counter()
            res = prog(Uh, q0, v0, w0)
            jax.block_until_ready(res)
            times.append(time.perf_counter() - t)
        mem = dict(peak_mem_framework_mib=peak(), peak_mem_process_mib=C.process_gpu_mem_mib())
        (loss_value, final), g = res[0], np.asarray(res[1])
        ms = [1e3 * t / H for t in times]
        st = settings(Uh, final)
        st["finite"] = bool(st["finite"] and np.isfinite(g).all())
        if variant == "iter1":  # how far one iteration is from the forward measurement's solve: the same rollout's loss
            mjm.opt.iterations = REF_IT
            mx_ref = mjx.put_model(mjm, impl="jax")
            mjm.opt.iterations = IT

            def loss_ref(U, q, v, w):
                def one(q, v, w, u):
                    d = mjx.step(mx_ref, d0.replace(qpos=q, qvel=v, qacc_warmstart=w, ctrl=u))
                    return d.qpos, d.qvel, d.qacc_warmstart

                def ctrl(s, u):
                    s = jax.lax.scan(lambda s, _: (jax.vmap(one)(*s, u), None), s, (), S)[0]
                    return s, 0.5 * jnp.sum((s[0][:, 7:] - qref) ** 2) + 0.5 * C.W_V * jnp.sum(s[1] ** 2)
                return jnp.sum(jax.lax.scan(ctrl, (q, v, w), U)[1]) / N
            try:
                alternative += (f"; the same rollout's loss with a cap of {REF_IT} iterations (forward only): "
                             f"{float(jax.jit(loss_ref)(Uh, q0, v0, w0)):.6g}")
            except Exception as e:  # noqa: BLE001
                if not C.is_oom(e):
                    raise
                alternative += "; the matched-iterations reference loss did not fit in memory"
        out.row(measure="grad", status="ok", env_steps_per_s=round(N / (1e-3 * C.median(ms)), 1),
                ms_per_control_step=round(C.median(ms), 5), repeat_values=C.fmt(ms, 5), repeat_unit="ms_per_control_step",
                compile_s=round(comp, 2), loss=float(loss_value), grad_abs_mean=float(np.abs(g).mean()),
                knobs=";".join(knobs + [f"grad_variant={variant}", "checkpoint_per_substep;state_only_carry" if ckpt else
                                        ("no_checkpoint;state_only_carry" if a.tuned else "no_checkpoint;data_carry")]),
                notes=base["notes"] + "; " + alternative, **mem, **st)
        if a.smoke_cpu:  # development check: central differences of the loss in a few target entries
            f = jax.jit(lambda *x: loss_fn(*x)[0])
            Un, eps = np.asarray(Uh), 2e-3
            for (t, w, j) in ((0, 0, 16), (H - 1, N - 1, 20), (H // 2, 0, 3)):
                up, dn = Un.copy(), Un.copy()
                up[t, w, j] += eps
                dn[t, w, j] -= eps
                fd = (float(f(put(up), q0, v0, w0)) - float(f(put(dn), q0, v0, w0))) / (2 * eps)
                C.say(f"FD check target[{t},{w},{j}]: reverse mode {g[t, w, j]:+.6e}  finite difference {fd:+.6e}")
    except Exception as e:  # noqa: BLE001
        if not C.is_oom(e):
            raise
        out.row(measure="grad", status="oom", knobs=";".join(knobs + [f"grad_variant={variant}"]),
                notes=alternative + "; " + str(e).split("\n")[0][:300])


def _proxy(real_jax, while_loop):
    """A stand-in for the name `jax` inside mujoco.mjx._src.solver: everything is the real module's, except
    lax.while_loop."""
    class _Lax:
        def __getattr__(self, name):
            return getattr(real_jax.lax, name)

    class _Jax:
        def __getattr__(self, name):
            return getattr(real_jax, name)
    lax, j = _Lax(), _Jax()
    lax.while_loop = while_loop
    j.lax = lax
    return j


def _calibrate(mujoco, np, mjm, wl, cullable, nworld, seg, substeps):
    """MuJoCo C (float64) on the first nworld worlds of the workload: per substep, the most pairs of one cullable group
    whose bounding spheres overlap (MJX's cull criterion) and the most contacts per condim."""
    d = mujoco.MjData(mjm)
    rb = np.asarray(mjm.geom_rbound)
    pairs, contacts = 0, {1: 0, 3: 0}
    for w in range(nworld):
        mujoco.mj_resetData(mjm, d)
        d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = wl["qpos0"][w], wl["qvel0"][w], wl["warm0"][w]
        for t in range(seg):
            d.ctrl[:] = wl["targets"][t, w]
            for _ in range(substeps):
                mujoco.mj_step(mjm, d)  # d.contact and d.geom_xpos are those of the state the step started from
                dims = d.contact.dim[:d.ncon]
                for c in (1, 3):
                    contacts[c] = max(contacts[c], int((dims == c).sum()))
                if d.ncon and not np.isin(dims, (1, 3)).all():
                    raise RuntimeError("the calibration expects contacts of condim 1 and 3 only")
                x = d.geom_xpos
                for _k, g1, g2 in cullable:
                    dist = np.linalg.norm(x[g2] - x[g1], axis=1) - (rb[g1] + rb[g2])
                    pairs = max(pairs, int((dist < 0).sum()))
    return dict(pairs=pairs, contacts=contacts)


if __name__ == "__main__":
    main()
