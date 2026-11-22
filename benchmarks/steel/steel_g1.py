"""Our G1 (LAFAN1 tracking) in Steel for benchmarks/steel/results.md: the model, the replay of recorded windows for the
accuracy rows and the timed control step for the speed rows.

Runs in the Steel environment (Skild's Newton fork, Warp and NumPy; no torch, no ndm_gym). The windows and the G1's
constants come from ``python -m ndm_gym.wmbench.replay_engine export`` in the main environment, and the accuracy rollouts
go back to its ``score``.

The model: g1_29dof.xml through Newton's MJCF importer, self-collisions off, mesh colliders as convex hulls (as MuJoCo
collides meshes), friction 1.0 on every shape (the recording's evaluation conditions), joint limits on; per joint the drive
gains, armature and effort limit the export carries (ndm_gym/robots/g1.py). States convert between our layout (pelvis
position, quaternion wxyz, world velocity of the pelvis frame's origin, world angular velocity, joints in G1_JOINT_NAMES
order) and Newton's (the free joint's xyzw quaternion and its linear velocity referenced at the pelvis's center of mass).

A control step (20 ms) copies the step's PD targets (default pose + recorded joint offsets + action scale x setpoint) into
``Control.joint_target_q`` and replays one CUDA graph of collision and --substeps Steel steps (collision before every step,
or once per control step with --collide control). Modes:

  accuracy  one world per window, in chunks of --chunk worlds: --len control steps from each window's first state with its
            recorded setpoints; writes the states after every step (n, len, 71) in our layout, and per chunk whether a row or
            a contact was dropped (check_constraint_capacity) with the watermarks.
  speed     N worlds replaying the windows as wmbench/speed_pair.py --replay does (segments of --len steps, robot i taking
            window (i + k N) mod n in segment k; the restarts untimed): 20 warm-up control steps and --warm_seconds, then
            --steps timed control steps, once per process; appends a row to --out. A dropped row or contact makes the row
            invalid (and the exit status 3).

Run with the Steel environment's Python (benchmarks/steel/results.md, Reproducing): ``python -I steel_g1.py --mode speed
--windows W.npz --mjcf data/g1_description/g1_29dof.xml --worlds 4096 --substeps 4 --iterations 12 --out steel.csv``.
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode, JointType

STATE_DIM, NUM_J, CONTROL_DT = 71, 29, 0.02


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4) in w, x, y, z order."""
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack((np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
                     np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
                     np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)  # fmt: skip


def parse_opt(items: list[str]) -> dict:
    """--opt key=value pairs as SolverSteel keyword arguments (values read as JSON: true, 0.1, "rows")."""
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)  # Newton's xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def merged_parts(mjcf: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """ndm_gym.sim.rigid.steel_sim.merged_parts in NumPy (this environment has no ndm_gym): the bodies that merging fixed-joint
    links grows, each as the composite of its unmerged parts (mass, centre of mass, inertia about it; body frame, fp64). Newton's
    merge at the pin adds a part's inertia with its frame origin's offset, not its centre of mass's (newton-physics/newton#5280)."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.mu = 1.0
    b.add_mjcf(mjcf, enable_self_collisions=False)
    names = [lab.split("/")[-1] for lab in b.body_label]
    incoming = {int(b.joint_child[j]): j for j in range(len(b.joint_type))}
    groups: dict[int, list] = {}
    for body in range(len(names)):
        R, p, root = np.eye(3), np.zeros(3), body
        while root in incoming and JointType(int(b.joint_type[incoming[root]])) == JointType.FIXED:
            j = incoming[root]
            Xp, Xc = b.joint_X_p[j], b.joint_X_c[j]
            Rj = _quat_mat(Xp[3:7]) @ _quat_mat(Xc[3:7]).T  # the child's frame in its parent's: X_p inv(X_c)
            pj = np.array([float(v) for v in Xp[0:3]]) - Rj @ np.array([float(v) for v in Xc[0:3]])
            R, p, root = Rj @ R, Rj @ p + pj, int(b.joint_parent[j])
        com = np.array([float(v) for v in b.body_com[body]])
        inertia = np.array([[float(v) for v in r] for r in b.body_inertia[body]])
        groups.setdefault(root, []).append((float(b.body_mass[body]), R @ com + p, R @ inertia @ R.T))
    out = {}
    for root, parts in groups.items():
        if len(parts) > 1:
            M = sum(m for m, _, _ in parts)
            C = sum(m * c for m, c, _ in parts) / M
            out[names[root]] = (M, C, sum(ip + m * ((c - C) @ (c - C) * np.eye(3) - np.outer(c - C, c - C)) for m, c, ip in parts))
    return out


class G1Steel:
    """``worlds`` copies of our G1 on a ground plane with SolverSteel, a captured control step and the state conversions;
    ``merge_fix`` (with ``collapse_fixed``): each merged body's inertia the composite of its parts (merged_parts), set before the
    solver is built."""

    def __init__(self, mjcf: str, consts: dict, worlds: int, device: str, substeps: int, solver_opts: dict, collide: str = "step",
                 collapse_fixed: bool = False, contact_max: int | None = None, graph: bool = True, engine: str = "steel",
                 deterministic: bool = False, merge_fix: bool = False):
        one = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(one)  # the MJCF's MuJoCo attributes, for either engine's model
        one.default_shape_cfg.mu = 1.0
        one.add_mjcf(mjcf, enable_self_collisions=False, collapse_fixed_joints=collapse_fixed)
        names = [lab.split("/")[-1] for lab in one.joint_label]
        if JointType(int(one.joint_type[0])) != JointType.FREE:
            raise ValueError(f"{mjcf}: the first joint must be the pelvis's free joint")
        want = [str(n) for n in consts["joint_names"]]
        missing = [n for n in want if n not in names]
        if missing:
            raise ValueError(f"{mjcf}: no joints {missing}")
        jid = [names.index(n) for n in want]
        self.q_idx = np.array([one.joint_q_start[j] for j in jid])  # our joint order -> Newton coordinates of a world
        self.qd_idx = np.array([one.joint_qd_start[j] for j in jid])
        for k, j in enumerate(jid):
            if JointType(int(one.joint_type[j])) != JointType.REVOLUTE:
                raise ValueError(f"{want[k]} is not a hinge")
            d = one.joint_qd_start[j]
            one.joint_target_ke[d] = float(consts["kp"][k])
            one.joint_target_kd[d] = float(consts["kd"][k])
            one.joint_armature[d] = float(consts["armature"][k])
            one.joint_effort_limit[d] = float(consts["effort"][k])
            one.joint_target_mode[d] = int(JointTargetMode.POSITION)
        one.approximate_meshes("convex_hull")
        self.nq, self.nqd = one.joint_coord_count, one.joint_dof_count
        self.com = np.array(one.body_com[one.joint_child[0]], dtype=np.float64)  # the pelvis's center of mass, its frame
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.0
        builder.replicate(one, worlds)
        builder.add_ground_plane()
        self.N, self.device, self.S, self.engine = worlds, device, substeps, engine
        self.collide = collide if engine == "steel" else "none"  # MuJoCo Warp collides itself (use_mujoco_contacts)
        self.dt = CONTROL_DT / substeps
        self.model = builder.finalize(device=device)
        if merge_fix:
            if not collapse_fixed:
                raise ValueError("--merge_fix corrects merged links' inertia: with --collapse_fixed")
            bodies = [lab.split("/")[-1] for lab in one.body_label]
            inertia, inv = self.model.body_inertia.numpy(), self.model.body_inv_inertia.numpy()
            for name, (M, C, I_parts) in merged_parts(mjcf).items():
                k = bodies.index(name)
                if abs(M - float(one.body_mass[k])) > 1e-6 * M or np.abs(C - np.array(one.body_com[k], dtype=np.float64)).max() > 1e-6:
                    raise RuntimeError(f"{name}: the parts' mass and centre of mass are not Newton's merge's")
                ids = np.arange(worlds) * len(bodies) + k
                inertia[ids], inv[ids] = I_parts.astype(np.float32), np.linalg.inv(I_parts).astype(np.float32)
            self.model.body_inertia.assign(inertia)
            self.model.body_inv_inertia.assign(inv)
        if engine == "steel":
            det = {"deterministic": wp.DeterministicMode.RUN_TO_RUN} if deterministic else {}
            self.solver = newton.solvers.SolverSteel(self.model, **solver_opts, **det)
        else:
            self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=True, **solver_opts)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=contact_max, deterministic=deterministic)
        self.contacts = self.pipeline.contacts()
        self.s0, self.s1, self.control = self.model.state(), self.model.state(), self.model.control()
        self.default_q = np.asarray(consts["default_q"], np.float64)
        self.alpha = np.asarray(consts["alpha"], np.float64)
        self.all_worlds = wp.array(np.ones(worlds + 1, dtype=bool), dtype=wp.bool, device=device)
        self.targets = None  # a segment's targets, (len x N x nq) on the device
        self.graph = None
        if graph:
            self.physics()  # loads every module before the capture
            with wp.ScopedCapture(device=device) as cap:
                self.physics()
            self.graph = cap.graph

    # ---- states
    def to_newton(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Our states (n, 71) -> (joint_q (n, nq), joint_qd (n, nqd)) of n worlds."""
        p, quat, v, w = s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13]
        R = quat_to_mat(quat.astype(np.float64))
        jq, jqd = np.zeros((len(s), self.nq)), np.zeros((len(s), self.nqd))
        jq[:, 0:3], jq[:, 3:7] = p, quat[:, [1, 2, 3, 0]]
        jq[:, self.q_idx] = s[:, 13:13 + NUM_J]
        jqd[:, 0:3], jqd[:, 3:6] = v + np.cross(w, R @ self.com), w  # the free joint's velocity at the center of mass
        jqd[:, self.qd_idx] = s[:, 13 + NUM_J:]
        return jq.astype(np.float32), jqd.astype(np.float32)

    def from_newton(self, jq: np.ndarray, jqd: np.ndarray) -> np.ndarray:
        """(joint_q (n, nq), joint_qd (n, nqd)) -> our states (n, 71)."""
        quat = jq[:, [6, 3, 4, 5]].astype(np.float64)
        R = quat_to_mat(quat / np.linalg.norm(quat, axis=-1, keepdims=True))
        w = jqd[:, 3:6].astype(np.float64)
        v = jqd[:, 0:3] - np.cross(w, R @ self.com)
        return np.concatenate((jq[:, 0:3], quat, v, w, jq[:, self.q_idx], jqd[:, self.qd_idx]), -1).astype(np.float32)

    def read(self) -> np.ndarray:
        n = self.N
        return self.from_newton(self.s0.joint_q.numpy().reshape(n, self.nq), self.s0.joint_qd.numpy().reshape(n, self.nqd))

    def restart(self, s: np.ndarray, a: np.ndarray, xi: np.ndarray):
        """Every world from our state s (N, 71), with the setpoints a (N, len, 29) and contexts xi (N, 33) of its window:
        the state written and kinematics evaluated, the solver's history of every world cleared, the targets staged."""
        jq, jqd = self.to_newton(s)
        self.s0.joint_q.assign(jq.reshape(-1))
        self.s0.joint_qd.assign(jqd.reshape(-1))
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        if self.engine == "steel":
            self.solver.reset(self.s0, world_mask=self.all_worlds)
        L = a.shape[1]
        tq = np.zeros((L, self.N, self.nq), np.float32)
        tq[:, :, self.q_idx] = (self.default_q + xi[:, 1:1 + NUM_J])[None] + self.alpha * a.transpose(1, 0, 2)
        self.targets = wp.array(tq.reshape(-1), dtype=wp.float32, device=self.device)

    # ---- the control step
    def physics(self):
        a, b = self.s0, self.s1
        if self.collide == "control":
            self.pipeline.collide(a, self.contacts)
        for k in range(self.S):
            if self.collide == "step":
                self.pipeline.collide(a, self.contacts)
            a.clear_forces()
            self.solver.step(a, b, self.control, self.contacts, self.dt)
            if self.S % 2 == 1 and k == self.S - 1:
                a.assign(b)  # an odd count ends in s1: copy it back, so that the state is s0 after every control step
            else:
                a, b = b, a

    def step(self, j: int):
        """Control step j of the staged segment: its targets, then the physics."""
        n = self.N * self.nq
        wp.copy(self.control.joint_target_q, self.targets, dest_offset=0, src_offset=j * n, count=n)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.physics()

    def capacity(self) -> tuple[bool, str]:
        """(dropped, message): whether any world dropped a row or a contact since the last full reset (Steel only)."""
        if self.engine != "steel":
            return False, ""
        try:
            self.solver.check_constraint_capacity()
        except RuntimeError as e:
            return True, str(e).split(". ")[0]
        return False, ""


def barrier(path: str, n: int, lead: float = 2.0) -> float:
    """--barrier: as wmbench/speed_pair.py's (this environment has no ndm_gym): ready, wait for n, then the common start, written once."""
    import threading

    os.makedirs(path, exist_ok=True)
    me = f"{os.getpid()}.{threading.get_ident()}"
    open(os.path.join(path, f"ready.{me}"), "w").close()
    go = os.path.join(path, "go")
    while not os.path.exists(go):
        if len([f for f in os.listdir(path) if f.startswith("ready.")]) >= n and not os.path.exists(go):
            tmp = os.path.join(path, f"go.{me}")
            with open(tmp, "w") as f:
                f.write(f"{time.time() + lead:.3f}")
            try:  # one start for all: a link fails if another caller made go first
                os.link(tmp, go)
            except FileExistsError:
                pass
            os.remove(tmp)
            break
        time.sleep(0.01)
    t_go = float(open(go).read())
    while time.time() < t_go:
        time.sleep(0.001)
    return t_go


def gpu_info() -> tuple[str, str]:
    """The GPU's name and the driver version ($NDM_GPU_DRIVER when set: a run that starts no nvidia-smi while others time)."""
    dev = wp.get_device()
    if not dev.is_cuda:
        return dev.name, ""
    if os.environ.get("NDM_GPU_DRIVER"):
        return dev.name, os.environ["NDM_GPU_DRIVER"]
    uuid = dev.uuid if hasattr(dev, "uuid") else None
    q = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"] + (["-i", str(uuid)] if uuid else [])
    return dev.name, subprocess.run(q, capture_output=True, text=True).stdout.strip().splitlines()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["accuracy", "speed"])
    p.add_argument("--engine", default="steel", choices=["steel", "mujoco"],
                   help="mujoco: Newton's SolverMuJoCo (MuJoCo Warp) on the same Newton model, with MuJoCo's contacts")
    p.add_argument("--mj", nargs="*", default=["solver=\"newton\"", "integrator=\"implicitfast\"", "cone=\"pyramidal\"", "iterations=100",
                                               "ls_iterations=50", "njmax=500", "nconmax=128"],
                   help="--engine mujoco: SolverMuJoCo options, key=value (default: LAFAN1's MJWarpSolverCfg with Isaac Lab's defaults)")
    p.add_argument("--windows", required=True, help="replay_engine export's .npz")
    p.add_argument("--mjcf", required=True, help="g1_29dof.xml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--substeps", type=int, default=4, help="Steel steps per 20 ms control step")
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--art_max", type=int, default=256, help="articulation_constraint_max (rows per world)")
    p.add_argument("--collide", default="step", choices=["step", "control"])
    p.add_argument("--collapse_fixed", action="store_true", help="merge fixed-joint links at import")
    p.add_argument("--merge_fix", action="store_true",
                   help="with --collapse_fixed: each merged body's inertia the composite of its parts (off: Newton's merge)")
    p.add_argument("--contact_max", type=int, default=None, help="the collision pipeline's rigid_contact_max (default: Newton's)")
    p.add_argument("--opt", nargs="*", default=[], help="more SolverSteel options, key=value")
    p.add_argument("--watermarks", action="store_true", help="row_watermark=True (costs time: never on timed rows)")
    p.add_argument("--no_graph", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="Steel's RUN_TO_RUN determinism and a deterministic collision pipeline")
    p.add_argument("--chunk", type=int, default=4096, help="accuracy: worlds per chunk")
    p.add_argument("--worlds", type=int, default=4096, help="speed: worlds")
    p.add_argument("--len", type=int, default=32, help="steps per window and segment")
    p.add_argument("--steps", type=int, default=200, help="speed: timed control steps")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--warm_seconds", type=float, default=1.0)
    p.add_argument("--setting", default="", help="a name for the row (run.sh: fastest-passing, task-physics, g1-preset, mujoco-newton)")
    p.add_argument("--barrier", default=None, help="speed: a folder where --barrier_n processes meet after the warm-up (multi-GPU rows)")
    p.add_argument("--barrier_n", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if (args.barrier is None) != (args.barrier_n == 0) or (args.barrier is not None and args.mode != "speed"):
        p.error("--barrier and --barrier_n go together, in the speed mode")

    wp.config.quiet = True
    wp.init()
    win = np.load(args.windows, allow_pickle=False)
    consts = {k: win[k] for k in ("joint_names", "default_q", "alpha", "kp", "kd", "armature", "effort")}
    s0, a, xi = win["s0"], win["a"][:, : args.len], win["xi"]
    n_win = len(s0)
    if args.engine == "steel":
        opts = dict(solve_mode="serial", iterations=args.iterations, articulated_contact_response="immediate",
                    articulation_constraint_max=args.art_max, enable_joint_limits=True, row_watermark=args.watermarks)  # fmt: skip
        opts.update(parse_opt(args.opt))
    else:
        opts = parse_opt(args.mj)
    common = dict(mjcf=args.mjcf, consts=consts, device=args.device, substeps=args.substeps, solver_opts=opts, collide=args.collide,
                  collapse_fixed=args.collapse_fixed, merge_fix=args.merge_fix, contact_max=args.contact_max, graph=not args.no_graph,
                  engine=args.engine, deterministic=args.deterministic)  # fmt: skip
    setting = {"engine": args.engine, "substeps": args.substeps, "iterations": args.iterations, "art_max": args.art_max, "collide": args.collide,
               "collapse_fixed": args.collapse_fixed, "merge_fix": args.merge_fix, "contact_max": args.contact_max, "opt": parse_opt(args.opt),
               "graph": not args.no_graph,
               "deterministic": args.deterministic}  # fmt: skip
    name, driver = gpu_info()
    dropped_any = False

    if args.mode == "accuracy":
        out = np.zeros((n_win, args.len, STATE_DIM), np.float32)
        chunks = []
        for c0 in range(0, n_win, args.chunk):
            sel = np.arange(c0, min(c0 + args.chunk, n_win))
            g = G1Steel(worlds=len(sel), **common)
            g.restart(s0[sel], a[sel], xi[sel])
            for j in range(args.len):
                g.step(j)
                out[sel, j] = g.read()
            dropped, msg = g.capacity()
            wm = {k: int(v) for k, v in g.solver.constraint_row_watermarks().items()} if args.watermarks and args.engine == "steel" else {}
            chunks.append({"worlds": [int(sel[0]), int(sel[-1]) + 1], "dropped": dropped, "message": msg, "watermarks": wm})
            dropped_any |= dropped
            del g
        np.savez(args.out, states=out, finite=np.isfinite(out).all(axis=(1, 2)))
        meta = {"setting": setting, "solver": opts, "chunks": chunks, "gpu": name, "driver": driver, "windows": args.windows,
                "newton": newton.__version__, "warp": wp.config.version}  # fmt: skip
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[steel_g1] accuracy {n_win} windows x {args.len} steps -> {args.out}; dropped rows or contacts: {dropped_any}", flush=True)
    else:
        N = args.worlds
        lengths = [args.len] * (args.steps // args.len) + ([args.steps % args.len] if args.steps % args.len else [])
        sched = [(np.arange(N) + k * N) % n_win for k in range(len(lengths))]
        t_build = time.perf_counter()
        g = G1Steel(worlds=N, **common)
        t_build = time.perf_counter() - t_build
        sync = wp.synchronize_device

        def restart(k):
            sel = sched[k]
            g.restart(s0[sel], a[sel], xi[sel])

        restart(0)
        for j in range(args.warmup):
            g.step(j % lengths[0])
        sync()
        t0, calls = time.perf_counter(), 0
        while time.perf_counter() - t0 < args.warm_seconds:
            g.step(0)
            calls += 1
            if calls % 16 == 0:
                sync()
        sync()
        if args.barrier:
            barrier(args.barrier, args.barrier_n)
        total = 0.0
        for k, n in enumerate(lengths):
            restart(k)
            sync()
            t0 = time.perf_counter()
            for j in range(n):
                g.step(j)
            sync()
            total += time.perf_counter() - t0
        dropped_any, msg = g.capacity()
        finite = bool(np.isfinite(g.s0.joint_q.numpy()).all())
        mem = wp.get_mempool_used_mem_high(args.device) / 2**30 if wp.get_device(args.device).is_cuda else 0.0
        rate = N * args.steps / total
        row = {"engine": args.engine, "setting": args.setting, "worlds": N, "steps": args.steps, "seconds": round(total, 5),
               "steps_per_s": round(rate, 1), "ms_per_step": round(1e3 * total / args.steps, 4), "valid": int(not dropped_any and finite),
               "gpu": name, "driver": driver, "peak_mem_gb": round(mem, 3), "build_s": round(t_build, 1),
               "detail": json.dumps({"solver": opts, "setting": setting, "segments": lengths, "windows": os.path.basename(args.windows),
                                     "warm_s": args.warm_seconds, "dropped": msg})}  # fmt: skip
        new = not os.path.exists(args.out)
        with open(args.out, "a") as f:
            if new:
                f.write(",".join(row) + "\n")
            f.write(",".join(str(v).replace(",", ";") if k != "detail" else '"' + str(v).replace('"', "'") + '"' for k, v in row.items()) + "\n")
        print(f"[steel_g1] speed {args.setting} {N} worlds: {rate:,.0f} control steps/s, {1e3 * total / args.steps:.3f} ms per step, "
              f"valid {row['valid']} {msg}", flush=True)
    if dropped_any:
        print("[steel_g1] ALARM: rows or contacts were dropped; the result is invalid (the criteria's first)", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
