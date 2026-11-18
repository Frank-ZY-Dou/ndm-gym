"""The mbrl_lib benchmark's simulator comparison: one of mbrl-lib's MuJoCo environments stepped
from identical states in mbrl-lib's own environment (MuJoCo 2.3.3, *own*), in the study's (*study*) and on MuJoCo Warp
(*Warp*), and the three compared. Four steps, each a subcommand:

    inputs  the states and controls, from a model's recordings (wmbench/collect_mbrl.py's <prefix>_s<seed>_<kind>.npz):
            <out>_transitions.npz holds every executed transition of --seeds and --rollouts candidates' transitions drawn from
            each of --rollout_seeds' rollout recordings; <out>_sequences.npz, for each seed:step of --starts, the executed
            state there and the next --horizon executed controls (the replays).
    step    the environment as mbrl-lib's handler builds it, on these inputs: every numeric field of its model and options, the
            resets of five seeds, one environment step from each transition (the solver's warm start zeroed), the replays
            stepped on, and the body at rest (200 steps under zero controls from reset(seed=0)). It needs mbrl-lib, gymnasium,
            mujoco and numpy only, so the same file runs in mbrl-lib's own environment; --own_margins and --own_mujoco build
            the study's environment through ndm_gym.sim.mbrl_envs.patch_envs.
    warp    the model the study's side saved (--save_model), on MuJoCo Warp as sim.mbrl_ndm steps it, one world per start.
    report  the comparison printed; with --threshold the exit code is 1 unless the study's side is mbrl-lib's own by this
            threshold: every transition with equal contact counts within 1e-7 in a position, other contact counts
            on fewer than 1 % of the transitions, the same terminations.

    python compare_sim.py inputs --prefix data/hop --starts 0:0 0:60 1:0 2:0 --out cmp/hop
    python compare_sim.py step --env gym___Hopper-v4 --transitions cmp/hop_transitions.npz --sequences cmp/hop_sequences.npz \\
        --out cmp/hop_own.npz                                        # in mbrl-lib's own environment
    python compare_sim.py step --env gym___Hopper-v4 --local_models data/mbrl_models --own_mujoco --save_model cmp/hop.mjb \\
        --transitions cmp/hop_transitions.npz --sequences cmp/hop_sequences.npz --out cmp/hop_study.npz
    python compare_sim.py warp --env hopper --model cmp/hop.mjb --cpu cmp/hop_study.npz --transitions cmp/hop_transitions.npz \\
        --sequences cmp/hop_sequences.npz --out cmp/hop_warp.npz
    python compare_sim.py report --own cmp/hop_own.npz --study cmp/hop_study.npz --warp cmp/hop_warp.npz \\
        --transitions cmp/hop_transitions.npz --sequences cmp/hop_sequences.npz --threshold
"""

import argparse
import os
import sys

import numpy as np

LOCAL = {"hopper.xml": "hopper_local.xml", "walker2d.xml": "walker2d_local.xml"}
KEYS = ("qpos", "qvel", "obs", "reward", "terminated", "ncon", "dist")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inputs")
    q.add_argument("--prefix", required=True, help="the recordings' prefix, e.g. data/hop")
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    q.add_argument("--rollouts", type=int, default=2000, help="candidates' transitions drawn from each of --rollout_seeds' rollout recordings")
    q.add_argument("--rollout_seeds", type=int, nargs="+", default=[0, 2])
    q.add_argument("--starts", nargs="+", required=True, help="seed:step of the executed recordings")
    q.add_argument("--horizon", type=int, default=30)
    q.add_argument("--out", required=True)
    q = sub.add_parser("step")
    q.add_argument("--env", required=True, help="mbrl-lib's name of the environment (pets_halfcheetah, gym___Hopper-v4, ...)")
    q.add_argument("--local_models", default=None, help="the directory of gymnasium's hopper.xml and walker2d.xml in local coordinates")
    q.add_argument("--own_margins", action="store_true", help="the study's environment with mbrl-lib's own MuJoCo's contact margins")
    q.add_argument("--own_mujoco", action="store_true", help="the study's environment with its model as mbrl-lib's own MuJoCo steps it")
    q.add_argument("--transitions", required=True, help=".npz with qpos (N, nq), qvel (N, nv), ctrl (N, nu)")
    q.add_argument("--sequences", required=True, help=".npz with qpos (M, nq), qvel (M, nv), ctrl (M, H, nu)")
    q.add_argument("--reset_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    q.add_argument("--rest", type=int, default=200)
    q.add_argument("--max_size", type=int, default=100000, help="the largest model array kept")
    q.add_argument("--save_model", default=None, help="the environment's MjModel as an .mjb")
    q.add_argument("--out", required=True)
    q = sub.add_parser("warp")
    q.add_argument("--env", required=True, help="the study's name of the environment (sim.mbrl_envs.ENVS)")
    q.add_argument("--model", required=True, help="the .mjb the study's side saved")
    q.add_argument("--cpu", required=True, help="the study's side's output (its frame skip and reset state)")
    q.add_argument("--transitions", required=True)
    q.add_argument("--sequences", required=True)
    q.add_argument("--rest", type=int, default=200)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("--own", required=True)
    q.add_argument("--study", required=True)
    q.add_argument("--warp", default=None)
    q.add_argument("--transitions", required=True)
    q.add_argument("--sequences", required=True)
    q.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 30])
    q.add_argument("--threshold", action="store_true", help="exit code 1 unless the study's side is mbrl-lib's own by the threshold (the module's docstring)")
    return p.parse_args(argv)


# ---------------- inputs ----------------
def inputs(args):
    q, v, u, nq, nv, ex = [], [], [], [], [], []
    executed = {}
    for s in args.seeds:
        z = np.load(f"{args.prefix}_s{s}_executed.npz")
        if z["reset"].any():
            raise ValueError(f"{args.prefix}_s{s}_executed.npz holds resets: not one trajectory")
        executed[s] = (z["qpos"][:, 0], z["qvel"][:, 0], z["ctrl"][:, 0])
        q.append(z["qpos"][:-1, 0])
        v.append(z["qvel"][:-1, 0])
        u.append(z["ctrl"][:, 0])
        nq.append(z["qpos"][1:, 0])
        nv.append(z["qvel"][1:, 0])
        ex.append(np.ones(len(z["ctrl"]), bool))
    for s in args.rollout_seeds:
        z = np.load(f"{args.prefix}_s{s}_rollout.npz")
        ok = np.argwhere(~z["reset"])
        pick = ok[np.random.default_rng(s).choice(len(ok), size=args.rollouts, replace=False)]
        t, n = pick[:, 0], pick[:, 1]
        q.append(z["qpos"][t, n])
        v.append(z["qvel"][t, n])
        u.append(z["ctrl"][t, n])
        nq.append(z["qpos"][t + 1, n])
        nv.append(z["qvel"][t + 1, n])
        ex.append(np.zeros(len(t), bool))
    np.savez(f"{args.out}_transitions.npz", qpos=np.concatenate(q), qvel=np.concatenate(v), ctrl=np.concatenate(u), next_qpos=np.concatenate(nq),
             next_qvel=np.concatenate(nv), executed=np.concatenate(ex))  # fmt: skip
    sq, sv, su, rq, rv, names = [], [], [], [], [], []
    for item in args.starts:
        s, t0 = (int(x) for x in item.split(":"))
        eq, ev, eu = executed[s]
        if t0 + args.horizon > len(eu):
            raise ValueError(f"--starts {item}: the executed recording of seed {s} has {len(eu)} steps")
        sq.append(eq[t0])
        sv.append(ev[t0])
        su.append(eu[t0 : t0 + args.horizon])
        rq.append(eq[t0 + 1 : t0 + args.horizon + 1])
        rv.append(ev[t0 + 1 : t0 + args.horizon + 1])
        names.append(item)
    np.savez(f"{args.out}_sequences.npz", qpos=np.stack(sq), qvel=np.stack(sv), ctrl=np.stack(su), rec_qpos=np.stack(rq), rec_qvel=np.stack(rv),
             names=np.array(names))  # fmt: skip
    print(f"[compare_sim inputs] {args.prefix}: {sum(len(x) for x in q)} transitions ({int(np.concatenate(ex).sum())} executed), {len(sq)} replays of "
          f"{args.horizon} steps")  # fmt: skip


# ---------------- step: mbrl-lib's own environment, or the study's ----------------
def make_env(args):
    import mbrl.util

    if args.own_margins or args.own_mujoco:  # the study's own patches, as its recorder and planner apply them
        from ndm_gym.sim.mbrl_envs import patch_envs

        patch_envs(args.local_models, own_margins=args.own_margins, own_mujoco=args.own_mujoco)
    elif args.local_models is not None:  # MuJoCo >= 2.3.4 refuses gymnasium 0.26's global-coordinate models
        from gymnasium.envs.mujoco import mujoco_env

        init = mujoco_env.MujocoEnv.__init__

        def mujoco_init(self, model_path, *a, **kw):
            if model_path in LOCAL:
                model_path = os.path.join(os.path.abspath(args.local_models), LOCAL[model_path])
            init(self, model_path, *a, **kw)

        mujoco_env.MujocoEnv.__init__ = mujoco_init
    return mbrl.util.create_handler_from_str(args.env).make_env_from_str(args.env)


def numeric(obj, prefix: str, max_size: int) -> dict:
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            x = getattr(obj, name)
        except Exception:  # noqa: BLE001  (a field this version does not expose)
            continue
        if isinstance(x, (bool, int, float)):
            out[prefix + name] = np.array(x, dtype=np.float64)
        elif isinstance(x, np.ndarray) and x.dtype.kind in "biuf" and 0 < x.size <= max_size:
            out[prefix + name] = np.array(x, dtype=np.float64)
    return out


def stepped(u, a):
    obs, reward, terminated, _, _ = u.step(a)
    d = u.data
    n = int(d.ncon)
    dist = min(float(d.contact[i].dist) for i in range(n)) if n else np.nan  # the deepest penetration, negative
    return d.qpos.copy(), d.qvel.copy(), np.asarray(obs, dtype=np.float64), float(reward), bool(terminated), n, dist


def step(args):
    import gymnasium
    import mujoco

    env = make_env(args)
    u = env.unwrapped
    m, d = u.model, u.data
    out = {**numeric(m, "model__", args.max_size), **numeric(m.opt, "opt__", args.max_size), "frame_skip": np.array(u.frame_skip),
           "versions": np.array([f"mujoco {mujoco.__version__}", f"gymnasium {gymnasium.__version__}", f"numpy {np.__version__}"])}  # fmt: skip
    if args.save_model:
        mujoco.mj_saveModel(m, args.save_model, None)
    rq, rv = [], []
    for s in args.reset_seeds:
        np.random.seed(s)  # the PETS environments reset from global numpy
        env.reset(seed=s)
        rq.append(d.qpos.copy())
        rv.append(d.qvel.copy())
    out.update(reset_qpos=np.stack(rq), reset_qvel=np.stack(rv))

    def start(q, v):
        u.set_state(np.array(q, dtype=np.float64), np.array(v, dtype=np.float64))
        d.qacc_warmstart[:] = 0

    z = np.load(args.transitions)
    rows = []
    for q, v, a in zip(z["qpos"], z["qvel"], z["ctrl"]):
        start(q, v)
        rows.append(stepped(u, a))
    for i, k in enumerate(KEYS):
        out["one_" + k] = np.stack([np.asarray(r[i]) for r in rows])
    z = np.load(args.sequences)
    seqs = []
    for q, v, acts in zip(z["qpos"], z["qvel"], z["ctrl"]):
        start(q, v)
        seqs.append([stepped(u, a) for a in acts])
    for i, k in enumerate(KEYS):
        out["seq_" + k] = np.stack([np.stack([np.asarray(r[i]) for r in s]) for s in seqs])
    np.random.seed(0)
    env.reset(seed=0)
    d.qacc_warmstart[:] = 0
    zero = np.zeros(m.nu)
    rest = [stepped(u, zero) for _ in range(args.rest)]
    for i, k in enumerate(KEYS):
        out["rest_" + k] = np.stack([np.asarray(r[i]) for r in rest])
    np.savez(args.out, **out)
    print(f"[compare_sim step] {args.env}: mujoco {mujoco.__version__}, {len(rows)} transitions, {len(seqs)} replays, {args.rest} resting steps -> {args.out}")


# ---------------- warp ----------------
def warp(args):
    import mujoco
    import mujoco_warp as mjw
    import torch
    import warp as wp

    from ndm_gym.sim.mbrl_envs import ENVS, Fields, Kinematics, Scorer

    spec = ENVS[args.env]
    mjm = mujoco.MjModel.from_binary_path(args.model)
    cpu = np.load(args.cpu)
    substeps = int(cpu["frame_skip"])
    scorer = Scorer(spec, mjm, substeps)
    derived = sorted({r.split(".")[1] for r in spec.reads} - {"qpos", "qvel"})
    if any(r.split(".")[0] != "pre" for r in spec.reads if r.split(".")[1] in derived):
        raise SystemExit(f"compare_sim warp: {spec.name}'s reward reads {sorted(spec.reads)}: only the start state's derived arrays are built here")
    kin = Kinematics(mjm, dtype=torch.float64) if derived else None  # the start state's kinematics, as set_state's mj_forward leaves them
    device = wp.get_device(args.device)

    def worlds(q, v):
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        with wp.ScopedDevice(device):
            m = mjw.put_model(mjm)
            d = mjw.put_data(mjm, mjd, nworld=len(q))
            d.qpos.assign(q.astype(np.float32))
            d.qvel.assign(v.astype(np.float32))
            d.qacc_warmstart.zero_()
        return m, d

    def one(m, d, u, pre_q):
        with wp.ScopedDevice(device):
            d.ctrl.assign(u.astype(np.float32))
            for _ in range(substeps):
                mjw.step(m, d)
            q, v = d.qpos.numpy().copy(), d.qvel.numpy().copy()
            n = int(d.nacon.numpy()[0])
            if n > d.naconmax:
                raise RuntimeError(f"compare_sim warp: {n} contacts for a capacity of {d.naconmax}")
            wid, dist = d.contact.worldid.numpy()[:n], d.contact.dist.numpy()[:n]
        ncon = np.bincount(wid, minlength=len(q)) if n else np.zeros(len(q), int)
        deepest = np.full(len(q), np.nan)
        for w, x in zip(wid, dist):
            deepest[w] = x if np.isnan(deepest[w]) else min(deepest[w], x)
        pq = torch.as_tensor(pre_q, dtype=torch.float64)
        pre = Fields(qpos=pq, **(kin(pq, names=tuple(derived)) if kin else {}))
        post = Fields(qpos=torch.as_tensor(q, dtype=torch.float64), qvel=torch.as_tensor(v, dtype=torch.float64))
        r = scorer.reward(pre, post, torch.as_tensor(u, dtype=torch.float64)).numpy()
        return q, v, r, scorer.terminated(post).numpy(), ncon, deepest

    keys = ("qpos", "qvel", "reward", "terminated", "ncon", "dist")
    out = {}
    z = np.load(args.transitions)
    m, d = worlds(z["qpos"], z["qvel"])
    for k, x in zip(keys, one(m, d, z["ctrl"], z["qpos"])):
        out["one_" + k] = x
    z = np.load(args.sequences)
    m, d = worlds(z["qpos"], z["qvel"])
    rows, pre = [], z["qpos"]
    for t in range(z["ctrl"].shape[1]):
        rows.append(one(m, d, z["ctrl"][:, t], pre))
        pre = rows[-1][0]
    for i, k in enumerate(keys):
        out["seq_" + k] = np.stack([r[i] for r in rows], 1)
    m, d = worlds(cpu["reset_qpos"][:1], cpu["reset_qvel"][:1])
    rows, pre = [], cpu["reset_qpos"][:1]
    for _ in range(args.rest):
        rows.append(one(m, d, np.zeros((1, mjm.nu)), pre))
        pre = rows[-1][0]
    for i, k in enumerate(keys):
        out["rest_" + k] = np.stack([r[i][0] for r in rows])
    out["versions"] = np.array([f"mujoco {mujoco.__version__}", f"mujoco_warp {mjw.__version__}", f"warp {wp.config.version}", f"device {device}"])
    np.savez(args.out, **out)
    print(f"[compare_sim warp] {args.env}: {len(out['one_qpos'])} transitions, {out['seq_qpos'].shape[0]} replays on {device} -> {args.out}")


# ---------------- report ----------------
def stats(x):
    x = np.abs(np.asarray(x, dtype=np.float64)).reshape(len(x), -1).max(1)
    return f"max {x.max():.3g}, 99th {np.quantile(x, 0.99):.3g}, median {np.median(x):.3g} (worst row {int(x.argmax())})"


def is_own(a, b) -> tuple[bool, str]:
    """The threshold on the one-step maps of mbrl-lib's own side ``a`` and the study's ``b``, and its line."""
    dq, dv = np.abs(a["one_qpos"] - b["one_qpos"]).max(1), np.abs(a["one_qvel"] - b["one_qvel"]).max(1)
    same = a["one_ncon"] == b["one_ncon"]
    bad, named = int(((dq > 1e-7) & same).sum()), int((~same).sum())
    term = int((a["one_terminated"].astype(bool) != b["one_terminated"].astype(bool)).sum())
    ok = bad == 0 and named < 0.01 * len(dq) and term == 0
    line = (f"{len(dq)} transitions: with equal contact counts a position is off by at most {dq[same].max() if same.any() else 0.0:.3g} (more than 1e-7 on "
            f"{bad}, more than 1e-9 on {int(((dq > 1e-9) & same).sum())}) and a velocity by {dv[same].max() if same.any() else 0.0:.3g}; contact counts "
            f"differ on {named} ({100 * named / len(dq):.2f} %), where a position is off by at most {dq[~same].max() if named else 0.0:.3g}; terminations "
            f"differ on {term}; rewards within {np.abs(a['one_reward'] - b['one_reward']).max():.3g}: {'own' if ok else 'NOT own'}")  # fmt: skip
    return ok, line


def report(args) -> bool:
    a, b = np.load(args.own), np.load(args.study)
    c = np.load(args.warp) if args.warp else None
    tr, sq = np.load(args.transitions), np.load(args.sequences)
    print("own:", " ".join(a["versions"]), "| study:", " ".join(b["versions"]), "| warp:", " ".join(c["versions"]) if c is not None else "-")
    print(f"frame skip: own {int(a['frame_skip'])}, study {int(b['frame_skip'])}")
    print("\n== model constants and options (fields both versions have)")
    names = sorted(k for k in a.files if k.startswith(("model__", "opt__")))
    both = [k for k in names if k in b.files]
    diff = []
    for k in both:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            diff.append((k, f"shapes {x.shape} and {y.shape}"))
        elif not np.array_equal(x, y, equal_nan=True):
            d = np.abs(x - y)
            i = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.ndim else ()
            diff.append((k, f"max |diff| {np.nanmax(d):.3g} (own {x[i]:.6g}, study {y[i]:.6g} at {i}); relative {np.nanmax(d / np.maximum(np.abs(x), 1e-300)):.3g}"))
    print(f"{len(both)} fields compared, {len(both) - len(diff)} equal bit for bit; fields only in own: {len(names) - len(both)}; only in study: "
          f"{len([k for k in b.files if k.startswith(('model__', 'opt__')) and k not in a.files])}")  # fmt: skip
    for k, s in diff:
        print(f"   {k}: {s}")
    for k in ("opt__timestep", "opt__integrator", "opt__solver", "opt__iterations", "opt__tolerance", "opt__cone", "opt__impratio", "opt__noslip_iterations",
              "opt__disableflags", "opt__enableflags", "opt__gravity"):  # fmt: skip
        print(f"   {k}: own {a[k].tolist() if k in a.files else None}, study {b[k].tolist() if k in b.files else None}")
    for k in ("model__body_mass", "model__geom_friction", "model__geom_solref", "model__geom_solimp", "model__geom_margin", "model__geom_condim",
              "model__actuator_gear", "model__actuator_ctrlrange", "model__actuator_ctrllimited", "model__dof_damping", "model__dof_armature",
              "model__jnt_stiffness", "model__jnt_range"):  # fmt: skip
        if k in a.files and k in b.files:
            print(f"   {k}: equal {np.array_equal(a[k], b[k])}; own sum {a[k].sum():.10g}, study sum {b[k].sum():.10g}")
    print("\n== resets (np.random.seed(s); env.reset(seed=s))")
    print(f"   qpos: {stats(a['reset_qpos'] - b['reset_qpos'])}; qvel: {stats(a['reset_qvel'] - b['reset_qvel'])}")
    print("\n== one step from identical states")
    ex = tr["executed"]
    print(f"   {len(ex)} transitions, {int(ex.sum())} executed ones")
    print(f"   study, stepped again, against its recording (executed): qpos {stats((b['one_qpos'] - tr['next_qpos'])[ex])}; qvel "
          f"{stats((b['one_qvel'] - tr['next_qvel'])[ex])}")  # fmt: skip
    for name, x in (("own against study", a), ("warp against study", c)):
        if x is None:
            continue
        print(f"   {name}: next qpos {stats(x['one_qpos'] - b['one_qpos'])}")
        print(f"   {name}: next qvel {stats(x['one_qvel'] - b['one_qvel'])}")
        print(f"   {name}: reward {stats(x['one_reward'] - b['one_reward'])}; terminations that differ "
              f"{int((x['one_terminated'].astype(bool) != b['one_terminated'].astype(bool)).sum())} of {int(b['one_terminated'].sum())} in the study's")  # fmt: skip
        same = x["one_ncon"] == b["one_ncon"]
        dd = np.abs(x["one_dist"] - b["one_dist"])[same & (b["one_ncon"] > 0)]
        print(f"   {name}: contact counts equal on {same.mean():.4f} of the transitions (study's counts {np.bincount(b['one_ncon'].astype(int)).tolist()}); "
              f"deepest distance differs by at most {dd.max() if len(dd) else float('nan'):.3g} where they are")  # fmt: skip
        dq, dv = np.abs(x["one_qpos"] - b["one_qpos"]).max(1), np.abs(x["one_qvel"] - b["one_qvel"]).max(1)
        rq, rv = dq / np.maximum(np.abs(b["one_qpos"]).max(1), 1.0), dv / np.maximum(np.abs(b["one_qvel"]).max(1), 1.0)
        print(f"   {name}: a position more than 1e-9 off on {int((dq > 1e-9).sum())} transitions, more than 1e-7 on {int((dq > 1e-7).sum())}, more than 1e-6 on "
              f"{int((dq > 1e-6).sum())}; more than 1e-3 of the state's size off: positions {int((rq > 1e-3).sum())} ({100 * (rq > 1e-3).mean():.2f} %), "
              f"velocities {int((rv > 1e-3).sum())} ({100 * (rv > 1e-3).mean():.2f} %); contact counts differ on {int((~same).sum())}")  # fmt: skip
        worst = np.argsort(dq)[-3:][::-1]
        print(f"   {name}: worst rows", [(int(i), f"{dq[i]:.3g}", f"index {int(np.abs(x['one_qpos'][i] - b['one_qpos'][i]).argmax())}",
                                          f"contacts {int(b['one_ncon'][i])} / {int(x['one_ncon'][i])}", "executed" if ex[i] else "rollout") for i in worst])  # fmt: skip
    print(f"   own against study: observation {stats(a['one_obs'] - b['one_obs'])}")
    print("\n== replays: max |qpos difference| after n steps")
    names = sq["names"].tolist()
    for i, nm in enumerate(names):
        row = [f"start {nm}:"]
        for n in args.steps:
            if n > b["seq_qpos"].shape[1]:
                continue
            j = n - 1
            cell = f"n={n}: own-study {np.abs(a['seq_qpos'][i, j] - b['seq_qpos'][i, j]).max():.2g}"
            if c is not None:
                cell += f", warp-study {np.abs(c['seq_qpos'][i, j] - b['seq_qpos'][i, j]).max():.2g}"
            cell += f", study-recorded {np.abs(b['seq_qpos'][i, j] - sq['rec_qpos'][i, j]).max():.2g}"
            row.append(cell)
        print("   " + " | ".join(row))
    deepest = lambda x, i: np.nanmin(x["seq_dist"][i]) if x is not None and np.isfinite(x["seq_dist"][i]).any() else float("nan")  # noqa: E731
    for i, nm in enumerate(names):
        line = f"   start {nm}: study's largest |qvel| {np.abs(b['seq_qvel'][i]).max():.3g}, rewards sum own {a['seq_reward'][i].sum():.4g}, study {b['seq_reward'][i].sum():.4g}"
        if c is not None:
            line += f", warp {c['seq_reward'][i].sum():.4g}"
        print(line + f"; contacts per step (study) {b['seq_ncon'][i].astype(int).tolist()[:12]}, deepest {deepest(b, i):.4g} (own {deepest(a, i):.4g}, warp "
              f"{deepest(c, i):.4g})")  # fmt: skip
    print("\n== at rest (zero controls from reset(seed=0))")
    for name, x in (("own", a), ("study", b), ("warp", c)):
        if x is None:
            continue
        end = int(np.argmax(x["rest_terminated"])) if x["rest_terminated"].any() else None
        print(f"   {name}: after {len(x['rest_qpos'])} steps qpos {np.round(x['rest_qpos'][-1], 5).tolist()}; contacts {int(x['rest_ncon'][-1])}, deepest distance "
              f"{float(x['rest_dist'][-1]):.5g}; largest |qvel| at the end {np.abs(x['rest_qvel'][-1]).max():.3g}; terminated at step {end}")  # fmt: skip
    print(f"   own against study: qpos along the rest {stats(a['rest_qpos'] - b['rest_qpos'])}")
    if c is not None:
        print(f"   warp against study: qpos along the rest {stats(c['rest_qpos'] - b['rest_qpos'])}")
    ok, line = is_own(a, b)
    print(f"\n== the threshold\n   {line}")
    return ok


def main(argv=None):
    args = parse_args(argv)
    if args.command == "report":
        ok = report(args)
        if args.threshold and not ok:
            sys.exit(1)
    else:
        {"inputs": inputs, "step": step, "warp": warp}[args.command](args)


if __name__ == "__main__":
    main()
