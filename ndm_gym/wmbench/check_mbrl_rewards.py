"""A check of the batched rewards and terminations of benchmarks/mbrl_lib (``sim.mbrl_envs.Scorer``) against the
environments' own ``step``, on transitions recorded from CPU MuJoCo with each environment built as control_env.py builds it
(mbrl-lib's handler, with the run-time patches of ``sim.mbrl_envs``). Episodes run from the environments' resets under uniform random actions;
with ``--perturb``, every other step starts from a state spread around the episode's (qpos and qvel plus Gaussian noise), so
that the rewards' rarer branches occur. MuJoCo's data is copied before and after each step (dm_control: after each of its
control steps) and scored in float64. A step agrees when the batched reward is within ``--tol`` of the environment's,
absolutely or relatively (the environments compute their control costs in float32), and the batched termination is the
environment's (dm_control's time limit, which ends its episodes, is the rollouts' step count, not a termination). One line per
environment: PASS, FAIL or REFUSED with the reason; the exit status is 1 if any fails.

Two measurements beside it, outside PASS and FAIL:
- the pusher's named difference: its reward from the forward kinematics of the state before the step (what the world model's
  reward reads) against ``env.step``'s, over the steps whose data a previous step left, in episodes of uniform random actions
  without perturbations (the reward, and the bodies' positions);
- ``--ant_forces``: the Ant's contact forces on MuJoCo Warp (``rne_postconstraint`` after one environment step from each
  recorded state, with its controls and the solver's warm start) against CPU MuJoCo's ``cfrc_ext`` after the same step; the
  relative error of a transition is its largest absolute difference over the largest of CPU MuJoCo's forces and torques, or
  over a floor of 1e-3 times the Ant's weight where those are smaller (no contact); the criterion is 1e-4. Reported beside it:
  the transitions in contact, those where one simulator's forces exceed the floor and the other's do not, and the largest
  difference of the states after the step.

Run: ``python -m ndm_gym.wmbench.check_mbrl_rewards --local_models converted/ --steps 2000 --perturb --ant_forces --out check.json``
"""

import argparse
import contextlib
import json
import sys

import numpy as np
import torch

from ..sim.mbrl_envs import ENVS, Fields, Kinematics, Scorer, make_env, patch_envs, physics, recording_dm_control


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=list(ENVS), help=f"of {list(ENVS)}")
    p.add_argument("--local_models", default=None,
                   help="the directory of hopper_local.xml and walker2d_local.xml (tools/convert_mbrl_models.py makes them, into data/mbrl_models)")
    p.add_argument("--own_margins", action="store_true", help="the contact margins of mbrl-lib's own MuJoCo (sim.mbrl_envs.set_own_margins)")
    p.add_argument("--own_mujoco", action="store_true",
                   help="the models as mbrl-lib's own MuJoCo steps them: its margins, inverse weights and convex collision (sim.mbrl_envs.set_own_mujoco)")
    p.add_argument("--steps", type=int, default=2000, help="environment steps per environment")
    p.add_argument("--perturb", action="store_true", help="every other step from a perturbed state")
    p.add_argument("--spread", type=float, nargs=2, default=[0.2, 5.0], help="the perturbation's std of qpos and of qvel")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--ant_forces", action="store_true", help="also compare the Ant's contact forces on MuJoCo Warp with CPU MuJoCo's")
    p.add_argument("--ant_steps", type=int, default=1000, help="--ant_forces: recorded transitions")
    p.add_argument("--device", default="cpu", help="--ant_forces: Warp's device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help=".json of the results")
    return p.parse_args(argv)


def perturb(spec, env, rng, spread):
    m, d, _, _ = physics(spec, env)
    q, v = d.qpos + spread[0] * rng.standard_normal(m.nq), d.qvel + spread[1] * rng.standard_normal(m.nv)
    if spec.dm_control:
        ph = env.unwrapped.gym_env.unwrapped.env._env.physics
        with ph.reset_context():
            ph.data.qpos[:], ph.data.qvel[:] = q, v
    else:
        env.unwrapped.set_state(q, v)


def check(spec, args) -> dict:
    if spec.name == "humanoid":
        return {"result": "REFUSED", "reason": f"no batched reward: {spec.warp_refused}; {spec.ndm_refused}"}
    handler, env = make_env(spec, args.seed)
    rng = np.random.default_rng(args.seed)
    err, sub_err, worst, bad_term, n_term, episodes, rewards, rejected = 0.0, 0.0, 0.0, 0, 0, 1, [], 0
    for t in range(args.steps):
        m, d, substeps, _ = physics(spec, env)
        sc = Scorer(spec, m, substeps)
        if args.perturb and t % 2:
            perturb(spec, env, rng, args.spread)
        a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        goal = torch.tensor(np.asarray(env.unwrapped.goal, dtype=np.float64))[None] if "goal" in spec.reads else None
        pre, sub = Fields.of_mjdata(m, d), []
        try:
            with recording_dm_control(spec, env, sub) if spec.dm_control else contextlib.nullcontext():
                _, r, term, trunc, _ = env.step(a)
        except Exception as e:  # dm_control refuses a state diverging from a perturbation (PhysicsError): counted, a new episode
            if not (spec.dm_control and args.perturb and type(e).__name__ == "PhysicsError"):
                raise
            rejected += 1
            env.reset(seed=args.seed + episodes)
            episodes += 1
            continue
        post = Fields.of_mjdata(m, d)
        ctrl = torch.tensor(a, dtype=torch.float64)[None]
        if spec.dm_control:
            parts = [sc.reward(None, f, torch.tensor(u)[None])[0].item() for f, u, _ in sub]
            sub_err = max([sub_err] + [abs(x - (ts or 0.0)) for x, (_, _, ts) in zip(parts, sub)])
            mine, mine_term = sum(parts), bool(sc.terminated(post)[0])
            dmc = env.unwrapped.gym_env.unwrapped.env._env
            mine_term = mine_term or dmc._step_count >= dmc._step_limit
        else:
            mine, mine_term = sc.reward(pre, post, ctrl, goal)[0].item(), bool(sc.terminated(post)[0])
        e = abs(mine - r)
        err, worst = max(err, e), max(worst, e / max(1.0, abs(r)))
        bad_term += int(mine_term != bool(term))
        n_term += int(bool(term))
        rewards.append(r)
        if term or trunc:
            np.random.seed(args.seed + episodes)  # the PETS environments reset from global numpy
            env.reset(seed=args.seed + episodes)
            episodes += 1
    ok = (worst <= args.tol or err <= args.tol) and bad_term == 0 and sub_err <= max(args.tol, args.tol * max(abs(x) for x in rewards))
    rw = np.asarray(rewards)
    out = {"result": "PASS" if ok else "FAIL", "steps": len(rewards), "rejected": rejected, "episodes": episodes, "max_abs_err": err,
           "max_rel_err": worst, "control_step_max_abs_err": sub_err if spec.dm_control else None, "terminated": n_term,
           "termination_mismatches": bad_term, "reward_mean": float(rw.mean()), "reward_nonzero": float((rw != 0).mean()),
           "reward_range": [float(rw.min()), float(rw.max())]}  # fmt: skip
    if spec.named:
        out["named_difference"] = named_difference(spec, args)
    return out


def named_difference(spec, args) -> dict:
    """The pusher's reward with the kinematics of the state before the step against env.step's (the module's docstring)."""
    handler, env = make_env(spec, args.seed + 7)
    rng = np.random.default_rng(args.seed + 7)
    out, fresh, episodes = {"reward": [], "position": []}, True, 1  # fresh: the data is the state's own (after a reset's mj_forward)
    for _ in range(args.steps):
        m, d, substeps, _ = physics(spec, env)
        sc, kin = Scorer(spec, m, substeps), Kinematics(m, dtype=torch.float64)
        a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        pre = Fields.of_mjdata(m, d)
        _, r, term, trunc, _ = env.step(a)
        if not fresh:
            fk = kin(pre.qpos, names=("xpos",))["xpos"]
            out["reward"].append(abs(sc.reward(Fields(xpos=fk), None, torch.tensor(a, dtype=torch.float64)[None])[0].item() - r))
            out["position"].append(float((fk - pre.xpos)[0, [sc.obj, sc.tip, sc.goal]].norm(dim=-1).max()))
        fresh = False
        if term or trunc:
            np.random.seed(args.seed + 7 + episodes)
            env.reset(seed=args.seed + 7 + episodes)
            fresh, episodes = True, episodes + 1
    return {"steps": len(out["reward"]), **{f"{k}_{s}": float(f(out[k])) for k in out for s, f in (("max", np.max), ("median", np.median))}}


def ant_forces(args) -> dict:
    """The Ant's contact forces on MuJoCo Warp against CPU MuJoCo's on recorded transitions (the module's docstring)."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    spec = ENVS["ant"]
    handler, env = make_env(spec, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    rows, episodes = [], 1
    for _ in range(args.ant_steps):
        m, d, substeps, _ = physics(spec, env)
        a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        pre = (d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), a.astype(np.float64))
        _, _, term, trunc, _ = env.step(a)
        rows.append(pre + (d.cfrc_ext.copy(), int(d.ncon), d.qpos.copy()))
        if term or trunc:
            env.reset(seed=args.seed + 1000 + episodes)
            episodes += 1
    m, _, substeps, _ = physics(spec, env)
    mjd = mujoco.MjData(m)
    mujoco.mj_forward(m, mjd)
    with wp.ScopedDevice(wp.get_device(args.device)):
        wm, wd = mjw.put_model(m), mjw.put_data(m, mjd, nworld=len(rows))
        for i, k in enumerate(("qpos", "qvel", "qacc_warmstart", "ctrl")):
            wp.to_torch(getattr(wd, k)).copy_(torch.tensor(np.stack([r[i] for r in rows]), dtype=torch.float32))
        for _ in range(substeps):
            mjw.step(wm, wd)
        mjw.rne_postconstraint(wm, wd)
        got = wp.to_torch(wd.cfrc_ext).cpu().numpy().astype(np.float64)
        qpos = wp.to_torch(wd.qpos).cpu().numpy().astype(np.float64)
    ref, ncon = np.stack([r[4] for r in rows]), np.array([r[5] for r in rows])
    diff, scale, warp_scale = np.abs(got - ref).max((1, 2)), np.abs(ref).max((1, 2)), np.abs(got).max((1, 2))
    floor = 1e-3 * float(m.body_mass.sum()) * float(np.linalg.norm(m.opt.gravity))
    rel, contact = diff / np.maximum(scale, floor), scale > floor
    return {"transitions": len(rows), "episodes": episodes, "in_contact": int(contact.sum()), "floor_N": floor, "max_rel": float(rel.max()),
            "median_rel": float(np.median(rel)), "p99_rel": float(np.quantile(rel, 0.99)),
            "median_rel_in_contact": float(np.median(rel[contact])) if contact.any() else None,
            "contact_disagreements": int((contact != (warp_scale > floor)).sum()), "max_abs": float(diff.max()), "max_force": float(scale.max()),
            "state_max_abs": float(np.abs(qpos - np.stack([r[6] for r in rows])).max()), "cpu_contacts_max": int(ncon.max()),
            "criterion": 1e-4, "match": bool(rel.max() <= 1e-4), "device": args.device}  # fmt: skip


def main(argv=None):
    args = parse_args(argv)
    bad = [e for e in args.envs if e not in ENVS]
    if bad:
        raise ValueError(f"check_mbrl_rewards: unknown environments {bad}")
    patch_envs(args.local_models, own_margins=args.own_margins, own_mujoco=args.own_mujoco)
    out = {}
    for name in args.envs:
        res = out[name] = check(ENVS[name], args)
        if res["result"] == "REFUSED":
            print(f"[check_mbrl_rewards] {name}: REFUSED ({res['reason']})", flush=True)
            continue
        sub = f", per control step {res['control_step_max_abs_err']:.2g}" if res["control_step_max_abs_err"] is not None else ""
        rej = f" ({res['rejected']} perturbed steps refused by dm_control's physics check)" if res["rejected"] else ""
        print(f"[check_mbrl_rewards] {name}: {res['result']} {res['steps']} steps in {res['episodes']} episodes{rej}; reward max |err| "
              f"{res['max_abs_err']:.2g} (rel {res['max_rel_err']:.2g}){sub}; terminated {res['terminated']}, mismatches "
              f"{res['termination_mismatches']}; rewards {res['reward_range'][0]:.3g}..{res['reward_range'][1]:.3g}, nonzero "
              f"{100 * res['reward_nonzero']:.0f}%", flush=True)  # fmt: skip
        if "named_difference" in res:
            n = res["named_difference"]
            print(f"[check_mbrl_rewards] {name}: named difference, the world model's kinematics of the state before the "
                  f"step against env.step's, over {n['steps']} steps after a step: reward |diff| max {n['reward_max']:.3g}, median "
                  f"{n['reward_median']:.3g}; body positions max {n['position_max']:.3g} m, median {n['position_median']:.3g} m",
                  flush=True)  # fmt: skip
    if args.ant_forces:
        a = out["ant_forces"] = ant_forces(args)
        print(f"[check_mbrl_rewards] ant: contact forces on MuJoCo Warp ({a['device']}) against CPU MuJoCo's cfrc_ext, {a['transitions']} "
              f"transitions ({a['in_contact']} in contact): relative error max {a['max_rel']:.3g}, p99 {a['p99_rel']:.3g}, median "
              f"{a['median_rel']:.3g}, median in contact {a['median_rel_in_contact']:.3g} (floor {a['floor_N']:.3g} N); absolute max "
              f"{a['max_abs']:.3g} N of forces up to {a['max_force']:.3g} N; contact disagreements {a['contact_disagreements']}; states after "
              f"the step differ by up to {a['state_max_abs']:.3g}; {'match' if a['match'] else 'no match'} at 1e-4", flush=True)  # fmt: skip
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": out}, f, indent=1)
    if any(isinstance(r, dict) and r.get("result") == "FAIL" for r in out.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
