"""Checkpoint evaluation on HDMI's human-object interaction tasks, the way HDMI evaluates a policy (LeCAR-Lab/HDMI @
32282f6: scripts/eval.py with cfg/eval.yaml, which runs scripts/helpers.py:evaluate). One protocol scores the SuperTrack
and SuperMimic checkpoints of ndm_gym.train_hoi and the PPO-ROA checkpoints of ndm_gym.train_ppo_hoi. References are
ALIAS:line as in docs/hdmi_task_spec.md, plus EVY = cfg/eval.yaml and DIST = active_adaptation/learning/modules/distributions.py.

HDMI's evaluation, from its code:
* 2048 envs (EVY:41-42, which overrides Y:13 because EVY's own entries come last in its defaults, EVY:36-39). The
  paper's statistics are over 4096 environments (arXiv 2509.16757, §V), the count of the evaluation that
  scripts/train.py runs on its training env after training (TRN:194-196); ``--num_envs 4096`` gives that count.
* ``env.eval()`` (HLP:215-216) sets training = False (torchrl's EnvBase is an nn.Module). Then every episode starts at
  frame 0 (CMD:144-145) without noise on the robot's root pose and velocity (CMD:169-170, 177-178). Nothing else reads
  the flag: the robot's joint noise (CMD:187-191), the object's pose noise and a hinge's angle and rate noise
  (CMD:463-467, 503-507), the reset draws of the joint offsets, the action delay and filter and the hinge's friction
  (RND:742-744, ACT:86-95, HRN:107-112), the startup randomization (§7) and the observation noise (OC:27-34, 54-62,
  113-114; OBS:282-300, 338-349, 368-380) stay on. The observation normalizer is frozen (EVY:20, HLP:155-157).
* Actions are the policy distribution's mode (HLP:198, 226), the Gaussian's mean (DIST:139-158): a teacher's actor on
  its privileged encoder, a student's on the adaptation EMA's features (PPO:406-439).
* The seed is set (HLP:217, ENV:550-553; seed 0, EVY:18) and all envs are reset; then max_episode_length control steps
  (HLP:227), each env reset as soon as its episode is done (HLP:234). Only every env's first episode is scored
  (HLP:246-250); ``episode_cnt`` counts all episode ends in the window (HLP:245).
* Terminations as in training (§6); truncation at the clip's end (t >= L) or at max_episode_length (ENV:503-505,
  CMD:229-233). traverse_door's clip (1463 frames) outlasts its 1000-step episodes, so it cannot succeed.
* Statistics (HLP:252-282): for every stats key (ENV:161-169, 276, 287, 315), the mean and std over envs of its value at
  the first episode's last step: the episode length; the success flag; and, divided by the episode length, every reward
  term's sum of weight x value (disabled terms too, ENV:620-647, so also the debug group's tracking errors, Y:232-241),
  every group's return (its reward times dt) and the termination flags of that step. Success is HDMI's frame counter
  at L - 1 when that step's rewards are computed (CMD:225-227, ENV:421-424): the episode reached the clip's end, also
  if a termination fires at that same step.

HDMI's paper reports the success rate ("finishes the task without triggering any termination") and "joint and body
tracking errors" without definitions; the code computes errors only as the debug group's terms, all reported here.

Here: HDMISim in eval_mode (frame 0, no root pose or velocity noise) with one spare env whose index pads the done envs'
indices to a fixed count (as ndm_gym.train_ppo_hoi), HDMITask with its observation noise, the run's policy acting
deterministically, HDMI's loop and statistics (``hdmi`` in the result, under HDMI's keys), and per checkpoint: success
rate and count, mean episode length and fraction of the clip reached, the termination breakdown (the fraction of first
episodes that ended with each term's flag on), each reward group's mean per step, HDMI's tracking errors, the keypoint
position error (the mean over the 16 keypoints of the world-frame error against the frame HDMI compares with, averaged
over the episode; its mean and 90th percentile over envs), the object's position and orientation error or the hinge
angle error (the state after the step against that frame, as the terminations compare), and the evaluation's time.
Errors in m and rad; group rewards per step include HDMI's factor dt.

Not reproduced: the physics engine (MuJoCo-Warp instead of PhysX) and the per-env object scale (scale 1, as in
training; see ndm_gym.sim.rigid.hdmi_sim). HDMI's second observation of an episode reads its previous episode's command
buffers (§0); here the reset state, as in training. The first observation is HDMI's all-zero one for the PPO checkpoints
(ENV:385-386) and the reset state's for SuperTrack and SuperMimic, each as it was trained (train_ppo_hoi, train_hoi).
HDMI draws its startup randomization before it sets the seed (HLP:130-162); here the seed also fixes it.

Reproducibility: the startup randomization, the resets' draws and the observation noise come from ``--seed``, drawn in
a fixed order and amount whatever the policy does, so that every checkpoint and algorithm starts each env's first
episode from the same reset and sees the same observation noise at each step. MuJoCo-Warp accumulates contact and
constraint forces with atomics, so two evaluations of one checkpoint agree only up to the GPU's summation order, which
can grow along an episode.

Run: ``python -m ndm_gym.evaluate_hoi --run_dir logs/<run> --watch`` (appends to <run>/eval_hoi.csv), or one checkpoint:
``python -m ndm_gym.evaluate_hoi --run_dir logs/<run> --ckpt logs/<run>/ckpt_<it>.pt --out_json <file>``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import torch

from ndm_gym.algos.ppo_hoi import OBS_GROUPS, residual_base
from ndm_gym.tasks.hdmi import (DECIMATION, GROUPS, JOINT_NAMES, KEYPOINT_NAMES, NUM_ACTIONS, PHYSICS_DT, TRACKING_JOINT_NAMES, FeetContact,
                                HDMIInput, HDMITask, feet_air_time, heading_pos, heading_rot, joint_pos_limits, object_ori_error, object_pos_error,
                                relative_rot, rot_angle)  # fmt: skip
from ndm_gym.tasks.hdmi_tasks import LOWER_BODY, TRACKED_JOINTS, UPPER_BODY

NUM_ENVS = 2048  # EVY:41-42
SEED = 0  # EVY:18
HOI_ALGOS = ("supertrack", "supermimic")  # ndm_gym.train_hoi's --algo
PPO_ALGOS = {"train": "ppo_roa_train", "finetune": "ppo_roa_finetune"}  # train_ppo_hoi's --phase -> HDMI's algo name (PPO:89-91)
# HDMI's debug group (Y:232-241, in its order): every term disabled, so the group's reward is 0, but HDMI's statistics
# hold each term's value like any other's; the error terms are HDMI's tracking errors
DEBUG_TERMS = ("feet_air_time", "feet_contact_count", "joint_pos_limits", "eef_contact_all", "root_pos_error", "root_ori_error",
               "body_pos_error_local", "body_ori_error_local", "joint_pos_error")  # fmt: skip
DEBUG_CONTACT_POS_THRES = 0.1  # eef_contact_all's pos_thres (Y:236)
DEBUG_CONTACT_POS_THRES_TASK = {"roll_ball-hand": 0.4}  # T:roll_ball-hand:43-44


def _indices(patterns, names) -> list[int]:
    """Indices of ``names`` matched in full by one of ``patterns``; every pattern must match."""
    hits = [i for i, n in enumerate(names) if any(re.fullmatch(p, n) for p in patterns)]
    if not all(any(re.fullmatch(p, n) for n in names) for p in patterns):
        raise ValueError(f"patterns {patterns} must each match one of {names}")
    return hits


def feet_after_step(task: HDMITask) -> FeetContact:
    """The feet's contact state the terms of the step just taken read (HDMITask.advance_feet's result), from the task's
    timers after HDMITask.step."""
    cnt = task.contact_count
    return FeetContact((cnt > 0) & (cnt <= DECIMATION), cnt.to(task.dtype) * PHYSICS_DT, task.last_air_time)


class EvalTerms:
    """The per-step values the evaluation averages besides HDMITask's reward terms: HDMI's debug group (DEBUG_TERMS, each
    weight x value) and the errors added here, timed as HDMI's terms (§0): the state after step n against the reference
    of the previous command update (``x.ref`` index 0, frame f_{n-1}); the local terms in the heading frame of that
    update's root snapshot and the contact term on its snapshots, as HDMI's reward classes read them."""

    def __init__(self, task: HDMITask):
        dev, lim = task.device, task.joint_limits
        ix = lambda pats, names: torch.tensor(_indices(pats, names), device=dev)  # noqa: E731
        self.bodies = ix(LOWER_BODY + UPPER_BODY, KEYPOINT_NAMES)  # body_*_error_local's 12 bodies (Y:239-240)
        self.joints = ix(TRACKED_JOINTS, TRACKING_JOINT_NAMES)  # joint_pos_error's 19 joints (Y:241)
        self.all_joints = torch.arange(len(JOINT_NAMES), device=dev)
        mid, span = (lim[:, 0] + lim[:, 1]) / 2, lim[:, 1] - lim[:, 0]
        self.soft_lo, self.soft_hi = mid - 0.5 * span * 0.9, mid + 0.5 * span * 0.9  # joint_pos_limits' soft_factor 0.9 (Y:235, RC:69-74)
        self.contact_pos_thres = DEBUG_CONTACT_POS_THRES_TASK.get(task.cfg.name, DEBUG_CONTACT_POS_THRES)
        self.hinged = task.hinged
        self.object_errors = ("e_hinge",) if task.hinged else ("e_obj_pos", "e_obj_ori")
        self.names = [f"debug/{n}" for n in DEBUG_TERMS] + ["e_kp", *self.object_errors]

    def __call__(self, x: HDMIInput, feet: FeetContact) -> torch.Tensor:
        """(N, len(names))."""
        ref, now, prev = x.ref, x.now, x.prev
        dt = now.body_pos.dtype
        # eef_contact_all (REW:452-490; frc_thres 1.0, a float: the force-magnitude branch): an end effector flagged for
        # contact counts when it is near its target and pressed; outside the flagged frames the term is 1
        near = (prev.eef_pos - prev.target_pos).norm(dim=-1) < self.contact_pos_thres
        pressed = prev.eef_force.norm(dim=-1) >= 1.0
        g = ref.contact[:, 0].to(dt)
        contact = ((near & pressed).to(dt) * g + 1 - g).mean(-1)
        b, j = self.bodies, self.joints
        r_pos = heading_pos(ref.body_pos[:, 0, b], ref.body_pos[:, 0, 0], ref.body_rot[:, 0, 0])
        r_rot = heading_rot(ref.body_rot[:, 0, b], ref.body_rot[:, 0, 0])
        vals = [
            5.0 * feet_air_time(feet, 0.2),  # RF:143-154, Y:233
            feet.first_contact.sum(-1).to(dt),  # RF:272-276, Y:234
            joint_pos_limits(x, self.all_joints, self.soft_lo, self.soft_hi),  # RC:76-80, Y:235
            contact,
            (ref.body_pos[:, 0, 0] - now.body_pos[:, 0]).norm(dim=-1),  # root_pos_error: keypoint_pos_error, pelvis (REW:135-143)
            rot_angle(relative_rot(ref.body_rot[:, 0, 0], now.body_rot[:, 0])),  # root_ori_error: keypoint_ori_error (REW:210-219)
            (r_pos - heading_pos(now.body_pos[:, b], prev.root_pos, prev.root_rot)).norm(dim=-1).mean(-1),  # REW:145-172
            rot_angle(relative_rot(r_rot, heading_rot(now.body_rot[:, b], prev.root_rot))).mean(-1),  # REW:221-243
            (ref.joint_pos[:, 0, j] - now.joint_pos[:, j]).abs().mean(-1),  # REW:307-313
            (now.body_pos - ref.body_pos[:, 0]).norm(dim=-1).mean(-1),  # e_kp: the 16 keypoints, world frame
        ]  # fmt: skip
        if self.hinged:
            vals.append((ref.obj_joint_pos[:, 0] - now.obj_joint_pos).abs())
        else:
            vals += [object_pos_error(x), object_ori_error(x)]  # the terminations' errors (TRM:153-167)
        return torch.stack(vals, -1)


def reward_keys(task: HDMITask) -> list[str]:
    """HDMITask's reward terms as its step's ``terms`` names them, disabled ones included (HDMI logs them all)."""
    return [f"{g}/{n}" for g in GROUPS for n, *_ in task.reward_terms[g]]


def new_stats(n: int, values: int, terms: int, device) -> dict:
    """Per-env statistics of the first episode, all zero; ``done_count`` counts every episode end."""
    z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
    return {"sums": z(n, values), "ended": z(n, dtype=torch.bool), "length": z(n, dtype=torch.long), "success": z(n, dtype=torch.bool),
            "terminated": z(n, dtype=torch.bool), "flags": z(n, terms, dtype=torch.bool), "nonfinite": z(n, dtype=torch.bool),
            "done_count": z((), dtype=torch.long)}  # fmt: skip


def accumulate(st: dict, values, done, terminated, flags, t, ep_step, clip_len: int, n_scored: int, bad=None) -> dict:
    """One step of HDMI's episode statistics (ENV:406-442, 503-510; HLP:244-250), without writing into ``st``: the
    values of the envs still in their first episode are summed; at its last step (done) its length, success, termination
    and termination flags are kept. ``t`` and ``ep_step``: the frame counter and the episode's steps after the step;
    HDMI's success flag is its counter before the step's command update at L - 1, i.e. ``t >= L``. Every episode end of
    the first ``n_scored`` envs is counted (``episode_cnt``). ``bad``: the envs the guard flagged at this step
    (hoi_guard.nonfinite; their episode ends as terminated): their values of this step are left out and the flag kept."""
    bad = torch.zeros_like(done) if bad is None else bad
    values = torch.where(bad.unsqueeze(-1), 0.0, values)
    first = ~st["ended"]
    last = first & done
    keep = lambda new, old: torch.where(last.view(-1, *([1] * (old.dim() - 1))), new, old)  # noqa: E731
    return {"sums": st["sums"] + torch.where(first.unsqueeze(-1), values, 0.0), "ended": st["ended"] | done,
            "length": keep(ep_step.long(), st["length"]), "success": keep(t >= clip_len, st["success"]),
            "terminated": keep(terminated, st["terminated"]), "flags": keep(flags, st["flags"]), "nonfinite": keep(bad, st["nonfinite"]),
            "done_count": st["done_count"] + done[:n_scored].sum()}  # fmt: skip


def summarize(st: dict, names: list[str], term_names: list[str], n: int, clip_len: int) -> dict:
    """HDMI's statistics of the first episodes of the first ``n`` envs (HLP:252-282) under HDMI's keys (``hdmi``), and
    the metrics derived from them, read from the device at once."""
    length = st["length"][:n].float()
    per_step = st["sums"][:n] / length.unsqueeze(-1)  # a term's sum over the episode / its length (HLP:258-261)
    success, terminated, flags = st["success"][:n], st["terminated"][:n], st["flags"][:n]
    col = {k: per_step[:, i] for i, k in enumerate(names)}
    hdmi = {"episode_len": length, "success": success.float(), "debug/return": torch.zeros_like(length)}  # debug: no enabled term
    hdmi.update({k: v for k, v in col.items() if not k.startswith("e_")})
    hdmi.update({f"termination/{k}": flags[:, i].float() / length for i, k in enumerate(term_names)})
    out = {f"eval/{k}{s}": f(v) for k, v in hdmi.items() for s, f in (("", torch.mean), ("_std", torch.std))}
    out["episode_cnt"] = st["done_count"]
    out = dict(sorted(out.items()))  # HLP:281-282
    m = {"n_success": success.sum(), "success": success.float().mean(), "ep_len": length.mean(), "clip_frac": (length / clip_len).mean(),
         "terminated": terminated.float().mean(), "time_limit": (~terminated & ~success).float().mean(),
         "nonfinite": st["nonfinite"][:n].float().mean()}  # fmt: skip
    m.update({f"term_{k}": flags[:, i].float().mean() for i, k in enumerate(term_names)})
    m.update({f"r_{g}": col[f"{g}/return"].mean() for g in GROUPS})
    m.update({"e_kp": col["e_kp"].mean(), "e_kp_p90": torch.quantile(col["e_kp"], 0.9)})
    for k, d in (("e_root_pos", "root_pos_error"), ("e_root_ori", "root_ori_error"), ("e_body_pos_local", "body_pos_error_local"),
                 ("e_body_ori_local", "body_ori_error_local"), ("e_joint", "joint_pos_error")):  # fmt: skip
        m[k] = col[f"debug/{d}"].mean()
    m.update({k: v.mean() for k, v in col.items() if k.startswith("e_") and k != "e_kp"})
    m["eef_contact_all"] = col["debug/eef_contact_all"].mean()
    m["episode_cnt"] = st["done_count"]
    ks = list(out) + list(m)
    vals = torch.stack([v.float() for v in list(out.values()) + list(m.values())]).tolist()
    out, met = dict(zip(ks[: len(out)], vals[: len(out)])), dict(zip(ks[len(out) :], vals[len(out) :]))
    out["episode_cnt"] = int(out["episode_cnt"])
    for k in ("n_success", "episode_cnt"):
        met[k] = int(met[k])
    return {"metrics": met, "hdmi": out}


def csv_columns(term_names: list[str], object_errors: tuple[str, ...]) -> list[str]:
    """The columns of eval_hoi.csv for a task with these termination terms and object errors (EvalTerms)."""
    return (["iteration", "train_wall_h", "eval_at", "n_envs", "success", "n_success", "ep_len", "clip_frac", "terminated", "time_limit",
             "nonfinite"]
            + [f"term_{k}" for k in term_names] + [f"r_{g}" for g in GROUPS]
            + ["e_kp", "e_kp_p90", "e_root_pos", "e_root_ori", "e_body_pos_local", "e_body_ori_local", "e_joint", *object_errors]
            + ["eef_contact_all", "episode_cnt", "eval_s"])  # fmt: skip


def csv_row(columns: list[str], res: dict) -> str:
    """One line of eval_hoi.csv from an evaluation's result."""
    ints = ("iteration", "n_envs", "n_success", "episode_cnt")
    vals = {**res["metrics"], **{k: res[k] for k in ("iteration", "n_envs", "eval_s")}, "train_wall_h": res["wall_time"] / 3600,
            "eval_at": res["eval_at"]}  # fmt: skip
    fmt = lambda k: str(vals[k]) if k in ints or k == "eval_at" else f"{vals[k]:.5f}"  # noqa: E731
    return ",".join(fmt(k) for k in columns) + "\n"


def run_algo(a: dict) -> str:
    """The algorithm of a run from its args.json: "supertrack" or "supermimic" (ndm_gym.train_hoi), "ppo_roa_train" or
    "ppo_roa_finetune" (ndm_gym.train_ppo_hoi, HDMI's teacher and student)."""
    if "task" not in a:
        raise ValueError("args.json has no task: not a run of ndm_gym.train_hoi or ndm_gym.train_ppo_hoi")
    if "algo" in a:
        if a["algo"] not in HOI_ALGOS:
            raise ValueError(f"unknown algo {a['algo']!r}: ndm_gym.train_hoi trains {HOI_ALGOS}")
        return a["algo"]
    if "phase" in a:
        if a["phase"] not in PPO_ALGOS:
            raise ValueError(f"unknown phase {a['phase']!r}: ndm_gym.train_ppo_hoi trains {tuple(PPO_ALGOS)}")
        return PPO_ALGOS[a["phase"]]
    raise ValueError("args.json has neither ndm_gym.train_hoi's algo nor ndm_gym.train_ppo_hoi's phase")


def load_run_args(run_dir: str) -> dict:
    path = os.path.join(run_dir, "args.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: the run's arguments say which task, algorithm and policy to build")
    with open(path) as f:
        return json.load(f)


def check_ckpt(ck: dict, a: dict, path: str):
    """A checkpoint of the run whose args.json is ``a``: the same task and algorithm, and the policy's weights."""
    for k in ("task", "algo", "phase"):
        if ck["args"].get(k) != a.get(k):
            raise ValueError(f"{path}: {k} {ck['args'].get(k)!r}, the run's args.json has {a.get(k)!r}")
    need = "pi" if "algo" in a else "agent"
    if need not in ck:
        raise KeyError(f"{path} has no {need!r} entry")


def ckpt_iteration(path: str) -> int:
    m = re.fullmatch(r"ckpt_(\d+)\.pt", os.path.basename(path))
    if m is None:
        raise ValueError(f"{path} is not a checkpoint name ckpt_<iteration>.pt")
    return int(m.group(1))


class HOIActor:
    """ndm_gym.train_hoi's policy (SuperTrack, SuperMimic; HOIAgent.act): HOIPolicy on the command, policy and object
    groups (and the privileged group with --priv_obs) plus HDMI's residual base at the current frame, deterministic (no
    exploration noise). An episode's first observation is the reset state's (HDMISim.reset_input), as train_hoi trains
    it."""

    needs_reset_obs = True

    def __init__(self, a: dict, env, task: HDMITask, dev):
        import ndm_gym.core.nets as nets
        from ndm_gym.algos.hoi_algos import policy_groups
        from ndm_gym.policies.hoi_policy import HOIPolicy

        nets.AMP_DTYPE = torch.bfloat16 if a["amp"] == "bf16" else None
        d = task.obs_dims
        self.groups, self.N = policy_groups(a.get("priv_obs", False)), env.N - 1
        self.pi = HOIPolicy(d["command"], d["policy"], d["object"], priv_dim=d["priv"] if "priv" in self.groups else 0).to(dev).eval()
        self.base = (env.clip.joint_pos, env.q0[env.j_act].clone(), env.scale.clone())

    def load(self, ck: dict):
        self.pi.load_state_dict(ck["pi"])

    def begin(self, fresh: dict) -> dict:
        return fresh

    def act(self, obs: dict, t: torch.Tensor) -> torch.Tensor:
        return self.pi({k: obs[k][: self.N] for k in self.groups}) + residual_base(*self.base, t[: self.N])

    def observe(self, carry, obs: dict, t: torch.Tensor) -> dict:
        return obs

    def reset(self, obs: dict, done: torch.Tensor, fresh: dict) -> dict:
        return {k: torch.where(done.view(-1, *([1] * (v.dim() - 1))), fresh[k], v) for k, v in obs.items()}


class PPOActor:
    """ndm_gym.train_ppo_hoi's policy (PPOROA) as HDMI's evaluation runs it: the normalizer frozen (EVY:20, HLP:155-157),
    the mean action (HLP:198) of the teacher (residual over HDMI's ref_joint_pos_, PPO:223-236) or of the student (on the
    adaptation EMA's features). An episode's first observation is HDMI's: all groups zero (ENV:385-386), normalized,
    and a zero residual base (as train_ppo_hoi trains it)."""

    needs_reset_obs = False
    groups = OBS_GROUPS

    def __init__(self, a: dict, env, task: HDMITask, dev, compile_: bool):
        from ndm_gym.algos.ppo_hoi import PPOROA, PPOConfig

        self.agent = PPOROA(PPOConfig(phase=a["phase"], train_every=a["train_every"]), task.obs_dims, dev, compile=compile_)
        self.N, self.dev = env.N - 1, dev
        self.base = (env.clip.joint_pos, env.q0[env.j_act].clone(), env.scale.clone())

    def load(self, ck: dict):
        self.agent.load_state_dict(ck["agent"])
        self.zero = self.agent.vecnorm.normalize(torch.zeros(1, self.agent.obs_dim, device=self.dev))

    def begin(self, fresh) -> tuple:
        return self.zero.expand(self.N, -1), torch.zeros(self.N, NUM_ACTIONS, device=self.dev)

    def act(self, carry: tuple, t: torch.Tensor) -> torch.Tensor:
        return self.agent.act(*carry, deterministic=True)[0]

    def observe(self, carry, obs: dict, t: torch.Tensor) -> tuple:
        """The step's groups normalized and the residual base at the frame after the step (train_ppo_hoi's post_step)."""
        return self.agent.vecnorm.normalize(torch.cat([obs[k][: self.N] for k in self.groups], -1)), residual_base(*self.base, t[: self.N])

    def reset(self, carry: tuple, done: torch.Tensor, fresh) -> tuple:
        m = done[: self.N].unsqueeze(-1)
        return torch.where(m, self.zero, carry[0]), torch.where(m, 0.0, carry[1])


class Evaluator:
    """HDMI's evaluation of one run's checkpoints on ``num_envs`` envs (module docstring). Must be constructed inside
    ``launch_simulation``; the scene and its startup randomization (from ``seed``) serve every checkpoint."""

    def __init__(self, sim_cfg, a: dict, num_envs: int, seed: int, graph: bool = True, compile_: bool = True):
        from ndm_gym.algos.hoi_guard import nonfinite
        from ndm_gym.sim.rigid.hdmi_sim import HDMISim
        from ndm_gym.tasks.hdmi_tasks import get
        from ndm_gym.train_ppo_hoi import TASK_COMPILE_OPTIONS

        self.algo, self.seed, self.N = run_algo(a), seed, num_envs
        cfg = get(a["task"])
        dev = sim_cfg.device
        torch.manual_seed(seed)  # the startup randomization
        # the spare env (index N) has action 0 and is reset at every step, its index padding the done envs' to a fixed
        # count (torch.nonzero_static), so that the resets need no host synchronization
        self.env = env = HDMISim(sim_cfg, cfg, num_envs + 1, eval_mode=True, graph=graph)
        self.task = task = HDMITask(cfg, env.N, env.joint_limits, env.robot_mass, noise=True, device=dev)
        self.actor = HOIActor(a, env, task, dev) if self.algo in HOI_ALGOS else PPOActor(a, env, task, dev, compile_)
        self.terms = terms = EvalTerms(task)
        keys = reward_keys(task)
        self.names = keys + [f"{g}/return" for g in GROUPS] + terms.names  # the columns of the per-step values
        self.columns = csv_columns(task.term_names, terms.object_errors)
        groups, L, N = self.actor.groups, env.L, num_envs

        def step(x, noise, truncated, t, ep_step, st, a):
            """HDMITask's step (the observation noise drawn outside, so that compiled it is eager's random stream), the
            values it and EvalTerms give, the statistics' update; the policy's groups and the done flags. An env the
            training's guard flags (hoi_guard.nonfinite: a non-finite or diverged state, which HDMI never meets with
            PhysX) ends its episode as terminated, its values of the step left out and its observations zeroed."""
            o = task.step(x, noise)
            bad = nonfinite(x, o.obs, a, t - 1)[0].any(-1)
            values = torch.cat((torch.stack([o.terms[k] for k in keys], -1), o.reward, terms(x, feet_after_step(task))), -1)
            terminated = o.terminated | bad
            done = terminated | truncated
            flags = torch.stack([o.term_flags[k] for k in task.term_names], -1)
            obs = {k: torch.where(bad.view(-1, *([1] * (o.obs[k].dim() - 1))), 0.0, o.obs[k]) for k in groups}
            return obs, done, accumulate(st, values, done, terminated, flags, t, ep_step, L, N, bad)

        def reset_obs(x, noise):
            return {k: v for k, v in task.observations(x, noise).items() if k in groups}

        cmp = (lambda f: torch.compile(f, dynamic=False, options=TASK_COMPILE_OPTIONS)) if compile_ else (lambda f: f)  # noqa: E731
        self.step_fn, self.reset_obs_fn = cmp(step), cmp(reset_obs)

    @torch.no_grad()
    def __call__(self, ck: dict) -> dict:
        """The checkpoint's evaluation: the result of ``summarize`` plus the protocol and the time it took."""
        from ndm_gym.wmbench.collect_hoi import merge_states

        env, task, actor, N = self.env, self.task, self.actor, self.N
        Ns, dev, steps = env.N, env.device, task.cfg.max_episode_length
        actor.load(ck)
        torch.cuda.synchronize()
        t0 = time.time()
        torch.manual_seed(self.seed)  # HLP:217: the resets' draws and the observation noise, the same for every checkpoint
        ids = torch.arange(Ns, device=dev)
        env.reset(ids)
        env.refresh()
        prev = env.read()
        task.reset(ids, prev)
        # the reset observation's noise is drawn for every algorithm, so that all of them see the same random stream
        noise = task.draw_noise(push=False)
        carry = actor.begin(self.reset_obs_fn(env.reset_input(prev), noise) if actor.needs_reset_obs else None)
        st = new_stats(Ns, len(self.names), len(task.term_names), dev)
        a_sim = torch.zeros(Ns, NUM_ACTIONS, device=dev)
        done_sim = torch.ones(Ns, dtype=torch.bool, device=dev)
        for _ in range(steps):  # HLP:227-235
            a_sim[:N].copy_(actor.act(carry, env.t))
            x, truncated = env.step(a_sim, prev)
            obs, done, st = self.step_fn(x, task.draw_noise(), truncated, env.t, env.ep_step, st, a_sim)
            carry = actor.observe(carry, obs, env.t)
            done_sim[:N].copy_(done[:N])
            ids = torch.nonzero_static(done_sim, size=Ns, fill_value=N).squeeze(-1)  # the done envs, then the spare env
            env.reset(ids)
            env.refresh()
            prev = merge_states(done_sim, x.now, env.read())
            task.reset(ids, prev)
            noise = task.draw_noise(push=False)
            carry = actor.reset(carry, done_sim, self.reset_obs_fn(env.reset_input(prev), noise) if actor.needs_reset_obs else None)
        if not bool(st["ended"][:N].all()):
            raise RuntimeError(f"after {steps} steps not every env's first episode has ended")
        res = summarize(st, self.names, task.term_names, N, env.L)
        torch.cuda.synchronize()
        res.update({"task": task.cfg.name, "algo": self.algo, "n_envs": N, "seed": self.seed, "steps": steps, "clip_len": env.L,
                    "max_episode_length": task.cfg.max_episode_length, "eval_s": time.time() - t0})  # fmt: skip
        return res


def parse_args():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="a run of ndm_gym.train_hoi or ndm_gym.train_ppo_hoi (args.json, ckpt_*.pt)")
    parser.add_argument("--watch", action="store_true", help="keep evaluating new checkpoints as they appear")
    parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
    parser.add_argument("--out_json", type=str, default=None, help="--ckpt mode: also write the full result here")
    parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint (iterations in units of the run's --ckpt_every)")
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS, help="HDMI's evaluation script: 2048 (cfg/eval.yaml); its paper: 4096")
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the startup randomization, the resets and the observation noise")
    parser.add_argument("--eager_sim", action="store_true", help="HDMISim without its CUDA graph (the same values)")
    parser.add_argument("--no_compile", action="store_true", help="no torch.compile (the same computations)")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.out_json is not None and args.ckpt is None:
        parser.error("--out_json needs --ckpt")
    if args.ckpt is not None and args.watch:
        parser.error("--ckpt evaluates one checkpoint, --watch the run's")
    return args


def main():
    args = parse_args()

    from isaaclab.app import launch_simulation

    from ndm_gym.sim.rigid.hdmi_sim import make_hdmi_sim

    a = load_run_args(args.run_dir)
    algo = run_algo(a)
    torch.backends.cuda.matmul.allow_tf32 = True  # as the trainers (TRN:27-28)
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda:0"
    sim_cfg = make_hdmi_sim(dev)
    load = lambda path: torch.load(path, map_location=dev, weights_only=False)  # noqa: E731
    with launch_simulation(sim_cfg, args):
        ev = Evaluator(sim_cfg, a, args.num_envs, args.seed, graph=not args.eager_sim, compile_=not args.no_compile)
        if args.ckpt:
            ck = load(args.ckpt)
            check_ckpt(ck, a, args.ckpt)
            res = ev(ck)
            res.update({"ckpt": os.path.abspath(args.ckpt), "iteration": ck["iteration"], "wall_time": ck["wall_time"],
                        "eval_at": time.strftime("%Y-%m-%d %H:%M:%S")})  # fmt: skip
            print(json.dumps({k: v for k, v in res.items() if k != "hdmi"}, indent=1), flush=True)
            if args.out_json:
                with open(args.out_json, "w") as f:
                    json.dump(res, f, indent=1)
            return
        out_csv = os.path.join(args.run_dir, "eval_hoi.csv")
        header = ",".join(ev.columns) + "\n"
        done = set()
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                lines = f.read().splitlines()
            if not lines or lines[0] + "\n" != header:
                raise ValueError(f"{out_csv} does not start with the columns this evaluator writes:\n{header}")
            done = {int(ln.split(",")[0]) for ln in lines[1:]}
        else:
            with open(out_csv, "w") as f:
                f.write(header)
        print(f"{a['task']} ({algo}): {args.num_envs} envs, {ev.task.cfg.max_episode_length} steps per evaluation, clip of {ev.env.L} frames",
              flush=True)  # fmt: skip
        while True:
            ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")), key=ckpt_iteration)
            todo = [c for c in ckpts if ckpt_iteration(c) not in done and (ckpt_iteration(c) // a["ckpt_every"]) % args.every == 0]
            for c in todo:
                it = ckpt_iteration(c)
                ck = load(c)
                check_ckpt(ck, a, c)
                if ck["iteration"] != it:
                    raise ValueError(f"{c} holds iteration {ck['iteration']}")
                res = ev(ck)
                res.update({"ckpt": os.path.abspath(c), "iteration": it, "wall_time": ck["wall_time"], "eval_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                with open(out_csv, "a") as f:
                    f.write(csv_row(ev.columns, res))
                with open(os.path.splitext(c)[0] + "_eval_hoi.json", "w") as f:
                    json.dump(res, f, indent=1)
                m = res["metrics"]
                print(f"[eval_hoi] it {it} ({ck['wall_time'] / 3600:.2f} h): success {m['success']:.3f} ({m['n_success']}/{args.num_envs}), "
                      f"ep_len {m['ep_len']:.1f} of {ev.env.L}, e_kp {1e3 * m['e_kp']:.1f} mm (p90 {1e3 * m['e_kp_p90']:.1f}), "
                      f"e_joint {m['e_joint']:.3f} rad ({res['eval_s']:.0f} s)", flush=True)  # fmt: skip
                done.add(it)
            if not args.watch:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
