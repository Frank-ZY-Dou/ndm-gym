"""Replay MuscleVAE's released tracking policy (ndm_gym.policies.musclevae_pretrained) in ndm_gym's muscle simulator
(ndm_gym.sim.rigid.muscle_sim.MuscleSim) on their walk, run and jump references, and report how it tracks.

Every environment starts at a reference frame and runs --steps control steps (20 Hz) without resets. The policy and the
simulator's step are one CUDA graph, and the metrics are written into device buffers inside it: nothing synchronizes
with the host until the end. Starts (--start): "spread", frames evenly spaced over each clip that leave --steps frames
before its end, so that the reference follows the clip without a jump; "demo", every environment at frame 10, as their
tracking playground's README runs it (track_something.py --start_frame 10). --switch switches the reference to a random
frame every 96 steps and at a clip's end, as their training environment does (CFG env_random_count, play_random_mode 1).
The simulator is theirs by default: ball joints with their Euler limits, ODE's damped step (MuscleSim's ball_joints
"euler", joint_damping "ode"). --exclude_resets leaves out of the starts and resets the reference frames near the Euler
hinges' singularity ("gimbal") or beyond the ball joints' Euler limits ("limits") (MuscleSim's reset_exclude).

Reported per clip and overall (replay.json), after the reset frames' Euler statistics (MuscleSim.reset_frame_stats: the
frames near the singularity and beyond the limits, per ball joint):
* the episodes their termination rule ends within --steps (the head's height off the reference's by more than 0.5 m
  for 20 steps, after 26 steps; or a non-finite state), the steps to termination, the first step off track;
* errors against the reference on our skeleton (the forward kinematics of the tracked frame's joint positions), at
  REPORT_STEPS and over the episode, over the episodes not yet ended: the MPJPE in the root's frame (body positions
  relative to the root, rotated into it, as their observation's), in the heading frame (rotated by the root's yaw only:
  the tilt counts), and in the world (the drift counts); the mean joint rotation error (geodesic, each body relative to
  its parent);
* muscle activation (MA after the control step, the last substep's activation; the fraction above 0.5), fatigue (MF),
  the action's magnitude, the ball joints' Euler-limit excess;
* with --their_dataset (their walk_run_jump.pickle, tools/download_musclevae_pretrained.sh --dataset): the policy's
  target observations against their dataset's (its arrays read from the pickle's opcodes, never unpickled) and the
  checkpoint's normalization against their dataset's statistics (MDS:33-52).
metrics.npz holds the per-step values (steps, environments); states_<clip>_<k>.npz the recorded environments in
tools/render_musclevae.py's --states format (body_state, fatigue_state, ref_body_state, fps).

Run: python tools/replay_musclevae.py --out_dir results/musclevae_replay [--num_envs 768 --steps 512 --start spread]
References: MDS = MuscleVAECore/Utils/motion_dataset.py, CFG = Data/NNModel/config.yml of wangchek/MuscleVAE_ @ 54c001c.
"""

from __future__ import annotations

import argparse
import json
import os
import pickletools
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ndm_gym.core.kinematics import mat_vec  # noqa: E402
from ndm_gym.core.rotations import mat_to_quat, quat_to_mat, yaw_mat  # noqa: E402
from ndm_gym.motion.sources import musclevae as clip_source  # noqa: E402
from ndm_gym.policies import musclevae_pretrained as mp  # noqa: E402
from ndm_gym.robots import musclevae_char as mc  # noqa: E402
from ndm_gym.sim.rigid.muscle_sim import ERR_THRESHOLD, RANDOM_COUNT, RESET_EXCLUSIONS, RESET_TAIL, MuscleSim  # noqa: E402

REPORT_STEPS = (1, 5, 10, 20, 40, 80, 160, 320, 512)
DEMO_FRAME = 10
THEIR_CLIP_ORDER = ("lafan_jump", "lafan_run", "lafan_walk")  # their dataset's first half (the second is mirrored)
THEIR_FIELDS = {"state": ("<f4", 3), "observation": ("<f4", 2), "done": ("|b1", 2)}
OBS_BLOCKS = {"pos": slice(0, 69), "rot": slice(69, 207), "vel": slice(207, 276), "avel": slice(276, 345), "height": slice(345, 368), "up": slice(368, 371)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=768)
    p.add_argument("--steps", type=int, default=512, help="control steps per environment (their episodes' maximum length)")
    p.add_argument("--start", type=str, default="spread", choices=["spread", "demo"])
    p.add_argument("--switch", action="store_true", help="their training environment's random reference switches")
    p.add_argument("--ball_joints", type=str, default="euler", choices=["euler", "ball"])
    p.add_argument("--joint_damping", type=str, default="ode", choices=["ode", "implicit"])
    p.add_argument("--exclude_resets", type=str, nargs="*", default=[], choices=list(RESET_EXCLUSIONS),
                   help="reference frames left out of the starts and resets: near the Euler singularity, beyond the Euler limits")  # fmt: skip
    p.add_argument("--muscle_model", type=str, default="released", choices=["released", "paper"])
    p.add_argument("--latent", type=str, default="mean", choices=list(mp.LATENT_MODES))
    p.add_argument("--action_noise", type=float, default=0.0)
    p.add_argument("--prior_prob", type=float, default=0.0)
    p.add_argument("--record_envs", type=int, default=2, help="environments per clip recorded for tools/render_musclevae.py")
    p.add_argument("--their_dataset", type=str, default=None, help="their walk_run_jump.pickle: check the targets against it")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def their_dataset_arrays(path: str) -> dict:
    """The state, observation and done arrays of their pickled MotionDataSet, read from the pickle's opcodes
    (pickletools) without unpickling: each field's raw bytes with the shape and dtype the opcodes before them state."""
    with open(path, "rb") as f:
        ops = list(pickletools.genops(f.read()))
    out = {}
    for i, (op, arg, _) in enumerate(ops):
        if op.name not in ("SHORT_BINUNICODE", "BINUNICODE") or arg not in THEIR_FIELDS or arg in out:
            continue
        dtype, ndim = THEIR_FIELDS[arg]
        j = next(k for k in range(i + 1, len(ops)) if ops[k][0].name in ("BINBYTES", "SHORT_BINBYTES", "BINBYTES8") and len(ops[k][1]) >= 16)
        ints = [x for o, x, _ in ops[i + 1 : j] if o.name in ("BININT", "BININT1", "BININT2")]
        k = ints.index(1, ints.index(0) + 1)  # ndarray._reconstruct's (0,), then __setstate__'s version 1 and the shape
        shape = tuple(ints[k + 1 : k + 1 + ndim])
        if [o.name for o, _, _ in ops[i + 1 : j] if o.name in ("NEWTRUE", "NEWFALSE")][-1:] != ["NEWFALSE"]:
            raise ValueError(f"{path}: {arg} is not in C order")
        raw = ops[j][1]
        if int(np.prod(shape)) * np.dtype(dtype).itemsize != len(raw):
            raise ValueError(f"{path}: {arg} has {len(raw)} bytes, which its shape {shape} and dtype {dtype} do not fit")
        out[arg] = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
    if set(out) != set(THEIR_FIELDS):
        raise ValueError(f"{path}: fields {sorted(out)}, expected {sorted(THEIR_FIELDS)}")
    return out


def check_their_dataset(path: str, policy: mp.MuscleVAEPolicy) -> dict:
    """Our target observations of their clips against their dataset's, and the checkpoint's normalization against their
    dataset's statistics (MDS:33-52: the mean and standard deviation over every frame, deviations below 0.1 set to 0.1)."""
    d = their_dataset_arrays(path)
    obs, done = d["observation"].astype(np.float64), d["done"][:, 0]
    ours = mp.reference_observations(THEIR_CLIP_ORDER).numpy()
    ends = np.flatnonzero(done)
    if len(ends) != 2 * len(THEIR_CLIP_ORDER) or ends[len(THEIR_CLIP_ORDER) - 1] + 1 != len(ours):
        raise ValueError(f"{path}: clip ends {ends.tolist()}, expected their {len(THEIR_CLIP_ORDER)} clips ({len(ours)} frames) and their mirrors")
    err = np.abs(ours - obs[: len(ours)])
    std = obs.std(0)
    std[std < 0.1] = 0.1
    return {"frames": int(len(ours)), "observation_max_abs_error": float(err.max()), "per_block_max": {k: float(err[:, s].max()) for k, s in OBS_BLOCKS.items()},
            "obs_mean_max_diff": float(np.abs(policy.obs_mean.cpu().numpy() - obs.mean(0)).max()),
            "obs_std_max_diff": float(np.abs(policy.obs_std.cpu().numpy() - std).max())}  # fmt: skip


def start_frames(sim: MuscleSim, n: int, steps: int, mode: str, switch: bool) -> torch.Tensor:
    """n start frames, environment i in clip i mod C: frame 10 ("demo"), or evenly spaced over the frames that leave
    --steps frames before the clip's end ("spread"; with --switch, over the frames a reset may draw); with the
    simulator's reset_exclude, only frames it keeps (evenly spaced over them; an excluded demo frame is an error)."""
    ref, C = sim.ref, len(sim.ref.names)
    kept = torch.zeros(len(ref.clip), dtype=torch.bool)
    kept[ref.reset_frames.cpu()] = True
    out = torch.empty(n, dtype=torch.long)
    for c in range(C):
        idx = torch.arange(c, n, C)
        first = int(ref.start[c])
        last = (int(ref.start[c + 1]) if c + 1 < C else len(ref.clip)) - 1
        if mode == "demo":
            if sim.reset_exclude and not kept[first + DEMO_FRAME]:
                raise ValueError(f"clip {ref.names[c]}: the demo frame {DEMO_FRAME} is excluded by --exclude_resets {list(sim.reset_exclude)}")
            out[idx] = first + DEMO_FRAME
            continue
        hi = last - RESET_TAIL - 1 if switch else last - steps - 1
        if hi <= first + 1:
            raise ValueError(f"clip {ref.names[c]}: too short for {steps} steps from a spread start")
        if not sim.reset_exclude:
            out[idx] = torch.linspace(first + 1, hi, len(idx)).round().long()
            continue
        cand = first + 1 + torch.nonzero(kept[first + 1 : hi + 1]).squeeze(-1)
        if len(cand) == 0:
            raise ValueError(f"clip {ref.names[c]}: --exclude_resets {list(sim.reset_exclude)} leaves no start frame")
        out[idx] = cand[torch.linspace(0, len(cand) - 1, len(idx)).round().long()]
    return out


class Metrics:
    """Per-step values of every environment written inside the control step's graph (PolicyRunner's ``after``) into
    (steps, N) device buffers at a step counter on the device, and the recorded environments' states."""

    VALUES = ("head_err", "mpjpe_root", "mpjpe_heading", "mpjpe_world", "joint_err", "act_mean", "act_high", "fatigue", "action_abs", "ball_excess")

    def __init__(self, sim: MuscleSim, steps: int, record: torch.Tensor):
        dev, N, R = sim.device, sim.N, len(record)
        self.sim, self.record = sim, record.to(dev)
        self.t = torch.zeros(1, dtype=torch.long, device=dev)
        self.val = {k: torch.zeros(steps, N, device=dev) for k in self.VALUES}
        self.term = torch.zeros(steps, N, dtype=torch.bool, device=dev)
        self.rec_body = torch.zeros(steps, R, mc.NUM_BODIES, 13, device=dev)
        self.rec_fatigue = torch.zeros(steps, R, mc.NUM_MUSCLES, 3, device=dev)
        self.rec_frame = torch.zeros(steps, R, dtype=torch.long, device=dev)
        # the reference's bodies on our skeleton: forward kinematics of each frame's joint positions, raised as a reset is
        pos, rot = [], []
        for name in sim.ref.names:
            clip = clip_source.load_clip(os.path.join(mc.CLIPS_DIR, f"{name}.npz"))
            p, r = mc.forward_kinematics(clip.root_p.double(), clip.root_quat.double(), clip.joints.double(), sim.char)
            pos.append(p)
            rot.append(r)
        pos = torch.cat(pos)
        pos[..., 2] += sim.ref.lift.double().cpu()[:, None]
        self.ref_pos, self.ref_rot = pos.float().to(dev), torch.cat(rot).float().to(dev)
        self.parents = torch.tensor(mc.BODY_PARENTS[1:], device=dev)

    def __call__(self, runner: mp.PolicyRunner):
        """After the simulator's step: the step's values into row t (the frame it tracked: runner.tracked)."""
        sim, f = self.sim, runner.tracked
        q = sim.views[0].body_q
        pos, R = q[..., :3], quat_to_mat(q[..., sim.wxyz])
        rp, rR = self.ref_pos[f], self.ref_rot[f]
        Ls, Lr = R[:, self.parents].transpose(-1, -2) @ R[:, 1:], rR[:, self.parents].transpose(-1, -2) @ rR[:, 1:]
        cos = (((Ls * Lr).sum((-1, -2)) - 1.0) / 2.0).clamp(-1.0, 1.0)

        def in_frame(F, p):  # F^T (p - p_root) for frames F (N, 3, 3)
            return mat_vec(F.transpose(-1, -2).unsqueeze(1), p - p[:, :1])

        values = {"head_err": (pos[:, mc.HEAD, 2] - sim.ref.head_z[f]).abs(),
                  "mpjpe_root": (in_frame(R[:, 0], pos) - in_frame(rR[:, 0], rp)).norm(dim=-1).mean(-1),
                  "mpjpe_heading": (in_frame(yaw_mat(R[:, 0]), pos) - in_frame(yaw_mat(rR[:, 0]), rp)).norm(dim=-1).mean(-1),
                  "mpjpe_world": (pos - rp).norm(dim=-1).mean(-1), "joint_err": torch.rad2deg(torch.acos(cos)).mean(-1), "act_mean": sim.MA.mean(-1),
                  "act_high": (sim.MA > 0.5).float().mean(-1), "fatigue": sim.MF.mean(-1), "action_abs": sim.action.abs().mean(-1),
                  "ball_excess": torch.rad2deg(sim.ball_limit_excess().amax((1, 2)))}  # fmt: skip
        for k, v in values.items():
            self.val[k].index_copy_(0, self.t, v.unsqueeze(0))
        self.term.index_copy_(0, self.t, sim.terminated.unsqueeze(0))
        self.rec_body.index_copy_(0, self.t, sim.body_state()[self.record].unsqueeze(0))
        self.rec_fatigue.index_copy_(0, self.t, sim.fatigue_state()[self.record].unsqueeze(0))
        self.rec_frame.index_copy_(0, self.t, f[self.record].unsqueeze(0))
        self.t.add_(1)


def summarize(m: Metrics, clip_of: np.ndarray, names: tuple) -> dict:
    """Per clip and overall: terminations, steps to termination, the first step off track, the metrics at REPORT_STEPS
    and over the episode (the steps up to and with the terminating one)."""
    term = m.term.cpu().numpy()
    val = {k: v.cpu().numpy() for k, v in m.val.items()}
    T, N = term.shape
    ended = term.any(0)
    end_step = np.where(ended, term.argmax(0) + 1, T)
    off = val["head_err"] > ERR_THRESHOLD
    first_off = np.where(off.any(0), off.argmax(0) + 1, T)
    alive = np.arange(T)[:, None] < end_step[None, :]

    def part(sel: np.ndarray) -> dict:
        a = alive[:, sel]
        out = {"episodes": int(sel.sum()), "terminated": float(ended[sel].mean()), "steps_to_termination_mean": float(end_step[sel].mean()),
               "steps_to_termination_median": float(np.median(end_step[sel])), "first_off_track_median": float(np.median(first_off[sel]))}  # fmt: skip
        for k, v in val.items():
            x = v[:, sel]
            out[k] = {"episode_mean": float(x[a].mean()),
                      **{f"step_{s}": (float(x[s - 1][a[s - 1]].mean()) if s <= T and a[s - 1].any() else None) for s in REPORT_STEPS}}  # fmt: skip
        return out

    return {"all": part(np.ones(N, dtype=bool)), **{name: part(clip_of == c) for c, name in enumerate(names)}}


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    torch.set_grad_enabled(False)
    dev = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    sim = MuscleSim(a.num_envs, dev, muscle_model=a.muscle_model, random_count=RANDOM_COUNT if a.switch else 0, seed=a.seed, ball_joints=a.ball_joints,
                    joint_damping=a.joint_damping, reset_exclude=tuple(a.exclude_resets), capture_step=False)  # fmt: skip
    resets = sim.reset_frame_stats()
    worst = sorted(resets["joints"].items(), key=lambda kv: -kv[1]["beyond_limits"])[:4]
    print(f"[replay_musclevae] reset frames: {resets['reset_frames']}, {resets['near_gimbal']} within {resets['gimbal_margin']} rad of the Euler "
          f"singularity, {resets['beyond_limits']} beyond the Euler limits (" + ", ".join(f"{k} {v['beyond_limits']}" for k, v in worst) +
          f"); {resets['kept']} kept (--exclude_resets {a.exclude_resets})", flush=True)  # fmt: skip
    policy = mp.MuscleVAEPolicy.load(device=dev)
    C = len(sim.ref.names)
    if a.num_envs < C * a.record_envs:
        raise ValueError(f"--record_envs {a.record_envs} per clip needs at least {C * a.record_envs} environments")
    per_clip = a.num_envs // C  # the recorded environments spread over each clip's starts: the (2k + 1) / 2R quantiles
    record = torch.tensor([c + C * ((2 * k + 1) * per_clip // (2 * a.record_envs)) for c in range(C) for k in range(a.record_envs)])
    metrics = Metrics(sim, a.steps, record)
    runner = mp.PolicyRunner(sim, policy, latent=a.latent, action_noise=a.action_noise, prior_prob=a.prior_prob, after=metrics)
    build_s = time.time() - t0
    frames = start_frames(sim, a.num_envs, a.steps, a.start, a.switch)
    runner.reset(frames=frames.to(dev))
    metrics.t.zero_()  # the capture's warm-up step wrote row 0
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        runner.step()
    torch.cuda.synchronize()
    run_s = time.time() - t0
    solver = sim.check_solver()  # raises if a contact buffer overflowed
    clip_of = sim.ref.clip[frames.to(dev)].cpu().numpy()
    settings = ("num_envs", "steps", "start", "switch", "ball_joints", "joint_damping", "exclude_resets", "muscle_model", "latent", "action_noise",
                "prior_prob", "seed")  # fmt: skip
    result = {"settings": {k: getattr(a, k) for k in settings}, "checkpoint_sha256": mp.CHECKPOINT_SHA256, "gpu": torch.cuda.get_device_name(),
              "reset_frames": resets,
              "build_seconds": build_s, "run_seconds": run_s, "env_steps_per_second": a.num_envs * a.steps / run_s, "solver_iteration_limit_worlds": solver,
              "metrics": summarize(metrics, clip_of, sim.ref.names)}  # fmt: skip
    if a.their_dataset:
        result["their_dataset"] = check_their_dataset(a.their_dataset, policy)
    with open(os.path.join(a.out_dir, "replay.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez(os.path.join(a.out_dir, "metrics.npz"), term=metrics.term.cpu().numpy(), frames=frames.numpy(), clip=clip_of, names=np.array(sim.ref.names),
             **{k: v.cpu().numpy() for k, v in metrics.val.items()})  # fmt: skip
    for i, e in enumerate(record.tolist()):
        fr = metrics.rec_frame[:, i]
        ref_state = torch.cat((metrics.ref_pos[fr], mat_to_quat(metrics.ref_rot[fr])), -1)
        np.savez(os.path.join(a.out_dir, f"states_{sim.ref.names[e % C]}_{i % a.record_envs}.npz"), body_state=metrics.rec_body[:, i].cpu().numpy(),
                 fatigue_state=metrics.rec_fatigue[:, i].cpu().numpy(), ref_body_state=ref_state.cpu().numpy(), fps=np.float32(mc.CONTROL_HZ),
                 start_frame=np.int64(frames[e]), env=np.int64(e))  # fmt: skip
    m = result["metrics"]["all"]
    print(f"[replay_musclevae] {a.num_envs} envs x {a.steps} steps ({a.start} starts{', switching' if a.switch else ''}; {a.ball_joints} ball joints, "
          f"{a.joint_damping} damping): {a.num_envs * a.steps / run_s:,.0f} env-steps/s; terminated {m['terminated']:.2f}, steps to termination median "
          f"{m['steps_to_termination_median']:.0f}; MPJPE in the root's frame step 10 {m['mpjpe_root']['step_10']:.3f} m, over the episode "
          f"{m['mpjpe_root']['episode_mean']:.3f} m -> {a.out_dir}/replay.json", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
