"""Evaluate a policy trained by ndm_gym.train_muscles in MuscleSim on MuscleVAE's references, deterministically (no action
noise), with the metrics tools/replay_musclevae.py reports for their released policy (its Metrics and summarize): per clip
and overall, the episodes their termination rule ends within --steps, the steps to termination, the first step off track,
and the errors against the reference on our skeleton at REPORT_STEPS and over the episode (MPJPE in the root's frame, the
heading frame and the world; the joint rotation error), with the muscles' activation, fatigue and the actions.

Every environment starts at a reference frame (--start spread: evenly spaced over each clip, leaving --steps frames; or
"demo", frame 10) and runs --steps control steps without resets; the reference follows the clip (--switch: MuscleVAE's
random switches every 96 steps and at a clip's end). The simulator is the run's (ball joints, joint damping, muscle model,
clips). Policy and step are one CUDA graph; nothing synchronizes with the host until the end. The policy runs with the
run's autocast (--amp), as ndm_gym.evaluate does.

The recorded environments (--record_envs per clip) are written as tools/replay_musclevae.py writes them, states_<clip>_<k>.npz in
eval_<checkpoint>_states/ beside the report, for tools/render_musclevae.py --states.

Run: python tools/evaluate_muscles.py --ckpt logs/<run>/ckpt_XXXXXX.pt [--num_envs 768 --steps 512] (eval.json beside it)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import ndm_gym.core.nets as nets  # noqa: E402
from ndm_gym.algos.muscle_algos import MUSCLE_ALGOS  # noqa: E402
from ndm_gym.core.rotations import mat_to_quat  # noqa: E402
from ndm_gym.policies.muscle_policy import MusclePolicy, goal_table, policy_observation  # noqa: E402
from ndm_gym.robots import musclevae_char as mc  # noqa: E402
from ndm_gym.sim.rigid.muscle_sim import RANDOM_COUNT, MuscleSim  # noqa: E402
from ndm_gym.tasks import muscle_tracking as mt  # noqa: E402
from replay_musclevae import Metrics, start_frames, summarize  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="a checkpoint of ndm_gym.train_muscles (ckpt_*.pt or last.pt)")
    p.add_argument("--out", type=str, default=None, help="the report (default: eval_<checkpoint>.json beside the checkpoint)")
    p.add_argument("--num_envs", type=int, default=768)
    p.add_argument("--steps", type=int, default=512, help="control steps per environment (MuscleVAE's episodes' maximum length)")
    p.add_argument("--start", type=str, default="spread", choices=["spread", "demo"])
    p.add_argument("--switch", action="store_true", help="MuscleVAE's random reference switches")
    p.add_argument("--record_envs", type=int, default=2, help="environments per clip recorded for tools/render_musclevae.py")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class Runner:
    """The trained policy driving a MuscleSim (constructed with capture_step=False), policy and step as one CUDA graph:
    the state from the simulator's buffers (tasks.muscle_tracking.body_states), the deterministic action into its action
    buffer, the step, then ``after`` (the metrics). ``tracked``: the reference frame of the last step."""

    def __init__(self, sim: MuscleSim, pi: MusclePolicy, goal: torch.Tensor, ref_action: torch.Tensor | None, after):
        self.sim, self.pi, self.goal, self.ref_action, self.after = sim, pi, goal, ref_action, after
        self.tracked = torch.zeros(sim.N, dtype=torch.long, device=sim.device)
        self.graph = sim.capture(self._body)
        sim.reset()

    def _body(self):
        sim, v = self.sim, self.sim.views[0]
        S = mt.unflatten_bodies(mt.body_states(v.body_q, v.body_qd, sim.before, sim.just_reset))
        f = sim.ref_frame
        a = self.pi(policy_observation(S, sim.part_fatigue()), self.goal[f])
        if self.ref_action is not None:
            a = a + self.ref_action[f]
        sim.action.copy_(a)
        self.tracked.copy_(f)
        sim.step_body()
        self.after(self)

    def step(self):
        self.graph.replay()


@torch.no_grad()
def main():
    a = parse_args()
    dev = "cuda:0"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    targs = ck["args"]
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    nets.AMP_DTYPE = torch.bfloat16 if targs["amp"] == "bf16" else None
    t0 = time.time()
    sim = MuscleSim(a.num_envs, dev, clips=tuple(targs["mv_clips"]), muscle_model=targs["muscle_model"], random_count=RANDOM_COUNT if a.switch else 0,
                    seed=a.seed, ball_joints=targs["ball_joints"], joint_damping=targs["joint_damping"], capture_step=False)  # fmt: skip
    ref = mt.reference_from_sim(sim)
    pi = MusclePolicy().to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    C = len(sim.ref.names)
    if a.num_envs < C * a.record_envs:
        raise ValueError(f"--record_envs {a.record_envs} per clip needs at least {C * a.record_envs} environments")
    per_clip = a.num_envs // C
    record = torch.tensor([c + C * ((2 * k + 1) * per_clip // (2 * a.record_envs)) for c in range(C) for k in range(a.record_envs)])
    metrics = Metrics(sim, a.steps, record)
    residual = MUSCLE_ALGOS[targs["algo"]].residual
    runner = Runner(sim, pi, goal_table(ref), ref.action if residual else None, metrics)
    frames = start_frames(sim, a.num_envs, a.steps, a.start, a.switch)
    sim.reset(frames=frames.to(dev))
    metrics.t.zero_()  # the capture's warm-up step wrote row 0
    build_s = time.time() - t0
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        runner.step()
    torch.cuda.synchronize()
    run_s = time.time() - t0
    solver = sim.check_solver()
    clip_of = sim.ref.clip[frames.to(dev)].cpu().numpy()
    result = {"checkpoint": os.path.abspath(a.ckpt), "iteration": ck["iteration"], "algo": targs["algo"], "wall_time_h": ck["wall_time"] / 3600,
              "settings": {k: getattr(a, k) for k in ("num_envs", "steps", "start", "switch", "seed")}, "gpu": torch.cuda.get_device_name(),
              "build_seconds": build_s, "run_seconds": run_s, "env_steps_per_second": a.num_envs * a.steps / run_s,
              "solver_iteration_limit_worlds": solver, "metrics": summarize(metrics, clip_of, sim.ref.names)}  # fmt: skip
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), f"eval_{os.path.splitext(os.path.basename(a.ckpt))[0]}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    states_dir = os.path.splitext(out)[0] + "_states"
    os.makedirs(states_dir, exist_ok=True)
    for i, e in enumerate(record.tolist()):  # as tools/replay_musclevae.py writes them
        fr = metrics.rec_frame[:, i]
        ref_state = torch.cat((metrics.ref_pos[fr], mat_to_quat(metrics.ref_rot[fr])), -1)
        np.savez(os.path.join(states_dir, f"states_{sim.ref.names[e % C]}_{i % a.record_envs}.npz"), body_state=metrics.rec_body[:, i].cpu().numpy(),
                 fatigue_state=metrics.rec_fatigue[:, i].cpu().numpy(), ref_body_state=ref_state.cpu().numpy(), fps=np.float32(mc.CONTROL_HZ),
                 start_frame=np.int64(frames[e]), env=np.int64(e))  # fmt: skip
    lines = []
    for name, m in result["metrics"].items():
        step10 = 1e3 * (m["mpjpe_root"]["step_10"] or float("nan"))
        lines.append(f"  {name:>10}: ended {m['terminated']:.2f} (median steps {m['steps_to_termination_median']:.0f}), MPJPE root frame "
                     f"over the episode {1e3 * m['mpjpe_root']['episode_mean']:.1f} mm (step 10 {step10:.1f}), joint error "
                     f"{m['joint_err']['episode_mean']:.2f} deg")  # fmt: skip
    print(f"[evaluate_muscles] {targs['algo']} it {ck['iteration']}: {a.num_envs} envs x {a.steps} steps ({a.start} starts"
          f"{', switching' if a.switch else ''}) -> {out}\n" + "\n".join(lines), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
