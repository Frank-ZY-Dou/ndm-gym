"""Train a score-conditioned wrist policy through learned piano dynamics on CUDA."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import torch
from torch import nn
import warp as wp

from check_contact import CONTROL_SUBSTEPS, KEY_COUNT, LOWEST_PITCH, MIDDLE_C, PHYSICS_STEP, build_scene

from piano_score import KEY_DOWN_FRACTION

DEVICE = "cuda:0"
STEPS = 50
CONTROL_DT = PHYSICS_STEP * CONTROL_SUBSTEPS
SCORE = (0.60, 1.05, 1.60, 2.05)


class RepeatedNoteSim:
    def __init__(self, model, worlds):
        self.gpu_model = mjw.put_model(model)
        self.data = mjw.put_data(model, mujoco.MjData(model), nworld=worlds, nconmax=128, njmax=512)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.wrist_q = model.joint("right_height").qposadr[0]
        self.wrist_v = model.joint("right_height").dofadr[0]
        key = model.joint(f"key_{MIDDLE_C - LOWEST_PITCH}_joint")
        self.key_q, self.key_v, self.key_limit = key.qposadr[0], key.dofadr[0], key.range[1]
        self.key_indices = [model.joint(f"key_{i}_joint").qposadr[0] for i in range(KEY_COUNT)]
        limits = torch.as_tensor([model.joint(f"key_{i}_joint").range.tolist() for i in range(KEY_COUNT)], device=DEVICE)
        self.key_lower = limits[:, 0]
        self.key_travel = limits[:, 1] - self.key_lower
        self.wrist_actuator = model.actuator("right_height").id
        mjw.step(self.gpu_model, self.data)
        with wp.ScopedCapture() as capture:
            for _ in range(CONTROL_SUBSTEPS):
                mjw.step(self.gpu_model, self.data)
        self.graph = capture.graph

    def reset(self, wrist_height):
        mjw.reset_data(self.gpu_model, self.data)
        self.qpos[:, self.wrist_q] = wrist_height

    def state(self):
        return torch.stack(
            (
                self.qpos[:, self.wrist_q] / 0.03,
                self.qvel[:, self.wrist_v] / 0.5,
                self.qpos[:, self.key_q] / self.key_limit,
                self.qvel[:, self.key_v] * CONTROL_DT / self.key_limit,
            ),
            dim=-1,
        )

    def step(self, action):
        self.ctrl[:, self.wrist_actuator] = -0.035 * action
        wp.capture_launch(self.graph)
        return self.state()


def score_features(frame):
    time = frame * CONTROL_DT
    edges = torch.tensor(SCORE, device=frame.device)
    return ((time[..., None] - edges) / 0.25).clamp(-1, 1)


def due_at(frame):
    start1, end1, start2, end2 = (round(t / CONTROL_DT) for t in SCORE)
    return ((frame >= start1) & (frame < end1)) | ((frame >= start2) & (frame < end2))


def imagined_key_reward(travel, due):
    # Eden's key-press term specialized to a world model of middle C. Keep its
    # binary forward value while supplying a gradient at the wrong-key boundary.
    travel = travel.clamp(0, 1)
    due_credit = torch.exp(-math.log(10) * (0.95 - travel).clamp_min(0).square())
    soft_down = torch.sigmoid(16 * (travel - KEY_DOWN_FRACTION))
    hard_down = (travel >= KEY_DOWN_FRACTION).to(travel.dtype)
    down = hard_down + (soft_down - soft_down.detach())
    return 0.5 * due * due_credit + 0.5 * (1 - (~due) * down)


def frame_metrics(played, goal):
    hits = (played & goal).sum(-1)
    down = played.sum(-1)
    due = goal.sum(-1)

    def ratio(count, total):
        return torch.where(total > 0, count / total.clamp_min(1), 1.0)

    return torch.stack((ratio(hits, down), ratio(hits, due), ratio(2 * hits, down + due)), dim=-1)


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, outputs))


def predict(model, state, action):
    return state + model(torch.cat((state, action[:, None]), dim=-1))


def action_of(policy, state, frame):
    return policy(torch.cat((state, score_features(frame)), dim=-1)).squeeze(-1).sigmoid()


def collect(sim, worlds):
    sim.reset(torch.empty(worlds, device=DEVICE).uniform_(-0.003, 0.003))
    transitions = [sim.state()]
    actions = []
    jitter = torch.randint(-3, 4, (worlds, 2), device=DEVICE)
    depths = torch.empty(worlds, 2, device=DEVICE).uniform_(0.35, 0.89)
    for step in range(STEPS):
        phase = step - jitter
        pressed = (phase >= torch.tensor([10, 30], device=DEVICE)) & (phase < torch.tensor([20, 40], device=DEVICE))
        scripted = (pressed * depths).sum(-1)
        action = (scripted + 0.07 * torch.randn_like(scripted)).clamp(0, 0.95)
        actions.append(action)
        transitions.append(sim.step(action))
    return torch.stack(transitions), torch.stack(actions)


def fit_world_model(model, states, actions, updates):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    worlds = actions.shape[1]
    for _ in range(updates):
        frame = torch.randint(STEPS - 2, (384,), device=DEVICE)
        env = torch.randint(worlds, (384,), device=DEVICE)
        state = states[frame, env]
        loss = 0
        for offset in range(3):
            state = predict(model, state, actions[frame + offset, env])
            error = state - states[frame + offset + 1, env]
            loss += error.square().mean() + error[:, 2].square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()


@torch.no_grad()
def model_error(model, states, actions):
    state = states[:-3].flatten(0, 1)
    for offset in range(3):
        state = predict(model, state, actions[offset : STEPS - 2 + offset].flatten())
    return (state[:, 2] - states[3:].flatten(0, 1)[:, 2]).abs().mean().item()


def train_policy(policy, model, states, updates, on_update):
    model.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    for update in range(1, updates + 1):
        frame = torch.randint(STEPS - 7, (384,), device=DEVICE)
        env = torch.randint(states.shape[1], (384,), device=DEVICE)
        state = states[frame, env]
        loss = 0
        for offset in range(8):
            action = action_of(policy, state, frame + offset)
            state = predict(model, state, action)
            reward = imagined_key_reward(state[:, 2], due_at(frame + offset))
            loss = loss - reward.mean() + 0.015 * action.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1)
        optimizer.step()
        if update % 100 == 0 or update == updates:
            on_update(update)


@torch.no_grad()
def evaluate(sim, policy, starts):
    sim.reset(starts)
    history = []
    for frame in range(STEPS):
        action = action_of(policy, sim.state(), torch.full_like(starts, frame))
        sim.step(action)
        depth = ((sim.qpos[:, sim.key_indices] - sim.key_lower) / sim.key_travel).clamp(0, 1)
        history.append(depth >= KEY_DOWN_FRACTION)
    played = torch.stack(history)
    goal = torch.zeros((STEPS, 1, KEY_COUNT), device=DEVICE, dtype=torch.bool)
    goal[:, 0, MIDDLE_C - LOWEST_PITCH] = due_at(torch.arange(STEPS, device=DEVICE))
    scores = frame_metrics(played, goal).mean(0).tolist()
    # Keep the per-frame keyboard state so the reported metrics can be audited.
    cases = []
    for metrics, keyboard in zip(scores, played.transpose(0, 1).cpu().numpy()):
        cases.append(
            dict(zip(("precision", "recall", "f1"), metrics), played_midi_by_frame=[(row.nonzero()[0] + LOWEST_PITCH).tolist() for row in keyboard])
        )
    return cases


def mean_scores(cases):
    return {metric: sum(case[metric] for case in cases) / len(cases) for metric in ("precision", "recall", "f1")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", type=Path, required=True)
    parser.add_argument("--keyboard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-model-updates", type=int, default=1800)
    parser.add_argument("--policy-updates", type=int, default=900)
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()
    if args.world_model_updates <= 0 or args.policy_updates <= 0:
        parser.error("training update counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch and a visible NVIDIA GPU are required")
    torch.manual_seed(args.seed)
    wp.init()
    torch.cuda.set_stream(torch.cuda.Stream())
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_scene(args.hand, args.keyboard)
    starts = torch.cat((torch.zeros(1, device=DEVICE), torch.linspace(-0.003, 0.003, 16, device=DEVICE)))
    validation_starts = torch.linspace(-0.002, 0.002, 9, device=DEVICE)
    with wp.ScopedDevice(DEVICE):
        evaluation = RepeatedNoteSim(model, len(starts))
        policy = network(8, 1).to(DEVICE)
        if args.evaluate:
            policy.load_state_dict(torch.load(args.evaluate, map_location=DEVICE, weights_only=True)["policy"])
        else:
            validation = RepeatedNoteSim(model, len(validation_starts))
            sim = RepeatedNoteSim(model, 256)
            states, actions = collect(sim, 256)
            dynamics = network(5, 4).to(DEVICE)
            fit_world_model(dynamics, states[:, :224], actions[:, :224], args.world_model_updates)
            print(f"Held-out 3-step key travel MAE: {model_error(dynamics, states[:, 224:], actions[:, 224:]):.4f}", flush=True)
            best = -1.0

            def save_best(update):
                nonlocal best
                cases = evaluate(validation, policy, validation_starts)
                f1 = mean_scores(cases)["f1"]
                print(f"Policy update {update}: mean validation F1 {f1:.4f}", flush=True)
                if f1 > best:
                    best = f1
                    torch.save(
                        {"policy": policy.state_dict(), "world_model": dynamics.state_dict(), "score": SCORE, "update": update},
                        args.out / "checkpoint.pt",
                    )

            train_policy(policy, dynamics, states[:, :224], args.policy_updates, save_best)
            policy.load_state_dict(torch.load(args.out / "checkpoint.pt", map_location=DEVICE, weights_only=True)["policy"])
        cases = evaluate(evaluation, policy, starts)
        report = {
            "score_s": SCORE,
            "key_down_fraction": KEY_DOWN_FRACTION,
            "control_dt": CONTROL_DT,
            "device": torch.cuda.get_device_name(),
            "seed": args.seed,
            "mean": mean_scores(cases),
            "starts_m": starts.tolist(),
            "cases": cases,
        }
        (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Final evaluation: {mean_scores(cases)}", flush=True)


if __name__ == "__main__":
    main()
