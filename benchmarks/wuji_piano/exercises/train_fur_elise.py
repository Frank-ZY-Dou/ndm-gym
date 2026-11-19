"""Train and evaluate a two-hand score-conditioned wrist controller on Für Elise."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .check_contact import CONTROL_SUBSTEPS, LOWEST_PITCH, KEY_COUNT, PHYSICS_STEP, mount_hand
from ndm_gym.tasks.piano import KEY_DOWN_FRACTION, frame_metrics
from .train_repeated_note import network

# Mutopia 931. Sixteenth-note slots, 0 = rest. Adjacent equal pitches here are tied durations.
RIGHT = [76, 75, 76, 75, 76, 71, 74, 72, 69, 69, 0, 60, 64, 69, 71, 71, 0, 64, 68, 71, 72, 72, 0, 64, 76, 75, 76, 75, 76, 71, 74, 72]
LEFT = [0] * 8 + [45, 52, 57, 0, 0, 0, 40, 52, 56, 0, 0, 0, 45, 52, 57, 0, 0, 0] + [0] * 6
DEVICE = "cuda:0"
DT = PHYSICS_STEP * CONTROL_SUBSTEPS
WARMUP = 10
MUSIC_FRAMES = round(len(RIGHT) * 60 / 72 / 4 / DT)
SIDES = ("right", "left")
KEY_SPLIT = 60 - LOWEST_PITCH
# Units used to normalize wrist residuals (metres along the three axes).
SCALE = (0.021, 0.016, 0.012)


def score():
    """Round each score boundary to the nearest control frame."""
    boundaries = np.rint(np.arange(33) * 60 / 72 / 4 / DT).astype(int)
    parts = [np.repeat(part, np.diff(boundaries)) for part in (RIGHT, LEFT)]
    return np.pad(np.stack(parts, axis=1), ((WARMUP, 5), (0, 0)))


def build(hand_dir, keyboard):
    spec = mujoco.MjSpec.from_file(str(keyboard))
    spec.option.timestep = PHYSICS_STEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for i in range(KEY_COUNT):
        spec.body(f"key_{i}").gravcomp = 1
    for side in SIDES:
        mount_hand(spec, hand_dir / f"{side}.xml", side)
        carrier = spec.body(f"{side}_carrier")
        # A near-vertical index can reach the black-key row without dragging the
        # palm across neighboring keys. Keep the other digits curled.
        carrier.pos = [
            0.111 if side == "right" else -0.050,
            (-0.0073 if side == "left" else 0.0073) + (0.15 if side == "right" else -0.32),
            0.205 if side == "right" else 0.191,
        ]
        carrier.add_joint(name=f"{side}_reach", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.12, 0.055])
        act = spec.add_actuator(name=f"{side}_reach", target=f"{side}_reach", trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-0.12, 0.055])
        act.set_to_position(kp=5000, kv=90)
        for axis, limits, kp, kv in (("lateral", [-0.55, 0.55], 6500, 100), ("height", [-0.05, 0.045], 3500, 65)):
            name = f"{side}_{axis}"
            spec.joint(name).range = limits
            act = spec.actuator(name)
            act.ctrlrange = limits
            act.set_to_position(kp=kp, kv=kv)
    model = spec.compile()
    for side in SIDES:
        a = np.deg2rad(16.5 if side == "right" else -10)
        model.body(f"{side}/{side[0]}_wrist").quat[:] = [np.cos(a), 0, np.sin(a), 0]
    # Hold the flexed digits during fast wrist moves, retaining the MJCF effort limits.
    for i in range(model.nu):
        if "/r_" in model.actuator(i).name or "/l_" in model.actuator(i).name:
            gain = 12 if "_FFJ" in model.actuator(i).name else 5
            model.actuator_gainprm[i, 0] *= gain
            model.actuator_biasprm[i, 1] *= gain
            model.actuator_biasprm[i, 2] *= np.sqrt(gain)
    return model


def reference(model, parts):
    """Score geometry initializes setpoints; future notes position resting hands."""
    out = np.zeros((len(parts), 2, 3), np.float32)
    for side in range(2):
        next_pitch = parts[:, side].max()
        for t in reversed(range(len(parts))):
            due = parts[t, side]
            if due:
                next_pitch = due
            elif t == len(parts) - 1:
                next_pitch = parts[:, side][parts[:, side] > 0][-1]
            body = model.body(f"key_{next_pitch-LOWEST_PITCH}")
            black = next_pitch % 12 in (1, 3, 6, 8, 10)
            out[t, side] = [-0.045 if black else 0, body.pos[1] - (0.15 if side == 0 else -0.32), (-0.017 if black else -0.030) if due else 0.014]
    # Feed forward one part of the lateral step to compensate for PD lag.
    out[1:, :, :2] += 0.20 * np.diff(out[:, :, :2], axis=0)
    return torch.as_tensor(out, device=DEVICE)


class Scene:
    def __init__(self, model, worlds):
        self.m = model
        self.gm = mjw.put_model(model)
        self.gd = mjw.put_data(model, mujoco.MjData(model), nworld=worlds, nconmax=256, njmax=1024)
        self.q = wp.to_torch(self.gd.qpos)
        self.v = wp.to_torch(self.gd.qvel)
        self.ctrl = wp.to_torch(self.gd.ctrl)
        axes = [f"{s}_{a}" for s in SIDES for a in ("reach", "lateral", "height")]
        self.aidx = [model.actuator(a).id for a in axes]
        self.qidx = [model.joint(a).qposadr[0] for a in axes]
        self.vidx = [model.joint(a).dofadr[0] for a in axes]
        keys = [model.joint(f"key_{i}_joint") for i in range(KEY_COUNT)]
        self.kidx = [k.qposadr[0] for k in keys]
        self.limits = torch.tensor([k.range[1].item() for k in keys], device=DEVICE)
        self.curl = [
            (a, model.jnt_qposadr[model.actuator(a).trnid[0]], min(1.2, model.actuator(a).ctrlrange[1]))
            for a in range(model.nu)
            if any(part in model.actuator(a).name for part in ("_RFJ", "_MFJ", "_LFJ", "_THJ")) and model.actuator(a).name[-1] in "023"
        ]
        self.set_fingers()
        mjw.step(self.gm, self.gd)
        self.graphs = []
        for n in (3, CONTROL_SUBSTEPS - 3):
            with wp.ScopedCapture() as capture:
                for _ in range(n):
                    mjw.step(self.gm, self.gd)
            self.graphs.append(capture.graph)

    def set_fingers(self):
        for actuator, joint, angle in self.curl:
            self.q[:, joint] = angle
            self.ctrl[:, actuator] = angle

    def reset(self, initial):
        mjw.reset_data(self.gm, self.gd)
        self.set_fingers()
        self.q[:, self.qidx] = initial.flatten(1)
        self.ctrl[:, self.aidx] = initial.flatten(1)

    def advance(self, ctrl):
        # Unload the fingertip during the first 15 ms of a traverse.
        moving = (self.q[:, self.qidx].reshape(-1, 2, 3)[:, :, :2] - ctrl[:, :, :2]).abs().amax(-1) > 0.012
        approach = ctrl.clone()
        approach[:, :, 2] = torch.where(moving, 0.012, ctrl[:, :, 2])
        self.ctrl[:, self.aidx] = approach.flatten(1)
        wp.capture_launch(self.graphs[0])
        self.ctrl[:, self.aidx] = ctrl.flatten(1)
        wp.capture_launch(self.graphs[1])
        return (self.q[:, self.kidx] / self.limits).clamp(0, 1)


def targets(parts):
    midi = torch.arange(LOWEST_PITCH, LOWEST_PITCH + KEY_COUNT, device=DEVICE)
    return (torch.as_tensor(parts, device=DEVICE)[:, :, None] == midi).any(1)


def wrist_features(scene, ref, scale):
    pos = scene.q[:, scene.qidx].reshape(-1, 2, 3)
    vel = scene.v[:, scene.vidx].reshape(-1, 2, 3)
    return torch.cat(((pos - ref) / scale, vel * DT / scale), -1).clamp(-12, 12)


class ScorePolicy(torch.nn.Module):
    """A learned correction per score frame plus shared neural wrist feedback."""

    def __init__(self, frames):
        super().__init__()
        self.offsets = torch.nn.Embedding(frames, 6)
        self.feedback = network(9, 3)
        torch.nn.init.zeros_(self.offsets.weight)
        torch.nn.init.zeros_(self.feedback[-1].weight)
        torch.nn.init.zeros_(self.feedback[-1].bias)

    def forward(self, frame, features, reference):
        context = torch.cat((features, reference.expand_as(features[..., :3]) * 10), -1)
        return self.offsets.weight[frame].reshape(-1, 2, 3) + 0.2 * self.feedback(context).tanh()


@torch.no_grad()
def rollout(scene, ref, residual, perturb, record=False):
    scale = torch.tensor(SCALE, device=DEVICE)
    scene.reset(ref[0] + perturb)
    travels, inputs, snapshots = [], [], []
    for i in range(len(ref)):
        features = wrist_features(scene, ref[i], scale)
        correction = residual(i, features, ref[i]) if isinstance(residual, ScorePolicy) else residual[i]
        # Bounded state feedback around a learned score-indexed setpoint trajectory.
        act = (correction - 0.04 * features[:, :, :3]).clamp(-2.8, 2.8)
        depth = scene.advance(ref[i] + act * scale)
        travels.append(depth)
        inputs.append(torch.cat((features, act), -1))
        if record:
            snapshots.append(scene.q[0].clone())
    return torch.stack(travels), torch.stack(inputs), snapshots


def per_hand_quality(depth, goal):
    """Local key-depth objective, including the deepest undesired key in each register."""
    quality = []
    for section in (slice(KEY_SPLIT, None), slice(None, KEY_SPLIT)):
        d = depth[:, :, section]
        g = goal[:, None, section]
        desired = (d * g).sum(-1)
        wrong = (d * (~g)).amax(-1)
        grade = frame_metrics(d >= KEY_DOWN_FRACTION, g)[..., 2]
        quality.append(grade + 0.45 * desired.clamp(max=0.98) - 0.45 * wrong)
    return torch.stack(quality, -1)


def stats(depth, goal):
    music = slice(WARMUP, WARMUP + MUSIC_FRAMES)
    played = depth[music] >= KEY_DOWN_FRACTION
    goal = goal[music, None, :]
    result = {}
    for name, section in (("all", slice(None)), ("right_register", slice(KEY_SPLIT, None)), ("left_register", slice(None, KEY_SPLIT))):
        result[name] = frame_metrics(played[..., section], goal[..., section]).mean(0).cpu().tolist()
    result["perfect_rollouts"] = (played == goal).all(2).all(0).sum().item()
    result["exact_frame_share"] = (played == goal).all(2).float().mean().item()
    return result


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def search_actions(scene, ref, goal, center, perturb, spread):
    """Branch wrist commands from the best partial trajectories every control step."""
    worlds = len(perturb)
    scale = torch.tensor(SCALE, device=DEVICE)
    scene.reset(ref[0] + perturb)
    beam = min(6, worlds // 4)
    scores = torch.zeros(worlds, device=DEVICE)
    trajectory = torch.zeros((worlds, len(ref), 2, 3), device=DEVICE)
    transitions, depths = [], []
    warm = wp.to_torch(scene.gd.qacc_warmstart)
    noise = torch.randn_like(trajectory) * spread
    for t in range(len(ref)):
        features = wrist_features(scene, ref[t], scale)
        candidate = (center[t] + noise[:, t]).clamp(-2.5, 2.5)
        candidate[:beam] = center[t]
        if t:
            # Some trajectories keep the previous frame's correction.
            candidate[-beam:] = trajectory[-beam:, t - 1]
        act = (candidate - 0.04 * features[:, :, :3]).clamp(-2.8, 2.8)
        depth = scene.advance(ref[t] + act * scale)
        transitions.append(torch.cat((features, act), -1))
        depths.append(depth)
        grade = per_hand_quality(depth[None], goal[t : t + 1])[0].sum(-1)
        # Contact alone cannot guide a fingertip that has not yet reached the key.
        position_error = ((scene.q[:, scene.qidx].reshape(-1, 2, 3)[:, :, :2] - ref[t, :, :2]) / scale[:2]).abs().mean((1, 2))
        scores += grade - 0.06 * position_error
        trajectory[:, t] = candidate
        if t + 1 < len(ref):
            winners = scores.topk(beam).indices
            parent = winners[torch.arange(worlds, device=DEVICE) % beam]
            for buf in (scene.q, scene.v, warm, scene.ctrl):
                buf.copy_(buf[parent])
            trajectory = trajectory[parent]
            scores = scores[parent]
    return trajectory[scores.argmax()][:, None], torch.stack(transitions), torch.stack(depths)


def learn(scene, ref, goal, worlds, rounds, seed, out):
    torch.manual_seed(seed)
    zero = torch.zeros(len(ref), 1, 2, 3, device=DEVICE)
    best = zero.clone()
    actor = ScorePolicy(len(ref)).to(DEVICE)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=0.0005)
    # A local neural transition model predicts next key travel around the score
    # from wrist state, six continuous actions and phase. Train anew per seed.
    key_ids = goal.any(0).nonzero().flatten()
    phase = torch.eye(len(ref), device=DEVICE)
    dynamics = network(len(ref) + 18, len(key_ids)).to(DEVICE)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=0.002)
    perturb = torch.empty(worlds, 2, 3, device=DEVICE).uniform_(-0.002, 0.002)
    perturb[0] = 0
    validation = torch.empty_like(perturb).uniform_(-0.002, 0.002, generator=torch.Generator(device=DEVICE).manual_seed(456))
    validation[0] = 0
    history = []
    first_hit = None
    best_f1 = -1.0
    start = sync_time()
    for iteration in range(rounds + 1):
        spread = max(0.16, 0.85 * (0.80**iteration))
        depth, _, _ = rollout(scene, ref, actor, validation)
        exact = float(np.mean(stats(depth, goal)["all"], axis=0)[2])
        elapsed = sync_time() - start
        if exact >= 0.95 and first_hit is None:
            first_hit = elapsed
        history.append({"round": iteration, "validation_f1": exact, "seconds": elapsed})
        print(f"seed={seed} round={iteration} F1={exact:.4f} training={elapsed:.1f}s", flush=True)
        if iteration == 0 or exact > best_f1:
            best_f1 = exact
            torch.save(
                {"residual": best[:, 0].cpu(), "actor": actor.state_dict(), "dynamics": dynamics.state_dict(), "seed": seed, "round": iteration},
                out / "checkpoint.pt",
            )
        if iteration == rounds or exact == 1.0:
            break
        if exact < best_f1:
            saved = torch.load(out / "checkpoint.pt", map_location=DEVICE, weights_only=True)
            best = saved["residual"][:, None]
            actor.load_state_dict(saved["actor"])
        elite_actions, inputs, depth = search_actions(scene, ref, goal, best, perturb, spread)
        # Fit the forward model to this batch; differentiate predicted due and
        # extra key travel to refine the elite command at each score frame.
        flat = inputs.flatten(2)
        train_target = depth[:, :, key_ids]
        for _ in range(140):
            t = torch.randint(len(ref), (384,), device=DEVICE)
            w = torch.randint(worlds, (384,), device=DEVICE)
            prediction = dynamics(torch.cat((phase[t], flat[t, w]), -1))
            label = train_target[t, w]
            loss = ((prediction - label).square() * (1 + 4 * label)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # Distill the best complete trajectory and refine the score-conditioned actor
        # through the learned key transition model, regularizing it to the physics search.
        with torch.no_grad():
            actor.offsets.weight.copy_(elite_actions[:, 0].flatten(1))
        dynamics.requires_grad_(False)
        for _ in range(80):
            t = torch.randint(len(ref), (384,), device=DEVICE)
            w = torch.randint(worlds, (384,), device=DEVICE)
            anchor = inputs[t, w, :, :6]
            correction = actor(t, anchor, ref[t])
            act = (correction - 0.04 * anchor[:, :, :3]).clamp(-2.8, 2.8)
            pred = dynamics(torch.cat((phase[t], torch.cat((anchor, act), -1).flatten(1)), -1))
            due = goal[t][:, key_ids]
            track = ((pred - due.float()).square() * (1 + 4 * due)).mean()
            loss = track + 4 * (correction - elite_actions[t, 0]).square().mean()
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            actor_opt.step()
        dynamics.requires_grad_(True)
        best = actor.offsets.weight.detach().reshape(len(ref), 1, 2, 3).clone()
    return {"training_s": sync_time() - start, "time_to_f1_095_s": first_hit, "history": history}


def main():
    from benchmarks.wuji_piano.training_options import training_defaults

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-dir", type=Path, required=True)
    parser.add_argument("--keyboard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--reference", action="store_true", help="Evaluate the untrained geometric wrist controller")

    args = parser.parse_args()
    training_defaults(parser, args, training=args.replay is None and not args.reference, rounds=12, seeds=[0, 1, 2])
    if args.worlds < 8 or 256 % args.worlds:
        parser.error("--worlds must be at least 8 and divide 256")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    wp.init()
    torch.cuda.set_stream(torch.cuda.Stream())
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    args.out.mkdir(parents=True, exist_ok=True)
    with wp.ScopedDevice(DEVICE):
        t0 = sync_time()
        model = build(args.hand_dir, args.keyboard)
        parts = score()
        ref = reference(model, parts)
        goal = targets(parts)
        scene = Scene(model, args.worlds)
        setup = sync_time() - t0
        seeds = args.seeds
        if args.replay:
            seeds = [torch.load(args.replay, map_location="cpu", weights_only=True)["seed"]]
        elif args.reference:
            seeds = [0]
        for seed in seeds:
            folder = args.out / ("reference" if args.reference else f"seed{seed}")
            folder.mkdir(exist_ok=True)
            info = {}
            if not args.replay and not args.reference:
                info = learn(scene, ref, goal, args.worlds, args.rounds, seed, folder)
            actor = ScorePolicy(len(ref)).to(DEVICE)
            selected_round = 0
            if not args.reference:
                ck = torch.load(args.replay or folder / "checkpoint.pt", map_location=DEVICE, weights_only=True)
                actor.load_state_dict(ck["actor"])
                selected_round = ck["round"]
            evaluation_worlds = 256
            perturb = torch.empty(evaluation_worlds, 2, 3, device=DEVICE)
            perturb.uniform_(-0.003, 0.003, generator=torch.Generator(device=DEVICE).manual_seed(789))
            perturb[0] = 0
            start = sync_time()
            batches = []
            for i, starts in enumerate(perturb.split(args.worlds)):
                depth, _, snapshots = rollout(scene, ref, actor, starts, i == 0)
                batches.append(depth)
                if i == 0:
                    q = snapshots
            depth = torch.cat(batches, 1)
            info.update(stats(depth, goal))
            info["evaluation_starts"] = evaluation_worlds
            info["active_note_recall"] = {}
            for side, section in (("right", slice(KEY_SPLIT, None)), ("left", slice(None, KEY_SPLIT))):
                music = slice(WARMUP, WARMUP + MUSIC_FRAMES)
                target = goal[music, None, section]
                info["active_note_recall"][side] = ((depth[music, :, section] >= KEY_DOWN_FRACTION) & target).sum().item() / (
                    target.sum().item() * evaluation_worlds
                )
            np.savez_compressed(
                folder / "key_states.npz", played=(depth >= KEY_DOWN_FRACTION).cpu().numpy(), parts=parts, perturb=perturb.cpu().numpy()
            )
            info.update(
                seed=seed,
                selected_round=selected_round,
                setup_s=setup,
                evaluation_s=sync_time() - start,
                gpu=torch.cuda.get_device_name(),
                worlds=args.worlds,
                score_frames=MUSIC_FRAMES,
                control_hz=20,
                physics_hz=200,
            )
            np.savez_compressed(folder / "rollout.npz", qpos=torch.stack(q).cpu().numpy(), depth=depth[:, 0].cpu().numpy(), parts=parts)
            (folder / "results.json").write_text(json.dumps(info, indent=2) + "\n")
            print("final", seed, np.mean(info["all"], axis=0), flush=True)


if __name__ == "__main__":
    main()
