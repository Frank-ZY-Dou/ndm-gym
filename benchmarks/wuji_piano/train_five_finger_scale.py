"""Train NDM Gym's hand SuperTrack update on the five active fingers and passive piano keys."""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from ndm_gym.algos.hand_fast import optimizer, update_functions
from ndm_gym.algos.supertrack import LossBalancer
from ndm_gym.worldmodels.piano import FingerDynamics
from ndm_gym.policies.hand_policy import HandPolicy

from training_options import training_defaults
from piano_updates import PianoTrackUpdates, make_replay
from five_finger_scale import CONTROL_SUBSTEPS, KEY_COUNT, LOWEST_PITCH, MELODY, PITCHES, build, finger_ranges, piano_roll
from ndm_gym.tasks.piano import KEY_DOWN_FRACTION, frame_metrics

DEV = "cuda:0"
N = len(PITCHES)
K = [key - LOWEST_PITCH for key in PITCHES]
LOOKAHEAD = 11
STATE_DIM = 4 * N
HORIZON = 4
WINDOW = 2


class PianoSim:
    def __init__(self, model, worlds):
        self.model = model
        self.worlds = worlds
        self.gm = mjw.put_model(model)
        self.gd = mjw.put_data(model, mujoco.MjData(model), nworld=worlds, nconmax=384, njmax=1024)
        hover, press, ids = finger_ranges(model)
        self.hover = torch.as_tensor(hover, device=DEV)
        self.ids = torch.as_tensor(ids, device=DEV)
        self.delta = torch.as_tensor((press - hover)[ids], device=DEV)
        self.finger_q = model.jnt_qposadr[model.actuator_trnid[ids, 0]].tolist()
        self.all_q = model.jnt_qposadr[model.actuator_trnid[:, 0]].tolist()
        keys = [model.joint(f"key_{key}_joint") for key in range(KEY_COUNT)]
        self.key_q = [k.qposadr[0] for k in keys]
        self.limits = torch.tensor([k.range[1].item() for k in keys], device=DEV)
        self.ctrl, self.q = wp.to_torch(self.gd.ctrl), wp.to_torch(self.gd.qpos)
        mjw.step(self.gm, self.gd)
        with wp.ScopedCapture() as capture:
            for _ in range(CONTROL_SUBSTEPS):
                mjw.step(self.gm, self.gd)
        self.graph = capture.graph

    def depths(self):
        return self.q[:, self.key_q] / self.limits

    def positions(self):
        # Position along each finger's four-joint stroke, and its corresponding passive key.
        pos = ((self.q[:, self.finger_q] - self.hover[self.ids]) * self.delta).sum(-1) / self.delta.square().sum(-1)
        return torch.cat([pos, self.depths()[:, K]], -1)

    def reset(self, perturb):
        mjw.reset_data(self.gm, self.gd)
        initial, self.offset = perturb.unbind(1)
        self.q[:, self.all_q] = self.hover + initial
        self.ctrl[:] = self.hover + self.offset
        self.pos = self.positions()
        return torch.cat((self.pos, torch.zeros_like(self.pos)), dim=-1)

    def step(self, action):
        self.ctrl[:, self.ids] = self.hover[self.ids] + ((action.clamp(-1, 1) + 1) / 2).unsqueeze(-1) * self.delta + self.offset[:, self.ids]
        wp.capture_launch(self.graph)
        pos = self.positions()
        state = torch.cat((pos, pos - self.pos), dim=-1)
        self.pos = pos
        return state


class PianoSuperTrack(PianoTrackUpdates):
    """Inherit the hand trainer's replay-window model update and imagined policy update with piano-specific states and losses."""

    def __init__(self, roll):
        self.args = SimpleNamespace(wm_batches=1, wm_batch=256, wm_window=WINDOW, pi_batch=192, lr=0.0015, fused_adam=roll.is_cuda, update_graphs=False)
        self.dev = roll.device
        self.horizon = HORIZON
        self.goal = torch.cat((roll[:, K], torch.zeros(LOOKAHEAD + HORIZON, N, device=self.dev, dtype=torch.bool))).float()
        self.future = self.goal.unfold(0, LOOKAHEAD, 1).transpose(1, 2).reshape(-1, LOOKAHEAD * N)
        self.length = len(roll)
        self.wm = FingerDynamics(N).to(self.dev)
        self.pi = HandPolicy(STATE_DIM + N, LOOKAHEAD * N, N, hidden=(128, 96)).to(self.dev)
        self.opt_wm = optimizer(self.args, self.dev, self.wm, "wm")
        self.opt_pi = optimizer(self.args, self.dev, self.pi, "pi")
        self.no_sync = False
        self.reg = 0.001
        self.imag_noise = self.fail_w = 0
        self.task = SimpleNamespace(max_step_reward=1.0)
        self.term_scale = torch.ones(2, device=self.dev)
        self.bal = LossBalancer(2).to(self.dev)
        self.bal.w.copy_(torch.tensor([1.0, 0.6], device=self.dev) / HORIZON)
        self.bal.ready.fill_(True)

    def action(self, state, prev, f):
        return self.pi(torch.cat((state, prev), -1), self.future[f].expand(len(state), -1))

    def st_step(self, state, fk, f, prev, noise):
        action = self.action(state, prev, f)
        state = self.wm(state, action + noise)
        target = self.goal[f]
        depth = state[:, N : 2 * N]
        due = ((depth - 1.0).abs() * target).sum(-1)
        other = (depth.abs() * (1 - target)).amax(-1)
        errors = torch.stack((due, other), -1)
        return state, None, (f + 1).clamp(max=self.length), action, errors, torch.zeros_like(due), 1 - errors.mean(-1)

    def wm_rollout(self, state, actions, next_states, mask):
        loss = 0
        for j in range(actions.shape[1]):
            state = self.wm(state, actions[:, j])
            err = (state - next_states[:, j]).square()
            loss = loss + ((err.mean(-1) + err[:, N : 2 * N].mean(-1)) * mask[:, j]).mean()
        return loss


def perturbations(n, model, seed, width):
    g = torch.Generator(device=DEV).manual_seed(seed)
    v = torch.rand((n, 2, model.nu), device=DEV, generator=g) * 2 - 1
    v[:, 0] *= 0.08
    v[:, 1] *= width
    v[0] = 0
    return v


@torch.no_grad()
def collect(sim, algo, buffer, iteration):
    n = sim.worlds
    # Independent random note sequences excite contact and release dynamics. The rest of the worlds use the live policy.
    state = sim.reset(perturbations(n, sim.model, int(torch.randint(1000000, (1,))), 0.017))
    prev = torch.full((n, N), -1.0, device=DEV)
    explore_n = n if iteration == 1 else n // 2
    random_action = prev.clone()
    phases = torch.randint(0, algo.length, (n,), device=DEV)
    for t in range(algo.length):
        f = (phases + t) % algo.length
        new = torch.rand(n, 1, device=DEV) < 0.25
        pick = torch.randint(N, (n,), device=DEV)
        strong = torch.rand(n, N, device=DEV) * 0.55 + 0.45
        background = torch.rand(n, N, device=DEV) * 0.55 - 1
        proposal = background.scatter(1, pick[:, None], strong.gather(1, pick[:, None]))
        random_action = torch.where(new, proposal, random_action)
        action = (algo.action(state, prev, f) + torch.randn_like(prev) * 0.14).clamp(-1, 1)
        action[:explore_n] = (random_action[:explore_n] + 0.08 * torch.randn_like(prev[:explore_n])).clamp(-1, 1)
        s1 = sim.step(action)
        done = (f == algo.length - 1) | (t == algo.length - 1)
        buffer.add(dict(s=state, s1=s1, u=action, f=f, done=done, a_prev=prev))
        prev, state = action, s1


@torch.no_grad()
def play(sim, algo, roll, starts, record=False):
    all_depth = []
    poses = []
    for batch in starts.split(sim.worlds):
        n = len(batch)
        if n < sim.worlds:
            batch = torch.cat((batch, batch[:1].expand(sim.worlds - n, -1, -1)))
        state = sim.reset(batch)
        prev = torch.full((sim.worlds, N), -1.0, device=DEV)
        one = []
        for f in range(len(roll)):
            action = algo.action(state, prev, f)
            state = sim.step(action)
            one.append(sim.depths()[:n])
            if record and not all_depth:
                poses.append(sim.q[0].clone())
            prev = action
        all_depth.append(torch.stack(one, dim=1))
    depth = torch.cat(all_depth)
    down = depth >= KEY_DOWN_FRACTION
    stats = frame_metrics(down, roll).mean(1)
    return depth, stats, torch.stack(poses) if poses else None


def seconds():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-dir", required=True, type=Path)
    parser.add_argument("--keyboard", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--wm-updates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--pi-updates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--eval-starts", type=int, default=256)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    training_defaults(parser, args, training=args.replay is None, seeds=[0, 1, 2], rounds=8, wm_updates=105, pi_updates=75)
    if min(args.worlds, args.rounds, args.wm_updates, args.pi_updates, args.eval_starts) < 1:
        parser.error("All counts must be positive")
    if not torch.cuda.is_available():
        parser.error("An NVIDIA CUDA GPU is required")
    wp.init()
    torch.cuda.set_stream(torch.cuda.Stream())
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    torch.backends.cuda.matmul.allow_tf32 = True
    with wp.ScopedDevice(DEV):
        m = build(args.hand_dir, args.keyboard)
        sim = PianoSim(m, args.worlds)
        roll = torch.as_tensor(piano_roll(), device=DEV)
        val = perturbations(args.worlds, m, 456, 0.018)
        held = perturbations(args.eval_starts, m, 789, 0.025)
        seeds = [torch.load(args.replay, map_location="cpu", weights_only=True)["seed"]] if args.replay else args.seeds
        for seed in seeds:
            torch.manual_seed(seed)
            dest = args.out / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            ckpath = dest / "best.pt"
            algo = PianoSuperTrack(roll)
            metrics = {}
            if (dest / "results.json").exists():
                raise FileExistsError(dest / "results.json")
            if not args.replay:
                buffer = make_replay(4 * len(roll), args.worlds, DEV, state_dim=STATE_DIM, num_actions=N)
                wm_update, pi_update = update_functions(algo, buffer, algo.args)
                if ckpath.exists():
                    raise FileExistsError(ckpath)
                _, untrained, _ = play(sim, algo, roll, held)
                metrics["untrained_held"] = untrained.mean(0).tolist()
                history = []
                start = seconds()
                _, base, _ = play(sim, algo, roll, val)
                baseline = base.mean(0).tolist()
                best = -1
                for iteration in range(1, args.rounds + 1):
                    collect(sim, algo, buffer, iteration)
                    for _ in range(args.wm_updates):
                        wm_loss = wm_update()
                    for _ in range(args.pi_updates):
                        pi_loss = pi_update()[0]
                    _, quality, _ = play(sim, algo, roll, val)
                    score = quality[:, 2].mean().item()
                    row = dict(round=iteration, validation_f1=score, seconds=seconds() - start, wm_loss=wm_loss.item(), pi_loss=pi_loss.item())
                    print(seed, row, flush=True)
                    history.append(row)
                    checkpoint = dict(pi=algo.pi.state_dict(), wm=algo.wm.state_dict(), seed=seed, round=iteration)
                    if iteration == args.rounds:
                        torch.save(checkpoint, dest / "last.tmp")
                        os.replace(dest / "last.tmp", dest / "last.pt")
                    if score > best:
                        best = score
                        torch.save(checkpoint, dest / "temporary.pt")
                        os.replace(dest / "temporary.pt", ckpath)
                metrics.update(training_s=seconds() - start, baseline_validation=baseline, history=history)
            ck = torch.load(args.replay or dest / "last.pt", map_location=DEV, weights_only=True)
            algo.pi.load_state_dict(ck["pi"])
            begin = seconds()
            depth, quality, poses = play(sim, algo, roll, held, record=True)
            elapsed = seconds() - begin
            d = depth.cpu().numpy()
            goals = roll.cpu().numpy()
            played = d >= KEY_DOWN_FRACTION
            hits = played & goals
            exact = (played == goals).all(-1)
            note_starts = np.flatnonzero(goals.any(-1) & ~np.r_[False, goals.any(-1)[:-1]])
            coverage = np.stack(
                [
                    played[:, f : f + int(goals[f:, pitch - LOWEST_PITCH].cumprod().sum()), pitch - LOWEST_PITCH].any(-1)
                    for pitch, f in zip(MELODY, note_starts, strict=True)
                ],
                axis=1,
            )
            onset = played & ~np.pad(played[:, :-1], ((0, 0), (1, 0), (0, 0)))
            metrics.update(
                seed=seed,
                selected_round=ck["round"],
                gpu=torch.cuda.get_device_name(),
                eval_starts=len(held),
                eval_s=elapsed,
                worlds=args.worlds,
                rounds=args.rounds,
                wm_updates_per_round=args.wm_updates,
                pi_updates_per_round=args.pi_updates,
                precision_recall_f1=quality.mean(0).tolist(),
                exact_share=float(exact.mean()),
                pitch_recall=[float(hits[:, :, k].sum() / (len(held) * goals[:, k].sum())) for k in K],
                all_notes_reached=int(coverage.all(-1).sum()),
                nominal_coverage=int(coverage[0].sum()),
                nominal_onsets=(np.argwhere(onset[0]) + [0, LOWEST_PITCH]).tolist(),
                nominal_scores=quality[0].tolist(),
                nominal_exact=int(exact[0].sum()),
                frames=len(goals),
            )
            np.savez_compressed(dest / "rollout.npz", qpos=poses.cpu().numpy(), depth=d[0], goal=goals)
            np.savez_compressed(dest / "evaluation.npz", played=played, goal=goals, perturb=held.cpu().numpy())
            (dest / "results.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print("final", seed, metrics["precision_recall_f1"], metrics["nominal_coverage"], metrics["nominal_onsets"], flush=True)


if __name__ == "__main__":
    main()
