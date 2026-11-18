"""Train two floating Wuji hands on Eden's Ode to Joy using the NDM Gym hand trainers."""

import argparse
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import torch
from torch import nn
import warp as wp

from ndm_gym.algos.hand_algos import HandSuperMimic, HandSuperTrack
from ndm_gym.algos.hand_fast import optimizer, update_functions
from ndm_gym.algos.supertrack import LossBalancer
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.nets import mlp
from ndm_gym.policies.critic import Critic
from ndm_gym.policies.hand_policy import HandPolicy

from piano_score import CONTROL_SUBSTEPS
from ode_to_joy import DIGITS, KEY_COUNT, SIDES, WRIST_SCALE, action_layout, build, piano_roll
from train_repeated_note import KEY_DOWN_FRACTION, frame_metrics
from train_five_finger_scale import seconds

DEV = "cuda:0"
A = 52
D = 2 * A + KEY_COUNT
H = 3
LOOK = 11


def ten_sites(m):
    return [m.site(f"{side}/{side[0]}_{name}_tip").id for side in SIDES for name, _ in DIGITS]


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEV)


class FreeHands:
    def __init__(self, m, n):
        self.model, self.worlds = m, n
        ids, rest, scale = action_layout(m)
        self.ids = ids.tolist()
        self.rest, self.scale = tensor(rest), tensor(scale)
        self.hand_indices = [i for i, a in enumerate(ids) if m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT]
        self.joint_q = m.jnt_qposadr[m.actuator_trnid[ids[self.hand_indices], 0]].tolist()
        self.wrist_q = [int(m.joint(side + "_free").qposadr[0]) for side in SIDES]
        self.keys = [int(m.joint(f"key_{k}_joint").qposadr[0]) for k in range(KEY_COUNT)]
        self.travel = tensor([m.joint(f"key_{k}_joint").range[1].item() for k in range(KEY_COUNT)])
        self.gm = mjw.put_model(m)
        self.gd = mjw.put_data(m, mj.MjData(m), nworld=n, nconmax=384, njmax=1024)
        self.q = wp.to_torch(self.gd.qpos)
        self.ctrl = wp.to_torch(self.gd.ctrl)
        self.length = wp.to_torch(self.gd.actuator_length)
        self.jitter_q0 = tensor(m.qpos0)
        mjw.step(self.gm, self.gd)
        with wp.ScopedCapture() as cap:
            for _ in range(CONTROL_SUBSTEPS):
                mjw.step(self.gm, self.gd)
        self.graph = cap.graph

    def reset(self, starts):
        mjw.reset_data(self.gm, self.gd)
        self.q[:, self.joint_q] = self.rest[self.hand_indices] + starts[:, self.hand_indices] * self.scale[self.hand_indices]
        for h, adr in enumerate(self.wrist_q):
            base = h * 26
            self.q[:, adr : adr + 3] = self.jitter_q0[adr : adr + 3] + starts[:, base : base + 3] * self.scale[base : base + 3]
            vec = starts[:, base + 3 : base + 6] * self.scale[base + 3 : base + 6]
            angle = vec.norm(dim=-1, keepdim=True)
            self.q[:, adr + 3 : adr + 7] = torch.cat((torch.cos(angle * 0.5), 0.5 * torch.sinc(angle / (2 * np.pi)) * vec), -1)
        self.ctrl[:, self.ids] = self.rest
        self.pos = starts.clone()
        return torch.cat((self.pos, torch.zeros_like(self.pos), self.depths()), -1)

    def depths(self):
        return self.q[:, self.keys] / self.travel

    def step(self, act):
        self.ctrl[:, self.ids] = self.rest + act * self.scale
        wp.capture_launch(self.graph)
        new = (self.length[:, self.ids] - self.rest) / self.scale
        state = torch.cat((new, new - self.pos, self.depths()), -1)
        self.pos = new
        return state


class TipGeometry(nn.Module):
    """A differentiable local FK fit to each physical digit, with arbitrary wrist translation and rotation."""

    def __init__(self, model):
        super().__init__()
        ids, rest, scale = action_layout(model)
        d = mj.MjData(model)
        sites = ten_sites(model)
        axes = np.array([i for i, a in enumerate(ids) if model.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT])
        qids = model.jnt_qposadr[model.actuator_trnid[ids[axes], 0]]
        rng = np.random.default_rng(514)
        samples = rng.uniform(-1.25, 1.25, (256, 4))
        pairs = list(itertools.combinations_with_replacement(range(4), 2))
        self.ii = [i for i, _ in pairs]
        self.jj = [j for _, j in pairs]
        design = np.c_[np.ones(len(samples)), samples, samples[:, self.ii] * samples[:, self.jj]]
        origin = np.array([model.qpos0[int(model.joint(s + "_free").qposadr[0]) : int(model.joint(s + "_free").qposadr[0]) + 3] for s in SIDES])
        xyz = []
        for x in samples:
            d.qpos[qids] = rest[axes] + np.tile(x, 10) * scale[axes]
            mj.mj_kinematics(model, d)
            xyz.append(d.site_xpos[sites] - origin.repeat(5, axis=0))
        coeff = np.linalg.lstsq(design, np.array(xyz).reshape(len(samples), -1), rcond=None)[0].reshape(-1, 2, 5, 3)
        self.register_buffer("coeff", tensor(coeff.transpose(1, 2, 0, 3)))
        self.register_buffer("origin", tensor(origin))
        self.register_buffer("scales", tensor(WRIST_SCALE))
        self.register_buffer("centers", tensor([model.body(f"key_{i}").pos.tolist() for i in range(KEY_COUNT)]))
        self.register_buffer("sizes", tensor([model.geom(f"key_{i}_geom").size.tolist() for i in range(KEY_COUNT)]))

    def forward(self, q):
        hands = q.reshape(-1, 2, 26)
        digits = hands[:, :, 6:].reshape(-1, 2, 5, 4)
        poly = torch.cat((torch.ones_like(digits[..., :1]), digits, digits[..., self.ii] * digits[..., self.jj]), -1)
        local = (poly[..., None, :] @ self.coeff).squeeze(-2)
        pose = hands[:, :, :6] * self.scales
        r = pose[:, :, None, 3:].expand_as(local)
        theta = r.norm(dim=-1, keepdim=True)
        cross = torch.cross(r, local, dim=-1)
        rot = local + torch.sinc(theta / np.pi) * cross + 0.5 * torch.sinc(theta / (2 * np.pi)).square() * torch.cross(r, cross, dim=-1)
        return (rot + (self.origin + pose[:, :, :3])[:, :, None, :]).flatten(1, 2)

    def distances(self, tips):
        diff = tips[:, :, None, :] - self.centers
        # Approximate distance to reachable points on every key surface. Any of the ten tips can approach any key.
        dx = (diff[..., 0].abs() - self.sizes[:, 0] * 0.80).clamp_min(0)
        dy = diff[..., 1]
        dz = diff[..., 2] - self.sizes[:, 2]
        return dx, dy, dz


class PianoDynamics(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.geom = TipGeometry(m)
        self.actuator = mlp(3 * A, [128, 128], A, act=nn.SiLU)
        self.key_residual = mlp(3 * A + KEY_COUNT, [192, 128], KEY_COUNT, act=nn.SiLU)
        self.tip_cal = nn.Parameter(torch.zeros(10))
        for net in (self.actuator, self.key_residual):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, state, action):
        q, v, last = torch.split(state, [A, A, KEY_COUNT], -1)
        features = torch.cat((q, v, action), -1)
        vel = 0.5 * (action - q) + 0.12 * v + self.actuator(features)
        q1 = q + vel
        tips = self.geom(q1 + 0.18 * (action - q1))
        dx, dy, dz = self.geom.distances(tips)
        yerr = (dy.abs() - self.geom.sizes[:, 1] * 0.50).clamp_min(0)
        align = torch.exp(-((yerr / 0.007).square() + (dx / 0.012).square()).clamp_max(35))
        depth = torch.sigmoid((-dz - 0.004 + self.tip_cal[:, None] * 0.012) / 0.0027)
        contact = (align * depth).amax(1)
        keys = 0.91 * contact + 0.09 * last + self.key_residual(torch.cat((features, last), -1))
        return torch.cat((q1, vel, keys), -1)


class OdePiano:
    def __init__(self, m, roll):
        self.dev = DEV
        self.args = SimpleNamespace(wm_batches=1, wm_batch=96, wm_window=2, pi_batch=96, lr=0.0008, fused_adam=True, update_graphs=False)
        self.horizon = H
        self.goal = torch.cat((roll, torch.zeros(LOOK + H, KEY_COUNT, device=DEV, dtype=torch.bool))).float()
        self.length = len(roll)
        self.future = self.goal.unfold(0, LOOK, 1).transpose(1, 2).reshape(-1, LOOK * KEY_COUNT)
        self.wm = PianoDynamics(m).to(DEV)
        self.pi = HandPolicy(D + A, LOOK * KEY_COUNT, A, hidden=(192, 128)).to(DEV)
        self.opt_wm = optimizer(self.args, DEV, self.wm, "wm")
        self.opt_pi = optimizer(self.args, DEV, self.pi, "pi")
        self.no_sync = False
        self.task = SimpleNamespace(max_step_reward=1.0)
        self.weights = tensor([1.0, 0.85, 0.20])

    def action(self, state, prev, f):
        return self.pi(torch.cat((state, prev), -1), self.future[f].expand(len(state), -1))

    def _sample_policy_start(self, buffer, batch):
        age, e = buffer.sample_start_points(batch)
        return buffer.at("s", age, e), buffer.at("f", age, e), buffer.at("a_prev", age, e)

    def tracking_errors(self, state, target):
        k = state[:, 2 * A :]
        count = target.sum(-1).clamp_min(1)
        due = ((k - 1).abs() * target).sum(-1) / count
        wrong = (k.abs() * (1 - target)).amax(-1)
        dx, dy, dz = self.wm.geom.distances(self.wm.geom(state[:, :A]))
        dist = (dx.square() + dy.square() + dz.square() + 1e-8).amin(1).sqrt()
        reach = (dist * target).sum(-1) / count / 0.035
        return torch.stack((due, wrong, reach), -1)

    def wm_rollout(self, state, actions, truth, mask):
        loss = 0
        for j in range(actions.shape[1]):
            state = self.wm(state, actions[:, j])
            err = (state - truth[:, j]).square()
            depth = truth[:, j, 2 * A :]
            keyerr = err[:, 2 * A :]
            term = (
                err[:, : 2 * A].mean(-1)
                + keyerr.mean(-1)
                + 0.6 * keyerr.amax(-1)
                + (keyerr * depth.clamp(0, 1)).sum(-1) / depth.clamp(0, 1).sum(-1).clamp_min(1)
            )
            loss += (term * mask[:, j]).mean()
        return loss


class OdeSuperTrack(OdePiano, HandSuperTrack):
    def __init__(self, m, roll):
        super().__init__(m, roll)
        self.imag_noise = self.fail_w = self.reg = 0.0
        self.term_scale = torch.ones(3, device=DEV)
        self.bal = LossBalancer(3).to(DEV)
        self.bal.w.copy_(self.weights / H)
        self.bal.ready.fill_(True)

    def _policy_start(self, buffer, batch):
        state, f, prev = self._sample_policy_start(buffer, batch)
        return state, None, f, prev

    def st_step(self, state, fk, f, prev, noise):
        act = self.action(state, prev, f)
        state = self.wm(state, act + noise)
        errors = self.tracking_errors(state, self.goal[f])
        return state, None, (f + 1).clamp(max=self.length), act, errors, torch.zeros_like(f, dtype=state.dtype), 1.0 - errors.mean(-1)


class OdeSuperMimic(OdePiano, HandSuperMimic):
    def __init__(self, m, roll):
        super().__init__(m, roll)
        self.args.gamma, self.args.lam = 0.99, 0.95
        self.args.critic_epochs, self.args.critic_minibatches = 5, 4
        self.critic = Critic(D + A + LOOK * KEY_COUNT).to(DEV)
        self.opt_v = optimizer(self.args, DEV, self.critic, "critic")

    def start_features(self, state, f, prev):
        return None, torch.cat((state, prev, self.future[f]), -1)

    def imag_step(self, state, fk, f, prev):
        act = self.action(state, prev, f)
        state = self.wm(state, act)
        reward = 1.0 - (self.tracking_errors(state, self.goal[f]) * self.weights).sum(-1)
        done = (f >= self.length - 1).to(state.dtype)
        f = (f + 1).clamp(max=self.length)
        _, x = self.start_features(state, f, act)
        return state, None, f, act, reward, done, x


def starts(n, seed, amplitude):
    rng = torch.Generator(device=DEV).manual_seed(seed)
    x = (2 * torch.rand(n, A, generator=rng, device=DEV) - 1) * amplitude
    x[0] = 0
    return x


@torch.no_grad()
def collect(sim, algo, buffer, round_idx, steps):
    n = sim.worlds
    state = sim.reset(starts(n, int(torch.randint(1000000, (1,))), 0.14))
    prev = torch.zeros(n, A, device=DEV)
    phase = torch.randint(algo.length, (n,), device=DEV)
    random = prev.clone()
    for t in range(steps):
        f = (phase + t) % algo.length
        refresh = torch.rand(n, 1, device=DEV) < 0.23
        random = torch.where(refresh, torch.rand_like(random) * 2 - 1, random)
        pi = algo.action(state, prev, f)
        noise = 0.28 if round_idx < 5 else 0.18
        action = (pi + noise * torch.randn_like(pi) + 0.22 * random).clamp(-1, 1)
        explore = n if round_idx == 1 else n // 4
        action[:explore] = random[:explore]
        state1 = sim.step(action)
        buffer.add(dict(s=state, s1=state1, u=action, a_prev=prev, f=f, done=(f == algo.length - 1) | (t == steps - 1)))
        state, prev = state1, action


@torch.no_grad()
def evaluate(sim, algo, roll, initial, record=False):
    masks, poses, depths, controls, observed = [], [], [], [], []
    for batch in initial.split(sim.worlds):
        n = len(batch)
        if n < sim.worlds:
            batch = torch.cat((batch, batch[:1].expand(sim.worlds - n, -1)))
        state = sim.reset(batch)
        prev = torch.zeros_like(batch)
        if record and not masks:
            observed.append(state[0].clone())
        hist = []
        for f in range(len(roll)):
            prev = algo.action(state, prev, f)
            state = sim.step(prev)
            k = state[:, 2 * A :]
            hist.append(k[:n] >= KEY_DOWN_FRACTION)
            if record and not masks:
                poses.append(sim.q[0].clone())
                depths.append(k[0].clone())
                controls.append(prev[0].clone())
                observed.append(state[0].clone())
        masks.append(torch.stack(hist, 1))
    down = torch.cat(masks)
    score = frame_metrics(down, roll).mean((0, 1)).tolist()
    exact = (down == roll).all(-1)
    poly = roll.sum(-1) == 2
    both = (down[:, poly] | ~roll[poly]).all(-1)
    ons = roll & ~torch.cat((torch.zeros_like(roll[:1]), roll[:-1]))
    reached = []
    for start, key in ons.nonzero().tolist():
        end = start + int(roll[start:, key].int().cumprod(0).sum())
        reached.append(down[:, start:end, key].any(1))
    reached = torch.stack(reached, 1)
    result = dict(
        precision=score[0],
        recall=score[1],
        f1=score[2],
        exact_frames=exact.float().mean().item(),
        perfect=int(exact.all(1).sum()),
        both_due=both.float().mean().item(),
        exact_poly=exact[:, poly].float().mean().item(),
        all_notes=int(reached.all(1).sum()),
        notes_reached=reached.float().sum(1).mean().item(),
        nominal_coverage=int(reached[0].sum()),
        nominal_exact=int(exact[0].sum()),
        nominal_f1=frame_metrics(down[0], roll)[:, 2].mean().item(),
        starts=len(down),
    )
    out = None
    if record:
        out = dict(
            qpos=torch.stack(poses).cpu().numpy(),
            depth=torch.stack(depths).cpu().numpy(),
            actions=torch.stack(controls).cpu().numpy(),
            states=torch.stack(observed).cpu().numpy(),
            goal=roll.cpu().numpy(),
        )
    return result, down, out


@torch.no_grad()
def model_errors(model, recorded):
    """Open-loop key-depth errors under the nominal policy rollout's recorded actions."""
    states, actions, goal = (torch.as_tensor(recorded[k], device=DEV) for k in ("states", "actions", "goal"))
    errors = {}
    for horizon in (1, 3, 10):
        n = len(actions) - horizon + 1
        predicted = states[:n]
        for j in range(horizon):
            predicted = model(predicted, actions[j : j + n])
        real = states[horizon:, 2 * A :]
        err = (predicted[:, 2 * A :] - real).abs()
        due = goal[horizon - 1 :]
        errors[horizon] = dict(all_key_mae=err.mean().item(), due_key_mae=err[due].mean().item())
    return errors


def save(path, stuff):
    tmp = path.with_suffix(".tmp")
    torch.save(stuff, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand-dir", required=True, type=Path)
    p.add_argument("--keyboard", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--algo", choices=("supertrack", "supermimic"), help="Defaults to SuperTrack or the replay checkpoint's algorithm.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--wm-updates", type=int, default=90)
    p.add_argument("--pi-updates", type=int, default=50)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--eval-starts", type=int, default=256)
    p.add_argument("--replay", type=Path)
    a = p.parse_args()
    replay = torch.load(a.replay, map_location="cpu", weights_only=True) if a.replay else None
    if replay and a.algo and a.algo != replay.get("algo", "supertrack"):
        p.error("The requested algorithm does not match the replay checkpoint")
    a.algo = a.algo or (replay.get("algo", "supertrack") if replay else "supertrack")
    if min(a.worlds, a.rounds, a.wm_updates, a.pi_updates, a.steps, a.eval_starts) < 1 or a.steps < 2:
        p.error("Positive counts and at least two collection steps are required")
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required")
    wp.init()
    torch.cuda.set_stream(torch.cuda.Stream())
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    torch.backends.cuda.matmul.allow_tf32 = True
    with wp.ScopedDevice(DEV):
        m = build(a.hand_dir, a.keyboard)
        sim = FreeHands(m, a.worlds)
        roll = torch.as_tensor(piano_roll(), device=DEV)
        val = starts(a.worlds, 456, 0.15)
        test = starts(a.eval_starts, 789, 0.20)
        fields = {k: (dim, torch.float32) for k, dim in (("s", D), ("s1", D), ("u", A), ("a_prev", A))}
        fields.update(f=((), torch.long), done=((), torch.bool))
        seeds = [replay["seed"]] if replay else a.seeds
        for seed in seeds:
            torch.manual_seed(seed)
            dest = a.out / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "results.json").exists() or (not a.replay and (dest / "best.pt").exists()):
                raise FileExistsError(dest)
            algo = OdeSuperMimic(m, roll) if a.algo == "supermimic" else OdeSuperTrack(m, roll)
            rec = dict(seed=seed, gpu=torch.cuda.get_device_name(), settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
            rec["optimizer_settings"] = vars(algo.args)
            if not a.replay:
                buffer = ReplayBuffer(6 * a.steps, a.worlds, DEV, fields=fields, priorities=False)
                buffer.window_keys = ("s", "u", "s1", "done")
                wm_update, pi_update = update_functions(algo, buffer, algo.args)
                baseline, _, _ = evaluate(sim, algo, roll, test)
                rec["baseline"] = baseline
                hist = []
                best = -1
                start = seconds()
                for i in range(1, a.rounds + 1):
                    # Decay the actor learning rate in the second half of the fixed budget.
                    lr = algo.args.lr * (1 - 0.8 * max(0.0, 2 * i / a.rounds - 1))
                    for group in algo.opt_pi.param_groups:
                        group["lr"] = lr
                    t0 = seconds()
                    collect(sim, algo, buffer, i, a.steps)
                    t1 = seconds()
                    for _ in range(a.wm_updates):
                        wm_loss = wm_update()
                    t2 = seconds()
                    for _ in range(a.pi_updates):
                        pi_loss, value_loss, imagined_reward, grad_norm, survival = pi_update()
                    t3 = seconds()
                    stats, _, _ = evaluate(sim, algo, roll, val)
                    t4 = seconds()
                    item = dict(round=i, seconds=seconds() - start, wm=float(wm_loss), pi=float(pi_loss), **stats)
                    item.update(actor_lr=lr, collect_s=t1 - t0, wm_s=t2 - t1, pi_s=t3 - t2, validate_s=t4 - t3)
                    diagnostics = dict(wm=item["wm"], pi=item["pi"], f1=item["f1"])
                    if a.algo == "supermimic":
                        diagnostics.update(
                            value_loss=float(value_loss), imagined_reward=float(imagined_reward), grad_norm=float(grad_norm), survival=float(survival)
                        )
                        item.update(diagnostics)
                    if not all(np.isfinite(v) for v in diagnostics.values()):
                        raise RuntimeError(item)
                    hist.append(item)
                    print(
                        seed,
                        i,
                        round(item["seconds"], 1),
                        {k: round(item[k], 4) for k in ("wm", "pi", "f1", "exact_poly", "notes_reached")},
                        flush=True,
                    )
                    ck = dict(algo=a.algo, seed=seed, round=i, pi=algo.pi.state_dict(), wm=algo.wm.state_dict())
                    if a.algo == "supermimic":
                        ck["critic"] = algo.critic.state_dict()
                    if stats["f1"] > best:
                        save(dest / "best.pt", ck)
                        best = stats["f1"]
                    if i == a.rounds:
                        save(dest / "last.pt", ck)
                    (dest / "history.json").write_text(json.dumps(hist, indent=2))
                rec.update(history=hist, training_s=seconds() - start)
            ck = replay or torch.load(dest / "last.pt", map_location=DEV, weights_only=True)
            algo.pi.load_state_dict(ck["pi"])
            algo.wm.load_state_dict(ck["wm"])
            if a.algo == "supermimic":
                algo.critic.load_state_dict(ck["critic"])
            start = seconds()
            stats, masks, out = evaluate(sim, algo, roll, test, record=True)
            rec.update(round=ck["round"], test=stats, evaluation_s=seconds() - start)
            rec["model_errors"] = model_errors(algo.wm, out)
            np.savez_compressed(dest / "rollout.npz", **out)
            np.savez_compressed(dest / "evaluation.npz", played=masks.cpu().numpy(), goal=roll.cpu().numpy(), starts=test.cpu().numpy())
            (dest / "results.json").write_text(json.dumps(rec, indent=2) + "\n")
            print("final", seed, stats, flush=True)


if __name__ == "__main__":
    main()
