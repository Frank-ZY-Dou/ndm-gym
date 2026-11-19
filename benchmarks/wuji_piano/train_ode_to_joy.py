"""Train two floating Wuji hands on Eden's Ode to Joy using the NDM Gym hand trainers."""

import argparse
import itertools
import json
import os
from pathlib import Path

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from ndm_gym.algos.common import set_lr
from ndm_gym.algos.hand_fast import update_functions
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.worldmodels.piano import NUM_ACTUATORS as A, TipGeometry

from training_options import training_defaults
from piano_updates import H, REACH_W, OdeSuperMimic, OdeSuperTrack, make_replay
from ndm_gym.tasks.piano import CONTROL_SUBSTEPS
from ode_to_joy import DIGITS, KEY_COUNT, SIDES, WRIST_SCALE, action_layout, build, piano_roll
from ndm_gym.tasks.piano import KEY_DOWN_FRACTION, frame_metrics
from train_five_finger_scale import seconds

DEV = "cuda:0"


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


def fit_geometry(model):
    """Fit the digit polynomials to the MJCF sites once, before training."""
    ids, rest, scale = action_layout(model)
    d = mj.MjData(model)
    sites = ten_sites(model)
    axes = np.array([i for i, a in enumerate(ids) if model.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT])
    qids = model.jnt_qposadr[model.actuator_trnid[ids[axes], 0]]
    rng = np.random.default_rng(514)
    samples = rng.uniform(-1.25, 1.25, (256, 4))
    pairs = list(itertools.combinations_with_replacement(range(4), 2))
    ii = [i for i, _ in pairs]
    jj = [j for _, j in pairs]
    design = np.c_[np.ones(len(samples)), samples, samples[:, ii] * samples[:, jj]]
    origin = np.array([model.qpos0[int(model.joint(s + "_free").qposadr[0]) : int(model.joint(s + "_free").qposadr[0]) + 3] for s in SIDES])
    xyz = []
    for x in samples:
        d.qpos[qids] = rest[axes] + np.tile(x, 10) * scale[axes]
        mj.mj_kinematics(model, d)
        xyz.append(d.site_xpos[sites] - origin.repeat(5, axis=0))
    coeff = np.linalg.lstsq(design, np.array(xyz).reshape(len(samples), -1), rcond=None)[0].reshape(-1, 2, 5, 3)
    centers = [model.body(f"key_{i}").pos.tolist() for i in range(KEY_COUNT)]
    sizes = [model.geom(f"key_{i}_geom").size.tolist() for i in range(KEY_COUNT)]
    return TipGeometry(coeff.transpose(1, 2, 0, 3), origin, WRIST_SCALE, centers, sizes)


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
    p.add_argument("--seeds", nargs="+", type=int, default=argparse.SUPPRESS)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--rounds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--wm-updates", type=int, default=argparse.SUPPRESS)
    p.add_argument("--pi-updates", type=int, default=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval-starts", type=int, default=256)
    p.add_argument("--update-graphs", action="store_true", default=argparse.SUPPRESS,
                   help="Capture native model and policy updates in CUDA graphs. Replay sampling and rounding can change learning results.")
    p.add_argument("--horizon", type=int, default=argparse.SUPPRESS, help="Imagined steps of the policy update, both algorithms (default 3).")
    p.add_argument("--st-imag-noise", type=float, default=argparse.SUPPRESS, help="SuperTrack: Gaussian noise on the imagined actions (default 0).")
    p.add_argument("--st-reg", type=float, default=argparse.SUPPRESS, help="SuperTrack: weight of the L2 and L1 action regularizer (default 0).")
    p.add_argument("--st-reach-weight", type=float, default=argparse.SUPPRESS, help="SuperTrack: the reach term's weight in its loss (default 0.20).")
    p.add_argument("--gamma", type=float, default=argparse.SUPPRESS, help="SuperMimic: the discount (default 0.99).")
    p.add_argument("--bptt-decay", type=float, default=argparse.SUPPRESS, help="SuperMimic: the gradient through each imagined state, scaled by it (default 1).")
    p.add_argument("--critic-epochs", type=int, default=argparse.SUPPRESS, help="SuperMimic: critic epochs per actor update (default 5).")
    p.add_argument("--critic-minibatches", type=int, default=argparse.SUPPRESS, help="SuperMimic: critic minibatches per epoch (default 4).")
    p.add_argument("--replay", type=Path)
    a = p.parse_args()
    supplied = set(vars(a))
    training_defaults(p, a, training=a.replay is None, seeds=[0, 1, 2], rounds=20, wm_updates=90, pi_updates=50, steps=160, update_graphs=False,
                      horizon=H, st_imag_noise=0.0, st_reg=0.0, st_reach_weight=REACH_W, gamma=0.99, bptt_decay=1.0, critic_epochs=5,
                      critic_minibatches=4)  # fmt: skip
    replay = torch.load(a.replay, map_location="cpu", weights_only=True) if a.replay else None
    if replay and a.algo and a.algo != replay.get("algo", "supertrack"):
        p.error("The requested algorithm does not match the replay checkpoint")
    a.algo = a.algo or (replay.get("algo", "supertrack") if replay else "supertrack")
    own = dict(supertrack=("st_imag_noise", "st_reg", "st_reach_weight"), supermimic=("gamma", "bptt_decay", "critic_epochs", "critic_minibatches"))
    other = [k for k in own["supermimic" if a.algo == "supertrack" else "supertrack"] if k in supplied]
    if other:
        p.error(f"--{other[0].replace('_', '-')} is not an option of {a.algo}")
    if a.horizon < 1 or a.st_imag_noise < 0 or a.st_reg < 0 or a.st_reach_weight < 0 or not 0 < a.gamma <= 1 or not 0 < a.bptt_decay <= 1:
        p.error("--horizon >= 1, --st-imag-noise, --st-reg, --st-reach-weight >= 0, 0 < --gamma <= 1 and 0 < --bptt-decay <= 1 are required")
    if min(a.critic_epochs, a.critic_minibatches) < 1 or a.critic_minibatches > a.horizon * 96:
        p.error("--critic-epochs >= 1 and 1 <= --critic-minibatches <= the imagined states per update are required")
    if a.algo == "supermimic":
        options = dict(gamma=a.gamma, critic_epochs=a.critic_epochs, critic_minibatches=a.critic_minibatches, bptt_decay=a.bptt_decay)
    else:
        options = dict(imag_noise=a.st_imag_noise, reg=a.st_reg, reach_w=a.st_reach_weight)
    if min(a.worlds, a.rounds, a.wm_updates, a.pi_updates, a.steps, a.eval_starts) < 1 or a.steps < 2:
        p.error("Positive counts and at least two collection steps are required")
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required")
    RunningNorm.no_sync = a.update_graphs
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
        geometry = fit_geometry(m)
        seeds = [replay["seed"]] if replay else a.seeds
        for seed in seeds:
            torch.manual_seed(seed)
            dest = a.out / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "results.json").exists() or (not a.replay and (dest / "best.pt").exists()):
                raise FileExistsError(dest)
            trainer = OdeSuperMimic if a.algo == "supermimic" else OdeSuperTrack
            algo = trainer(geometry, roll, horizon=a.horizon, update_graphs=a.update_graphs, **options)
            rec = dict(seed=seed, gpu=torch.cuda.get_device_name(), settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
            rec["optimizer_settings"] = vars(algo.args)
            if not a.replay:
                buffer = make_replay(6 * a.steps, a.worlds, DEV, update_graphs=a.update_graphs)
                wm_update, pi_update = update_functions(algo, buffer, algo.args)
                baseline, _, _ = evaluate(sim, algo, roll, test)
                rec["baseline"] = baseline
                rec["time_to_target_s"] = None
                hist = []
                best = -1
                start = seconds()
                for i in range(1, a.rounds + 1):
                    # Decay the actor learning rate in the second half of the fixed budget.
                    lr = algo.args.lr * (1 - 0.8 * max(0.0, 2 * i / a.rounds - 1))
                    set_lr(algo.opt_pi, lr)
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
                    if best < 0.95 <= stats["f1"]:
                        save(dest / "target.pt", ck)
                        rec["time_to_target_s"] = item["seconds"]
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
