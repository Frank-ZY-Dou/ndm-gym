"""Policy-gradient statistics of a training run's checkpoint at learning rate 0: how often the gradient of the BPTT
through the world model spikes, and how well each update points along the mean gradient, with and without the tricks
against spikes (``--bptt_clip_k``, ``--rf_rope_detach``, ``--wm_bptt_fp32``).

Loads the run's full state (``last.pt``) and a replay dataset recorded with the run's own policy
(``python -m ndm_gym.wmbench.collect``), sets every learning rate to 0 and runs ``--updates`` policy updates, update i
on its own batch drawn with seed ``--seed0 + i``, so that every setting sees the same batches and noise. The policy,
world model and critic stay fixed: their normalizers, which an update may refit, are restored after every update, and
the parameters are checked to be unchanged at the end. Per update: the policy-gradient norm before clipping (the
training log's ``pi_gradnorm``), the policy loss, and the cosine between the gradient and a reference direction. As
training clips nearly every update to norm 1, directions are averaged as unit vectors: the reference is the mean unit
gradient of the run's own setting over ``--ref_updates`` other batches (seeds from 10^6; ``--save_ref`` writes it,
``--ref`` reads it), and the length of a setting's own mean unit gradient (1: every update points the same way) measures
how consistent its updates are. ``--set key=value`` overrides the run's training arguments.

``--wm_finetune n`` first fine-tunes the world model: n world-model updates on the replay at the run's last world-model
learning rate, with the run's settings and ``--set`` (for example ``wm_smooth_w=w``), from a fixed seed. Before them it
reports the world model's fit loss L and smoothness penalty P on 20 replay batches, and before and after them the
8- and 32-step open-loop errors on the held-out datasets of ``--wm_test``.

``--anatomy`` takes every update's gradient apart (``Anatomy``): by imagined step (which steps of the horizon carry
it), by rollout (how the rollouts share it), and the gradients that reach the policy's output and the world model's
state at every step. It runs the imagination step eagerly, so it needs ``--set pi_cudagraphs=0`` (the CUDA graphs do
not change the gradient; ``--save_ref`` takes this and ``rf_bptt_keep_blocks=0``, which do not either), and a
reference saved with it. ``--fp32_twin`` also computes every update's gradient on the same batch and noise with the
world model in fp32 (``--wm_bptt_fp32``) and compares the two. ``--max_gpu_gb`` caps the process's GPU memory.

Usage: python tools/grad_stats.py --ckpt logs/<run>/last.pt --data <dataset.pt> --out <dir>/<setting>.json
       (--save_ref <dir>/ref.pt | --ref <dir>/ref.pt [--set bptt_clip_k=3 | rf_rope_detach=1 | wm_bptt_fp32=1 ...])
       [--wm_finetune 2000 --wm_test <held-out dataset.pt> --set wm_smooth_w=<w>] [--anatomy --set pi_cudagraphs=0 [--fp32_twin]]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
REF_SEED0 = 10**6
FINETUNE_SEED = 2 * 10**6
SAME_GRADIENT = ("pi_cudagraphs", "rf_bptt_keep_blocks")  # speed options that leave the policy gradient unchanged


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="the run's last.pt (the full training state)")
    p.add_argument("--data", nargs="+", required=True, help="replay dataset(s) recorded with the run's policy (ndm_gym.wmbench.collect)")
    p.add_argument("--out", required=True, help="JSON with the per-update values and their summary")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--ref_updates", type=int, default=500)
    p.add_argument("--seed0", type=int, default=0)
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--save_ref", help="compute the reference direction with the run's own setting (no --set) and write it here")
    ref.add_argument("--ref", help="reference direction written by --save_ref for the same checkpoint")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="training arguments to override")
    p.add_argument("--wm_finetune", type=int, default=0, help="world-model updates on the replay before the measurement (0: none)")
    p.add_argument("--wm_test", nargs="*", default=[], help="held-out replay datasets: open-loop errors before and after --wm_finetune")
    p.add_argument("--motion_dir", default=os.path.join(REPO, "data", "lafan1_g1"))
    p.add_argument("--anatomy", action="store_true", help="every update's gradient by imagined step and by rollout (Anatomy)")
    p.add_argument("--anatomy_chunk", type=int, default=512, help="--anatomy: rollouts per rerun of the policy")
    p.add_argument("--fp32_twin", action="store_true", help="also every update's gradient with the world model in fp32, on the same batch")
    p.add_argument("--max_gpu_gb", type=float, default=None, help="cap on the process's GPU memory (GB)")
    p.add_argument("--save_every", type=int, default=0, help="also write the JSON of the updates so far every this many updates (0: at the end)")
    return p.parse_args()


def run_arguments(ck_args: dict, overrides: list[str], launched: set[str]) -> dict:
    """The run's training arguments over the current defaults (flags added after the run take their defaults, which
    keep its behavior), then ``key=value`` overrides parsed with the argument's type. ``launched``: keys the simulator's
    launcher added to the arguments after train.py wrote ``args.json``."""
    from isaaclab.app import add_launcher_args

    from ndm_gym.train import add_speed_args, add_train_args

    parser = argparse.ArgumentParser()  # as ndm_gym.train.parse_args builds it
    add_train_args(parser)
    add_speed_args(parser)
    add_launcher_args(parser)
    defaults = vars(parser.parse_args(["--algo", ck_args["algo"]]))
    unknown = sorted(set(ck_args) - set(defaults) - launched)
    if unknown:
        raise ValueError(f"the checkpoint has training arguments this code does not know: {unknown}")
    a = {**defaults, **ck_args}
    for kv in overrides:
        k, sep, v = kv.partition("=")
        if not sep or k not in defaults or defaults[k] is None:
            raise ValueError(f"--set {kv!r}: expected key=value for a training argument with a typed default")
        d = defaults[k]
        if isinstance(d, bool):
            if v.lower() not in ("0", "1", "false", "true"):
                raise ValueError(f"--set {kv!r}: a flag takes 0/1/false/true")
            a[k] = v.lower() in ("1", "true")
        else:
            a[k] = type(d)(v)
    return a


def wm_finetune(algo, buf, a: dict, iteration: int, args, dev: str) -> dict:
    """The world model's fit loss L (the world-model objective's loss term) and smoothness penalty P on 20 replay
    batches, its held-out open-loop errors, ``--wm_finetune`` world-model updates at the run's last world-model learning
    rate, and the held-out errors again."""
    import numpy as np
    import torch

    from ndm_gym.algos.common import lr_at, set_lr
    from ndm_gym.wmbench.fit import fixed_windows, load_buffer, window_errors

    if a["lr_decay_hours"] is not None:
        raise SystemExit("--wm_finetune: the run decayed its learning rate by training hours, which the checkpoint does not record")
    lr = lr_at(iteration, a["lr"], a["lr_decay_iters"])
    wm_lr = lr if a["wm_lr"] is None else lr * a["wm_lr"] / a["lr"]
    torch.manual_seed(FINETUNE_SEED)
    fit, pen = [], []
    with torch.no_grad():
        for _ in range(20):
            w = buf.sample_windows(a["wm_batch"], a["wm_window"])
            fit.append(float(algo.wm_loss(algo.wm_rollout(w["s"], w["a"], w["xi"]), w)[0]))
            pen.append(float(algo._wm_smoothness(w["s"][:, 0], w["a"][:, 0], w["xi"][:, 0])))
    tests = {}
    for path in args.wm_test:
        tb, _ = load_buffer([path], dev)
        tests[os.path.splitext(os.path.basename(path))[0]] = {k: fixed_windows(tb, 8192, k, seed=k) for k in (8, 32)}

    def errors():
        out = {}
        with torch.no_grad():
            for name, by_k in tests.items():
                for k, w in by_k.items():
                    eb, eq = window_errors(algo, w)
                    out[f"{name}_k{k}"] = {"body_mm": 1e3 * float(eb.mean()), "q_mrad": 1e3 * float(eq.mean()), "n": len(eb)}
        return out

    res = {"fit_loss": float(np.mean(fit)), "smoothness": float(np.mean(pen)), "test_before": errors()}
    print(json.dumps({"wm_finetune": res}), flush=True)
    set_lr(algo.opt_wm, wm_lr)
    losses = [algo.wm_update(buf)[0] for _ in range(args.wm_finetune)]
    res.update({"updates": args.wm_finetune, "lr": wm_lr, "test_after": errors(),
                "fit_loss_per_step_last_200": float(torch.stack(losses[-200:]).mean()) if losses else None})  # fmt: skip
    return res


class Anatomy:
    """--anatomy: an update's policy gradient g taken apart by imagined step t and by rollout b, g = sum_t g_t =
    sum_b G_b (``ndm_gym.algos.rollout_grads``). The imagination step runs eagerly with the world model's step compiled
    on its own (the run compiles the whole imagination step; equal up to rounding), so that tensor hooks see, at every
    step, the gradient that reaches the policy's output from all its later uses (the world model, the observations, the
    losses: the output's adjoint) and the one that reaches the world model's next state. After the update the policy runs
    again on every step's inputs as in the update (``RolloutGrads``), which gives the parts; both sums are checked
    against g."""

    def __init__(self, algo, chunk: int):
        import torch

        from ndm_gym.policies.tracking_policy import TrackingPolicy

        self.algo, self.chunk, self.on = algo, chunk, False
        self.steps, self.states, self.f = [], [], None
        self.vsum = {}  # per aggregation of take_apart: the sum of its unit gradients over the updates
        pi, wm, sample = algo.pi, algo.wm, algo.sample_valid_starts
        step = torch.compile(wm.step)

        def pi_forward(o_hist, g):
            a = TrackingPolicy.forward(pi, o_hist, g)
            if self.on and a.requires_grad:
                rec = {"o": o_hist.detach(), "g": g.detach()}
                self.steps.append(rec)
                a.register_hook(lambda d: rec.__setitem__("adj", d.detach()))
            return a

        def wm_step(S, a, xi_n, input_noise=0.0):
            S1 = step(S, a, xi_n, input_noise)
            if self.on and S1[0].requires_grad:
                rec = {"sq": None}
                self.states.append(rec)

                def hook(d):
                    sq = d.detach().float().flatten(1).square().sum(1)
                    rec["sq"] = sq if rec["sq"] is None else rec["sq"] + sq

                for x in S1:
                    x.register_hook(hook)
            return S1

        def starts(buffer, B):
            s, h = sample(buffer, B)
            if self.on:
                self.f = s["f"].clone()
            return s, h

        pi.forward, wm.step, algo.sample_valid_starts = pi_forward, wm_step, starts
        if algo.args.algo == "supertrack":
            algo.st_imag_step = algo._st_imag_step
        else:
            algo.imag_step = algo._imag_step

    def start(self):
        self.on, self.steps, self.states, self.f = True, [], [], None

    def parts(self, g, ref) -> dict:
        """The update's gradient g (flat, the policy's parameters in order) taken apart, against g and the reference
        direction ref: per step t |g_t|^2, g_t . g, g_t . ref (H each) and their sum over the steps (P); per rollout
        |G_b|^2, G_b . g, G_b . ref (B each); the norms of the policy outputs' and the next states' adjoints (H, B). The
        rerun stays in ``self.rg`` for ``RolloutGrads.weighted``."""
        import torch

        from ndm_gym.algos.rollout_grads import RolloutGrads

        self.on = False
        algo = self.algo
        H, B = len(self.steps), len(self.f)
        if H != algo.HZ or len(self.states) != H or any("adj" not in r for r in self.steps):
            raise RuntimeError(f"--anatomy saw {H} policy calls, {len(self.states)} world-model steps; expected {algo.HZ} with gradients")
        A = torch.stack([r["adj"] for r in self.steps]).float()  # (H, B, J) the policy outputs' adjoints
        rg = self.rg = RolloutGrads(algo.pi, torch.stack([r["o"] for r in self.steps]), torch.stack([r["g"] for r in self.steps]), A, self.chunk)
        W, bias = rg.step_parts()
        offs, lins = rg.offs, rg.lins
        part = {}
        for i, m in enumerate(lins):
            part[m.weight], part[m.bias] = W[i].sum(0), bias[i].sum(0)
        return {
            "step_gram": sum(W[i].flatten(1) @ W[i].flatten(1).T + bias[i] @ bias[i].T for i in range(len(lins))),  # g_t . g_t' (H, H)
            "step_sq": sum(W[i].flatten(1).square().sum(1) + bias[i].square().sum(1) for i in range(len(lins))),
            "step_dot_g": sum(W[i].flatten(1) @ g[offs[m.weight]] + bias[i] @ g[offs[m.bias]] for i, m in enumerate(lins)),
            "step_dot_ref": sum(W[i].flatten(1) @ ref[offs[m.weight]] + bias[i] @ ref[offs[m.bias]] for i, m in enumerate(lins)),
            "step_sum": rg.flat(part),
            "rollout_sq": rg.rollout_sq_norms(),
            "rollout_dot_g": rg.rollout_dots(g),
            "rollout_dot_ref": rg.rollout_dots(ref),
            "adj": A.norm(dim=-1),
            "state_adj": torch.stack([g.new_zeros(B) if r["sq"] is None else r["sq"] for r in self.states]).sqrt(),
        }

    def take_apart(self, g, ref) -> dict:
        """``parts`` summarized: per step (H values each: norm relative to |g|, share of |g|^2 summing to 1, cosine with
        the reference, the adjoints' median and maximum over the rollouts) and over the rollouts."""
        import torch

        P = self.parts(g, ref)
        algo, B = self.algo, len(self.f)
        gn2 = g @ g
        sq, an, sn = P["step_sq"], P["adj"], P["state_adj"]
        n = P["rollout_sq"].clamp_min(0).sqrt()
        c = P["rollout_dot_g"] / gn2  # each rollout's share of |g|^2 (summing to 1)
        cs, idx = c.sort(descending=True)
        q = torch.quantile(n, torch.tensor([0.5, 0.9, 0.99], device=n.device))
        top = []
        for j in idx[:3].tolist():
            f = int(self.f[j])
            k = int(algo.motion.clip_id[f])
            top.append([algo.motion.names[k], f - int(algo.motion.clip_start[k]), float(n[j] / q[0]), float(c[j])])
        agg = self.last = self.variant_vectors(P)
        variants = {}
        for k, v in agg.items():
            vn = v.norm()
            variants[k] = [float(vn), float(v @ ref / vn), float(v @ g / (vn * gn2.sqrt()))]  # norm, cosine with ref, with g
            self.vsum[k] = self.vsum.get(k, 0) + v / vn
        live, e = n > 0, an.square().sum(0).sqrt()
        both = live & (e > 0)
        k1 = max(1, B // 100)
        rnd = lambda x: [round(v, 6) for v in x.tolist()]  # noqa: E731
        offs = self.rg.offs
        layer_sq = torch.stack([g[offs[m.weight]].square().sum() + g[offs[m.bias]].square().sum() for m in self.rg.lins])
        return {
            "variants": variants,
            "layer_share": rnd(layer_sq / gn2),  # each linear layer's share of |g|^2 (the encoder's, then the decoder's)
            "proxy_logcorr": float(torch.corrcoef(torch.stack((n[both].log(), e[both].log())))[0, 1]),
            "proxy_top1pct_overlap": len(set(n.topk(k1).indices.tolist()) & set(e.topk(k1).indices.tolist())) / k1,
            "check_steps": float((P["step_sum"] - g).norm() / gn2.sqrt()),
            "check_rollouts": float((P["rollout_dot_g"].sum() - gn2).abs() / gn2),
            "step_norm": rnd(sq.sqrt() / gn2.sqrt()),
            "step_gram": [rnd(r) for r in P["step_gram"] / gn2],  # g_t . g_t' / |g|^2: any per-step reweighting's norm, offline
            "step_share": rnd(P["step_dot_g"] / gn2),
            "step_cos_ref": rnd(P["step_dot_ref"] / sq.sqrt().clamp_min(1e-30)),
            "adj_med": rnd(an.median(1).values),
            "adj_max": rnd(an.amax(1)),
            "state_adj_med": rnd(sn.median(1).values),
            "state_adj_max": rnd(sn.amax(1)),
            "rollout_norm_p50": float(q[0]),
            "rollout_norm_p90_rel": float(q[1] / q[0]),
            "rollout_norm_p99_rel": float(q[2] / q[0]),
            "rollout_norm_max_rel": float(n.max() / q[0]),
            "top1_share": float(cs[0]),
            "top10_share": float(cs[:10].sum()),
            "top1pct_share": float(cs[: max(1, B // 100)].sum()),
            "neff": float(n.sum() ** 2 / P["rollout_sq"].sum()),  # (sum |G_b|)^2 / sum |G_b|^2: B when the rollouts' norms are equal
            "align": float(gn2 / P["rollout_sq"].sum()),  # |sum G_b|^2 / sum |G_b|^2: about 1 for unrelated rollouts, B for equal ones
            "rollout_cos_g_mean": float((P["rollout_dot_g"] / (n * gn2.sqrt()).clamp_min(1e-30)).mean()),
            "rollout_cos_ref_mean": float((P["rollout_dot_ref"] / n.clamp_min(1e-30)).mean()),
            "top": top,  # the 3 rollouts with the largest shares: clip, start frame in it, norm / median, share
        }

    def variant_vectors(self, P: dict) -> dict:
        """Robust aggregations of the batch's rollout gradients (``parts``' P; flat): |G_b| clipped at k times the median
        over the rollouts with a gradient (rclip<k>; with the rollout's output adjoint over the horizon, the action-space
        proxy of |G_b|: _proxy), every |G_b| scaled to the median (rnorm), the largest 1% left out (trim1), the geometric
        median of the chunks' sums times their number (mom<chunks>); rerun: all weights 1, the check of the rerun."""
        import torch

        n, e = P["rollout_sq"].clamp_min(0).sqrt(), P["adj"].square().sum(0).sqrt()
        live = n > 0
        inv = torch.where(live, n[live].median() / n.clamp_min(1e-30), 0.0)
        weights = {"rerun": torch.ones_like(n), "rnorm": inv, "trim1": torch.ones_like(n).index_fill(0, n.topk(max(1, len(n) // 100)).indices, 0.0)}
        for k in (2, 4, 8):
            weights[f"rclip{k}"] = torch.clamp(k * inv, max=1.0)
        weights["rclip4_proxy"] = torch.where(e > 0, torch.clamp(4 * e[e > 0].median() / e.clamp_min(1e-30), max=1.0), 0.0)
        agg, groups = self.rg.weighted(weights)
        agg[f"mom{len(groups)}"] = len(groups) * geometric_median(torch.stack(groups))
        return agg

    def twin(self, g32, ref) -> dict:
        """--fp32_twin: every aggregation of the fp32 update's rollout gradients (recorded since ``start``), its cosine
        with the same aggregation of the bf16 update's (the last ``take_apart``)."""
        agg = self.variant_vectors(self.parts(g32, ref))
        return {k: float(v @ self.last[k] / (v.norm() * self.last[k].norm())) for k, v in agg.items()}


def geometric_median(X, iters: int = 100):
    """The point minimizing the sum of Euclidean distances to the rows of X (m, P), by Weiszfeld's iteration."""
    z = X.mean(0)
    for _ in range(iters):
        w = 1 / (X - z).norm(dim=1).clamp_min(1e-9 * float(z.norm()))
        z = (w[:, None] * X).sum(0) / w.sum()
    return z


def variant_summary(recs: list[dict], vsum: dict, ref, base_mean) -> dict:
    """Per aggregation of ``Anatomy.take_apart``: the statistics of the base setting's summary (norms, spikes above 3
    and 10 times its own median, cosine with the reference: mean, of the spikes; consistency) and the cosine of each
    update with the base gradient and of its mean direction with the base setting's."""
    import numpy as np

    out = {}
    for k in recs[0]["variants"]:
        v = np.array([r["variants"][k] for r in recs])
        fin = np.isfinite(v[:, 0])
        g, c, cg = v[fin, 0], v[fin, 1], v[fin, 2]
        med = float(np.median(g))
        sp = g > 3 * med
        mean = vsum[k] / int(fin.sum())
        out[k] = {"gradnorm_median": med, "gradnorm_p99": float(np.percentile(g, 99)), "gradnorm_max": float(g.max()),
                  "above_3x_median": int(sp.sum()), "above_10x_median": int((g > 10 * med).sum()), "cos_to_ref_mean": float(c.mean()),
                  "cos_to_ref_spikes_mean": float(c[sp].mean()) if sp.any() else None, "cos_to_base_mean": float(cg.mean()),
                  "mean_unit_gradient_length": float(mean.norm()), "mean_direction_cos_to_ref": float(mean @ ref / mean.norm()),
                  "mean_direction_cos_to_base": float(mean @ base_mean / (mean.norm() * base_mean.norm()))}  # fmt: skip
    return out


def anatomy_summary(recs: list[dict], gns, twins: list[dict]) -> dict:
    """Means (profiles: per step) and medians of --anatomy's per-update values over all updates, the spikes (norm above
    three times the median) and the others; with --fp32_twin, the agreement of the bf16 and fp32 gradients."""
    import numpy as np

    g = np.array(gns)
    fin = np.isfinite(g)
    med = float(np.median(g[fin]))
    groups = {"all": fin, "spikes": fin & (g > 3 * med), "normal": fin & (g <= 3 * med)}
    out = {}
    for name, sel in groups.items():
        ids = np.flatnonzero(sel)
        if not len(ids):
            out[name] = None
            continue
        r = [recs[i] for i in ids]
        prof = lambda k: np.mean([x[k] for x in r], 0).round(5).tolist()  # noqa: E731
        medv = lambda k: float(np.median([x[k] for x in r]))  # noqa: E731
        out[name] = {"updates": len(ids), **{k: prof(k) for k in ("step_share", "step_norm", "step_cos_ref", "adj_med", "adj_max", "state_adj_med",
                                                                  "layer_share")},  # fmt: skip
                     **{k: medv(k) for k in ("top1_share", "top10_share", "top1pct_share", "neff", "align", "rollout_norm_p99_rel",
                                             "rollout_norm_max_rel", "rollout_cos_g_mean", "rollout_cos_ref_mean", "check_steps",
                                             "check_rollouts", "proxy_logcorr", "proxy_top1pct_overlap")}}  # fmt: skip
        if twins:
            t = [twins[i] for i in ids]
            out[name]["fp32_cos_median"] = float(np.median([x["cos"] for x in t]))
            out[name]["fp32_cos_p10"] = float(np.percentile([x["cos"] for x in t], 10))
            out[name]["fp32_norm_ratio_median"] = float(np.median([x["norm_ratio"] for x in t]))
            for k in t[0]["variants_cos"]:  # each aggregation of the bf16 against the fp32 update: median, 10th percentile
                vc = [x["variants_cos"][k] for x in t]
                out[name][f"fp32_cos_{k}"] = [float(np.median(vc)), float(np.percentile(vc, 10))]
    if twins:
        g32 = np.array([x["gradnorm"] for x in twins])
        f32 = np.isfinite(g32)
        s16, s32 = fin & (g > 3 * med), f32 & (g32 > 3 * float(np.median(g32[f32])))
        out["fp32"] = {"gradnorm_median": float(np.median(g32[f32])), "above_3x_median": int(s32.sum()), "both_above_3x": int((s16 & s32).sum()),
                       "cos_to_ref_mean": float(np.mean([x["cos_ref"] for x, ok in zip(twins, f32) if ok]))}  # fmt: skip
    return out


def main():
    args = parse_args()
    if args.save_ref and (any(kv.partition("=")[0] not in SAME_GRADIENT for kv in args.set) or args.wm_finetune or args.wm_test):
        raise SystemExit(f"--save_ref computes the reference with the run's own setting and world model: no --set (but of {SAME_GRADIENT}), "
                         "--wm_finetune or --wm_test")  # fmt: skip
    if args.fp32_twin and (args.save_ref or not args.anatomy):
        raise SystemExit("--fp32_twin is part of the measurement with --anatomy (not of --save_ref)")

    import numpy as np
    import torch

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import set_lr
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1, read_clip_list, select
    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP
    from ndm_gym.train import combo_kernels, elementwise_3x3_products, training_reward_cfg
    from ndm_gym.wmbench.fit import load_buffer

    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.max_gpu_gb is not None:
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb * 2**30 / torch.cuda.get_device_properties(dev).total_memory))
    ck = torch.load(args.ckpt, map_location=dev)
    for key in ("pi", "wm", "opt_pi", "iteration", "args"):
        if key not in ck:
            raise SystemExit(f"{args.ckpt} has no {key!r}: pass the run's last.pt (the full training state), not a ckpt_*.pt")
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    with open(os.path.join(run_dir, "args.json")) as f:
        a = run_arguments(ck["args"], args.set, launched=set(ck["args"]) - set(json.load(f)))
    if args.anatomy and a["pi_cudagraphs"]:
        raise SystemExit("--anatomy runs the imagination step eagerly: pass --set pi_cudagraphs=0 (the CUDA graphs do not change the gradient)")
    if args.fp32_twin and a["wm_bptt_fp32"]:
        raise SystemExit("--fp32_twin compares the run's bf16 gradient with the fp32 one: the setting must not have wm_bptt_fp32")
    nets.AMP_DTYPE = torch.bfloat16 if a["amp"] == "bf16" else None
    if a["elementwise_3x3"]:  # the run's products in compiled functions, as train.py sets them up (the other speed options: Algo)
        elementwise_3x3_products()
    if a["combo_kernels"]:
        combo_kernels()
    kin = g1.kinematics(dev)
    clips = lafan1.load_dir(args.motion_dir)
    if a["clips"]:  # the clips the run trained on, as train.py recorded them
        clips = select(clips, read_clip_list(os.path.join(run_dir, "clips.txt")))
    motion = MotionLib(clips, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev), training_reward_cfg(a["reward_std_scale"]))
    algo = ALGOS[a["algo"]](argparse.Namespace(**a), kin, motion, mdp, dev)
    algo.load_state_dict(ck)
    buf, data_meta = load_buffer(args.data, dev)
    for m in data_meta:
        if m["algo"] != a["algo"]:
            raise SystemExit(f"dataset recorded with a {m['algo']} policy, the checkpoint is {a['algo']}")
    finetune = wm_finetune(algo, buf, a, ck["iteration"], args, dev) if (args.wm_finetune or args.wm_test) else None
    anatomy = Anatomy(algo, args.anatomy_chunk) if args.anatomy else None
    for o in algo.optimizers:
        set_lr(o, 0.0)
    mods = [m for m in (algo.pi, algo.wm, getattr(algo, "critic", None)) if m is not None]
    params0 = [p.detach().clone() for m in mods for p in m.parameters()]
    state0 = [{k: v.clone() for k, v in m.state_dict().items() if k not in dict(m.named_parameters())} for m in mods]

    # the raw policy gradient of every update, before policy_update clips it
    params = list(algo.pi.parameters())
    sizes = [p.numel() for p in params]
    offs = np.cumsum([0] + sizes).tolist()
    flat = torch.zeros(offs[-1], device=dev)

    def capture(i):
        def hook(p):
            flat[offs[i] : offs[i + 1]].copy_(p.grad.reshape(-1))

        return hook

    for i, p in enumerate(params):
        p.register_post_accumulate_grad_hook(capture(i))
    unit = lambda v: v / max(float(v.norm()), 1e-30)  # noqa: E731

    def update(seed: int):
        torch.manual_seed(seed)
        flat.zero_()
        t0 = time.time()
        out = algo.policy_update(buf)
        torch.cuda.synchronize()
        dt = time.time() - t0
        for m, s in zip(mods, state0):
            m.load_state_dict(s, strict=False)
        gn = float(out[3])
        if np.isfinite(gn) and abs(float(flat.norm()) - gn) > 1e-3 * max(gn, 1e-6):
            raise RuntimeError(f"captured gradient norm {float(flat.norm())} differs from the update's {gn}")
        return gn, float(out[0]), dt

    run = os.path.basename(run_dir)
    if args.save_ref:
        ref, secs_ref = torch.zeros_like(flat), []
        for j in range(args.ref_updates):
            gn, _, dt = update(REF_SEED0 + j)
            secs_ref.append(dt)
            if np.isfinite(gn):
                ref += unit(flat)
        ref = unit(ref)
        torch.save({"ref": ref.cpu(), "run": run, "iteration": ck["iteration"], "updates": args.ref_updates, "anatomy": args.anatomy,
                    "set": args.set}, args.save_ref)  # fmt: skip
        if args.updates == 0:  # the reference only
            print(f"[grad_stats] saved {args.save_ref} ({args.ref_updates} updates, {np.median(secs_ref):.2f} s each)", flush=True)
            return
    else:
        saved = torch.load(args.ref, map_location=dev)
        if (saved["run"], saved["iteration"]) != (run, ck["iteration"]):
            raise SystemExit(f"{args.ref} is for {saved['run']} at {saved['iteration']}, not {run} at {ck['iteration']}")
        if saved.get("anatomy", False) != args.anatomy:  # the eager imagination step of --anatomy rounds differently
            raise SystemExit(f"{args.ref} was saved {'with' if saved.get('anatomy', False) else 'without'} --anatomy; measure the same way")
        ref = saved["ref"]

    gns, losses, cos, secs, recs, twins = [], [], [], [], [], []
    mean = torch.zeros_like(flat)

    def write_out(final: bool):
        """The JSON of the updates so far (``--save_every``: also while running; ``summary.updates`` says how many)."""
        g = np.array(gns)
        fin = np.isfinite(g)
        med = float(np.median(g[fin]))
        c = np.array(cos)[fin]
        summary = {
            "updates": len(gns),
            "nonfinite": int((~fin).sum()),
            "gradnorm_median": med,
            "gradnorm_p90": float(np.percentile(g[fin], 90)),
            "gradnorm_p99": float(np.percentile(g[fin], 99)),
            "gradnorm_max": float(g[fin].max()),
            "above_3x_median": int((g[fin] > 3 * med).sum()),
            "above_10x_median": int((g[fin] > 10 * med).sum()),
            "cos_to_ref_mean": float(c.mean()),
            "cos_to_ref_median": float(np.median(c)),
            "cos_to_ref_spikes_mean": float(c[g[fin] > 3 * med].mean()) if (g[fin] > 3 * med).any() else None,
            "mean_unit_gradient_length": float(mean.norm()) / int(fin.sum()),
            "mean_unit_gradient_cos_to_ref": float(unit(mean) @ ref),
            "sec_per_update_median": float(np.median(secs)),
            "wm_finetune": finetune,
        }
        if anatomy is not None:
            summary["anatomy"] = anatomy_summary(recs, gns, twins)
            summary["variants"] = variant_summary(recs, anatomy.vsum, ref, mean)
            summary["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        meta = {"ckpt": os.path.abspath(args.ckpt), "run": run, "iteration": ck["iteration"], "algo": a["algo"], "backbone": a["wm_backbone"],
                "set": args.set, "data": [os.path.abspath(d) for d in args.data], "data_meta": data_meta, "ref": args.save_ref or args.ref,
                "wm_test": [os.path.abspath(d) for d in args.wm_test], "anatomy": args.anatomy, "fp32_twin": args.fp32_twin,
                "seed0": args.seed0, "updates_requested": args.updates, "final": final, "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__}  # fmt: skip
        out = {"meta": meta, "summary": summary, "gradnorm": gns, "loss": losses, "cos_to_ref": cos, "sec": secs}
        if anatomy is not None:
            out.update({"anatomy": recs, "fp32_twin": twins})
        with open(args.out + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(args.out + ".tmp", args.out)
        if final:
            print(json.dumps({"run": run, "set": args.set, **{k: v for k, v in summary.items() if k not in ("anatomy", "variants")}}))

    t_loop = time.time()
    for i in range(args.updates):
        if anatomy is not None:
            anatomy.start()
        gn, loss, dt = update(args.seed0 + i)
        gns.append(gn), losses.append(loss), secs.append(dt)
        cos.append(float(unit(flat) @ ref))
        if np.isfinite(gn):
            mean += unit(flat)
        if anatomy is not None:
            recs.append(anatomy.take_apart(flat, ref))
        if args.fp32_twin:  # the same batch and noise (same seed), the world model without autocast in the BPTT
            g16 = flat.clone()
            algo.args.wm_bptt_fp32 = True
            anatomy.start()
            gn32 = update(args.seed0 + i)[0]
            algo.args.wm_bptt_fp32 = False
            twins.append({"gradnorm": gn32, "cos": float(unit(flat) @ unit(g16)), "norm_ratio": gn32 / gn, "cos_ref": float(unit(flat) @ ref),
                          "variants_cos": anatomy.twin(flat, ref)})  # fmt: skip
        if anatomy is not None and (i + 1) % 50 == 0:
            print(f"[grad_stats] {i + 1}/{args.updates} updates, {(time.time() - t_loop) / (i + 1):.2f} s each with --anatomy"
                  f"{' --fp32_twin' if args.fp32_twin else ''} (the update alone {np.median(secs):.2f} s), peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)  # fmt: skip
        if args.save_every and (i + 1) % args.save_every == 0 and i + 1 < args.updates:
            write_out(final=False)
    for p0, p in zip(params0, (p for m in mods for p in m.parameters())):
        if not torch.equal(p0, p):
            raise RuntimeError("a parameter changed at learning rate 0")
    write_out(final=True)


if __name__ == "__main__":
    main()
