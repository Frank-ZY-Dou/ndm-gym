"""Score a neural dynamics model fitted by ``wmbench/fit_hoi.py`` (its ``<name>_wm.pt``) on held-out datasets through
one inference path: fit_hoi's scores (``window_scores``: keypoint, joint, object and contact force errors after k
open-loop steps) on its fixed windows (``fixed_windows``, seed k), with the step run as ``wmbench/speed_hoi.py`` times
it (``speed_hoi.inference_model``): compiled by --compile (reduce-overhead: CUDA graphs), the model itself or its
inference copy (--inference_copy, --rf_attention, --fused), the RigidFormer's blocks or whole network as fused kernels
(--rf_kernels), bf16 autocast by --amp (default: the fit's setting). The eager path (--compile none, no copy, the
fit's --amp) is fit_hoi's own scoring and reproduces the fit's last rows; the other paths show the accuracy that a
timed path keeps. Appends one row per test set and k to --out: fit_hoi's score columns and the path.

Run: ``python -m ndm_gym.wmbench.score_hoi --model results/wmbench_hoi/<name>_wm.pt --test held=data_wm/hoi/b.pt
       --inference_copy amp --fused --rf_kernels blocks --compile reduce-overhead --out results/wmbench_hoi/<name>_paths.csv``
"""

from __future__ import annotations

import argparse
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="fit_hoi.py's <name>_wm.pt")
    p.add_argument("--test", nargs="+", required=True, help="name=path held-out datasets")
    p.add_argument("--eval_windows", type=int, default=4096)
    p.add_argument("--eval_k", type=int, nargs="+", default=[8, 32])
    p.add_argument("--compile", type=str, default="none", choices=["none", "default", "reduce-overhead"], help="torch.compile mode of the step")
    p.add_argument("--inference_copy", type=str, default="none", choices=["none", "amp", "bf16"],
                   help="HOIWorldModel.inference_copy with the network in bf16 end to end (bf16) or in fp32 under autocast (amp)")
    p.add_argument("--rf_attention", type=str, default="token", choices=["token", "sdpa"], help="inference copy of the RigidFormer: attention")
    p.add_argument("--fused", action="store_true", help="inference copy: the step outside the network as two Triton kernels")
    p.add_argument("--rf_kernels", type=str, default="none", choices=["none", "blocks", "whole"],
                   help="RigidFormer: its blocks or whole network as fused Triton kernels, in the model as fitted or --inference_copy amp")
    p.add_argument("--amp", type=str, default=None, choices=["none", "bf16"], help="bf16 autocast (default: the fit's --amp)")
    p.add_argument("--out", type=str, required=True, help="csv to append the rows to")
    return p.parse_args()


def main():
    args = parse_args()

    import ndm_gym.core.nets as nets
    from ndm_gym.robots import g1
    from ndm_gym.tasks.hdmi import KEYPOINT_NAMES
    from ndm_gym.tasks.hdmi_tasks import TASKS
    from ndm_gym.wmbench.fit_hoi import SCORE_COLUMNS, fixed_windows, load_buffer, score_fields, window_scores, world_model
    from ndm_gym.wmbench.speed_hoi import inference_model
    from ndm_gym.worldmodels.hoi import ContextLayout

    ck = torch.load(args.model, map_location="cpu")
    fit, layout = ck["args"], ContextLayout(*ck["layout"])
    amp = fit["amp"] if args.amp is None else args.amp
    torch.backends.cuda.matmul.allow_tf32 = True  # as in fit_hoi
    nets.AMP_DTYPE = torch.bfloat16 if amp == "bf16" else None
    dev = "cuda:0"
    hinged = TASKS[fit["task"]].object_joint is not None
    wm = world_model(fit, layout)
    wm.load_state_dict(ck["wm"])
    wm, step, path = inference_model(wm.to(dev), args)
    path += f" amp {amp}"
    kin = g1.kinematics(dev, tracked=list(KEYPOINT_NAMES))
    graphs = args.compile == "reduce-overhead"

    def run(S, So, u, ctx):
        if args.compile == "none":  # as fit_hoi
            return step(S, So, u, ctx)
        if graphs:  # CUDA graphs reuse their output buffers: mark the step and keep copies (as speed_hoi's loop)
            torch.compiler.cudagraph_mark_step_begin()
        c = lambda z: tuple(t.contiguous() for t in z)  # noqa: E731  contiguous inputs, as speed_hoi's: one graph for every k
        out = step(c(S), c(So), u.contiguous(), ctx.contiguous())
        return tuple(tuple(t.clone() for t in z) for z in out) if graphs else out

    rows = []
    for spec in args.test:
        name, data = spec.split("=", 1)
        tb, tm = load_buffer([data], dev)
        if (tm[0]["task"], tuple(tm[0]["context_layout"])) != (fit["task"], tuple(layout)):
            raise ValueError(f"test set {data} is of another task or context layout than the model")
        for k in args.eval_k:
            s = window_scores(run, fixed_windows(tb, args.eval_windows, k, seed=k), kin, hinged)
            rows.append(f"{name},{k},{score_fields(s)},{path}\n")
            print(f"[score_hoi] {name} k{k}: keypoints {s['keypoint_mm']:.2f} mm, joints {s['joint_mrad']:.2f} mrad, object "
                  f"{s['obj_mrad'] if hinged else s['obj_mm']:.2f} ({path})", flush=True)  # fmt: skip
    new = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        if new:
            f.write(f"{SCORE_COLUMNS},path\n")
        f.writelines(rows)


if __name__ == "__main__":
    main()
