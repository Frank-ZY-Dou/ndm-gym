"""Render a saved physical reference or policy rollout in the native Adroit scene.

Visualization only: states are assigned for rendering, never used as feasibility
evidence. Supply the exact replay model with its assets resolved separately.
Writes a lossless WebP preview outside the source tree. A policy label requires
saved policy rollout states; rendering itself does not establish performance.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive endpoint; omit the terminal frame for README previews")
    parser.add_argument("--rollouts", type=Path, help="Saved physical-rollouts.npz from checkpoint evaluation")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--label", default="learned policy", help="Method and seed shown on saved policy trajectories")
    args = parser.parse_args()
    if args.stride < 1 or args.output.suffix != ".webp" or args.output.exists():
        parser.error("use positive stride and a new .webp output")

    from render import _font, draw_mark
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    from ndm_gym.motion.adroit_door import JOINT_NAMES, DoorReference

    ref = DoorReference.load(args.reference)
    end = args.end_frame if args.end_frame is not None else ref.frames
    if not 0 <= args.start_frame < end <= ref.frames:
        parser.error("frame window must be nonempty and within the saved trajectory")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt))
    if names != JOINT_NAMES or model.nq != 30 or model.nv != 30:
        raise ValueError("Model does not have the reference's native joint layout")
    qpos, qvel = ref.qpos, ref.qvel
    if args.rollouts:
        with np.load(args.rollouts, allow_pickle=False) as saved:
            qpos = saved[f"episode{args.episode}_qpos"]
            qvel = saved[f"episode{args.episode}_qvel"]
        if qpos.shape != ref.qpos.shape or qvel.shape != ref.qvel.shape:
            raise ValueError("Saved physical rollout must cover the complete reference")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Saved physical rollout contains nonfinite state")
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -25, 1.8
    camera.lookat[:] = [-.2, .1, .35]
    frames = []
    bold = _font(0, 20)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for f in range(args.start_frame, end, args.stride):
            data.qpos[:], data.qvel[:] = qpos[f], qvel[f]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(np.array(renderer.render(), copy=True)).convert("RGBA")
            overlay = Image.new("RGBA", frame.size)
            draw = ImageDraw.Draw(overlay)
            draw.text((12, 12), ("Adroit Door: " + args.label if args.rollouts else "Adroit Door: physical replay reference"), font=bold, fill=(255, 255, 255, 255))
            draw.text((12, 38), ("Saved physical control rollout" if args.rollouts else "Not a trained policy"), font=bold, fill=(255, 255, 255, 255))
            draw_mark(draw, 640, 480, bold, fill=(255, 255, 255, 220))
            frames.append(Image.alpha_composite(frame, overlay).convert("RGB"))
    frames[0].save(args.output, save_all=True, append_images=frames[1:], lossless=True,
                   duration=round(1000 * ref.metadata["control_dt"] * args.stride), loop=0)
    import hashlib
    import json
    inputs = {"reference": args.reference, "model": args.model}
    if args.rollouts:
        inputs["rollouts"] = args.rollouts
    provenance = dict(label=args.label if args.rollouts else "physical replay reference",
                      camera=dict(azimuth=camera.azimuth, elevation=camera.elevation,
                                  distance=camera.distance, lookat=camera.lookat.tolist()),
                      episode=args.episode if args.rollouts else None,
                      start_frame=args.start_frame, end_frame_exclusive=end, stride=args.stride,
                      contains_terminal_frame=bool(ref.frames-1 in range(args.start_frame, end, args.stride)),
                      input_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
                      output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f"Saved {len(frames)} saved trajectory frames to {args.output}; rendering is not physics replay")


if __name__ == "__main__":
    main()
