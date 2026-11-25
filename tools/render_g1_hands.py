"""Render a G1 hand task of tools/build_g1_hand_task.py in the look of the hand previews: the G1 at its table and a closer
view of the table and the hands, side by side; while the G1 walks the first view follows it.

    MUJOCO_GL=egl python tools/render_g1_hands.py --task <build_g1_hand_task.py's out dir> --out demo.mp4 [--webp demo.webp]
    MUJOCO_GL=egl python tools/render_g1_hands.py --task <dir> --rollout <evaluate_hand's rollout.npz> --out demo.mp4

The reference's sequence, or with --rollout a policy's: the walk up to the reach as the reference's, then the rollout's
arms, fingers and object, the pelvis, the legs and the waist held as at the reach. Also writes <out>.png, five frames in a
column, and with --webp an animated WebP at half the size.
"""

import argparse
import os
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_hand import SHOWN_GROUPS, studio_scene  # noqa: E402

STILLS = (0.03, 0.3, 0.5, 0.7, 0.97)  # the frames of the stills, as fractions of the sequence
WALK_SHADOW_CLIP = 12.0  # m, the shadow map's range while the G1 walks


def policy_sequence(task: str, model: mujoco.MjModel, qpos: np.ndarray, reach: int, rollout: str) -> np.ndarray:
    """The reference's frames before the reach, then the rollout's joints in the demo scene's order."""
    task_model = mujoco.MjModel.from_xml_path(os.path.join(task, "scene.xml"))
    played = np.load(rollout)["qpos"]
    policy = np.tile(qpos[reach], (len(played), 1))
    for joint in range(task_model.njnt):
        width = 7 if task_model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else 1
        source, target = task_model.jnt_qposadr[joint], model.joint(task_model.joint(joint).name).qposadr[0]
        policy[:, target : target + width] = played[:, source : source + width]
    return np.concatenate((qpos[:reach], policy))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--rollout", type=str, default=None, help="evaluate_hand's rollout.npz of a policy on the task")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--webp", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1: real time)")
    parser.add_argument("--shadow", type=int, default=2048, help="the shadow map's size")
    parser.add_argument("--samples", type=int, default=4, help="antialiasing samples")
    args = parser.parse_args()

    demo = os.path.join(args.task, "demo")
    reference = np.load(os.path.join(demo, "reference.npz"))
    qpos, dt, reach = reference["qpos"], float(reference["dt"]), int(reference["reach_starts"])
    width = args.width // 2
    model = studio_scene(os.path.join(demo, "scene.xml"), width, args.height, ghost=False)
    model.vis.quality.shadowsize, model.vis.quality.offsamples, model.vis.map.shadowclip = args.shadow, args.samples, 2.5
    walking = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint") >= 0
    if walking:  # the walk's shadows reach the far floor: a shorter range leaves acne and an edge on the floor where it ends
        model.vis.quality.shadowsize = max(args.shadow, 16384)
    if args.rollout:
        qpos = policy_sequence(args.task, model, qpos, reach, args.rollout)
    data = mujoco.MjData(model)
    x = np.arange(0, len(qpos) - 1 + 1e-9, args.speed / (args.fps * dt))  # the frames at the video's rate, linearly interpolated
    low = np.minimum(np.floor(x).astype(int), len(qpos) - 2)
    frames = qpos[low] + (qpos[low + 1] - qpos[low]) * (x - low)[:, None]
    for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        quat = slice(model.jnt_qposadr[joint] + 3, model.jnt_qposadr[joint] + 7)
        frames[:, quat] /= np.linalg.norm(frames[:, quat], axis=-1, keepdims=True)

    # the cameras: the wide one on the pelvis at the table (followed while walking), the close one on the hands and the object
    data.qpos[:] = qpos[reach]
    mujoco.mj_kinematics(model, data)
    pelvis_id = model.body("pelvis").id
    pelvis, heading = data.xpos[pelvis_id].copy(), data.xmat[pelvis_id].reshape(3, 3)[:, 0]
    facing = np.degrees(np.arctan2(heading[1], heading[0]))
    forward = np.array([np.cos(np.radians(facing)), np.sin(np.radians(facing))])
    bases = [model.body(f"{side}_hand_base").id for side in ("right", "left") if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base") >= 0]
    held = []
    for frame in qpos[reach::10]:
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        held.append(np.concatenate((data.xpos[bases], data.xpos[[model.body("right_object").id]])))
    held = np.concatenate(held)
    wide, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    wide.lookat[:] = (*(pelvis[:2] + 0.2 * forward), pelvis[2] - 0.05)
    wide.distance, wide.azimuth, wide.elevation = (2.7, facing + 180.0 - 62.0, -10.0) if walking else (2.1, facing + 180.0 - 42.0, -10.0)
    close.lookat[:] = (held.min(0) + held.max(0)) / 2
    close.distance, close.azimuth, close.elevation = (1.35, facing + 180.0 + 32.0, -24.0) if walking else (0.95, facing + 180.0 + 32.0, -30.0)
    follow = 1 - (1 - 0.18) ** (25 / args.fps)  # tools/render.py's smoothing of what the camera follows
    root = model.joint("floating_base_joint").qposadr[0] if walking else None
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[:SHOWN_GROUPS] = 1
    option.sitegroup[:] = 0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    encoder = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * width}x{args.height}",
                                "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out],
                               stdin=subprocess.PIPE)  # fmt: skip
    renderer = mujoco.Renderer(model, args.height, width)
    # the shadow map's range per frame: the walk's, then the table's finer one (the context copies it from the model when made)
    shadow_range = lambda source: WALK_SHADOW_CLIP if walking and source < reach else model.vis.map.shadowclip  # noqa: E731
    stills, column, small = {int(round(s * (len(frames) - 1))) for s in STILLS}, [], []
    for index, frame in enumerate(frames):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        renderer._mjr_context.shadowClip = shadow_range(x[index])
        if walking:
            ahead = frame[root : root + 2] + 0.2 * forward
            wide.lookat[:2] = ahead if index == 0 else wide.lookat[:2] + follow * (ahead - wide.lookat[:2])
        views = []
        for camera in (wide, close):
            renderer.update_scene(data, camera, option)
            views.append(renderer.render().copy())
        image = np.concatenate(views, axis=1)
        encoder.stdin.write(image.tobytes())
        if index in stills:
            column.append(image)
        if args.webp and index % 2 == 0:
            small.append(Image.fromarray(image).resize((width, args.height // 2), Image.LANCZOS))
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    Image.fromarray(np.concatenate(column, axis=0)).save(os.path.splitext(args.out)[0] + ".png")
    if args.webp:
        small[0].save(args.webp, save_all=True, append_images=small[1:], duration=int(round(2000 / args.fps)), loop=0, quality=70, method=4)
    print(args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
