"""Render recorded evaluation trajectories (python -m ndm_gym.evaluate --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost at the robot's position. A studio look
at little cost: a directional key light that follows the robot and casts shadows, a fill light, a softly reflecting floor with a
fine grid, multisample anti-aliasing, a smoothly following camera and a small NDM Gym mark.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance1_subject1 jumps1_subject1 --seconds 12
"""

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MJCF = os.path.join(ROOT, "data", "g1_description", "g1_29dof.xml")
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
GHOST_RGBA = "0.25 0.58 0.95 0.18"
INK, MUTED, RED = (28, 30, 34), (92, 98, 108), (200, 45, 45)


def two_robot_xml() -> str:
    xml = open(MJCF).read()
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="' + os.path.join(ROOT, 'data', 'g1_description') + '/"', xml)
    body = xml[xml.index('<body name="pelvis"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', f'rgba="{GHOST_RGBA}"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    # key light above the robot, following its center of mass (so its shadow stays in view), and a weak fill light
    head = re.search(r'<body name="pelvis"[^>]*>', xml).group(0)
    xml = xml.replace(head, head + '\n      <light name="key" mode="trackcom" directional="true" pos="0.8 -0.6 3" dir="-0.3 0.25 -1" '
                      'castshadow="true" diffuse="0.58 0.57 0.55" specular="0.2 0.2 0.2"/>', 1)  # fmt: skip
    scene = """
    <light name="fill" directional="true" pos="0 0 5" dir="0.5 0.6 -0.6" castshadow="false" diffuse="0.30 0.31 0.33" specular="0 0 0"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.985 0.987 0.99" rgb2="0.86 0.88 0.91" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.80 0.81 0.83" rgb2="0.77 0.78 0.80" mark="edge"
             markrgb="0.66 0.68 0.72" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="1 1" texuniform="true" reflectance="0.10" shininess="0.2" specular="0.1"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    visual = """<visual>
    <global offwidth="2560" offheight="1440"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.34 0.34 0.35" diffuse="0.16 0.16 0.16" specular="0.05 0.05 0.05"/>
    <map shadowclip="2.0" shadowscale="0.6" haze="0.4"/>
    <rgba haze="0.95 0.96 0.97 1"/>
  </visual>
  """
    assert "<visual>" not in xml, "the robot model has its own <visual> settings; merge them here"
    return xml.replace("<asset>", visual + "<asset>", 1)


def _font(i, size):
    return ImageFont.truetype(FONTS[i], size) if os.path.exists(FONTS[i]) else ImageFont.load_default(size=size)


def _pill(draw, xy, parts, pad=(10, 6), anchor="la"):
    """Text parts [(text, font, color)] on one rounded, translucent white label at xy (left- or right-aligned)."""
    widths = [draw.textlength(t, font=f) for t, f, _ in parts]
    gap = 8
    w = sum(widths) + gap * (len(parts) - 1)
    h = max(f.size for _, f, _ in parts)
    x0 = xy[0] if anchor == "la" else xy[0] - w - 2 * pad[0]
    y0 = xy[1]
    draw.rounded_rectangle((x0, y0, x0 + w + 2 * pad[0], y0 + h + 2 * pad[1]), radius=(h + 2 * pad[1]) // 2, fill=(255, 255, 255, 205))
    x = x0 + pad[0]
    for (t, f, c), wi in zip(parts, widths):
        draw.text((x, y0 + pad[1] + h - f.size), t, font=f, fill=c)
        x += wi + gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="", help="bold label, e.g. the method")
    ap.add_argument("--subtitle", type=str, default="", help="label after the title, e.g. the seed")
    ap.add_argument("--global_ghost", action="store_true", help="draw the reference at its global xy instead of anchored to the robot")
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    names = list(d["names"])
    model = mujoco.MjModel.from_xml_string(two_robot_xml())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 140.0, -12.0
    s = a.height / 432  # label sizes scale with the frame
    bold, regular, small = _font(0, round(20 * s)), _font(1, round(18 * s)), _font(1, round(15 * s))
    clips = a.clips or names
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = int(round(50 / a.fps))
    for clip in clips:
        c = names.index(clip)
        T = d["root_p"].shape[0]
        alive = d["alive"][:, c]
        t0 = int(a.start * 50)
        t1 = min(T, t0 + int(a.seconds * 50))
        fell_at = None
        look = None
        for t in range(t0, t1, stride):
            if not alive[t] and fell_at is None:
                fell_at = t
            tt = t if alive[t] else (fell_at if fell_at is not None else t)
            f = int(d["f"][tt, c])
            data.qpos[0:3] = d["root_p"][tt, c]
            data.qpos[3:7] = d["root_quat"][tt, c]
            data.qpos[7:36] = d["q"][tt, c]
            data.qpos[36:39] = d["ref_root_p"][f]
            if not a.global_ghost:  # anchor the reference to the robot's xy, as the tracking reward does
                data.qpos[36:38] = d["root_p"][tt, c][:2]
            data.qpos[39:43] = _mat_to_quat(d["ref_root_R"][f])
            data.qpos[43:72] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)  # the key light tracks the robot's center of mass
            target = np.array([data.qpos[0], data.qpos[1], 0.72])
            look = target if look is None else look + 0.18 * (target - look)  # smooth follow, no jitter
            cam.lookat[:] = look
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render()).convert("RGBA")
            over = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(over)
            m = round(12 * s)
            parts = [(a.title, bold, INK)] + ([(a.subtitle, regular, MUTED)] if a.subtitle else [])
            if a.title or a.subtitle:
                _pill(draw, (m, m), parts)
            clock = f"{clip}   {(f - int(d['clip_start'][c])) * 0.02:4.1f} s"
            _pill(draw, (a.width - m, m), [(clock, small, MUTED)], anchor="ra")
            if fell_at is not None:
                _pill(draw, (m, a.height - m - round(34 * s)), [("terminated", bold, RED)])
            mark = "NDM Gym"
            draw.text((a.width - m - draw.textlength(mark, font=bold), a.height - m - bold.size), mark, font=bold, fill=(28, 30, 34, 110))
            img = Image.alpha_composite(img, over).convert("RGB")
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


def _mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


if __name__ == "__main__":
    main()
