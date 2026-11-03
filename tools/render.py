"""Render recorded evaluation trajectories (python -m ndm_gym.evaluate --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost at the robot's position. A studio look
at little cost: a directional key light that follows the robot and casts shadows, a fill light, a softly reflecting floor with a
fine grid, multisample anti-aliasing, a smoothly following camera and a small NDM Gym mark.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance1_subject1 jumps1_subject1 --seconds 12
``--pair``: a two-robot evaluation (evaluate --paired --record): both robots of each named two-person clip in one view
(red and blue hands), each with its reference as a ghost at the reference's own place, seen from the side of the pair.
"""

import argparse
import ctypes
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


HANDS_RGBA = ("0.80 0.16 0.14 1", "0.13 0.34 0.82 1")  # the first and the second robot's hands


def pair_xml() -> str:
    """Two solid robots (the second's names prefixed b_), their two ghosts (ref_, ref_b_), with two_robot_xml's studio
    look; the key light is fixed above the scene, as two robots have no single center of mass to follow."""
    xml = two_robot_xml()
    xml = xml.replace(re.search(r'\n      <light name="key"[^>]*/>', xml).group(0), "", 1)
    start = xml.index('<body name="pelvis"')
    robot = xml[start : xml.index('<body name="ref_pelvis"')]
    ghost = xml[xml.index('<body name="ref_pelvis"') : xml.index('<light name="fill"')]

    def hands(body: str, rgba: str) -> str:  # the rubber hands' geoms in this robot's color
        return re.sub(r'(<geom [^>]*?)rgba="[^"]*"([^>]*mesh="(?:left|right)_rubber_hand")', r'\1rgba="' + rgba + r'"\2', body)

    second = re.sub(r' name="([^"]+)"', r' name="b_\1"', robot)
    ghost_b = re.sub(r' name="ref_([^"]+)"', r' name="ref_b_\1"', ghost)
    xml = xml.replace(robot, hands(robot, HANDS_RGBA[0]) + hands(second, HANDS_RGBA[1]), 1)
    xml = xml.replace(ghost, ghost + ghost_b, 1)
    key = ('<light name="key" directional="true" pos="0 0 4" dir="-0.3 0.25 -1" castshadow="true" diffuse="0.58 0.57 0.55" '
           'specular="0.2 0.2 0.2"/>')  # fmt: skip
    # the fixed light's shadow map covers 20 m around the scene, so no floor in view lies past its edge (where the shadow
    # lookup speckles the floor), at twice the resolution of the one-robot look
    xml = xml.replace('<light name="fill"', key + '\n    <light name="fill"', 1).replace('shadowclip="2.0"', 'shadowclip="20.0"', 1)
    return xml.replace('shadowsize="8192"', 'shadowsize="16384"', 1)


def render_pairs(a, d, renderer, model, data, cam, fonts, ff):
    """--pair: each named two-person clip (or every one) with both robots, their references as ghosts at the reference's
    own place; the camera looks at the pair's midpoint from its side."""
    bold, regular, small = fonts
    names = list(d["names"])
    pairs = a.clips or sorted({n.rsplit("_p", 1)[0] for n in names})
    stride = int(round(50 / a.fps))
    s = a.height / 432
    for pair in pairs:
        cols = [names.index(f"{pair}_p1"), names.index(f"{pair}_p2")]
        T = d["root_p"].shape[0]
        t0 = int(a.start * 50)
        t1 = min(T, t0 + int(a.seconds * 50))
        fell = [None, None]
        look = None
        d0 = d["root_p"][t0, cols[1], :2] - d["root_p"][t0, cols[0], :2]
        cam.azimuth = float(np.degrees(np.arctan2(d0[1], d0[0]))) + 90.0  # the line between them across the view
        for t in range(t0, t1, stride):
            for k, c in enumerate(cols):
                alive = d["alive"][t, c]
                if not alive and fell[k] is None:
                    fell[k] = t
                tt = t if alive else fell[k]
                f = int(d["f"][tt, c])
                o = 36 * k
                data.qpos[o : o + 3] = d["root_p"][tt, c]
                data.qpos[o + 3 : o + 7] = d["root_quat"][tt, c]
                data.qpos[o + 7 : o + 36] = d["q"][tt, c]
                g = 72 + 36 * k
                data.qpos[g : g + 3] = d["ref_root_p"][f]
                data.qpos[g + 3 : g + 7] = _mat_to_quat(d["ref_root_R"][f])
                data.qpos[g + 7 : g + 36] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            target = np.array([0.5 * (data.qpos[0] + data.qpos[36]), 0.5 * (data.qpos[1] + data.qpos[37]), 0.7])
            look = target if look is None else look + 0.18 * (target - look)
            cam.lookat[:] = look
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render()).convert("RGBA")
            over = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(over)
            m = round(12 * s)
            parts = [(a.title, bold, INK)] + ([(a.subtitle, regular, MUTED)] if a.subtitle else [])
            if a.title or a.subtitle:
                _pill(draw, (m, m), parts)
            f0 = int(d["f"][t if fell[0] is None else fell[0], cols[0]])
            _pill(draw, (a.width - m, m), [(f"{pair}   {(f0 - int(d['clip_start'][cols[0]])) * 0.02:4.1f} s", small, MUTED)], anchor="ra")
            for k, who in enumerate(("red", "blue")):
                if fell[k] is not None:
                    _pill(draw, (m, a.height - m - round(34 * s) * (k + 1) - k * m), [(f"{who} terminated", bold, RED)])
            draw_mark(draw, a.width, a.height, bold)
            img = Image.alpha_composite(img, over).convert("RGB")
            ff.stdin.write(np.asarray(img).tobytes())


EGL_CUDA_DEVICE_NV = 0x323A  # EGL_NV_device_cuda


def check_egl_device():
    """Refuse to render on a GPU other than the one CUDA_VISIBLE_DEVICES names: EGL enumerates every GPU whatever
    CUDA_VISIBLE_DEVICES says, and MuJoCo opens its context on MUJOCO_EGL_DEVICE_ID or on the first device. The EGL
    devices of the visible GPUs are those that have a CUDA id (EGL_NV_device_cuda)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if os.environ.get("MUJOCO_GL") != "egl" or not cvd:
        return
    from mujoco.egl import egl_ext
    from OpenGL import EGL

    proc = EGL.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not proc:
        raise RuntimeError("EGL lacks eglQueryDeviceAttribEXT: cannot tell which EGL device is the GPU CUDA_VISIBLE_DEVICES names")
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(ctypes.cast(proc, ctypes.c_void_p).value)
    visible = []
    for i, dev in enumerate(egl_ext.eglQueryDevicesEXT(max_devices=64)):
        cuda = ctypes.c_ssize_t(-1)
        if query(ctypes.cast(dev, ctypes.c_void_p).value, EGL_CUDA_DEVICE_NV, ctypes.byref(cuda)):
            visible.append(i)
    want = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if not visible or want is None or int(want) not in visible:
        fix = f"set MUJOCO_EGL_DEVICE_ID={visible[0]}" if visible else "no EGL device is a visible GPU"
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={cvd} is EGL device {visible} but MUJOCO_EGL_DEVICE_ID is {want}: EGL ignores "
                           f"CUDA_VISIBLE_DEVICES and would render on another GPU; {fix}")  # fmt: skip


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


def draw_mark(draw, width, height, bold, fill=(28, 30, 34, 110)):
    """The small NDM Gym mark at the bottom right of a width x height frame, in the bold font of the frame's labels
    (_font(0, round(20 * height / 432))); tools/render_musclevae.py and tools/render_hoi.py (in white, on a dark floor)
    draw it too."""
    m = round(12 * (height / 432))
    mark = "NDM Gym"
    draw.text((width - m - draw.textlength(mark, font=bold), height - m - bold.size), mark, font=bold, fill=fill)


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
    ap.add_argument("--pair", action="store_true", help="a two-robot evaluation: --clips names two-person clips (default: every one)")
    a = ap.parse_args()
    check_egl_device()

    d = dict(np.load(a.traj))
    names = list(d["names"])
    model = mujoco.MjModel.from_xml_string(pair_xml() if a.pair else two_robot_xml())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = (3.6 if a.pair else 3.0), 140.0, -12.0
    s = a.height / 432  # label sizes scale with the frame
    bold, regular, small = _font(0, round(20 * s)), _font(1, round(18 * s)), _font(1, round(15 * s))
    clips = a.clips or names
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    if a.pair:
        render_pairs(a, d, renderer, model, data, cam, (bold, regular, small), ff)
        clips = []
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
            draw_mark(draw, a.width, a.height, bold)
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
