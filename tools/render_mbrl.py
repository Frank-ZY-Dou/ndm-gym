"""Render an episode of mbrl-lib's planner (``ndm_gym.wmbench.collect_mbrl``'s ``<out>_executed.npz``) on mbrl-lib's own MJCF model,
in the light look of tools/render.py: the model's geoms and colours under three lights (a key light that casts soft shadows, a fill
light, a rim light, all placed relative to the camera), a light checker floor (below the cart-pole's rail; the pusher's table) and
a gradient sky, 8x multisample anti-aliasing, a fixed camera framed on the episode's moving bodies and their shadows, and the NDM
Gym mark (--no_mark: none). The joints are interpolated linearly between the control steps to --fps (--speed 0.5: half speed).
Written as an MP4 (H.264, CRF 16); a .png output is the frame at --start alone.

Usage: MUJOCO_GL=egl python tools/render_mbrl.py cp_s10_executed.npz pets_cartpole cartpole.mp4 [--size 1280 960] [--fps 20]
On a CPU, Mesa's EGL: __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json CUDA_VISIBLE_DEVICES= (no GPU).

The README's renders, from one episode per task (collect_mbrl --sim ndm on the final fits, planner seeds 10 and 17):
  python tools/render_mbrl.py cp_s10_executed.npz pets_cartpole cartpole.mp4 --seconds 3 --speed 0.5 --elevation -12
  python tools/render_mbrl.py push_s17_executed.npz pets_pusher pusher.mp4 --seconds 3 --speed 0.5 --azimuth 270 --elevation -25
and the teaser's, with --size 1376 512 --no_mark: the cart-pole at --elevation -8, the pusher at --azimuth 270 --elevation -20.
"""

import argparse
import os
import re
import subprocess

from render import _font, check_egl_device, draw_mark  # tools/render.py: the NDM Gym mark; importing it first sets up EGL as it does

import mbrl
import mujoco
import numpy as np
from PIL import Image, ImageDraw

# mbrl-lib's model of each environment, the height of the floor added below it (None: the model's own plane, the pusher's table,
# is the floor), the height where the shadows fall, and the camera's heading and pitch in degrees
SCENES = {
    "pets_cartpole": {"xml": "cartpole.xml", "floor_z": -0.70, "shadow_z": -0.70, "azimuth": 90.0, "elevation": -15.0},
    "pets_pusher": {"xml": "pusher.xml", "floor_z": None, "shadow_z": -0.325, "azimuth": 150.0, "elevation": -38.0},
}
ASSETS = """
    <texture name="ndm_sky" type="skybox" builtin="gradient" rgb1="0.985 0.987 0.99" rgb2="0.86 0.88 0.91" width="512" height="512"/>
    <texture name="ndm_floor" type="2d" builtin="checker" rgb1="0.80 0.81 0.83" rgb2="0.77 0.78 0.80" mark="edge" markrgb="0.66 0.68 0.72"
             width="512" height="512"/>
    <material name="ndm_floor" texture="ndm_floor" texrepeat="2 2" texuniform="true" reflectance="0" shininess="0.2" specular="0.1"/>
"""  # tools/render.py's sky and floor, matte, its squares 25 cm (tools/render.py's 50 cm), as these models are about a metre across
VISUAL = """<visual>
    <global offwidth="2560" offheight="1920"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.30 0.30 0.31" diffuse="0.10 0.10 0.10" specular="0 0 0"/>
    <map shadowclip="1.2" shadowscale="0.55" haze="0.4"/>
    <rgba haze="0.95 0.96 0.97 1"/>
  </visual>
"""
# (name, degrees around the camera's heading (0: from the camera), degrees above the horizon, diffuse, specular, casts a shadow)
LIGHTS = (("key", 35.0, 60.0, 0.56, 0.18, True), ("fill", -60.0, 25.0, 0.24, 0.0, False), ("rim", 180.0, 40.0, 0.30, 0.20, False))


def scene_xml(env: str) -> str:
    """mbrl-lib's model of ``env`` with the look: its own skybox and lights replaced, the floor added or its table's plane made
    the floor, the visual settings above. Its bodies, joints, geoms and colours are unchanged."""
    s = SCENES[env]
    xml = open(os.path.join(os.path.dirname(mbrl.__file__), "env", "assets", s["xml"])).read()
    if "<visual>" in xml:
        raise ValueError(f"render_mbrl: {s['xml']} has <visual> settings of its own; merge them")
    xml, n = re.subn(r'<texture type="skybox"[^>]*/>', "", xml)
    if n != 1:
        raise ValueError(f"render_mbrl: {s['xml']} has {n} skyboxes, not one")
    xml = re.sub(r"<light [^>]*/>", "", xml)
    lights = "".join(f'<light name="{name}" directional="true" pos="0 0 3" dir="0 0 -1" castshadow="{str(shadow).lower()}" '
                     f'diffuse="{d} {d} {d}" specular="{sp} {sp} {sp}"/>' for name, _, _, d, sp, shadow in LIGHTS)  # fmt: skip
    if s["floor_z"] is not None:  # the floor with a colour of its own: the model's default geom colour would tint it
        lights += (f'<geom name="ndm_floor" type="plane" pos="0 0 {s["floor_z"]}" size="0 0 0.05" material="ndm_floor" rgba="1 1 1 1" '
                   'contype="0" conaffinity="0"/>')  # fmt: skip
    else:  # the table: an endless plane in the floor's look, at its height
        table = r'<geom name="table" type="plane"\1 size="0 0 0.05" material="ndm_floor" rgba="1 1 1 1"'
        xml, n = re.subn(r'<geom name="table" type="plane"([^>]*?) size="[^"]*"', table, xml)
        if n != 1:
            raise ValueError(f"render_mbrl: {s['xml']} has no plane named table")
    xml = xml.replace("<worldbody>", "<worldbody>" + lights, 1).replace("<asset>", VISUAL + "<asset>" + ASSETS, 1)
    return xml


def light_from(azimuth: float, around: float, above: float) -> np.ndarray:
    """The unit vector toward a light at ``around`` degrees around the camera's heading (0: from the camera's side) and ``above``
    degrees above the horizon."""
    h, e = np.radians(azimuth + 180.0 + around), np.radians(above)
    return np.array([np.cos(e) * np.cos(h), np.cos(e) * np.sin(h), np.sin(e)])


def frame_camera(model, data, qs, azimuth: float, elevation: float, aspect: float, margin: float, shadow_z: float) -> tuple[np.ndarray, float]:
    """The look-at point and distance of a camera at ``azimuth`` and ``elevation`` whose view holds the visible geoms of every
    body but the world's over the poses ``qs``, each as the cube around its bounding sphere, and their shadows from the key light
    on the plane at ``shadow_z``, with ``margin`` of the view spare."""
    geoms = np.flatnonzero((model.geom_bodyid > 0) & (model.geom_rgba[:, 3] > 0))
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    pts = []
    for q in qs:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        pts.append((data.geom_xpos[geoms][:, None] + model.geom_rbound[geoms][:, None, None] * corners).reshape(-1, 3))
    pts = np.concatenate(pts)
    down = -light_from(azimuth, *LIGHTS[0][1:3])  # the key light's direction
    above = pts[pts[:, 2] > shadow_z]
    pts = np.concatenate([pts, above + down * ((above[:, 2] - shadow_z) / -down[2])[:, None]])
    az, el = np.radians(azimuth), np.radians(elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    centre = (pts.min(0) + pts.max(0)) / 2
    x, y = (pts - centre) @ right, (pts - centre) @ up
    centre = centre + right * (x.min() + x.max()) / 2 + up * (y.min() + y.max()) / 2  # centred in the view
    x, y, z = (pts - centre) @ right, (pts - centre) @ up, (pts - centre) @ fwd
    ty = np.tan(np.radians(model.vis.global_.fovy) / 2) * (1 - margin)
    return centre, float(np.max(np.maximum(np.abs(x) / (ty * aspect), np.abs(y) / ty) - z))


def place_lights(model, lookat: np.ndarray, azimuth: float, extent: float):
    """The lights of LIGHTS around ``lookat``: each at its angle around the camera's heading and above the horizon, pointing at it."""
    for name, around, above, *_ in LIGHTS:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        frm = light_from(azimuth, around, above)
        model.light_pos[i], model.light_dir[i] = lookat + extent * frm, -frm


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("traj", help="collect_mbrl's <out>_executed.npz")
    p.add_argument("env", choices=sorted(SCENES))
    p.add_argument("out", help=".mp4, or .png for the frame at --start")
    p.add_argument("--size", type=int, nargs=2, default=(1280, 960), metavar=("W", "H"))
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--start", type=float, default=0.0, help="seconds into the episode")
    p.add_argument("--seconds", type=float, default=None, help="of the episode; default: to its end")
    p.add_argument("--speed", type=float, default=1.0, help="the playback's speed (0.5: half speed)")
    p.add_argument("--azimuth", type=float, default=None, help="default: the scene's")
    p.add_argument("--elevation", type=float, default=None, help="default: the scene's")
    p.add_argument("--margin", type=float, default=0.06, help="the view's share left around the moving bodies")
    p.add_argument("--no_mark", action="store_true", help="no NDM Gym mark (a source for tools/make_grid.py --mark, which draws one)")
    a = p.parse_args(argv)
    check_egl_device()
    s = SCENES[a.env]
    az = s["azimuth"] if a.azimuth is None else a.azimuth
    el = s["elevation"] if a.elevation is None else a.elevation
    model = mujoco.MjModel.from_xml_string(scene_xml(a.env))
    z = np.load(a.traj)
    q, dt = z["qpos"], float(z["dt"])
    if q.ndim != 3 or q.shape[1] != 1 or q.shape[2] != model.nq:
        raise ValueError(f"render_mbrl: qpos {q.shape} is not one world's trajectory of the model's {model.nq} positions")
    if z["reset"].any():
        raise ValueError("render_mbrl: the recording holds a reset; render one episode")
    q = q[:, 0]
    end = (len(q) - 1) * dt if a.seconds is None else min(a.start + a.seconds, (len(q) - 1) * dt)
    times = a.start + np.arange(int(np.floor((end - a.start) * a.fps / a.speed + 1e-6)) + 1) * a.speed / a.fps
    k = np.minimum((times / dt).astype(int), len(q) - 2)
    w = (times / dt - k)[:, None]
    qs = (1 - w) * q[k] + w * q[k + 1]  # the joints (hinges and slides) between the control steps
    width, height = a.size
    data = mujoco.MjData(model)
    lookat, dist = frame_camera(model, data, qs, az, el, width / height, a.margin, s["shadow_z"])  # on the whole window, also for a .png
    if a.out.endswith(".png"):
        qs = qs[:1]
    place_lights(model, lookat, az, dist)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
    renderer = mujoco.Renderer(model, height, width)
    bold = _font(0, round(20 * height / 432))
    ff = None
    if not a.out.endswith(".png"):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                               "-r", f"{a.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", a.out],
                              stdin=subprocess.PIPE)  # fmt: skip
    for qi in qs:
        data.qpos[:] = qi
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)  # the lights' world positions and directions
        renderer.update_scene(data, cam)
        frame = Image.fromarray(renderer.render())
        if not a.no_mark:
            over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            draw_mark(ImageDraw.Draw(over), width, height, bold)
            frame = Image.alpha_composite(frame.convert("RGBA"), over).convert("RGB")
        if ff is None:
            frame.save(a.out)
        else:
            ff.stdin.write(frame.tobytes())
    if ff is not None:
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"render_mbrl: ffmpeg failed writing {a.out}")
    print(f"[render_mbrl] {a.out}: {len(qs)} frames at {a.fps:g} fps from {a.start:g} s at speed {a.speed:g}, {width} x {height}; "
          f"camera at {lookat.round(3)}, distance {dist:.2f} m, azimuth {az:g}, elevation {el:g}")  # fmt: skip


if __name__ == "__main__":
    main()
