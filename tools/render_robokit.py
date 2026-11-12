"""Render a RoboKit example's executed trajectory (``plan_robokit --save_executed``) in RoboKit's look: the scene on a seamless
white background with soft shadows, a slowly following three-quarter camera, and the NDM Gym mark of tools/render.py; written
as an animated WebP, as RoboKit's README animations are.

The look: no skybox or fog; the background (no geom in the segmentation pass) white; the floor light grey under the robot,
with grid lines every ``--grid`` metres, fading into the white background over ``--floor_fade`` metres from the camera's
target, so there is no horizon; shadows darken it (its pixels times the ratio of a render with shadows to one without); the
scenery fixed to the world (terrain, steps) light grey. The camera looks at the root body (the first free
joint's body), smoothed over time, from ``--azimuth`` and ``--elevation`` at ``--distance``. The world's shadow-casting lights
move horizontally with the camera's target: a light's shadow map covers the model's extent around the light, so a robot that
travels away from the origin (Go2 jumping, 1.6 m) would otherwise lose its shadow.

Usage: MUJOCO_GL=egl python tools/render_robokit.py run.npz out.webp [--fps 25] [--size 960 624] [--distance 3.2]
       (run.npz and run.mjb from plan_robokit --save_executed run.npz)
"""

import argparse
import os

from render import _font, check_egl_device, draw_mark  # tools/render.py: the NDM Gym mark; importing it first sets up EGL as it does

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def look(model: mujoco.MjModel):
    """RoboKit's look on a loaded model, in place: floor planes and the world body's other visible geoms (terrain, steps) light
    grey without texture, a bright headlight without specular. Returns the floor planes' geom ids."""
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if not len(planes):
        raise ValueError("render_robokit: the model has no floor plane")
    scenery = np.flatnonzero((model.geom_bodyid == 0) & (model.geom_rgba[:, 3] > 0))
    model.geom_matid[scenery] = -1
    model.geom_rgba[scenery] = (0.86, 0.86, 0.87, 1.0)
    model.geom_rgba[planes] = (0.96, 0.96, 0.96, 1.0)
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.35
    model.vis.headlight.specular[:] = 0.0
    return planes


def floor_grey(cam, fovy: float, size: tuple, origin, rot, target, grid: float, fade: float) -> np.ndarray:
    """(height, width) the floor's grey (1: white) at each pixel's ray through a floor plane at ``origin`` with axes ``rot`` (3 x 3,
    columns: its x, y and normal): 0.9 under the camera's target fading to 1 over ``fade`` metres, grid lines every ``grid``
    metres along the plane's axes (anti-aliased, and faded out where a pixel spans a large part of a grid cell)."""
    width, height = size
    pos, fwd, up = (np.asarray(x, dtype=np.float64) for x in (cam.pos, cam.forward, cam.up))
    right = np.cross(fwd, up)
    ty = np.tan(np.radians(fovy) / 2)
    u = ((np.arange(width) + 0.5) / width * 2 - 1) * ty * width / height
    v = (1 - (np.arange(height) + 0.5) / height * 2) * ty
    ray = fwd + u[None, :, None] * right + v[:, None, None] * up  # (height, width, 3)
    n = rot[:, 2]
    dn = ray @ n
    hits = dn < -1e-6  # rays that meet the plane in front of the camera
    t = np.where(hits, ((origin - pos) @ n) / np.minimum(dn, -1e-6), 0.0)  # distance along the ray to the plane
    hit = (pos + t[..., None] * ray - origin) @ rot[:, :2]  # the plane's coordinates of the hit point (height, width, 2)
    r = np.linalg.norm(hit - (target - origin) @ rot[:, :2], axis=-1)
    w = np.exp(-3.0 * (r / fade) ** 2)  # 1 under the target, 0.05 at fade metres
    grey = 1.0 - 0.10 * w
    if grid > 0:
        foot = np.maximum(t * 2 * ty / height / np.maximum(-dn / np.linalg.norm(ray, axis=-1), 0.05), 1e-9)  # a pixel on the floor, m
        off = np.abs(hit / grid - np.round(hit / grid)).min(-1) * grid  # to the nearest grid line, metres
        line = np.clip(1.0 - off / (0.75 * foot), 0.0, 1.0) * np.clip(1.0 - foot / (0.15 * grid), 0.0, 1.0)
        grey = grey - 0.12 * w * line
    return np.where(hits, grey, 1.0)


def root_body(model: mujoco.MjModel) -> int:
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if not len(free):
        raise ValueError("render_robokit: no free joint to follow")
    return int(model.jnt_bodyid[free[0]])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("traj", help=".npz of plan_robokit --save_executed (its .mjb beside it)")
    p.add_argument("out", help=".webp")
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--size", type=int, nargs=2, default=(960, 624), help="width height (twice RoboKit's 480 x 312)")
    p.add_argument("--distance", type=float, default=3.2)
    p.add_argument("--azimuth", type=float, default=135.0)
    p.add_argument("--elevation", type=float, default=-16.0)
    p.add_argument("--lookat_z", type=float, default=None, help="a fixed height of the look-at point (default: the root's)")
    p.add_argument("--smooth", type=float, default=0.08, help="the camera's share of the root's motion per frame")
    p.add_argument("--quality", type=int, default=82)
    p.add_argument("--frames", type=int, default=None, help="only the first this many frames (a sample)")
    p.add_argument("--grid", type=float, default=0.5, help="the floor's grid spacing, metres (0: no grid)")
    p.add_argument("--floor_fade", type=float, default=None, help="metres from the camera's target over which the floor fades to white (default: 2.5 x --distance)")
    a = p.parse_args(argv)

    check_egl_device()
    model = mujoco.MjModel.from_binary_path(os.path.splitext(a.traj)[0] + ".mjb")
    z = np.load(a.traj)
    t, q = z[f"time_{a.instance}"].reshape(-1), z[f"qpos_{a.instance}"]
    if q.shape[1] != model.nq:
        raise ValueError(f"render_robokit: qpos has {q.shape[1]} columns, the model {model.nq}")
    planes = look(model)
    body = root_body(model)
    lights = np.flatnonzero((model.light_bodyid == 0) & model.light_castshadow.astype(bool))  # the world's, casting shadows
    light_xy = model.light_pos[lights, :2].copy()
    data = mujoco.MjData(model)
    width, height = a.size
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)  # the offscreen buffer must hold a frame
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    for flag, on in ((mujoco.mjtRndFlag.mjRND_SKYBOX, 0), (mujoco.mjtRndFlag.mjRND_FOG, 0), (mujoco.mjtRndFlag.mjRND_SHADOW, 1),
                     (mujoco.mjtRndFlag.mjRND_REFLECTION, 0)):  # fmt: skip
        renderer.scene.flags[flag] = on
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = a.distance, a.azimuth, a.elevation
    times = np.arange(t[0], t[-1], 1.0 / a.fps)
    idx = np.minimum(np.searchsorted(t, times), len(t) - 1)
    if a.frames:
        idx = idx[: a.frames]
    bold = _font(0, round(20 * height / 432))
    frames, lookat, start = [], None, None
    fade = a.floor_fade if a.floor_fade is not None else 2.5 * a.distance
    for i in idx:
        data.qpos[:] = q[i]
        mujoco.mj_forward(model, data)
        target = data.xpos[body].copy()
        if a.lookat_z is not None:
            target[2] = a.lookat_z
        lookat = target if lookat is None else lookat + a.smooth * (target - lookat)
        start = lookat.copy() if start is None else start
        cam.lookat[:] = lookat
        if len(lights):  # the lights follow the target horizontally (see the module's docstring)
            model.light_pos[lights, :2] = light_xy + (lookat - start)[:2]
            mujoco.mj_camlight(model, data)
        renderer.update_scene(data, cam, opt)
        flags = renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX], flags[mujoco.mjtRndFlag.mjRND_FOG] = 0, 0
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        rgb = renderer.render().astype(np.float32)
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        lit = renderer.render().astype(np.float32)
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        shade = np.clip(rgb.mean(-1) / np.maximum(lit.mean(-1), 1.0), 0.0, 1.0)  # 1 lit, below 1 in a shadow
        for g in np.intersect1d(planes, geom):  # each visible floor plane: its grey, darkened by a softened shadow
            px = geom == g
            grey = floor_grey(renderer.scene.camera[0], model.vis.global_.fovy, (width, height), data.geom_xpos[g],
                              data.geom_xmat[g].reshape(3, 3), lookat, a.grid, fade)  # fmt: skip
            rgb[px] = 255.0 * (grey[px] * (1.0 - 0.6 * (1.0 - shade[px])))[:, None]
        rgb[seg[..., 0] < 0] = 255.0  # no geom: the white background
        rgb = rgb.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_mark(ImageDraw.Draw(over), width, height, bold)
        frames.append(Image.alpha_composite(img, over).convert("RGB"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=round(1000 / a.fps), loop=0, quality=a.quality, method=6)
    print(f"[render_robokit] {a.out}: {len(frames)} frames, {width} x {height}, {os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
