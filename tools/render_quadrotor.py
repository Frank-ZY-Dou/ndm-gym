"""Render a quadrotor rollout (ndm_gym.evaluate_quadrotor's rollout_<flight>.npz): the simulated drone solid, the flown
reference a translucent blue drone with its path drawn through the arena, the flight's racing gates where the motion capture
placed them, the simulated drone's trail of the last moments in orange, the NDM Gym mark (``--no_mark``: none).

    MUJOCO_GL=egl python tools/render_quadrotor.py --rollout logs/<name>/evaluation/rollout_<flight>.npz --out flight.mp4

``--source reference_command`` renders the same start flown under the reference's feedforward command, open loop, over the
same reference. ``--view chase`` (the default): the camera follows the reference from behind its direction of flight;
``follow``: from a fixed side; ``arena``: the whole path in view. The drones are drawn ``--scale`` times their size (1.5 by
default, 4 in the arena), with a dark disc under the simulated one for its shadow. ``--start``
and ``--seconds`` cut a window of the flight; ``--speed`` is the playback speed (1: real time); ``--path_seconds`` draws the
reference's path only that many seconds before and after the window (a pilot's dozen laps drawn whole are a tangle; the
map shows the whole flight either way). ``--hud`` draws over the
frame the simulated drone's speed, its distance from the reference and a map of the path from above with the two drones on
it (``--title``: a title too). ``--look sky`` (the default): tools/render_musclevae.py's sky and checker floor, with its
lights; ``--look noir``: a dark stage, as tools/render_hand.py's, with the path and the gates glowing; ``--look studio``:
tools/render.py's light studio. ``--preview``: also an animated WebP in the format of the
README's hand-object animations (960 pixels wide, 30 frames per second, at most ``--preview_mb``). Needs ffmpeg (or the
imageio-ffmpeg package's).
"""

import argparse
import os
import shutil
import subprocess
import sys

from render import _font, check_egl_device, draw_mark  # tools/render.py: the NDM Gym mark; importing it first sets up EGL as it does

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ndm_gym.core.rotations import mat_to_quat
from ndm_gym.motion.sources.drone_racing import GATE_FRAME, GATE_OPENING
from ndm_gym.robots.quadrotor import QuadrotorSpec

LOOKS = ("sky", "noir", "studio")
BLUE, ORANGE = (0.25, 0.62, 1.0), (1.0, 0.52, 0.12)
TRAIL = 0.7  # s of the simulated drone's path drawn behind it
FLOOR_CELL = 1.0  # the side of a floor tile (m)


def propeller_mesh(diameter: float = 0.1295, blades: int = 3, chord: float = 0.017, twist: tuple = (0.45, 0.12), n_r: int = 24, n_c: int = 10):
    """A propeller as vertex and face arrays: ``blades`` cambered plates swept along the radius with a chord that widens then
    tapers to the tip and a pitch that twists from ``twist[0]`` at the root to ``twist[1]`` at the tip, on a hub. The dataset's
    drone flies 5.1-inch tri-blade propellers (T-Motor T5147), which these follow in size; the shape is this module's own."""
    r0, r1 = 0.008, diameter / 2
    verts, faces = [], []
    for b in range(blades):
        base, phi = len(verts), 2 * np.pi * b / blades
        for i, r in enumerate(np.linspace(r0, r1, n_r)):
            t = (r - r0) / (r1 - r0)
            c = chord * (0.55 + 0.9 * np.sin(np.pi * min(1.0, 0.15 + t)) ** 0.7) * (1 - 0.6 * max(0.0, t - 0.85) / 0.15)
            tw = twist[0] * (1 - t) + twist[1] * t
            for u in np.linspace(-0.5, 0.5, n_c):
                y, z = u * c, 0.12 * c * (1 - (2 * u) ** 2)  # a cambered plate, turned by the pitch
                yw, zw = y * np.cos(tw) - z * np.sin(tw), y * np.sin(tw) + z * np.cos(tw)
                verts.append((r * np.cos(phi) - yw * np.sin(phi), r * np.sin(phi) + yw * np.cos(phi), zw))
        for i in range(n_r - 1):
            for j in range(n_c - 1):
                a = base + i * n_c + j
                faces += [(a, a + 1, a + n_c), (a + 1, a + n_c + 1, a + n_c), (a, a + n_c, a + 1), (a + 1, a + n_c, a + n_c + 1)]  # both sides
    hub, n_h = len(verts), 16
    for k in range(n_h):  # the hub: a short cylinder
        a = 2 * np.pi * k / n_h
        verts += [(0.009 * np.cos(a), 0.009 * np.sin(a), -0.003), (0.009 * np.cos(a), 0.009 * np.sin(a), 0.003)]
    verts += [(0, 0, -0.003), (0, 0, 0.003)]
    for k in range(n_h):
        a, b = hub + 2 * k, hub + 2 * ((k + 1) % n_h)
        faces += [(a, b, a + 1), (b, b + 1, a + 1), (hub + 2 * n_h, b, a), (hub + 2 * n_h + 1, a + 1, b + 1)]
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def add_drone(spec: mujoco.MjSpec, name: str, scale: float, ghost: bool, noir: bool):
    """A free body ``name`` with the drone's shapes, after the dataset's 5-inch racer (the specification's size): a carbon
    H-frame of two plates on standoffs with four flat arms, the battery strapped on top, the camera in a tilted mount at the
    front, the antenna at the back, four motors, and on each a tri-blade propeller (``propeller_mesh``) on a hinge
    (``<name>_prop<k>``, turned by the frame loop) in a faint disc of its sweep, the front pair's in orange. ``ghost``: all of
    it in translucent blue."""
    s = scale
    arm = QuadrotorSpec().diagonal / 2 * 0.7071 * s
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_root")
    body.mass, body.inertia = 1.0, (1e-3, 1e-3, 1e-3)

    def geom(rgba, parent=body, alpha=1.0, **kw):
        g = parent.add_geom(contype=0, conaffinity=0, **kw)
        g.rgba = (*BLUE, 0.30 * alpha) if noir and ghost else (0.20, 0.42, 0.92, 0.45 * alpha) if ghost else rgba
        if ghost:
            g.material = "ghost"
        elif noir:
            g.material = "satin"

    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    carbon, dark, alloy = (0.11, 0.11, 0.13, 1.0), (0.20, 0.21, 0.24, 1.0), (0.62, 0.64, 0.68, 1.0)
    # the frame: a bottom plate that the arms bolt to, a top plate on standoffs, the stack between them
    geom(carbon, type=box, size=(0.048 * s, 0.020 * s, 0.002 * s), pos=(0, 0, -0.012 * s))
    geom(carbon, type=box, size=(0.058 * s, 0.018 * s, 0.002 * s), pos=(0.004 * s, 0, 0.016 * s))
    for x in (0.036, -0.030):
        for y in (0.014, -0.014):
            geom(alloy, type=cyl, fromto=(x * s, y * s, -0.010 * s, x * s, y * s, 0.014 * s), size=(0.0025 * s, 0, 0))
    geom((0.08, 0.30, 0.16, 1.0), type=box, size=(0.016 * s, 0.016 * s, 0.009 * s), pos=(-0.002 * s, 0, 0.0))  # the flight controller stack
    # the battery on the top plate, with its strap
    geom((0.93, 0.74, 0.16, 1.0), type=box, size=(0.036 * s, 0.017 * s, 0.015 * s), pos=(-0.006 * s, 0, 0.033 * s))
    geom(dark, type=box, size=(0.006 * s, 0.0185 * s, 0.0165 * s), pos=(-0.006 * s, 0, 0.033 * s))
    # the camera in its tilted mount at the front, the antenna at the back
    geom(dark, type=box, size=(0.010 * s, 0.013 * s, 0.011 * s), pos=(0.060 * s, 0, 0.012 * s), euler=(0, -0.5, 0))
    geom((0.05, 0.05, 0.08, 1.0), type=cyl, size=(0.007 * s, 0.003 * s, 0), pos=(0.070 * s, 0, 0.017 * s), euler=(0, 1.07, 0))  # the lens
    geom((0.86, 0.90, 0.94, 1.0), type=cyl, fromto=(-0.060 * s, 0, 0.016 * s, -0.085 * s, 0, 0.050 * s), size=(0.0025 * s, 0, 0))
    # the arms: flat carbon bars to the motors, and the motors
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        x, y = sx * arm, sy * arm
        geom(carbon, type=box, size=(0.5 * np.hypot(x, y) + 0.006 * s, 0.0085 * s, 0.0025 * s), pos=(x / 2, y / 2, -0.012 * s),
             euler=(0, 0, np.arctan2(y, x)))  # fmt: skip
        geom(dark, type=cyl, pos=(x, y, -0.003 * s), size=(0.011 * s, 0.0065 * s, 0))  # the motor's bell
        geom(alloy, type=cyl, pos=(x, y, 0.006 * s), size=(0.0035 * s, 0.003 * s, 0))  # its shaft
        blade, disc = ((*ORANGE, 1.0), (*ORANGE, 0.16)) if sx > 0 else ((0.16, 0.17, 0.20, 1.0), (0.80, 0.84, 0.90, 0.14))
        geom(disc, alpha=0.4, type=cyl, pos=(x, y, 0.010 * s), size=(0.0648 * s, 0.0008 * s, 0))  # the sweep
        prop = body.add_body(name=f"{name}_prop{k}", pos=(x, y, 0.010 * s), quat=(1, 0, 0, 0) if sx * sy > 0 else (0, 1, 0, 0))  # mirrored pairs
        prop.add_joint(name=f"{name}_prop{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1))
        prop.mass, prop.inertia = 1e-3, (1e-6, 1e-6, 1e-6)
        geom(blade, parent=prop, type=mujoco.mjtGeom.mjGEOM_MESH, meshname="propeller")


def scene_model(path: np.ndarray, gates: np.ndarray, width: int, height: int, scale: float, look: str, extent: float, bold: bool = False) -> mujoco.MjModel:
    """The two drones (the simulated one, then ``ref``) over the floor, the reference's ``path`` (n, 3) as a line of capsules
    and the ``gates`` (g, 12: centre, rotation matrix) as square frames of the dataset's size. ``extent``: the size of the
    scene in view (m), which the shadow map covers. ``bold``: the path and the gates drawn to be seen from afar (the arena's view)."""
    if look not in LOOKS:
        raise ValueError(f"look must be one of {LOOKS}, got {look!r}")
    noir, sky_look = look == "noir", look == "sky"
    if sky_look:
        import render_musclevae as mv  # tools/render_musclevae.py: its sky, its floor's colors and its lights' levels

    spec = mujoco.MjSpec()
    satin = spec.add_material(name="satin")
    satin.specular, satin.shininess = 0.55, 0.75
    ghost = spec.add_material(name="ghost")
    ghost.rgba, ghost.emission = (*BLUE, 0.3), 0.6 if noir else 0.0
    path_rgb = BLUE if noir else (0.20, 0.42, 0.92)  # (a darker blue on the light floors)
    glow = spec.add_material(name="path")
    glow.rgba, glow.emission = (*path_rgb, 0.95 if bold else 0.55 if noir else 0.75), (1.0 if bold else 0.55) if noir else 0.0
    frame = spec.add_material(name="gate")
    frame.rgba, frame.emission, frame.specular = ((*ORANGE, 1.0), 0.22, 0.4) if noir else ((0.93, 0.47, 0.13, 1.0), 0.0, 0.2)
    mesh = spec.add_mesh(name="propeller")
    verts, faces = propeller_mesh()
    mesh.uservert, mesh.userface, mesh.scale = (verts * scale).flatten(), faces.flatten(), (1.0, 1.0, 1.0)
    add_drone(spec, "drone", scale, ghost=False, noir=noir)
    add_drone(spec, "ref", scale, ghost=True, noir=noir)

    sky = ((0.115, 0.125, 0.15), (0.012, 0.013, 0.017)) if noir else ((0.985, 0.987, 0.99), (0.86, 0.88, 0.91))
    if sky_look:  # a flat texture of the cube's size, filled with their sky after compiling
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=(1, 1, 1),
                         width=mv.SKY_SIZE, height=mv.SKY_SIZE)  # fmt: skip
    else:
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=sky[0],
                         rgb2=sky[1], width=512, height=512)  # fmt: skip
    if sky_look:
        checker = dict(rgb1=tuple(c / 255 for c in mv.FLOOR_RGB[0]), rgb2=tuple(c / 255 for c in mv.FLOOR_RGB[1]))
    elif noir:
        checker = dict(rgb1=(0.078, 0.081, 0.092), rgb2=(0.066, 0.069, 0.079), mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=(0.17, 0.18, 0.21))
    else:
        checker = dict(rgb1=(0.80, 0.81, 0.83), rgb2=(0.77, 0.78, 0.80), mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=(0.66, 0.68, 0.72))
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512, **checker)
    floor = spec.add_material(name="floor")
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    floor.texrepeat, floor.texuniform = ((2, 2) if sky_look else (1 / FLOOR_CELL, 1 / FLOOR_CELL)), True  # (sky: their floor's repeat)
    floor.reflectance, floor.shininess, floor.specular = (0.30, 0.85, 0.45) if noir else (0.0, 0.0, 0.0) if sky_look else (0.10, 0.2, 0.1)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), material="floor", contype=0, conaffinity=0)

    radius = 0.022 if bold else 0.004 * max(scale, 1.0)
    for a, b in zip(path[:-1], path[1:], strict=True):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=(*a, *b), size=(radius, 0, 0), material="path", rgba=glow.rgba,
                                contype=0, conaffinity=0)  # fmt: skip
    half, bar = GATE_OPENING / 2, 0.05 if bold else 0.03  # the opening's edge, drawn as a frame of bars (the gates' panels around it are left out)
    for i, g in enumerate(gates):  # the opening across the gate's x axis: the frame in its y-z plane
        body = spec.worldbody.add_body(name=f"gate{i + 1}", pos=g[:3], quat=mat_to_quat(torch.as_tensor(g[3:].reshape(3, 3))).numpy())
        for a, b in (((-half, half), (half, half)), ((-half, -half), (half, -half)), ((half, -half), (half, half)), ((-half, -half), (-half, half))):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=(0, a[0], a[1], 0, b[0], b[1]), size=(bar, 0, 0), material="gate",
                          rgba=frame.rgba, contype=0, conaffinity=0)  # fmt: skip
        for y in (-half, half):  # the legs, to the floor
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=(0, y, -half, 0, y, -float(g[2])), size=(bar * 0.7, 0, 0),
                          rgba=(0.30, 0.31, 0.34, 1.0), contype=0, conaffinity=0)  # fmt: skip

    light = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    if noir:  # a warm key light, a cool rim light from behind the drone (turned with the camera), a faint fill
        spec.worldbody.add_light(name="key", type=light, pos=(0.8, -0.6, 9), dir=(-0.35, 0.3, -1), castshadow=True,
                                 diffuse=(1.00, 0.95, 0.88), specular=(0.60, 0.57, 0.52))  # fmt: skip
        spec.worldbody.add_light(name="rim", type=light, pos=(0, 0, 9), dir=(1, 0, -0.35), castshadow=False,
                                 diffuse=(0.36, 0.50, 0.74), specular=(0.65, 0.80, 1.00))  # fmt: skip
        spec.worldbody.add_light(name="fill", type=light, pos=(0, 0, 9), dir=(0.5, 0.6, -0.6), castshadow=False,
                                 diffuse=(0.20, 0.21, 0.24), specular=(0, 0, 0))  # fmt: skip
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = (0.20, 0.20, 0.21), (0.16,) * 3, (0.05,) * 3
    elif sky_look:  # their warm key light from 77 degrees, their sky's light from above, from the horizon and from the ground, an ambient
        e = np.radians(mv.KEY_ELEVATION)
        spec.worldbody.add_light(name="key", type=light, pos=(0, 0, 9), dir=(np.cos(e), 0.3 * np.cos(e), -np.sin(e)), castshadow=True,
                                 diffuse=mv.KEY, specular=(0.3, 0.3, 0.3))  # fmt: skip
        spec.worldbody.add_light(name="zenith", type=light, pos=(0, 0, 9), dir=(0, 0, -1), castshadow=False, diffuse=mv.ZENITH, specular=(0, 0, 0))
        spec.worldbody.add_light(name="ground", type=light, pos=(0, 0, 9), dir=(0, 0, 1), castshadow=False, diffuse=mv.GROUND, specular=(0, 0, 0))
        spec.worldbody.add_light(name="rim", type=light, pos=(0, 0, 9), dir=(1, 0, 0), castshadow=False, diffuse=mv.HORIZON, specular=(0, 0, 0))
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = mv.AMBIENT, (0, 0, 0), (0, 0, 0)
    else:
        spec.worldbody.add_light(name="key", type=light, pos=(0.8, -0.6, 9), dir=(-0.3, 0.25, -1), castshadow=True,
                                 diffuse=(0.58, 0.57, 0.55), specular=(0.2, 0.2, 0.2))  # fmt: skip
        spec.worldbody.add_light(name="fill", type=light, pos=(0, 0, 9), dir=(0.5, 0.6, -0.6), castshadow=False,
                                 diffuse=(0.30, 0.31, 0.33), specular=(0, 0, 0))  # fmt: skip
        spec.visual.headlight.ambient, spec.visual.headlight.diffuse, spec.visual.headlight.specular = (0.34, 0.34, 0.35), (0.16,) * 3, (0.05,) * 3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.quality.shadowsize, spec.visual.quality.offsamples = 8192, 8
    model = spec.compile()
    model.stat.extent = extent  # the unit of the clipping planes, the shadow's range and the haze
    # the depth buffer's precision over the floor in view: a near plane closer than this speckles the far floor's shadow test
    model.vis.map.znear, model.vis.map.zfar = 0.15 / extent, 60.0 / extent
    model.vis.map.shadowclip, model.vis.map.shadowscale = 1.5, 0.6
    model.vis.map.haze = 0.9 if noir else 0.15 if sky_look else 0.4
    model.vis.rgba.haze[:] = (0.016, 0.017, 0.021, 1.0) if noir else (0.95, 0.96, 0.97, 1.0)
    if sky_look:
        cube, tex = mv.sky_cube(), model.texture("sky")
        model.tex_data[int(tex.adr[0]) : int(tex.adr[0]) + cube.size] = cube.ravel()
        model.vis.rgba.haze[:] = (*(c / 255 for c in mv.SKY[0][1]), 1.0)  # their horizon's color
    return model


def interpolate(pose: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Poses (T, 7: position, quaternion) at fractional steps ``x``: linear, the quaternion normalized (short arc)."""
    i0 = np.minimum(np.floor(x).astype(int), len(pose) - 2)
    t = (x - i0)[:, None]
    a, b = pose[i0], pose[i0 + 1].copy()
    b[:, 3:] *= np.where((a[:, 3:] * b[:, 3:]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    out = a + (b - a) * t
    out[:, 3:] /= np.linalg.norm(out[:, 3:], axis=-1, keepdims=True)
    return out


def add_trail(scene: mujoco.MjvScene, points: np.ndarray, width: float):
    """``points`` (n, 3), oldest first, as capsules that fade toward the oldest, added to the scene of the frame."""
    n = len(points) - 1
    for i in range(n):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(points[i + 1] - points[i]) < 1e-6:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array((*ORANGE, 0.85 * (i + 1) / n), dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, points[i], points[i + 1])
        g.emission = 0.8
        scene.ngeom += 1


def add_blob_shadow(scene: mujoco.MjvScene, p: np.ndarray, radius: float, alpha: float):
    """A dark disc on the floor under ``p``, fainter and wider with its height: the drone's shadow (a shadow map does not
    resolve a drone in an arena)."""
    if scene.ngeom >= scene.maxgeom:
        return
    h = max(float(p[2]), 0.0)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array((radius * (1 + 0.15 * h), 0.0008, 0.0)), np.array((p[0], p[1], 0.002)),
                        np.eye(3).flatten(), np.array((0.0, 0.0, 0.0, alpha / (1 + 0.35 * h)), dtype=np.float32))  # fmt: skip
    scene.ngeom += 1


class Hud:
    """The frame's overlay: a title, the simulated drone's speed and its distance from the reference, a map of the path from
    above (the gates as ticks, the part flown so far brighter, the reference as a ring and the simulated drone as a dot)."""

    def __init__(self, width: int, height: int, path: np.ndarray, gates: np.ndarray, noir: bool, title: str | None):
        self.w, self.h, self.noir, self.title = width, height, noir, title
        u = height / 432
        self.u, self.bold, self.big, self.small = u, _font(0, round(20 * u)), _font(0, round(30 * u)), _font(1, round(13 * u))
        self.ink = (236, 238, 242) if noir else (28, 30, 34)
        self.muted = (150, 156, 168) if noir else (96, 100, 110)
        # the map: the path's horizontal extent fitted into a box at the top right
        box_w = round(150 * u)
        lo, hi = path[:, :2].min(0) - 1.0, path[:, :2].max(0) + 1.0
        k = box_w / (hi[0] - lo[0])
        box_h, m = round(k * (hi[1] - lo[1])), round(12 * u)
        self.box = (width - m - box_w, m, width - m, m + box_h)
        self.to_map = lambda p: (self.box[0] + k * (p[..., 0] - lo[0]), self.box[3] - k * (p[..., 1] - lo[1]))  # noqa: E731
        self.path, self.gates, self.k = path, gates, k

    def draw(self, image: np.ndarray, i_path: int, p: np.ndarray, p_ref: np.ndarray, speed: float, error: float, mark: bool) -> np.ndarray:
        over = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d, u = ImageDraw.Draw(over), self.u
        panel = (10, 12, 16, 150) if self.noir else (255, 255, 255, 190)
        d.rounded_rectangle(self.box, radius=round(6 * u), fill=panel)
        xs, ys = self.to_map(self.path)
        line = list(zip(xs.tolist(), ys.tolist(), strict=True))
        d.line(line, fill=(90, 150, 235, 110), width=max(1, round(1.2 * u)))
        if i_path > 0:
            d.line(line[: i_path + 1], fill=(90, 170, 255, 255), width=max(1, round(1.8 * u)))
        for g in self.gates:  # a tick across the opening: the gate's y axis
            c, y = g[:3], g[3:].reshape(3, 3)[:, 1]
            a, b = self.to_map(c - y * GATE_FRAME / 2), self.to_map(c + y * GATE_FRAME / 2)
            d.line((float(a[0]), float(a[1]), float(b[0]), float(b[1])), fill=(255, 140, 40, 255), width=max(2, round(2.2 * u)))
        (rx, ry), (px, py), r = self.to_map(p_ref), self.to_map(p), 3.2 * u
        d.ellipse((rx - 1.6 * r, ry - 1.6 * r, rx + 1.6 * r, ry + 1.6 * r), outline=(120, 180, 255, 255), width=max(1, round(1.2 * u)))
        d.ellipse((px - r, py - r, px + r, py + r), fill=(255, 150, 50, 255))
        m = round(12 * u)
        if self.title:
            d.text((m, m), self.title, font=self.bold, fill=(*self.ink, 235))
        y0 = self.h - m - self.big.size - self.small.size - round(4 * u)
        for x0, value, unit, label in ((m, f"{speed:.1f}", "m/s", "speed"), (m + round(118 * u), _length(error)[0], _length(error)[1], "from the reference")):
            d.text((x0, y0), value, font=self.big, fill=(*self.ink, 240))
            d.text((x0 + d.textlength(value, font=self.big) + round(5 * u), y0 + self.big.size - self.small.size - round(3 * u)), unit, font=self.small,
                   fill=(*self.muted, 240))  # fmt: skip
            d.text((x0, y0 + self.big.size + round(4 * u)), label, font=self.small, fill=(*self.muted, 240))
        if mark:
            draw_mark(d, self.w, self.h, self.bold, fill=(236, 238, 242, 120) if self.noir else (28, 30, 34, 110))
        return np.asarray(Image.alpha_composite(Image.fromarray(image).convert("RGBA"), over).convert("RGB"))


def _length(metres: float) -> tuple[str, str]:
    """A distance as a number and its unit: millimetres below 0.1 m, then centimetres, then metres."""
    if metres < 0.1:
        return f"{1000 * metres:.1f}", "mm"
    return (f"{100 * metres:.0f}", "cm") if metres < 1.0 else (f"{metres:.1f}", "m")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollout", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--source", type=str, default="policy", choices=["policy", "reference_command"])
    parser.add_argument("--view", type=str, default="chase", choices=["chase", "follow", "arena"])
    parser.add_argument("--look", type=str, default="sky", choices=LOOKS)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1: real time)")
    parser.add_argument("--start", type=float, default=0.0, help="the window's start in the flight (s)")
    parser.add_argument("--seconds", type=float, default=None, help="the window's length (default: to the flight's end)")
    parser.add_argument("--path_seconds", type=float, default=None, help="the path drawn this many seconds before and after the window only (default: whole)")
    parser.add_argument("--azimuth", type=float, default=110.0, help="follow and arena: the camera's; chase: its offset from straight behind")
    parser.add_argument("--elevation", type=float, default=None, help="the camera's (degrees; default: -14 chasing, -20 following, -34 for the arena)")
    parser.add_argument("--distance", type=float, default=None, help="the camera's distance (m; default: 2, the path's extent for the arena)")
    parser.add_argument("--scale", type=float, default=None, help="the drones drawn this many times their size (default: 1.5, 4 for the arena)")
    parser.add_argument("--hud", action="store_true", help="the simulated drone's speed and distance from the reference and a map of the path, over the frame")
    parser.add_argument("--title", type=str, default=None, help="with --hud: a title at the top left")
    parser.add_argument("--no_mark", action="store_true", help="frames without the NDM Gym mark")
    parser.add_argument("--preview", type=str, default=None, help="also an animated WebP preview at this path, 960 pixels wide at 30 fps")
    parser.add_argument("--preview_mb", type=float, default=1.5, help="the preview's largest size (its quality steps down from 50 to fit)")
    args = parser.parse_args()
    check_egl_device()

    rollout = np.load(args.rollout)
    dt = float(rollout["control_dt"])
    state = rollout["state" if args.source == "policy" else "reference_command_state"].astype(np.float64)
    ref = np.concatenate((rollout["reference_p"], mat_to_quat(torch.as_tensor(rollout["reference_R"])).numpy()), -1).astype(np.float64)
    gates = rollout["gates"] if "gates" in rollout else np.zeros((0, 12))
    first = round(args.start / dt)
    last = len(state) - 1 if args.seconds is None else min(len(state) - 1, first + round(args.seconds / dt))
    steps = np.arange(first, last + 1e-9, args.speed / (args.fps * dt))  # control steps per video frame
    solid, ghost = interpolate(state[:, :7], steps), interpolate(ref, steps)
    # the propellers' angles: a nominal spin of 40 degrees per frame at 30 fps (a racer's thousands of turns a second would
    # strobe at any frame rate; a third of a turn is a tri-blade's own symmetry and would show it standing still), the pairs
    # turning opposite ways as a quadrotor's do
    spin = np.radians(40.0 / args.speed * 30.0 / args.fps) * np.arange(len(steps))[:, None] * np.array([1.0, -1.0, -1.0, 1.0])
    speeds = np.linalg.norm(state[:, 7:10], axis=-1)[np.round(steps).astype(int)]
    stride = max(1, round(0.02 / dt))
    path = ref[::stride, :3]  # the whole flight's path (the map's)
    drawn = path if args.path_seconds is None else ref[max(0, first - round(args.path_seconds / dt)) : last + round(args.path_seconds / dt) + 1 : stride, :3]
    arena, noir = args.view == "arena", args.look == "noir"
    lo, hi = path.min(0), path.max(0)
    distance = args.distance or (0.62 * float(np.linalg.norm(hi - lo)) + 2.0 if arena else 2.0)
    scale = args.scale or (4.0 if arena else 1.5)
    model = scene_model(drawn, gates, args.width, args.height, scale, args.look, extent=max(2.5, distance), bold=arena)
    addr = {n: int(model.joint(n).qposadr[0]) for n in [f"{d}_{j}" for d in ("drone", "ref") for j in ("root", *(f"prop{k}" for k in range(4)))]}

    camera = mujoco.MjvCamera()
    camera.elevation, camera.distance = args.elevation if args.elevation is not None else {"chase": -14.0, "follow": -20.0, "arena": -34.0}[args.view], distance
    # chase: behind the reference's direction of flight over the floor, smoothed with a half-life of 0.12 s (a hover keeps the last)
    vel = rollout["reference_v"][np.round(steps).astype(int)] if "reference_v" in rollout else np.gradient(ghost[:, :3], axis=0)
    heading, headings = None, []
    for v in vel:
        if np.linalg.norm(v[:2]) > 0.5:
            h = np.degrees(np.arctan2(v[1], v[0]))
            heading = h if heading is None else heading + (1 - 0.5 ** (1 / (0.12 * args.fps))) * ((h - heading + 180) % 360 - 180)
        headings.append(heading)
    start_heading = next((h for h in headings if h is not None), 0.0)
    headings = [start_heading if h is None else h for h in headings]
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    hud = Hud(args.width, args.height, path, gates, noir, args.title) if args.hud else None
    trail = round(TRAIL * args.fps / args.speed)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    data, bold = mujoco.MjData(model), _font(0, round(20 * args.height / 432))
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for i, (a, b) in enumerate(zip(solid, ghost, strict=True)):
            for d, pose in (("drone", a), ("ref", b)):
                data.qpos[addr[f"{d}_root"] : addr[f"{d}_root"] + 7] = pose
                for k in range(4):
                    data.qpos[addr[f"{d}_prop{k}"]] = spin[i, k]
            mujoco.mj_forward(model, data)
            apart = float(np.linalg.norm(a[:3] - b[:3]))
            # the camera stays on the reference; where the simulated drone leaves it, on the two, from as far as shows both
            look = (lo + hi) / 2 if arena else b[:3] if apart < 0.3 else (a[:3] + b[:3]) / 2
            camera.lookat[:] = look
            camera.distance = distance if arena else min(distance + 1.3 * max(apart - 0.3, 0.0), 14.0)
            camera.azimuth = args.azimuth if args.view != "chase" else headings[i] + (args.azimuth - 110.0) + 8.0  # (through the gates with it)
            model.light("key").pos[:2] = look[:2] + (0.8, -0.6)  # the shadow map stays on the scene in view
            az = np.radians(camera.azimuth)
            if noir:  # the rim light shines from behind the drone, toward the camera
                model.light("rim").dir[:] = (-np.cos(az), -np.sin(az), -0.35)
            elif args.look == "sky":  # the horizon's light from the camera's side
                model.light("rim").dir[:] = (np.cos(az), np.sin(az), 0.0)
            renderer.update_scene(data, camera, option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = not arena  # (the arena's far side would fade)
            add_blob_shadow(renderer.scene, a[:3], 0.14 * scale, 0.55 if noir else 0.35)  # (on the light floors a grey of the squares')
            add_trail(renderer.scene, solid[max(0, i - trail) : i + 1, :3], 0.012 * max(scale, 1.0) ** 0.5)
            image = renderer.render()
            if hud is not None:
                image = hud.draw(image, int(round(steps[i])) // stride, a[:3], b[:3], float(speeds[i]), float(np.linalg.norm(a[:3] - b[:3])),
                                 mark=not args.no_mark)  # fmt: skip
            elif not args.no_mark:
                over = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 0))
                draw_mark(ImageDraw.Draw(over), args.width, args.height, bold, fill=(236, 238, 242, 120) if noir else (28, 30, 34, 110))
                image = np.asarray(Image.alpha_composite(Image.fromarray(image).convert("RGBA"), over).convert("RGB"))
            encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"{args.out}: {len(solid)} frames, {args.width}x{args.height} at {args.fps} fps")
    if args.preview:
        for quality in range(50, 19, -5):
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", args.out, "-vf", "fps=30,scale=960:-2:flags=lanczos", "-c:v", "libwebp_anim",
                            "-quality", str(quality), "-loop", "0", "-an", args.preview], check=True)  # fmt: skip
            if os.path.getsize(args.preview) <= args.preview_mb * 1e6:
                break
        print(f"{args.preview}: quality {quality}, {os.path.getsize(args.preview) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
