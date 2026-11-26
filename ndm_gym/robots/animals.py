"""The animal bodies of MIMIC-MJX (arXiv 2511.20532): a rat, a fruit fly and a stick insect, built as its environments build
them (github.com/talmolab/vnl-playground, BSD-3-Clause; the fly is flybody's, github.com/TuragaLab/flybody, Apache-2.0) from
the files ``tools/download_mimic_mjx.py`` puts in ``<root>/<animal>/xmls``.

Each body is attached to its arena's floor at MIMIC-MJX's physics settings (Newton's solver, 5 iterations and 5 line-search
iterations; the rat and the stick insect at 2 ms, the fly at 0.2 ms) and with its edits:

- the rat: its motors made torque actuators (the gain the largest force, no bias), the body scaled by 0.9 (positions, sizes,
  meshes; the actuators' gear by 0.81), a free joint at its root;
- the fly: as its file holds it (position actuators), its root's free joint its own, spawned 5 cm above the floor;
- the stick insect: as its file holds it, its feet's contacts with the floor as explicit pairs (a 0.2 ms spring), and its
  actuators' gear scaled by 1e-5 with floors on the joints' damping (5e-7) and armature (1e-9) for its milligram legs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnimalSpec:
    body_xml: str  # in <root>/<animal>/xmls
    arena_xml: str
    root_body: str  # the body attached to the arena
    sim_dt: float  # s
    ctrl_dt: float  # s
    mocap_hz: float  # the reference clips' frame rate
    clip_length: int  # frames per reference clip
    end_effectors: tuple[str, ...]
    torque_actuators: bool = False
    rescale: float = 1.0
    add_free_joint: bool = False
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    foot_pairs: tuple[str, ...] = ()  # geoms given explicit floor contact pairs
    stick_si_rescaling: bool = False
    nconmax: int = 64  # MuJoCo-Warp's contacts and constraints per world
    njmax: int = 512
    suffix: str = ""  # appended to the body's names and assets when attached, as MIMIC-MJX's environments attach it


ANIMALS = {
    "rat": AnimalSpec("rodent.xml", "arena.xml", "walker", 0.002, 0.01, 50, 250,
                      ("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"),
                      torque_actuators=True, rescale=0.9, add_free_joint=True, suffix="-rodent", nconmax=64, njmax=512),  # fmt: skip
    "fly": AnimalSpec("fruitfly_fast.xml", "arena.xml", "thorax", 0.0002, 0.002, 500, 600,
                      ("claw_T1_left", "claw_T1_right", "claw_T2_left", "claw_T2_right", "claw_T3_left", "claw_T3_right"),
                      spawn_pos=(0.0, 0.0, 0.05), suffix="-fly", nconmax=32, njmax=256),  # fmt: skip
    "stick": AnimalSpec("stick_mesh_fast.xml", "arena.xml", "reference_base", 0.002, 0.01, 50, 225,
                        ("20-f-l-claws-l", "36-f-r-claws-l", "25-m-l-claws-l", "41-m-r-claws-l", "30-h-l-claws-l", "46-h-r-claws-l"),
                        foot_pairs=("claw_collide_fl", "claw_collide_ml", "claw_collide_hl", "claw_collide_fr", "claw_collide_mr",
                                    "claw_collide_hr"), stick_si_rescaling=True, suffix="-stick", nconmax=128, njmax=512),  # fmt: skip
}


def _scale_spec(spec, s: float):
    """vnl-playground's ``scale_spec``: the bodies' and geoms' positions, the geoms' sizes and fromto and the meshes by ``s``,
    the actuators' gear by s^2 (a muscle's force goes with its cross-section)."""
    spec = spec.copy()
    for mesh in spec.meshes:
        mesh.scale = mesh.scale * s
    for actuator in spec.actuators:
        actuator.gear = actuator.gear * s * s
    for key in spec.keys:
        qpos = key.qpos
        qpos[2] = qpos[2] * s
        key.qpos = qpos

    def scale(parent):
        body = parent.first_body()
        while body:
            if body.pos is not None:
                body.pos = body.pos * s
            for geom in body.geoms:
                geom.fromto = geom.fromto * s
                geom.size = geom.size * s
                if geom.pos is not None:
                    geom.pos = geom.pos * s
            scale(body)
            body = parent.next_body(body)

    scale(spec.worldbody.first_body())
    return spec


def _attach(arena, animal: str, root: str, activation: bool, suffix: str, rgba=None):
    """Attach ``animal``'s body, edited as MIMIC-MJX's environments edit it, to the spec ``arena`` with the name suffix
    ``suffix``; with ``rgba`` every geom recolored, without collisions or actuators (a ghost to pose)."""
    import mujoco

    a = ANIMALS[animal]
    body = mujoco.MjSpec.from_file(os.path.join(root, animal, "xmls", a.body_xml))
    if a.torque_actuators:
        for act in body.actuators:
            if act.forcerange.size >= 2:
                act.gainprm[0] = act.forcerange[1]
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.biasprm = np.zeros((10, 1))
    if not activation:
        for act in body.actuators:
            act.dyntype = mujoco.mjtDyn.mjDYN_NONE
    if a.rescale != 1.0:
        body = _scale_spec(body, a.rescale)
    if rgba is not None:
        for g in body.geoms:
            g.rgba, g.contype, g.conaffinity = list(rgba), 0, 0
        for act in list(body.actuators):
            body.delete(act)
    frame = arena.worldbody.add_frame(pos=list(a.spawn_pos), quat=[1, 0, 0, 0])
    attached = frame.attach_body(body.body(a.root_body), "", suffix)
    if a.add_free_joint:
        attached.add_freejoint(name="root" if rgba is None else "root_ghost")
    if rgba is None:
        for geom in a.foot_pairs:
            arena.add_pair(geomname1="floor", geomname2=geom + suffix, solref=[0.0002, 1.0], solimp=[0.95, 0.99, 0.01, 0.5, 2.0])


def build_model(animal: str, root: str, activation: bool = True, ghost_rgba=None):
    """The ``mujoco.MjModel`` of ``animal`` in its arena, at MIMIC-MJX's physics settings. The body's names carry its suffix
    (``ANIMALS[animal].suffix``), as in MIMIC-MJX's environments. The rat's actuators keep their first-order activation
    filters (``activation_filter``); ``activation=False`` builds the same body without them (na = 0), the layout the world
    model reads (worldmodels.mjcf.MjcfLayout refuses activation states), whose controls are then the activations.
    ``ghost_rgba``: a second copy of the body after it (its qpos after the body's), in that color, without collisions or
    actuators (tools/render_animal.py)."""
    import mujoco

    a = ANIMALS[animal]
    arena = mujoco.MjSpec.from_file(os.path.join(root, animal, "xmls", a.arena_xml))
    _attach(arena, animal, root, activation, a.suffix)
    if ghost_rgba is not None:
        _attach(arena, animal, root, activation, a.suffix + "-ghost", ghost_rgba)
    arena.option.noslip_iterations = 0  # as MIMIC-MJX compiles every body (the fly's file asks for 3; MuJoCo-Warp has no noslip)
    m = arena.compile()
    m.opt.timestep = a.sim_dt
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.iterations, m.opt.ls_iterations = 5, 5
    if a.stick_si_rescaling:
        m.actuator_gear[:, 0] *= 1e-5
        sub = m.body_mass.copy()
        for i in range(m.nbody - 1, 0, -1):
            sub[m.body_parentid[i]] += sub[i]
        m.body_subtreemass[:] = sub
        m.dof_armature[:] = np.maximum(m.dof_armature, 1e-9)
        m.dof_damping[6:] = np.maximum(m.dof_damping[6:], 5e-7)
    return m


def activation_filter(m, substeps: int):
    """The activation filters of ``m``'s actuators over a control step of ``substeps`` physics steps, the control held: per
    actuator (nu,) the factor ``r`` and the mean factor ``rbar`` such that, from activation a0 under control u, the step ends
    at u + (a0 - u) r and its physics steps' forces see the mean activation u + (a0 - u) rbar. MuJoCo's Euler step of a
    filter (dyntype filter) moves a by h (u - a) / tau after the forces are taken, so a_k = u + (a0 - u) c^k with
    c = 1 - h / tau, r = c^n and rbar = (1 + c + ... + c^(n-1)) / n. Actuators without a filter: r = rbar = 0 (the force
    is the control's). Returns numpy arrays."""
    import mujoco

    filt = m.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_FILTER)
    if (m.actuator_dyntype[~filt] != int(mujoco.mjtDyn.mjDYN_NONE)).any():
        raise ValueError("activation_filter: actuator dynamics other than a filter or none")
    if filt.any() and m.opt.integrator != int(mujoco.mjtIntegrator.mjINT_EULER):
        raise ValueError("activation_filter: the closed form is Euler's")
    if filt.any() and m.actuator_actearly[filt].any():
        raise ValueError("activation_filter: actearly actuators")
    tau = np.where(filt, m.actuator_dynprm[:, 0], 1.0)
    c = np.where(filt, 1.0 - m.opt.timestep / tau, 0.0)
    r = c**substeps
    rbar = np.where(filt, (1 - r) / np.maximum(substeps * (1 - c), 1e-12), 0.0)
    return r.astype(np.float32), rbar.astype(np.float32)
