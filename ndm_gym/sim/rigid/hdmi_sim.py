"""HDMI's human-object interaction scene on Isaac Lab 3 + Newton/MuJoCo-Warp (docs/hoi_design.md; HDMI's task:
docs/hdmi_task_spec.md, § below). The task terms are :mod:`ndm_gym.tasks.hdmi`; this module produces their inputs.

One control step is HDMI's: four 5 ms physics steps, before each the PD targets of the action delayed by d substeps and
low-pass filtered (§3.1), and the hinge torque of a door or folding chair (§9); the feet's contact forces are read after
every substep. The robot is the task's HDMI G1 variant, the tracked object and the static props are HDMI's assets
(tools/download_hdmi.sh), the reference is the task's converted clip (tools/convert_hdmi.py).

Engine differences from HDMI's PhysX, kept explicit: MuJoCo combines the friction of a contact pair by its maximum,
PhysX here by its product with the ground's 1.0, so the ground gets the lowest friction of the robot's and the object's
ranges and every pair keeps its own body's coefficient; MuJoCo has one sliding friction (the dynamic one is used) and
no restitution. HDMI's startup randomization (§7) is drawn once per env at construction, except the per-env object
scale (pending: every object keeps scale 1).

Joint velocity limits: PhysX enforces HDMI's (``velocity_limit_sim``, 32 / 20 / 37 / 22 rad/s by motor, G1:95-99, 122,
130, 138, 163-170); Newton's MuJoCo solver ignores them. After every physics substep the robot's joint velocities are
clamped to them in the simulation state, which the next substep starts from and which the substep joints read; after
the last substep the links' velocities are recomputed from the clamped joints (forward kinematics). Residual
difference: PhysX clamps inside its solver, so the clamped velocity also integrates that step's positions, whereas here
the substep that reaches a limit integrates positions and contacts with the unclamped velocity and the next one starts
from the limit. PhysX's per-link velocity caps (1000 m/s, 1000 deg/s) and its depenetration velocity of 1 m/s
(G1:58-60) are not reproduced.

Static props (§9): HDMI's are kinematic bodies, which PhysX never puts in contact with each other or with static
geometry. Newton's MuJoCo solver models a kinematic body as a free body locked by a large armature, and MuJoCo-Warp's
collision would make those contacts: in topple_wood_board_and_cross wall0 stands 8 cm into platform0, and every solve
then accelerated platform0 away from what stood on it (it carried no load) and left some worlds at the solver's
iteration limit. Collision filter pairs keep each prop out of contact with the other props and the ground
(exclude_prop_contacts); the robot and the object collide with the props as before.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg
from newton import BodyFlags, ModelFlags, ShapeFlags, eval_fk

from ...core.rotations import mat_to_quat, quat_to_mat
from ...robots.g1 import DEFAULT_JOINT_POS, G1_JOINT_NAMES, kd, kp, motor_of
from ...state import OBJ_DIM, STATE_DIM, flatten_object
from ...tasks.hdmi import (DECIMATION, FUTURE_STEPS, JOINT_NAMES, KEYPOINT_NAMES, NUM_ACTIONS, PHYSICS_DT, HDMIClip, HDMIInput,
                           HDMIState, action_pipeline, reference_window)  # fmt: skip
from ...tasks.hdmi_tasks import HDMITaskCfg
from .isaaclab_mjwarp import after_replay, before_replay, capture_step, flatten_states, on_capture_stream

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HDMI_DIR = os.path.join(REPO, "data", "hdmi")
GROUND_FRICTION = 0.3  # the lowest friction of the robot's feet and hands (§7) and of every task's object
WRISTS = ("left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint",
          "right_wrist_yaw_joint")  # fmt: skip
ACTION_SCALE = {"hip_yaw": 0.55, "hip_roll": 0.35, "hip_pitch": 0.55, "knee": 0.35, "ankle_pitch": 0.44, "ankle_roll": 0.44,
                "waist_roll": 0.44, "waist_pitch": 0.44, "waist_yaw": 0.55, "shoulder_pitch": 0.44, "shoulder_roll": 0.44,
                "shoulder_yaw": 0.44, "elbow": 0.44}  # Y:27-43  # fmt: skip


def make_hdmi_sim(device: str, substeps: int = 1) -> sim_utils.SimulationCfg:
    """5 ms physics steps. HDMISim needs ``substeps`` 1: the control step's four substeps are stepped one by one, each
    with its own PD targets (§0). ``substeps`` 4 runs a whole control step in one call with fixed targets, which only
    the throughput benchmark uses (wmbench/speed_hoi.py)."""
    physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=192, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True),
                        num_substeps=substeps)  # fmt: skip
    return sim_utils.SimulationCfg(dt=PHYSICS_DT * substeps, device=device, physics=physics)


def default_pose(cfg: HDMITaskCfg) -> list[float]:
    """The task's default joint pose in G1 order: the G1's, with the task's wrist-yaw override (scene_cfg)."""
    from ...robots.g1 import default_joint_pos_list

    q0 = default_joint_pos_list()
    if cfg.wrist_yaw_default is not None:
        for name, v in zip(("left_wrist_yaw_joint", "right_wrist_yaw_joint"), cfg.wrist_yaw_default):
            q0[G1_JOINT_NAMES.index(name)] = v
    return q0


def ndm_constants(cfg: HDMITaskCfg) -> dict:
    """The task's constants the token world model needs (worldmodels/hoi.HOIRigidFormerNet): the end effectors' bodies,
    the default pose in G1 order, the action scales in HDMI's action order and each G1 joint's action index (-1: a
    wrist, held at its default)."""
    acts = list(JOINT_NAMES[:NUM_ACTIONS])
    return {"eef_bodies": list(cfg.contact_eef_bodies), "q0": default_pose(cfg), "act_scale": [action_scale(n) for n in acts],
            "act_of_joint": [acts.index(n) if n in acts else -1 for n in G1_JOINT_NAMES]}  # fmt: skip


def action_scale(joint: str) -> float:
    key = joint.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    return ACTION_SCALE[key]


def load_clip(task: str) -> tuple[dict, dict]:
    """The task's converted clip (arrays as stored, Unitree joint order) and its sidecar."""
    path = os.path.join(HDMI_DIR, "clips", task)
    if not os.path.exists(path + ".npz"):
        raise FileNotFoundError(f"{path}.npz: run tools/download_hdmi.sh, then python tools/convert_hdmi.py")
    with np.load(path + ".npz", allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    with open(path + ".json") as f:
        side = json.load(f)
    return arrays, side


def hdmi_clip(arrays: dict, side: dict, device) -> HDMIClip:
    """The clip in HDMI's orders: its stored keypoints and velocities (HDMI's, not finite differences), the tracking
    joints, the object's root pose, the contact flags unshifted (reference_window applies the lead)."""
    t = lambda k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)  # noqa: E731
    kp_idx = [side["keypoint_names"].index(n) for n in KEYPOINT_NAMES]
    j_idx = [G1_JOINT_NAMES.index(n) for n in JOINT_NAMES[:NUM_ACTIONS]]
    wxyz = lambda q: quat_to_mat(q)  # noqa: E731
    return HDMIClip(t("keypoint_pos")[:, kp_idx], wxyz(t("keypoint_quat_wxyz")[:, kp_idx]), t("keypoint_lin_vel")[:, kp_idx],
                    t("keypoint_ang_vel")[:, kp_idx], t("joint_pos")[:, j_idx], t("joint_vel")[:, j_idx], t("object_pos"),
                    wxyz(t("object_quat_wxyz")), t("contact") > 0.5,
                    t("object_joint_pos")[:, 0] if "object_joint_pos" in arrays else None)  # fmt: skip


def _actuators(cfg: HDMITaskCfg) -> dict[str, ImplicitActuatorCfg]:
    acts = {}
    for name in G1_JOINT_NAMES:
        effort, vel, arm = motor_of(name)
        stiff = cfg.wrist_kp if (cfg.wrist_kp is not None and name in WRISTS) else kp(arm)  # §3.3: kd and armature unchanged
        acts[name.replace("_joint", "")] = ImplicitActuatorCfg(joint_names_expr=[name], effort_limit_sim=effort, velocity_limit_sim=vel,
                                                               stiffness=stiff, damping=kd(arm), armature=arm)  # fmt: skip
    return acts


def scene_cfg(cfg: HDMITaskCfg, side: dict, num_envs: int):
    usd = lambda rel: os.path.join(HDMI_DIR, rel)  # noqa: E731
    q0 = dict(DEFAULT_JOINT_POS)
    if cfg.wrist_yaw_default is not None:
        q0["left_wrist_yaw_joint"], q0["right_wrist_yaw_joint"] = cfg.wrist_yaw_default
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd(side["robot_usd"]), activate_contact_sensors=True,
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
                                   articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),  # fmt: skip
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_pos=q0, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators=_actuators(cfg),
    )
    obj_spawn = sim_utils.UsdFileCfg(usd_path=usd(side["object_usd"]), activate_contact_sensors=True)
    if cfg.object_joint is None:
        obj = RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{cfg.object_asset}", spawn=obj_spawn)
    else:  # door / folding chair: fixed base, the USD's drive replaced by the hinge torque of step() (§9)
        obj = ArticulationCfg(prim_path=f"{{ENV_REGEX_NS}}/{cfg.object_asset}", spawn=obj_spawn, init_state=ArticulationCfg.InitialStateCfg(),
                              actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=[cfg.object_joint], stiffness=0.0, damping=0.0,
                                                                      effort_limit_sim=100.0, velocity_limit_sim=20.0)})  # fmt: skip
    attrs = {"ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
                 physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION))),
             "robot": robot, "obj": obj}  # fmt: skip
    for name in cfg.extra_objects:  # static props: kinematic bodies (§9)
        attrs[f"extra_{name}"] = RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=sim_utils.UsdFileCfg(
            usd_path=usd(side["extra_object_usd"][name]), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)))  # fmt: skip
    for e, body in enumerate(cfg.contact_eef_bodies):  # the end effectors' forces from the object's contact body (LOC:89-100)
        attrs[f"eef{e}"] = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                                            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{cfg.object_asset}/{cfg.object_body}"])  # fmt: skip
    attrs["feet"] = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link")
    cls = configclass(type("HDMISceneCfg", (InteractiveSceneCfg,), {"__annotations__": {k: type(v) for k, v in attrs.items()}, **attrs}))
    return cls(num_envs=num_envs, env_spacing=0.0, replicate_physics=True)


def prop_filter_pairs(builder, patterns: dict[str, str]) -> list[tuple[int, int]]:
    """The Newton collision filter pairs that keep the static props out of contact with each other and with static
    shapes (module docstring): in every world each colliding shape of a prop against each colliding shape of the world's
    other props, and each against every colliding static shape (the ground). ``builder``: the scene's Newton
    ModelBuilder; ``patterns``: each prop's prim path as a regular expression, its bodies being that prim and the prims
    below it. Raises unless every prop body is kinematic and every world has a colliding shape of every prop."""
    collide = [bool(f & ShapeFlags.COLLIDE_SHAPES) for f in builder.shape_flags]
    static = [s for s, b in enumerate(builder.shape_body) if b == -1 and collide[s]]
    shapes = {w: {n: [] for n in patterns} for w in sorted({w for w in builder.body_world if w >= 0})}
    for name, pattern in patterns.items():
        rx = re.compile(f"{pattern}(/.*)?")
        for b, label in enumerate(builder.body_label):
            if rx.fullmatch(label):
                if not builder.body_flags[b] & BodyFlags.KINEMATIC:
                    raise RuntimeError(f"the prop {label} is not a kinematic body")
                shapes[builder.body_world[b]][name] += [s for s in builder.body_shapes[b] if collide[s]]
    pairs = []
    for w, props in shapes.items():
        missing = [n for n, s in props.items() if not s]
        if missing:
            raise RuntimeError(f"world {w} has no colliding shape of the props {missing} (prim paths {list(patterns.values())})")
        names = list(props)
        for i, a in enumerate(names):
            pairs += [(sa, sb) for b in names[i + 1:] for sa in props[a] for sb in props[b]]
            pairs += [(sa, ss) for sa in props[a] for ss in static]
    return pairs


def _prop_patterns(scene, cfg: HDMITaskCfg) -> dict[str, str]:
    """Each static prop's prim path in the scene, a regular expression over the envs."""
    return {n: scene[f"extra_{n}"].cfg.prim_path for n in cfg.extra_objects}


def exclude_prop_contacts(sim: SimulationContext, scene, cfg: HDMITaskCfg):
    """Add prop_filter_pairs for the task's static props to the Newton builder before the model is built (a one-shot
    MODEL_INIT callback, as Isaac Lab's articulations configure the builder); SolverMuJoCo turns them into MuJoCo
    collision masks and excludes. Call after the scene's construction and before sim.reset(); nothing for a task without
    props."""
    if not cfg.extra_objects:
        return
    patterns = _prop_patterns(scene, cfg)

    def add_pairs(_payload):
        handle.deregister()  # this scene's model only
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        for a, b in prop_filter_pairs(builder, patterns):
            builder.add_shape_collision_filter_pair(a, b)

    handle = sim.backend_manager.register_callback(add_pairs, PhysicsEvent.MODEL_INIT, name=f"hdmi_prop_filters_{cfg.name}")


def check_prop_contacts(sim: SimulationContext, scene, cfg: HDMITaskCfg):
    """Raise unless MuJoCo-Warp has a geom of every static prop and no collision candidate between two props or a prop
    and a static shape (after sim.reset(), which builds the solver): its candidate geom pairs, by the bodies of world 0's
    Newton shapes."""
    if not cfg.extra_objects:
        return
    solver, model = sim.backend_manager._solver, sim.backend_manager.get_model()  # the manager has no public solver accessor
    shape = solver.mjc_geom_to_newton_shape.numpy()[0]  # MuJoCo geom -> Newton shape of world 0, the template
    if (shape < 0).any():
        raise RuntimeError(f"MuJoCo geoms {np.flatnonzero(shape < 0).tolist()} have no Newton shape")
    body = model.shape_body.numpy()[shape]
    fixed = body == -1
    for name, pattern in _prop_patterns(scene, cfg).items():
        rx = re.compile(f"{pattern}(/.*)?")
        prop = np.array([b >= 0 and rx.fullmatch(model.body_label[b]) is not None for b in body])
        if not prop.any():
            raise RuntimeError(f"no MuJoCo geom of the prop {name} ({pattern})")
        fixed |= prop
    pairs = solver.mjw_model.nxn_geom_pair_filtered.numpy()
    bad = pairs[fixed[pairs[:, 0]] & fixed[pairs[:, 1]]]
    if len(bad):
        label = lambda g: "static" if body[g] == -1 else model.body_label[body[g]]  # noqa: E731
        raise RuntimeError(f"MuJoCo-Warp collides props with props or static shapes: {[(label(a), label(b)) for a, b in bad]}")


def place_scene(scene, cfg: HDMITaskCfg, frame: int = 0):
    """Frame ``frame`` of the task's clip in every env of a scene of scene_cfg, without HDMI's reset noise (§2): the
    robot's root pose and joints (clamped to the soft limits, as HDMISim.reset), the object's root pose and a hinge's
    angle, the props' poses; every velocity zero."""
    arrays, side = load_clip(cfg.name)
    dev, n = scene.device, scene.num_envs
    c = hdmi_clip(arrays, side, dev)
    ids, org, zero6 = torch.arange(n, device=dev), scene.env_origins, torch.zeros(n, 6, device=dev)
    every = lambda x: x.expand(n, *x.shape).clone()  # noqa: E731  the frame's value in every env
    t = lambda k: torch.as_tensor(arrays[k][frame], dtype=torch.float32, device=dev)  # noqa: E731
    extras = [(every(t("extra_pos")[k]) + org, every(quat_to_mat(t("extra_quat_wxyz")[k]))) for k in range(len(cfg.extra_objects))]
    robot_pose, obj_pose, *extra_poses = _poses((every(c.body_pos[frame, 0]) + org, every(c.body_rot[frame, 0])),
                                                (every(c.obj_pos[frame]) + org, every(c.obj_rot[frame])), *extras)  # fmt: skip
    robot, obj = scene["robot"], scene["obj"]
    lim = robot.data.joint_pos_limits.torch[0]
    mid, half = lim.mean(-1), 0.5 * (lim[:, 1] - lim[:, 0])
    jq = t("joint_pos")[[G1_JOINT_NAMES.index(j) for j in robot.joint_names]]
    jq = torch.minimum(torch.maximum(jq, mid - 0.9 * half), mid + 0.9 * half)
    robot.write_root_link_pose_to_sim_index(root_pose=robot_pose, env_ids=ids)
    robot.write_root_com_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)
    robot.write_joint_position_to_sim_index(position=every(jq), env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(n, jq.numel(), device=dev), env_ids=ids)
    obj.write_root_link_pose_to_sim_index(root_pose=obj_pose, env_ids=ids)
    if cfg.object_joint is None:
        obj.write_root_com_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)
    else:
        hq = c.obj_joint_pos[frame].clamp(*(float(v) for v in obj.data.joint_pos_limits.torch[0, 0]))
        obj.write_joint_position_to_sim_index(position=hq.repeat(n, 1), env_ids=ids)
        obj.write_joint_velocity_to_sim_index(velocity=torch.zeros(n, 1, device=dev), env_ids=ids)
    for name, pose in zip(cfg.extra_objects, extra_poses):
        scene[f"extra_{name}"].write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        scene[f"extra_{name}"].write_root_com_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)
    scene.reset(ids)


def _euler_xyz(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices of quat_from_euler_xyz(roll, pitch, yaw) = Rz Ry Rx [IL]."""
    cr, sr, cp, sp, cy, sy = r[:, 0].cos(), r[:, 0].sin(), r[:, 1].cos(), r[:, 1].sin(), r[:, 2].cos(), r[:, 2].sin()
    one, zero = torch.ones_like(cr), torch.zeros_like(cr)
    Rx = torch.stack((one, zero, zero, zero, cr, -sr, zero, sr, cr), -1).view(-1, 3, 3)
    Ry = torch.stack((cp, zero, sp, zero, one, zero, -sp, zero, cp), -1).view(-1, 3, 3)
    Rz = torch.stack((cy, -sy, zero, sy, cy, zero, zero, zero, one), -1).view(-1, 3, 3)
    return Rz @ Ry @ Rx


def _bounds(ranges: dict, keys, device) -> tuple[torch.Tensor, torch.Tensor]:
    """A reset range's low ends and widths, made once on the device."""
    lo = torch.tensor([ranges.get(k, (0.0, 0.0))[0] for k in keys], device=device)
    hi = torch.tensor([ranges.get(k, (0.0, 0.0))[1] for k in keys], device=device)
    return lo, hi - lo


def _uniform(bounds: tuple[torch.Tensor, torch.Tensor], n: int) -> torch.Tensor:
    lo, width = bounds
    return lo + width * torch.rand(n, len(lo), device=lo.device)


def _poses(*pr: tuple[torch.Tensor, torch.Tensor]) -> list[torch.Tensor]:
    """xyzw poses of (position, rotation matrix) pairs with one quaternion conversion; mat_to_quat works row by row, so
    each pose is the one a conversion of its own gives, bit for bit."""
    q = mat_to_quat(torch.cat([R for _, R in pr]))  # wxyz
    out, i = [], 0
    for p, _ in pr:
        out.append(torch.cat((p, q[i:i + len(p), 1:4], q[i:i + len(p), 0:1]), -1))  # Isaac Lab 3: xyzw
        i += len(p)
    return out


def _t(x) -> torch.Tensor:
    """Asset data are warp-backed arrays with a torch view; sensor data are torch tensors."""
    return x.torch if hasattr(x, "torch") else x


def _rot(pose_xyzw: torch.Tensor) -> torch.Tensor:
    q = pose_xyzw[..., 3:7]
    return quat_to_mat(torch.cat((q[..., 3:4], q[..., :3]), -1))


def _rots(*poses: torch.Tensor) -> list[torch.Tensor]:
    """_rot of each (N, ..., 7) pose with one conversion: quat_to_mat works row by row, so every part is its own _rot bit
    for bit; the parts after the first are copied out, fresh tensors as separate calls return."""
    if len(poses) == 1:
        return [_rot(poses[0])]
    R = _rot(torch.cat([p.reshape(-1, 7) for p in poses]))
    out, i = [], 0
    for p in poses:
        n = p.numel() // 7
        r = R[i:i + n].view(*p.shape[:-1], 3, 3)
        out.append(r if i == 0 else r.clone())
        i += n
    return out


class HDMISim:
    """The scene of one HDMI task with ``num_envs`` copies. Must be constructed inside ``launch_simulation(sim_cfg, ...)``
    with ``sim_cfg`` from :func:`make_hdmi_sim`. ``eval_mode``: HDMI's evaluation reset (no robot pose and velocity
    noise, §2) from frame 0 unless frames are given. ``randomize``: HDMI's startup randomization of masses, centres of
    mass, friction and the hinge's armature (§7); off only for kinematic checks. ``graph``: each control step as one
    CUDA graph (step; the same values): the frame counter ``t``, ``ep_step``, ``action_buf`` and ``a_filt`` are then
    tensors the step updates in place, and the physics model must not change after the third step."""

    def __init__(self, sim_cfg, cfg: HDMITaskCfg, num_envs: int, eval_mode: bool = False, randomize: bool = True, graph: bool = False):
        if abs(sim_cfg.dt - PHYSICS_DT) > 1e-9:
            raise ValueError(f"HDMISim steps {PHYSICS_DT} s substeps itself; got a simulation step of {sim_cfg.dt} s")
        self.cfg, self.N, self.eval_mode, self.device = cfg, num_envs, eval_mode, sim_cfg.device
        arrays, self.side = load_clip(cfg.name)
        if self.side["task"] != cfg.name:
            raise ValueError(f"clip {self.side['task']} for task {cfg.name}")
        self.clip = hdmi_clip(arrays, self.side, self.device)
        self.L = self.clip.body_pos.shape[0]
        dev = self.device
        t = lambda k: torch.as_tensor(arrays[k], dtype=torch.float32, device=dev)  # noqa: E731
        self.ref_extra_p = t("extra_pos") if cfg.extra_objects else None  # (T, n, 3)
        self.ref_extra_R = quat_to_mat(t("extra_quat_wxyz")) if cfg.extra_objects else None
        self.ref_joint_vel_all = t("joint_vel")  # (T, 29) Unitree order: the reset's joint velocities (§2)
        self.ref_joint_pos_all = t("joint_pos")
        self.ref_obj_joint_vel = t("object_joint_vel")[:, 0] if "object_joint_vel" in arrays else None

        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(scene_cfg(cfg, self.side, num_envs))
        exclude_prop_contacts(self.sim, self.scene, cfg)
        self.sim.reset()
        check_prop_contacts(self.sim, self.scene, cfg)
        self.robot, self.obj = self.scene["robot"], self.scene["obj"]
        self.extras = [self.scene[f"extra_{n}"] for n in cfg.extra_objects]
        self.eef_sensors = [self.scene[f"eef{e}"] for e in range(cfg.num_eefs)]
        self.feet_sensor = self.scene["feet"]
        names = list(self.robot.joint_names)
        if sorted(names) != sorted(JOINT_NAMES):
            raise ValueError(f"the robot's joints {names} are not HDMI's")
        self.j_hdmi = torch.tensor([names.index(n) for n in JOINT_NAMES], device=dev)  # HDMI order -> robot index
        self.j_act = self.j_hdmi[:NUM_ACTIONS]
        bodies = list(self.robot.body_names)
        self.kp_ids = torch.tensor([bodies.index(n) for n in KEYPOINT_NAMES], device=dev)
        self.eef_ids = torch.tensor([bodies.index(b) for b in cfg.contact_eef_bodies], device=dev)
        self.feet_order = torch.tensor([self.feet_sensor.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")], device=dev)
        self.future_steps = torch.tensor(FUTURE_STEPS, device=dev)  # made once: reference_window would copy it every call
        obj_bodies = list(self.obj.body_names)
        if cfg.object_body not in obj_bodies:
            raise ValueError(f"{cfg.object_asset} has no body {cfg.object_body}: {obj_bodies}")
        self.obj_body_id = obj_bodies.index(cfg.object_body)
        self.hinged = cfg.object_joint is not None
        if self.hinged and list(self.obj.joint_names) != [cfg.object_joint]:  # the hinge is the object's joint 0 everywhere below
            raise ValueError(f"{cfg.object_asset} must have exactly the joint {cfg.object_joint}, has {list(self.obj.joint_names)}")
        self.q0 = self.robot.data.default_joint_pos.torch[0].clone()  # robot order
        self.scale = torch.tensor([action_scale(n) for n in JOINT_NAMES[:NUM_ACTIONS]], device=dev)
        lim = self.robot.data.joint_pos_limits.torch[0]  # (29, 2) robot order
        self.joint_limits = lim[self.j_hdmi].clone()  # HDMI order, for HDMITask
        mid, half = lim.mean(-1), 0.5 * (lim[:, 1] - lim[:, 0])
        self.soft_lo, self.soft_hi = mid - 0.9 * half, mid + 0.9 * half  # §2
        self.vel_limit = self.robot.data.joint_vel_limits.torch[0].clone()
        if not torch.equal(self.vel_limit.cpu(), torch.tensor([motor_of(n)[1] for n in names])):  # HDMI's (module docstring), in rad/s
            raise RuntimeError(f"the robot's joint velocity limits {self.vel_limit.tolist()} are not HDMI's")
        self._vel_lo = -self.vel_limit
        qd, state_qd = self.robot.data.joint_vel.torch, wp.to_torch(self.sim.backend_manager.get_state_0().joint_qd)
        if not state_qd.data_ptr() <= qd.data_ptr() < state_qd.data_ptr() + state_qd.numel() * state_qd.element_size():
            raise RuntimeError("Isaac Lab's joint velocities are not a view of the simulation state, which the velocity limits write")
        self.j_g1 = torch.tensor([names.index(n) for n in G1_JOINT_NAMES], device=dev)  # G1 (Unitree) order -> robot index
        self.g1_of_robot = torch.tensor([G1_JOINT_NAMES.index(n) for n in names], device=dev)  # robot order -> G1 index
        r = cfg.reset
        rpy = ("roll", "pitch", "yaw")
        self._bounds = {"p": _bounds(r.pose_range, "xyz", dev), "r": _bounds(r.pose_range, rpy, dev), "v": _bounds(r.velocity_range, "xyz", dev),
                        "w": _bounds(r.velocity_range, rpy, dev), "op": _bounds(r.object_pose_range, "xyz", dev),
                        "or": _bounds(r.object_pose_range, rpy, dev)}  # fmt: skip
        self.hinge_lim = tuple(float(v) for v in self.obj.data.joint_pos_limits.torch[0, 0]) if self.hinged else None
        self.com_ids = [bodies.index(n) for n in ("pelvis", "torso_link")]
        self._mass0 = self.robot.data.body_mass.torch.clone()  # before the startup randomization
        self._com0 = self.robot.data.body_com_pos_b.torch[:, self.com_ids].clone()
        if randomize:
            self._randomize_startup()
        self.kp_com_b = self.robot.data.body_com_pos_b.torch[:, self.kp_ids].clone()  # (N, 16, 3) keypoint centres of mass
        self.robot_mass = float(self.robot.data.body_mass.torch[0].sum())  # env 0's, after the startup randomization (§5.5)
        self.eef_offsets = torch.tensor(cfg.contact_eef_offsets, device=dev)
        self.target_offsets = torch.tensor(cfg.contact_target_offsets, device=dev)
        self.obj_scale = torch.ones(self.N, 3, device=dev)  # per-env object scale (§7; pending, see the module docstring)
        self.startup_context = self._startup_context()

        N = self.N
        self.t = torch.zeros(N, dtype=torch.long, device=dev)  # HDMI's frame counter
        self.ep_step = torch.zeros(N, dtype=torch.long, device=dev)
        self.offset = torch.zeros(N, 29, device=dev)  # joint offsets, robot order
        self.delay = torch.full((N,), 2, dtype=torch.long, device=dev)
        self.alpha = torch.ones(N, 1, device=dev)
        self.action_buf = torch.zeros(N, NUM_ACTIONS, 3, device=dev)
        self.a_filt = torch.zeros(N, NUM_ACTIONS, device=dev)
        self.hinge_f = torch.zeros(N, device=dev)
        self.hinge_c = torch.zeros(N, device=dev)
        self.hinge_tau = torch.zeros(N, device=dev)
        self.graph, self._graph, self._graph_steps = graph, None, 0
        self._graph_action = torch.zeros(N, NUM_ACTIONS, device=dev) if graph else None
        self.reset(torch.arange(N, device=dev))

    def _randomize_startup(self):
        """HDMI's startup randomization (§7), one draw per env, env 0 included.

        Robot: every link's mass and inertia x U[0.9, 1.1] (RND:342-369); the sliding friction of the shapes of the
        ankle-roll and wrist-yaw links from 64 buckets of U[0.3, 1.2], an independent bucket per env and shape
        (RND:242-293); the pelvis's and torso's centres of mass shifted by U[-0.02, 0.02]^3 in the link frame
        (RND:371-383). Object: every body's mass from the task's range, its inertia x m_new / m_old; the sliding friction
        of all its shapes from 64 buckets of the task's dynamic range, one bucket per env (HRN:12-83); a hinge's
        armature from the task's range (HRN:103-105). MuJoCo has one friction coefficient, so the dynamic ranges are
        used, and no restitution, which is not drawn."""
        dev, N, rz = self.device, self.N, self.cfg.randomization
        r, o = self.robot, self.obj
        model = self.sim.backend_manager.get_model()
        m0, I0 = r.data.body_mass.torch.clone(), r.data.body_inertia.torch.clone()
        u = torch.empty_like(m0).uniform_(0.9, 1.1)
        r.set_masses_index(masses=m0 * u)
        r.set_inertias_index(inertias=I0 * u.unsqueeze(-1))
        mu = wp.to_torch(r._root_view.get_attribute("shape_material_mu", model)[:, 0])  # (N, shapes)
        shapes = [s for b, name in enumerate(r.backend_body_names) if name.endswith(("ankle_roll_link", "_wrist_yaw_link"))
                  for s in r._root_view.body_shapes[b]]  # fmt: skip
        if len(shapes) == 0:
            raise ValueError("no shapes on the robot's ankle-roll and wrist-yaw links")
        buckets = torch.empty(64, device=dev).uniform_(0.3, 1.2)
        mu[:, shapes] = buckets[torch.randint(0, 64, (N, len(shapes)), device=dev)]
        ids = [r.body_names.index(n) for n in ("pelvis", "torso_link")]
        com = r.data.body_com_pos_b.torch[:, ids].clone()
        r.set_coms_index(coms=com + torch.empty_like(com).uniform_(-0.02, 0.02), body_ids=ids)
        om0, oI0 = o.data.body_mass.torch.clone(), o.data.body_inertia.torch.clone()
        om = torch.empty_like(om0).uniform_(*rz.mass_range)
        o.set_masses_index(masses=om)
        o.set_inertias_index(inertias=oI0 * (om / om0).unsqueeze(-1))
        omu = wp.to_torch(o._root_view.get_attribute("shape_material_mu", model)[:, 0])
        obuckets = torch.empty(64, device=dev).uniform_(*rz.dynamic_friction_range)
        omu[:] = obuckets[torch.randint(0, 64, (N, 1), device=dev)]
        if self.hinged:
            o.write_joint_armature_to_sim_index(armature=torch.empty(N, 1, device=dev).uniform_(*rz.hinge_armature_range))
        self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def _shape_mu_means(self, asset, body_names: list[str] | None) -> torch.Tensor:
        """Mean sliding friction of the shapes of each named body of an articulation, (N, len(body_names)), or of all
        the shapes of ``asset``, (N, 1), with ``body_names`` None."""
        mu = wp.to_torch(asset._root_view.get_attribute("shape_material_mu", self.sim.backend_manager.get_model())[:, 0])
        if body_names is None:
            return mu.mean(-1, keepdim=True)
        backend = list(asset.backend_body_names)
        return torch.stack([mu[:, asset._root_view.body_shapes[backend.index(b)]].mean(-1) for b in body_names], -1)

    def _startup_context(self) -> torch.Tensor:
        """The startup draws (§7), fixed for the run: robot link mass factors, pelvis and torso centre-of-mass shifts,
        the feet's and wrist-yaw links' friction, the object's body masses, friction and scale, the hinge's armature."""
        r, o = self.robot, self.obj
        ends = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
        arm = o.data.joint_armature.torch[:, :1] if self.hinged else torch.zeros(self.N, 1, device=self.device)
        obj_mu = self._shape_mu_means(o, None)  # one draw per env for all its shapes
        return torch.cat((r.data.body_mass.torch / self._mass0, (r.data.body_com_pos_b.torch[:, self.com_ids] - self._com0).flatten(1),
                          self._shape_mu_means(r, list(ends)), o.data.body_mass.torch, obj_mu, self.obj_scale, arm), -1)  # fmt: skip

    def context(self) -> torch.Tensor:
        """Everything that stays fixed within an episode and shapes the dynamics, for the world models
        (docs/hoi_design.md §3): the startup draws, then the reset's joint offsets (G1 order), action delay and filter
        constant, the hinge's friction and damping, and, relative to the env origin, the poses (position, rotation matrix
        row by row) of a hinged object's fixed base and of the static props. (N, context_dim)."""
        poses = self._context_poses()
        return self._context(poses, _rots(*poses) if poses else [])

    def _context_poses(self) -> list[torch.Tensor]:
        return ([self.obj.data.root_link_pose_w.torch] if self.hinged else []) + [e.data.root_link_pose_w.torch for e in self.extras]

    def _context(self, poses: list[torch.Tensor], rots: list[torch.Tensor]) -> torch.Tensor:
        org, parts = self.scene.env_origins, [self.startup_context, self.offset[:, self.j_g1], self.delay.unsqueeze(-1).float(), self.alpha,
                                              self.hinge_f.unsqueeze(-1), self.hinge_c.unsqueeze(-1)]  # fmt: skip
        for pose, R in zip(poses, rots):
            parts += [pose[:, :3] - org, R.flatten(1)]
        return torch.cat(parts, -1)

    @property
    def context_layout(self) -> tuple[int, bool, int]:
        """(length of the startup part, whether a fixed base's pose follows, number of props) of ``context``."""
        return self.startup_context.shape[-1], self.hinged, len(self.extras)

    def robot_state(self, R: torch.Tensor | None = None):
        """The robot's state ``S = (p, R, v, w, q, qd)`` as the LAFAN1 tracking keeps it (ndm_gym.state): root link pose
        relative to the env origin and its world velocities, joints in G1 (Unitree) order. ``R``: the root's rotation
        matrix if already converted (_states)."""
        d = self.robot.data
        pose, vel = d.root_link_pose_w.torch, d.root_link_vel_w.torch
        q, qd = d.joint_pos.torch[:, self.j_g1], d.joint_vel.torch[:, self.j_g1]  # gathers: fresh tensors
        return (pose[:, :3] - self.scene.env_origins, _rot(pose) if R is None else R, vel[:, :3].clone(), vel[:, 3:6].clone(), q, qd)

    def object_state(self, R: torch.Tensor | None = None):
        """A free object's root link ``(p, R, v, w)`` (positions relative to the env origin), or a hinged object's
        ``(angle, rate)``, each (N, 1); its base is fixed and part of ``context``. ``R`` as in ``robot_state``."""
        d = self.obj.data
        if self.hinged:
            return (d.joint_pos.torch[:, :1].clone(), d.joint_vel.torch[:, :1].clone())
        pose, vel = d.root_link_pose_w.torch, d.root_link_vel_w.torch
        return (pose[:, :3] - self.scene.env_origins, _rot(pose) if R is None else R, vel[:, :3].clone(), vel[:, 3:6].clone())

    def _states(self, with_context: bool):
        """``robot_state()``, ``object_state()`` and, with ``with_context``, ``context()``, with one rotation conversion
        for all their poses (the same values)."""
        poses = [self.robot.data.root_link_pose_w.torch] + ([] if self.hinged else [self.obj.data.root_link_pose_w.torch])
        k = len(poses)
        ctx_poses = self._context_poses() if with_context else []
        rots = _rots(*poses, *ctx_poses)
        S, So = self.robot_state(rots[0]), self.object_state(None if self.hinged else rots[1])
        return S, So, (self._context(ctx_poses, rots[k:]) if with_context else None)

    def transition_fields(self) -> dict:
        """One HOI transition's fields for ``ndm_gym.buffer.ReplayBuffer`` (``transition``): the robot's state before
        and after the control step (``ndm_gym.state`` layout), the object's (``flatten_object``, or a hinge's angle
        and rate), the filtered action of each physics substep (HDMI order, the PD targets' action part), the raw
        action, the context, the contact forces (each end effector's from the object, in the contact body's frame;
        the feet's after each substep, world frame), the joints averaged over substeps 2 and 3 in HDMI's order (what
        HDMI's observation histories push), the frame counter and episode step before the step, the flags."""
        f32 = torch.float32
        o = 2 if self.hinged else OBJ_DIM
        return {"s": (STATE_DIM, f32), "s1": (STATE_DIM, f32), "o": (o, f32), "o1": (o, f32), "u": ((DECIMATION, NUM_ACTIONS), f32),
                "a": (NUM_ACTIONS, f32), "ctx": (self.context().shape[-1], f32), "f_eef": ((self.cfg.num_eefs, 3), f32),
                "f_feet": ((DECIMATION, 2, 3), f32), "q_obs": (len(JOINT_NAMES), f32), "t": ((), torch.long), "ep_step": ((), torch.int32),
                "done": ((), torch.bool),
                "term": ((), torch.bool)}  # fmt: skip

    def ndm_constants(self) -> dict:
        """``ndm_constants(cfg)``, checked against the simulator's default pose."""
        c = ndm_constants(self.cfg)
        if not torch.allclose(torch.tensor(c["q0"], device=self.device), self.q0[self.j_g1]):
            raise RuntimeError("the task's default pose differs from the simulator's")
        return c

    def hinge_frames(self) -> dict:
        """A hinged object's joint frames from the physics model, which place its contact body relative to its fixed
        base: T_base,body(q) = X_p Rot(axis, q) X_c^-1 (positions in m, rotation matrices; Newton's revolute joint)."""
        if not self.hinged:
            raise ValueError(f"{self.cfg.name}: the object has no hinge")
        model = self.sim.backend_manager.get_model()
        get = lambda name: wp.to_torch(self.obj._root_view.get_attribute(name, model))[0].reshape(-1)  # noqa: E731  env 0, the one joint
        if int(get("joint_type")[0]) != 1:  # newton.JointType.REVOLUTE
            raise ValueError(f"{self.cfg.object_joint} is not a revolute joint")
        if list(self.obj.body_names).index(self.cfg.object_body) != self.obj_body_id or self.obj_body_id == 0:
            raise ValueError(f"the contact body {self.cfg.object_body} is not the hinge's child")
        xp, xc = get("joint_X_p"), get("joint_X_c")
        to_R = lambda x: quat_to_mat(torch.cat((x[6:7], x[3:6])).unsqueeze(0))[0]  # noqa: E731  xyzw
        return {"p_p": xp[:3].tolist(), "R_p": to_R(xp).tolist(), "axis": get("joint_axis")[:3].tolist(), "p_c": xc[:3].tolist(),
                "R_c": to_R(xc).tolist()}  # fmt: skip

    def reset_input(self, state: HDMIState) -> HDMIInput:
        """The step input a policy observes right after a reset: the reset state as both now and prev, the reference
        windows around the start frame, no action yet. HDMI itself returns all-zero observations there and its PPO drops
        those transitions (§0); model-based training keeps them, so the policy observes the reset state instead."""
        z = lambda *s: torch.zeros(self.N, *s, device=self.device)  # noqa: E731
        q, qd = state.joint_pos, state.joint_vel
        return HDMIInput(state, state, reference_window(self.clip, self.t - 1, self.future_steps), reference_window(self.clip, self.t, self.future_steps),
                         torch.stack((q, q), 1),
                         torch.stack((qd, qd), 1), self.offset[:, self.j_hdmi], z(NUM_ACTIONS), z(len(JOINT_NAMES)), z(NUM_ACTIONS, 3),
                         z(DECIMATION, 2, 3), z() if self.hinged else None)  # fmt: skip

    def flat_object(self, obj_state) -> torch.Tensor:
        return torch.cat(obj_state, -1) if self.hinged else flatten_object(obj_state)

    def transition(self, before: dict, action: torch.Tensor, x: HDMIInput, terminated: torch.Tensor, truncated: torch.Tensor) -> dict:
        """The transition of the control step just taken; ``before`` holds ``robot_state``, ``object_state``,
        ``context``, the frame counter ``t`` and ``ep_step``, read before ``step``."""
        S1, O1, _ = self._states(with_context=False)
        if self.hinged:
            (s, s1), o, o1 = flatten_states(before["S"], S1), self.flat_object(before["O"]), self.flat_object(O1)
        else:
            s, s1, o, o1 = flatten_states(before["S"], S1, before["O"], O1)
        return {"s": s, "s1": s1, "o": o, "o1": o1, "u": self.a_sub, "a": action, "ctx": before["ctx"], "f_eef": x.now.eef_force,
                "f_feet": x.feet_force, "q_obs": x.joint_pos_sub.mean(1), "t": before["t"], "ep_step": before["ep_step"].int(),
                "done": terminated | truncated,
                "term": terminated}  # fmt: skip

    def before_step(self) -> dict:
        S, So, ctx = self._states(with_context=True)
        return {"S": S, "O": So, "ctx": ctx, "t": self.t.clone(), "ep_step": self.ep_step.clone()}

    # ------------------------------------------------------------------ reading
    def read(self) -> HDMIState:
        """The state now, positions relative to the env origins, joints in HDMI's order."""
        d, o = self.robot.data, self.obj.data
        org = self.scene.env_origins
        bpose = d.body_link_pose_w.torch[:, self.kp_ids]  # (N, 16, 7)
        epose = d.body_link_pose_w.torch[:, self.eef_ids]
        opose = o.body_link_pose_w.torch[:, 0]  # the root link
        cpose = o.body_link_pose_w.torch[:, self.obj_body_id]  # the contact body
        bR, eR, oR, cR = _rots(bpose, epose, opose, cpose)
        bp, op, cp = bpose[..., :3] - org[:, None], opose[:, :3] - org, cpose[:, :3] - org
        # a link's angular velocity is its centre of mass's (body_link_ang_vel_w copies it), so one buffer gives both
        cv = d.body_com_vel_w.torch
        blv, bav = cv[:, self.kp_ids, :3], cv[:, self.kp_ids, 3:]
        q, qd = d.joint_pos.torch[:, self.j_hdmi], d.joint_vel.torch[:, self.j_hdmi]  # gathers: fresh tensors
        ep = epose[..., :3] - org[:, None] + (eR @ self.eef_offsets.unsqueeze(-1)).squeeze(-1)
        tp = cp[:, None] + (cR[:, None] @ (self.target_offsets * self.obj_scale[:, None]).unsqueeze(-1)).squeeze(-1)
        fw = torch.stack([_t(s.data.force_matrix_w)[:, 0, 0] for s in self.eef_sensors], 1)  # (N, E, 3) world
        fb = (fw.unsqueeze(-2) @ cR[:, None]).squeeze(-2)  # in the contact body's frame (R^T f)
        hq = hqd = None
        if self.hinged:
            hq, hqd = o.joint_pos.torch[:, 0].clone(), o.joint_vel.torch[:, 0].clone()
        return HDMIState(bp, bR, blv, bav, q, qd, op, oR, ep, tp, fb, hq, hqd)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        """HDMI's reset (§2) of ``env_ids``: start frames uniform in [0, L - 33] (training) or 0 (evaluation)."""
        if len(env_ids) == 0:
            return
        n, dev, c, r = len(env_ids), self.device, self.clip, self.cfg.reset
        if frames is None:
            frames = torch.zeros(n, dtype=torch.long, device=dev) if self.eval_mode else torch.randint(0, self.L - 32, (n,), device=dev)
        s = frames
        org = self.scene.env_origins[env_ids]
        if self.eval_mode:
            dp, dr, dv, dw = (torch.zeros(n, 3, device=dev) for _ in range(4))
        else:
            dp, dr, dv, dw = (_uniform(self._bounds[k], n) for k in "prvw")
        p = c.body_pos[s, 0] + dp
        R = c.body_rot[s, 0] @ _euler_xyz(dr)  # q_ref (x) quat_from_euler_xyz(delta)
        v, w = c.body_lin_vel[s, 0] + dv, c.body_ang_vel[s, 0] + dw  # the pelvis's reference COM velocity (§2)
        jq = self.ref_joint_pos_all[s][:, self.g1_of_robot]
        jqd = self.ref_joint_vel_all[s][:, self.g1_of_robot]
        jq = jq + r.joint_pos_noise * (2 * torch.rand_like(jq) - 1)
        jqd = jqd + r.joint_vel_noise * (2 * torch.rand_like(jqd) - 1)
        jq = torch.minimum(torch.maximum(jq, self.soft_lo), self.soft_hi)
        jqd = torch.minimum(torch.maximum(jqd, -self.vel_limit), self.vel_limit)
        # object and props: reference pose plus one draw, zero velocity (§2)
        odp, odr = _uniform(self._bounds["op"], n), _uniform(self._bounds["or"], n)
        dRo = _euler_xyz(odr)
        zero6 = torch.zeros(n, 6, device=dev)
        if self.hinged:
            hq = c.obj_joint_pos[s] + r.object_joint_pos_noise * (2 * torch.rand(n, device=dev) - 1)
            hqd = self.ref_obj_joint_vel[s] + r.object_joint_vel_noise * (2 * torch.rand(n, device=dev) - 1)
            hq = hq.clamp(*self.hinge_lim)
        extras = [(self.ref_extra_p[s, k] + odp + org, self.ref_extra_R[s, k] @ dRo) for k in range(len(self.extras))]
        robot_pose, obj_pose, *extra_poses = _poses((p + org, R), (c.obj_pos[s] + odp + org, c.obj_rot[s] @ dRo), *extras)
        self.robot.write_root_link_pose_to_sim_index(root_pose=robot_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=jq, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=jqd, env_ids=env_ids)
        self.obj.write_root_link_pose_to_sim_index(root_pose=obj_pose, env_ids=env_ids)
        if self.hinged:
            self.obj.write_joint_position_to_sim_index(position=hq.unsqueeze(-1), env_ids=env_ids)
            self.obj.write_joint_velocity_to_sim_index(velocity=hqd.unsqueeze(-1), env_ids=env_ids)
        else:
            self.obj.write_root_com_velocity_to_sim_index(root_velocity=zero6, env_ids=env_ids)
        for ex, pose in zip(self.extras, extra_poses):
            ex.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            ex.write_root_com_velocity_to_sim_index(root_velocity=zero6, env_ids=env_ids)
        # action pipeline and offsets (§3.1, §7)
        self.offset[env_ids] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.delay[env_ids] = torch.randint(2, 7, (n,), device=dev)
        self.alpha[env_ids] = torch.empty(n, 1, device=dev).uniform_(0.8, 1.0)
        self.action_buf.index_fill_(0, env_ids, 0.0)  # index_fill_: a scalar assignment by index copies it to the device and syncs
        self.a_filt.index_fill_(0, env_ids, 0.0)
        if self.hinged:
            rz = self.cfg.randomization
            self.hinge_f[env_ids] = torch.empty(n, device=dev).uniform_(*rz.hinge_friction_range)
            self.hinge_c[env_ids] = torch.empty(n, device=dev).uniform_(*rz.hinge_damping_range)
        target = self.q0.expand(n, -1) + self.offset[env_ids]
        self.robot.actuators.target_command.set_position_index(value=target, env_ids=env_ids)
        self.t[env_ids] = s
        self.ep_step.index_fill_(0, env_ids, 0)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states so that read() is consistent right after a reset."""
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor, prev: HDMIState) -> tuple[HDMIInput, torch.Tensor]:
        """One control step with ``action`` (N, 23) in HDMI's order; ``prev`` is the state after the previous step.
        Returns the task's input and the truncation flags (clip end or episode length, §0)."""
        if self.graph:
            return self._step_graph(action, prev)
        o = self._control_step(action)
        self.a_sub, self.action_buf, self.a_filt, self.t, self.ep_step = o["a_sub"], o["buf"], o["filt"], o["t"], o["ep_step"]
        x = HDMIInput(o["now"], prev, o["ref"], o["ref_next"], o["q_sub"], o["qd_sub"], o["offset"], o["filt"].clone(), o["applied"],
                      o["buf"].clone(), o["feet"], self.hinge_tau.clone() if self.hinged else None)  # fmt: skip
        return x, o["trunc"]

    def _control_step(self, action: torch.Tensor) -> dict:
        """The control step's computations: the action pipeline, the four substeps (PD targets, hinge torque, physics,
        the feet's forces, the joints of the last two), the state read, the references. The new action buffer,
        filtered action, frame counter and episode step are returned, not stored."""
        N = self.N
        t_prev = self.t.clone()
        a_sub, buf, filt = action_pipeline(action, self.action_buf, self.a_filt, self.delay, self.alpha)
        # the four substeps' PD targets at once (DECIMATION, N, 29): element by element the additions of one at a time
        base = self.q0.expand(N, -1) + self.offset
        targets = base.expand(DECIMATION, N, -1).index_add(2, self.j_act, (self.scale * a_sub).transpose(0, 1))
        feet, q_sub, qd_sub = [], [], []
        for i in range(DECIMATION):
            self.robot.actuators.target_command.set_position_index(value=targets[i])
            if self.hinged:
                hqd = self.obj.data.joint_vel.torch[:, 0]
                self.hinge_tau = (-torch.sign(hqd) * (hqd.abs() > 0.01) * self.hinge_f - self.hinge_c * hqd).clamp(-100.0, 100.0)
                self.obj.actuators.target_command.set_effort_index(value=self.hinge_tau.unsqueeze(-1))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._limit_joint_velocities()
            self.scene.update(PHYSICS_DT)
            feet.append(_t(self.feet_sensor.data.net_forces_w)[:, self.feet_order])
            if i >= DECIMATION - 2:
                q_sub.append(self.robot.data.joint_pos.torch[:, self.j_hdmi])
                qd_sub.append(self.robot.data.joint_vel.torch[:, self.j_hdmi])
        self._link_velocities()
        now = self.read()
        ref = reference_window(self.clip, t_prev - 1, self.future_steps)  # CU(n-1): frames t_prev + F - 1, index 0 = f_{n-1}
        t, ep_step = self.t + 1, self.ep_step + 1
        ref_next = reference_window(self.clip, t_prev, self.future_steps)  # CU(n)
        applied = self.robot.data.applied_torque.torch[:, self.j_hdmi]
        return {"now": now, "ref": ref, "ref_next": ref_next, "q_sub": torch.stack(q_sub, 1), "qd_sub": torch.stack(qd_sub, 1),
                "offset": self.offset[:, self.j_hdmi], "applied": applied, "feet": torch.stack(feet, 1), "a_sub": a_sub, "buf": buf,
                "filt": filt, "t": t, "ep_step": ep_step, "trunc": (t >= self.L) | (ep_step >= self.cfg.max_episode_length),
                "targets": targets}  # fmt: skip

    def _limit_joint_velocities(self):
        """HDMI's joint velocity limits after a physics substep (module docstring): the robot's joint velocities clamped in
        the simulation state itself (Isaac Lab's joint_vel is a view of it), on the device."""
        qd = self.robot.data.joint_vel.torch
        torch.clamp(qd, self._vel_lo, self.vel_limit, out=qd)

    def _link_velocities(self):
        """The links' velocities recomputed from the joints after the control step's clamps (forward kinematics, as Newton
        runs it after each step): their positions are unchanged, their velocities those of the clamped joints."""
        state = self.sim.backend_manager.get_state_0()
        eval_fk(self.sim.backend_manager.get_model(), state.joint_q, state.joint_qd, state)

    def _graph_body(self) -> dict:
        """_control_step on the fixed buffers: the action from _graph_action, the step's state updated in place."""
        o = self._control_step(self._graph_action)
        self.action_buf.copy_(o["buf"])
        self.a_filt.copy_(o["filt"])
        self.t.copy_(o["t"])
        self.ep_step.copy_(o["ep_step"])
        o["tau"] = self.hinge_tau if self.hinged else None
        return o

    def _step_graph(self, action: torch.Tensor, prev: HDMIState) -> tuple[HDMIInput, torch.Tensor]:
        """step() with the control step as one CUDA graph (isaaclab_mjwarp.capture_step): steps 1 and 2 run eagerly
        (the second on the capture stream), step 3 is captured, later steps replay it. Its outputs are copied out, so
        the caller gets fresh tensors as from the eager step."""
        self._graph_action.copy_(action)
        self._graph_steps += 1
        if self._graph_steps <= 2:
            o = self._graph_body() if self._graph_steps == 1 else on_capture_stream(self.sim, self._graph_body)
        else:
            before_replay(self.sim)
            first = self._graph is None
            if first:  # the capture runs the step's Python side once; the replay below does its GPU work
                self._graph, self._graph_out, self._graph_record = capture_step(self.sim, self._graph_body)
            self._graph.replay()
            if not first:
                after_replay(self.sim, self.scene, PHYSICS_DT, DECIMATION)
            o = self._graph_out
        c = lambda v: None if v is None else v.clone()  # noqa: E731
        self.a_sub = c(o["a_sub"])
        if self.hinged:
            self.hinge_tau = c(o["tau"])
        now, ref, ref_next = (type(o[k])(*(c(v) for v in o[k])) for k in ("now", "ref", "ref_next"))
        x = HDMIInput(now, prev, ref, ref_next, c(o["q_sub"]), c(o["qd_sub"]), c(o["offset"]), c(o["filt"]), c(o["applied"]), c(o["buf"]),
                      c(o["feet"]), c(o["tau"]))  # fmt: skip
        return x, c(o["trunc"])


__all__ = ["HDMISim", "make_hdmi_sim", "hdmi_clip", "load_clip", "exclude_prop_contacts", "check_prop_contacts", "place_scene", "FUTURE_STEPS"]
