"""Motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`ndm_gym.tasks.tracking`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations


import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from ...core.rotations import mat_to_quat, quat_to_mat, so3_exp
from ...motion.library import MotionLib
from ...robots.g1 import DEFAULT_JOINT_POS, G1_JOINT_NAMES, USD_PATH, action_scale, kd, kp, motor_of
from ...robots.objects import ObjectSpec
from ...state import CONTROL_DT
from ...tasks.tracking import MDP


def flatten_states(*states) -> list[torch.Tensor]:
    """``ndm_gym.state.flatten_state`` of each robot state ``S`` and ``flatten_object`` of each object state ``So``
    (told apart by their length) with one quaternion conversion for all: mat_to_quat works row by row, so each result
    is bit for bit the one of its own call."""
    q = mat_to_quat(torch.cat([s[1].reshape(-1, 3, 3) for s in states]))
    out, i = [], 0
    for s in states:
        n = s[1].shape[:-2].numel()
        out.append(torch.cat((s[0], q[i:i + n].view(*s[1].shape[:-2], 4), *s[2:]), -1))
        i += n
    return out


# ---------------------------------------------------------------- simulation steps as CUDA graphs
# A function of the scene (Isaac Lab writes and reads, Newton physics, torch math, no host synchronization) is captured
# once and replayed: the same kernels on the same buffers, so the results are those of running it. A replay skips the
# function's Python side, which the caller restores: before_replay and after_replay for the simulation's, and state
# kept in fixed tensors updated in place.
def _capture_stream(sim) -> torch.cuda.Stream:
    """Warp's stream as a torch stream: torch's and Warp's work of a capture on one stream."""
    return torch.cuda.ExternalStream(wp.get_stream(sim.device).cuda_stream, device=sim.device)


def on_capture_stream(sim, fn):
    """``fn()`` run eagerly on the stream ``capture_step`` records on (the warm-up a capture needs first: cuBLAS
    workspaces, lazily allocated buffers)."""
    cap = _capture_stream(sim)
    cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        out = fn()
    torch.cuda.current_stream().wait_stream(cap)
    return out


def capture_step(sim, fn):
    """``fn()`` captured in a CUDA graph: torch's capture on Warp's stream with Warp's registered as external, so that
    MuJoCo-Warp's solver loops become conditional nodes; Newton's own physics graph is replaced by its kernels for the
    capture. Returns the graph, ``fn``'s outputs (graph memory: every replay rewrites them) and Warp's record of the
    capture (to be kept alive with the graph)."""
    mgr = sim.backend_manager
    stream = wp.get_stream(sim.device)
    newton_graph, mgr._graph = mgr._graph, None
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=_capture_stream(sim), capture_error_mode="thread_local"):
            wp.capture_begin(stream=stream, external=True, capture_mode=wp.CaptureMode.THREAD_LOCAL)
            try:
                out = fn()
            finally:
                record = wp.capture_end(stream=stream)
    finally:
        mgr._graph = newton_graph
    return graph, out, record


def before_replay(sim):
    """What sim.step does first that a replay cannot: apply pending model changes (refused: they need a new capture)
    and the forward kinematics of the envs written since the last step (run here, as sim.step would)."""
    mgr = sim.backend_manager
    if mgr._model_changes:
        raise RuntimeError("the physics model changed after the step was captured")
    if mgr.kinematics_dirty:
        sim.forward()


def after_replay(sim, scene, dt: float, n: int):
    """The Python side of n (sim.step, scene.update(dt)) that a replay skips: the assets' simulation timestamps (their
    lazily derived quantities compare against them) and the simulation's counters."""
    mgr = sim.backend_manager
    for _ in range(n):
        for asset in (*scene.articulations.values(), *scene.rigid_objects.values()):
            asset.data._sim_timestamp += dt
        sim._physics_step_count += 1
        mgr._sim_time += dt


def lazy_now(data, name: str):
    """Isaac Lab asset data's lazily derived quantity ``name`` (e.g. root_link_vel_w) recomputed now, by the property's
    own code. Inside a capture Isaac Lab refuses the property, since a replay skips its timestamp check; marked stale
    first, its kernel is recorded and every replay recomputes it."""
    getattr(data, "_" + name).timestamp = -1.0
    return getattr(type(data), name).fget.__wrapped__(data)


def _actuators() -> dict[str, ImplicitActuatorCfg]:
    acts = {}
    for name in G1_JOINT_NAMES:
        effort, vel, arm = motor_of(name)
        acts[name.replace("_joint", "")] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=effort,
            velocity_limit_sim=vel,
            stiffness=kp(arm),
            damping=kd(arm),
            armature=arm,
        )
    return acts


G1_CFG = ArticulationCfg(
    # USD generated by tools/urdf_to_mjcf.py + tools/mjcf_to_usd.py (the kit-less importers cannot
    # register PhysxSchema inside a running Isaac Lab process, so conversion is a separate step).
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        variants={"Physics": "mujoco"},
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators=_actuators(),
)


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def object_cfg(spec: ObjectSpec) -> RigidObjectCfg:
    """The interacted object as a free rigid cuboid with its nominal (evaluation) mass and friction."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5 * spec.size[2])),
    )


@configclass
class ObjectSceneCfg(TrackingSceneCfg):
    """The tracking scene with one interacted object (human-object interaction)."""

    obj: RigidObjectCfg = None


@configclass
class PairSceneCfg(TrackingSceneCfg):
    """The tracking scene with a second G1 (two-person clips). Within an env the two robots collide, as bodies of two
    articulations do; each robot's self-collisions stay off. The second robot spawns away from the first; resets place
    both."""

    robot_b = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/RobotB", init_state=replace(G1_CFG.init_state, pos=(2.0, 0.0, 0.76)))


def make_sim(device: str, args, paired: bool = False) -> SimulationContext:
    """``paired``: the two-robot scene (PairSceneCfg), with room for twice the constraints and contacts per env."""
    njmax, nconmax = (1000, 256) if paired else (500, 128)
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=njmax, nconmax=nconmax, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True),
        num_substeps=4,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT, device=device, physics=physics)


class TrackingSim:
    """Batched G1 motion-tracking simulator.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from :func:`make_sim`.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, eval_friction: float | None = None, eval_payload: float = 0.0,
                 obj: ObjectSpec | None = None):
        if motion.has_object != (obj is not None) or (obj is not None and obj.name != motion.obj_name):
            raise ValueError(f"the scene's object ({obj.name if obj else None}) must be the motion library's ({motion.obj_name})")
        if obj is not None and not getattr(mdp, "uses_object", False):
            raise ValueError("a scene with an object needs a task that scores the object (uses_object = True)")
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        if obj is None:
            self.scene = instantiate(TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True))
        else:
            self.scene = instantiate(ObjectSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True, obj=object_cfg(obj)))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.obj_spec = obj
        self.object = self.scene["obj"] if obj is not None else None
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        if not eval_mode and (eval_friction is not None or eval_payload):
            raise ValueError("eval_friction and eval_payload change the evaluation's dynamics; they need eval_mode")
        self.eval_friction = 1.0 if eval_friction is None else eval_friction  # evaluation: the same friction everywhere
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] ----
        self.xi = torch.zeros(self.N, 33, device=dev)
        self._setup_friction()
        if eval_mode:
            self.xi[:, 0] = self.eval_friction
            if eval_payload:  # extra mass on the torso, inertia unchanged; not part of xi (training never varies the mass)
                mass = self.robot.data.body_mass.torch[:, self.torso_id].clone() + eval_payload
                self.robot.set_masses_index(masses=mass.unsqueeze(1), body_ids=[self.torso_id])
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 30:33] = com
            default_com = self.robot.data.body_com_pose_b.torch[:, self.torso_id, :3].clone()
            self.robot.set_coms_index(coms=(default_com + com).unsqueeze(1), body_ids=[self.torso_id])
        self._apply_friction()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = mdp.max_step_reward / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reward_fn = mdp.reward  # the step's reward without an object; train.py --compile_eager sets its compiled version
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        mgr = self.sim.backend_manager
        model = mgr.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # ground shapes (body == -1): lowest friction of the range, since MuJoCo takes the max of a pair
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = 0.3 if not self.eval_mode else self.eval_friction

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ state
    def state(self, vel=None):
        """``vel``: the root link velocity to read instead of Isaac Lab's property, which a CUDA graph capture refuses
        (step(inplace=True) and evaluate --graph pass the property's own buffer, recomputed or current)."""
        d = self.robot.data
        pose = d.root_link_pose_w.torch  # read at once: the position difference and the quaternion are new tensors
        vel = (d.root_link_vel_w if vel is None else vel).torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # Isaac Lab 3 uses xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (p, R, vel[:, :3], vel[:, 3:6], d.joint_pos.torch.clone(), d.joint_vel.torch.clone())

    def object_state(self):
        """The object's state ``So = (p, R, v, w)``, positions relative to the environment origin."""
        d = self.object.data
        pose = d.root_link_pose_w.torch
        vel = d.root_link_vel_w.torch.clone()
        quat = pose[:, 3:7]  # xyzw
        return pose[:, :3] - self.scene.env_origins, quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1)), vel[:, :3], vel[:, 3:6]

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        last = m.clip_end[m.bin_clip[bins]]
        end = torch.minimum(start + 50, last)  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        # a clip of 50 k + 1 frames has a last bin of one frame, its last: a start there would step past the clip, so it
        # starts one frame earlier (no LAFAN1 clip has such a length, so their resets are unchanged)
        return torch.minimum(f, last - 1), bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        if self.object is not None:
            po, Ro, vo, wo = (x.clone() for x in m.ref_object(frames))
            if self.reset_noise:  # as HDMI: +-5 cm horizontally, +-0.1 rad about the vertical
                po = po + torch.stack((torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                                       torch.zeros(n, device=dev)), -1)  # fmt: skip
                Ro = so3_exp(torch.stack((torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.empty(n, device=dev).uniform_(-0.1, 0.1)), -1)) @ Ro
            oq = mat_to_quat(Ro)
            opose = torch.cat((po + self.scene.env_origins[env_ids], oq[:, 1:4], oq[:, 0:1]), -1)
            self.object.write_root_link_pose_to_sim_index(root_pose=opose, env_ids=env_ids)
            self.object.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((vo, wo), -1), env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)  # index_fill_: a scalar assignment by index copies it to the device and syncs
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor, S: tuple | None = None, inplace: bool = False) -> dict:
        """One control step with setpoints ``a``. ``S``: ``state()`` if the caller has just read it (the same values,
        one read fewer); None reads it here. ``inplace`` (evaluate --graph; evaluation mode, the caller's S, no object):
        the frame counter, episode step and previous action are updated in place, not replaced, and the state after the
        step recomputes the root velocity by Isaac Lab's own code (lazy_now), so that the step can run in a CUDA graph
        (the same values)."""
        if inplace and (not self.eval_mode or S is None or self.object is not None):
            raise ValueError("inplace steps are for evaluation without an object, with the state passed")
        S = self.state() if S is None else S
        So = self.object_state() if self.object is not None else None
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target_index(target=target)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        S1 = self.state(lazy_now(self.robot.data, "root_link_vel_w")) if inplace else self.state()
        So1 = self.object_state() if self.object is not None else None
        f1 = f + 1
        if So1 is None:
            r, terms = self.reward_fn(S1, f1, a, a_prev, terms=True)
            term = self.mdp.termination(S1, f1)
        else:
            r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True, So=So1)
            term = self.mdp.termination(S1, f1, So=So1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        flat = flatten_states(S, S1) if So is None else flatten_states(S, S1, So, So1)  # one quaternion conversion for all
        out = {
            "s": flat[0], "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flat[1], "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        if So is not None:
            out["o"], out["o1"] = flat[2], flat[3]
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        if inplace:
            self.f.copy_(f1)
            self.ep_step.copy_(ep_step + 1)
            self.a_prev.copy_(a)
        else:
            self.f = f1
            self.ep_step = ep_step + 1
            self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * self.mdp.max_step_reward)
            self.reset(ids)
        return out


def motion_pairs(motion: MotionLib) -> torch.Tensor:
    """(2, pairs) clip ids of each two-person clip's sides (``meta`` "pair" and "side", ``motion.sources.interhuman``),
    in the order of their first sides; every clip must be one side of one complete pair, both sides of equal length."""
    sides = {}
    for i, meta in enumerate(motion.meta):
        if not meta or "pair" not in meta or meta.get("side") not in (0, 1):
            raise ValueError(f"clip {motion.names[i]} is not one side of a two-person clip")
        if (meta["pair"], meta["side"]) in sides:
            raise ValueError(f"pair {meta['pair']} has two clips for side {meta['side']}")
        sides[(meta["pair"], meta["side"])] = i
    pairs = sorted((p for p, s in sides if s == 0), key=lambda p: sides[(p, 0)])
    if sorted(pairs) != sorted(p for p, s in sides if s == 1):
        raise ValueError("every pair needs both sides")
    ids = torch.tensor([[sides[(p, s)] for p in pairs] for s in (0, 1)])
    if not torch.equal(motion.clip_len[ids[0]].cpu(), motion.clip_len[ids[1]].cpu()):
        raise ValueError("a pair's two sides have different lengths")
    return ids


def pair_bins(motion: MotionLib, pair_clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pairs' reset bins: every first side's bin, the second side's bin at the same time (equal lengths: the same
    bins) and the frame offset from the first side's clip to the second's."""
    ba, bb, off = [], [], []
    for ca, cb in pair_clips.T.tolist():
        a, b = (motion.bin_clip == ca).nonzero().squeeze(-1), (motion.bin_clip == cb).nonzero().squeeze(-1)
        assert len(a) == len(b), (ca, cb)
        ba.append(a), bb.append(b), off.append(torch.full_like(a, int(motion.clip_start[cb] - motion.clip_start[ca])))
    return torch.cat(ba), torch.cat(bb), torch.cat(off)


class PairedTrackingSim(TrackingSim):
    """Two G1s per env, each tracking its own side of a two-person clip, in one scene where they collide
    (:class:`PairSceneCfg`). The rest of the pipeline sees 2P tracking instances: instance i < P is env i's first robot
    (side 1 of its pair), instance P + i its second (side 2). A pair resets together, to one frame of one two-person clip
    (return-based bin sampling over the first sides' bins, a bin weighing the mean of its two sides' returns), and in
    training an episode ends for both robots when either terminates or times out. Reward and termination are each
    robot's own (``tasks.tracking``); each robot's sim context (friction, joint offsets, torso CoM) is drawn on its own.
    Evaluation plays each instance to its own end or termination, the other robot still stepping.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from ``make_sim(..., paired=True)``.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, eval_friction: float | None = None, eval_payload: float = 0.0):
        if num_envs % 2:
            raise ValueError(f"num_envs counts tracking instances, two per env: {num_envs} is odd")
        if motion.has_object:
            raise ValueError("the two-robot scene has no object")
        self.pair_clips = motion_pairs(motion).to(sim_cfg.device)  # (2, pairs)
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.P = num_envs // 2
        self.scene = instantiate(PairSceneCfg(num_envs=self.P, env_spacing=0.0, replicate_physics=True))
        self.sim.reset()
        self.robot, self.robot_b = self.scene["robot"], self.scene["robot_b"]
        self.obj_spec, self.object = None, None
        assert list(self.robot.joint_names) == G1_JOINT_NAMES == list(self.robot_b.joint_names), (self.robot.joint_names, self.robot_b.joint_names)
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        if not eval_mode and (eval_friction is not None or eval_payload):
            raise ValueError("eval_friction and eval_payload change the evaluation's dynamics; they need eval_mode")
        self.eval_friction = 1.0 if eval_friction is None else eval_friction
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev, P = self.device, self.P
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        assert self.robot_b.find_bodies("torso_link")[0][0] == self.torso_id

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)], per instance ----
        self.xi = torch.zeros(self.N, 33, device=dev)
        self._setup_friction()
        if eval_mode:
            self.xi[:, 0] = self.eval_friction
            if eval_payload:
                for r in (self.robot, self.robot_b):
                    mass = r.data.body_mass.torch[:, self.torso_id].clone() + eval_payload
                    r.set_masses_index(masses=mass.unsqueeze(1), body_ids=[self.torso_id])
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 30:33] = com
            for r, c in ((self.robot, com[:P]), (self.robot_b, com[P:])):
                default_com = r.data.body_com_pose_b.torch[:, self.torso_id, :3].clone()
                r.set_coms_index(coms=(default_com + c).unsqueeze(1), body_ids=[self.torso_id])
        self._apply_friction()

        self.pbin_a, self.pbin_b, self.pbin_off = pair_bins(motion, self.pair_clips)

        # ---- episode bookkeeping, per instance ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = mdp.max_step_reward / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((len(self.pbin_a),), 1.0 / len(self.pbin_a), device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reward_fn = mdp.reward
        self.reset(torch.arange(self.N, device=dev))

    @property
    def instance_clip(self) -> torch.Tensor:
        """(2P,) the clip each instance plays from its first frame in evaluation: env i's robots play pair i (the pairs
        cycle when there are more envs than pairs)."""
        k = torch.arange(self.P, device=self.device) % self.pair_clips.shape[1]
        return torch.cat((self.pair_clips[0, k], self.pair_clips[1, k]))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        super()._setup_friction()  # the first robots' shapes and the ground
        self._mu_view_b = wp.to_torch(self.robot_b._root_view.get_attribute("shape_material_mu", self.sim.backend_manager.get_model())[:, 0])

    def _apply_friction(self):
        self._mu_view[:] = self.xi[: self.P, :1]
        self._mu_view_b[:] = self.xi[self.P :, :1]
        self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ state
    def _robot_state(self, r):
        d = r.data
        pose = d.root_link_pose_w.torch
        vel = d.root_link_vel_w.torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (p, R, vel[:, :3], vel[:, 3:6], d.joint_pos.torch.clone(), d.joint_vel.torch.clone())

    def state(self, vel=None):
        """Both robots' states, the first robots' rows first (instances 0..P-1, then P..2P-1)."""
        if vel is not None:
            raise ValueError("the two-robot scene reads its own root velocities (no CUDA-graph steps)")
        return tuple(torch.cat(x) for x in zip(self._robot_state(self.robot), self._robot_state(self.robot_b)))

    def object_state(self):
        raise ValueError("the two-robot scene has no object")

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """n pairs' start frames and bins: (first sides' frames, second sides' frames, their bins, their bins);
        TrackingSim's sampling over the first sides' bins, a bin's return the mean of its two sides'."""
        m = self.motion
        G = 0.5 * (self.bin_G[self.pbin_a] + self.bin_G[self.pbin_b])
        soft = torch.softmax(-25.0 * G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        nb = len(self.pbin_a)
        probs = 0.5 / nb + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        k = torch.multinomial(probs, n, replacement=True)
        bins = self.pbin_a[k]
        start = m.bin_start[bins]
        last = m.clip_end[m.bin_clip[bins]]
        end = torch.minimum(start + 50, last)
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        f = torch.minimum(f, last - 1)
        return f, f + self.pbin_off[k], bins, self.pbin_b[k]

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        """``env_ids``: instances. Without ``frames`` they must be both robots of each of their envs, which reset together
        to a sampled frame; with ``frames``, each instance to its frame (evaluation: each robot from its clip's start)."""
        if len(env_ids) == 0:
            return
        dev, P = self.device, self.P
        if frames is None:
            envs = env_ids[env_ids < P]
            if len(env_ids) != 2 * len(envs) or not torch.equal(torch.sort(env_ids[env_ids >= P]).values - P, torch.sort(envs).values):
                raise ValueError("an env's two robots reset together")
            fa, fb, ba, bb = self.sample_frames(len(envs))
            env_ids, frames, bins = torch.cat((envs, envs + P)), torch.cat((fa, fb)), torch.cat((ba, bb))
        else:
            bins = self.motion.frame_bin[frames]
        n = len(env_ids)
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        for r, sel, off in ((self.robot, env_ids < P, 0), (self.robot_b, env_ids >= P, P)):
            k = sel.nonzero().squeeze(-1)
            if len(k) == 0:
                continue
            e = env_ids[k] - off
            pose = torch.cat((p[k] + self.scene.env_origins[e], quat[k, 1:4], quat[k, 0:1]), -1)
            r.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=e)
            r.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v[k], w[k]), -1), env_ids=e)
            r.write_joint_position_to_sim_index(position=q[k], env_ids=e)
            r.write_joint_velocity_to_sim_index(velocity=qd[k], env_ids=e)
            r.set_joint_position_target_index(target=q[k], env_ids=e)
        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.scene.reset(torch.unique(env_ids % P))

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor, S: tuple | None = None, inplace: bool = False) -> dict:
        """One control step of both robots with setpoints ``a`` (2P, 29), the first robots' rows first. In training an
        env's episode ends for both robots when either terminates or times out (``done``; ``term`` stays each robot's)."""
        if inplace:
            raise ValueError("the two-robot scene has no in-place (CUDA-graph) steps")
        P = self.P
        S = self.state() if S is None else S
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target_index(target=target[:P])
        self.robot_b.set_joint_position_target_index(target=target[P:])
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.reward_fn(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        own = term | timeout
        done = (own[:P] | own[P:]).repeat(2)
        flat = flatten_states(S, S1)
        out = {
            "s": flat[0], "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flat[1], "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= own
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # each robot's bin return estimate; a robot whose partner ended the episode bootstraps as truncated
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * self.mdp.max_step_reward)
            self.reset(ids)
        return out
