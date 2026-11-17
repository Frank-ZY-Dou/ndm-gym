"""mbrl-lib's MuJoCo environments (facebookresearch/mbrl-lib at 3f93ccc, its code unmodified) as its true-dynamics controller
(``mbrl/diagnostics/control_env.py``) builds them, for planning on batched simulators: the models of benchmarks/mbrl_lib,
each environment's reward and termination as functions of batched states and actions (torch), and the run-time patches
that build the environments as the benchmark runs them, in this process and in control_env.py's pool workers:

- gymnasium 0.26.3's hopper.xml and walker2d.xml are in global coordinates, which MuJoCo >= 2.3.4 refuses; their
  local-coordinate copies (``local_models``, a directory with ``hopper_local.xml`` and ``walker2d_local.xml``) load instead.
  tools/convert_mbrl_models.py makes them, into data/mbrl_models; a missing copy is an error that names that command;
- a dm_control task steps PlaNet's action repeat (mbrl-lib's ``planet_*.yaml`` ``frame_skip``) per environment step, through
  the vendored dmc2gym's ``frame_skip``;
- with ``own_margins``, the contact margins of the MuJoCo mbrl-lib's own environment runs (2.x: a contact's margin is the
  larger of its two geoms' margins; MuJoCo 4.0.1 adds them, so gymnasium's Hopper touches the ground 1 mm earlier): each
  geom's margin halved (``set_own_margins``). Off by default, which keeps the models as the installed MuJoCo reads them;
- with ``own_mujoco``, the model as that MuJoCo (2.3.3) steps it (``set_own_mujoco``): these contact margins, its inverse
  weights of the bodies that move on slides only, whose joints' armature MuJoCo >= 3 leaves out (``set_own_invweight``), and
  its convex collision, libccd's with one contact per pair (``set_own_ccd``). The PETS pusher needs the three: its object
  is such a body, pushed by capsules against a cylinder. Off by default. Both options refuse a copy whose joint ranges
  are rounded (the copies of before that tool held six digits): its limits would not be that MuJoCo's.

A reward reads ``pre`` and ``post``, what the environment's MuJoCo data holds before and after the step: qpos and qvel, and
the derived arrays (xpos, xmat, site_xpos, subtree_linvel, cfrc_ext) where a reward reads them. gymnasium's MuJoCo
environments do not recompute kinematics after mj_step, so their derived arrays are those of the step's last physics step, not
of qpos (the PETS pusher's reward reads them before its step). dm_control recomputes them after each of its control steps
(mj_step1), and one dmc2gym step sums the rewards of its ``repeat`` control steps. The world model gives the derived arrays of
its own states (``Kinematics``), steps dm_control's control step ``repeat`` times per environment step, and gives the pusher the
kinematics of the state before the step (a named difference, ``PUSHER``); it gives no contact forces. What
each simulator refuses, with the reason: ``MbrlEnv.not_run`` (every simulator), ``ndm_refused``, ``warp_refused``.

``patch_envs(counter_fix=True)`` also fixes, at run time, mbrl-lib's dm_control step counter: its handler's
``set_env_state`` and ``FreezeDmcontrol`` restore ``_elapsed_steps`` on the outer gymnasium wrapper, not on gym's TimeLimit
inside dmc2gym, which therefore counts every rollout step of a pool worker and cuts all its rollouts to one step after
ceil(1000 / repeat) of them; the fix restores that TimeLimit's counter too."""

import contextlib
import dataclasses
import math
import os

import numpy as np
import torch

SLIDE, HINGE = 2, 3  # mjtJoint
PLANET = (1000, 12, 10)  # PlaNet's planner: population, horizon, iterations (mbrl-lib's planet_*.yaml)
LOCAL = {"hopper.xml": "hopper_local.xml", "walker2d.xml": "walker2d_local.xml"}
MAKE_COPIES = ("python tools/convert_mbrl_models.py --out {out}  (in mbrl-lib's own environment: its MuJoCo 2.3.3 still reads gymnasium's "
               "global-coordinate files)")  # fmt: skip


@dataclasses.dataclass(frozen=True)
class MbrlEnv:
    """One environment: ``env`` is mbrl-lib's name for it (control_env.py's ``--env``), ``episode`` its length in environment
    steps (the PETS override's trial_length, gymnasium's time limit, dm_control's divided by the repeat), ``pets`` the PETS
    override's (population, horizon, iterations), ``reads`` the fields its reward and termination read; ``not_run``,
    ``ndm_refused`` and ``warp_refused`` why it is not run at all, on the world model or on MuJoCo Warp; ``named`` the world
    model's named difference in its reward."""

    name: str
    env: str
    model: str
    episode: int
    reads: frozenset
    repeat: int = 1
    pets: tuple[int, int, int] | None = None
    override: str | None = None
    planar: bool = False
    not_run: str | None = None
    ndm_refused: str | None = None
    warp_refused: str | None = None
    named: str | None = None

    @property
    def dm_control(self) -> bool:
        return self.env.startswith("dmcontrol___")

    def settings(self) -> dict:
        """The optimizer settings to plan with: control_env.py's defaults, the PETS override's, PlaNet's."""
        out = {"control_env": (512, 30, 5)}
        if self.pets:
            out["pets"] = self.pets
        if self.dm_control:
            out["planet"] = PLANET
        return out


def _dmc(name, domain, task, repeat, reads, planar=False):
    return MbrlEnv(name, f"dmcontrol___{domain}--{task}", f"dm_control suite {domain}.xml ({task})", -(-1000 // repeat), frozenset(reads),
                   repeat=repeat, override=f"planet_{name[4:]}", planar=planar)  # fmt: skip


ANT_WARP = ("its contact forces on MuJoCo Warp (rne_postconstraint) do not equal CPU MuJoCo's cfrc_ext within 1e-4 "
            "relative on recorded transitions (check_mbrl_rewards --ant_forces): CPU MuJoCo only")
CONTACT = "its reward charges the contact forces cfrc_ext, which the world model does not predict (no world-model column)"
REACHER = ("pets_reacher is not run (it simulates pusher.xml, which pets_pusher covers, and its hidden goal is never synced "
           "to the pool's workers)")
PUSHER = ("a named difference: the reward reads the kinematics of the world model's state before the step, not MuJoCo's "
          "data of the previous step's last physics step")
ENVS = {
    e.name: e
    for e in [
        MbrlEnv("pets_cartpole", "pets_cartpole", "mbrl-lib cartpole.xml", 200, frozenset({"post.qpos"}), pets=(500, 30, 5),
                override="pets_cartpole_paper_version"),
        MbrlEnv("pets_halfcheetah", "pets_halfcheetah", "mbrl-lib half_cheetah.xml", 1000, frozenset({"pre.qpos", "post.qpos"}), pets=(400, 30, 5),
                override="pets_halfcheetah", planar=True),
        MbrlEnv("pets_pusher", "pets_pusher", "mbrl-lib pusher.xml", 150, frozenset({"pre.xpos"}), pets=(350, 25, 5), override="pets_pusher",
                named=PUSHER),
        MbrlEnv("pets_reacher", "pets_reacher", "mbrl-lib pusher.xml (pets_reacher)", 1000, frozenset({"post.qpos", "goal"}), pets=(350, 15, 5),
                override="pets_reacher", not_run=REACHER),
        MbrlEnv("hopper", "gym___Hopper-v4", "gymnasium hopper.xml (local coordinates)", 1000, frozenset({"pre.qpos", "post.qpos", "post.qvel"}),
                pets=(350, 30, 5), override="pets_hopper", planar=True),
        MbrlEnv("walker2d", "gym___Walker2d-v4", "gymnasium walker2d.xml (local coordinates)", 1000, frozenset({"pre.qpos", "post.qpos"}),
                planar=True),
        MbrlEnv("inverted_pendulum", "gym___InvertedPendulum-v4", "gymnasium inverted_pendulum.xml", 1000, frozenset({"post.qpos", "post.qvel"}),
                pets=(480, 45, 5), override="pets_inv_pendulum"),
        MbrlEnv("ant", "ant_truncated_obs", "gymnasium ant.xml (MBPO's Ant)", 1000,
                frozenset({"pre.xpos", "post.xpos", "post.qpos", "post.qvel", "post.cfrc_ext"}), override="mbpo_ant", ndm_refused=CONTACT,
                warp_refused=ANT_WARP),
        MbrlEnv("humanoid", "humanoid_truncated_obs", "gymnasium humanoid.xml (MBPO's Humanoid)", 1000,
                frozenset({"pre.xipos", "post.xipos", "post.cfrc_ext"}), override="mbpo_humanoid", ndm_refused=CONTACT,
                warp_refused="MuJoCo Warp does not support its PGS solver (CPU MuJoCo only)"),
        _dmc("dmc_cartpole_balance", "cartpole", "balance", 8, {"post.xmat", "post.qpos", "post.qvel"}),
        _dmc("dmc_cartpole_swingup", "cartpole", "swingup", 8, {"post.xmat", "post.qpos", "post.qvel"}),
        _dmc("dmc_cheetah_run", "cheetah", "run", 4, {"post.subtree_linvel"}, planar=True),
        _dmc("dmc_cup_catch", "ball_in_cup", "catch", 6, {"post.xpos", "post.site_xpos"}),
        _dmc("dmc_finger_spin", "finger", "spin", 2, {"post.qvel"}),
        _dmc("dmc_walker_walk", "walker", "walk", 2, {"post.xpos", "post.xmat", "post.subtree_linvel"}, planar=True),
    ]
}  # fmt: skip
REPEAT = {e.env.split("___")[1].split("--")[0]: e.repeat for e in ENVS.values() if e.dm_control}  # dm_control domain: repeat


def refusal(spec: MbrlEnv, sim: str) -> str | None:
    """Why ``spec`` is not run on ``sim`` (cpu, warp or ndm), or None."""
    if sim not in ("cpu", "warp", "ndm"):
        raise ValueError(f"refusal: no simulator {sim!r}")
    return spec.not_run or {"cpu": None, "warp": spec.warp_refused, "ndm": spec.ndm_refused}[sim]


# ---------------- the run-time patches (no file of mbrl-lib's or gymnasium's changed) ----------------
_PATCHED: dict = {}


def contact_margin_rule() -> str:
    """How the installed MuJoCo sets a contact's margin from its two geoms' margins: ``"max"`` (the larger, MuJoCo 2.x) or
    ``"sum"``, read off a sphere on a plane whose margins are both 0.01."""
    import mujoco

    xml = ('<mujoco><default><geom margin="0.01"/></default><worldbody><geom type="plane" size="1 1 1"/>'
           '<body pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>')  # fmt: skip
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if d.ncon != 1:
        raise RuntimeError(f"contact_margin_rule: {d.ncon} contacts of a sphere resting on a plane")
    margin = float(d.contact[0].includemargin)
    for rule, value in (("max", 0.01), ("sum", 0.02)):
        if abs(margin - value) < 1e-12:
            return rule
    raise RuntimeError(f"contact_margin_rule: a contact margin of {margin} from two geom margins of 0.01")


def set_own_margins(m):
    """The contact margins of mbrl-lib's own MuJoCo on the model ``m``, in place: where the installed MuJoCo adds the two
    geoms' margins, every geom's margin is halved, which gives each contact the larger of the two when all the model's geoms
    share one margin (gymnasium's Hopper, Walker2d, Ant and Humanoid; a model without margins stays as it is). A model whose
    geoms' margins differ is refused. Where the installed MuJoCo already takes the larger margin, nothing changes."""
    if contact_margin_rule() == "max":
        return
    g = m.geom_margin
    if m.ngeom and (g != g[0]).any():
        raise ValueError(f"set_own_margins: the geoms' margins differ ({sorted(set(g.tolist()))}): their halves do not give the larger of two")
    g[:] *= 0.5


def set_own_invweight(m):
    """MuJoCo 2.3.3's ``body_invweight0`` and ``dof_invweight0`` on the model ``m``, in place, for the bodies the installed
    MuJoCo marks as moving on slides only (``body_simple`` 2, MuJoCo >= 3). It gives them the inverse of the body's mass,
    without the joints' armature; MuJoCo 2.3.3 computes the general values, the mean diagonal of J M^-1 J^T at qpos0 with the
    armature in M, and so does this. The solver scales a constraint's regularization by these weights: the PETS pusher's
    object (1.3e-8 kg on two slides with an armature of 0.04) got contacts 4.4e6 times softer. The other bodies' values are
    the same in both versions and stay; a model without such a body stays as it is."""
    import mujoco

    simple = [b for b in range(m.nbody) if m.body_simple[b] == 2]
    if not simple:
        return
    d = mujoco.MjData(m)  # at qpos0
    mujoco.mj_forward(m, d)
    minv = np.zeros((m.nv, m.nv))
    mujoco.mj_solveM(m, d, minv, np.eye(m.nv))
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for b in simple:
        mujoco.mj_jacBodyCom(m, d, jacp, jacr, b)
        m.body_invweight0[b] = np.trace(jacp @ minv @ jacp.T) / 3, np.trace(jacr @ minv @ jacr.T) / 3
        for a in range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b]):  # slides: one degree of freedom each
            m.dof_invweight0[a] = minv[a, a]


def set_own_ccd(m):
    """MuJoCo 2.3.3's convex collision on the model ``m``, in place: libccd's penetration with one contact per pair and 50
    iterations. MuJoCo 4.0.1 runs its native pipeline and returns several contacts for a pair unless both are disabled
    (mjDSBL_NATIVECCD, mjDSBL_MULTICCD). It matters for the pairs without an analytic collision function (the PETS pusher's
    capsules against its cylinder). MuJoCo Warp has its own pipeline and reads the second flag only. A MuJoCo without the
    native pipeline (2.x) keeps its model; one with it and without the second flag is refused."""
    import mujoco

    bits = mujoco.mjtDisableBit
    if not hasattr(bits, "mjDSBL_NATIVECCD"):
        return
    if not hasattr(bits, "mjDSBL_MULTICCD"):
        raise RuntimeError(f"set_own_ccd: MuJoCo {mujoco.__version__} has a native convex collision and no mjDSBL_MULTICCD")
    m.opt.disableflags |= int(bits.mjDSBL_NATIVECCD) | int(bits.mjDSBL_MULTICCD)
    m.opt.ccd_iterations = 50


def set_own_mujoco(m):
    """The model ``m`` as mbrl-lib's own MuJoCo (2.3.3) steps it, in place: its contact margins, its inverse weights of
    slider bodies and its convex collision."""
    set_own_margins(m)
    set_own_invweight(m)
    set_own_ccd(m)


def reads_global_coordinates() -> bool:
    """Whether the installed MuJoCo compiles a model in global coordinates (up to 2.3.3) or refuses it."""
    import mujoco

    try:
        mujoco.MjModel.from_xml_string('<mujoco><compiler coordinate="global"/><worldbody/></mujoco>')
    except ValueError:
        return False
    return True


def check_exact_ranges(m, original: str, copy: str):
    """Refuse the model ``m`` of a local-coordinate ``copy`` whose joint ranges are not the ``original`` file's: every joint
    the original gives a range (in degrees for a hinge, as gymnasium's files do) must hold it within 1e-12. The copies made
    before tools/convert_mbrl_models.py hold ranges rounded to six digits (1.6e-7 rad off at 45 degrees)."""
    import xml.etree.ElementTree as ET

    import mujoco

    root = ET.parse(original).getroot()
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") == "degree"
    for joint in root.iter("joint"):
        name, rng = joint.get("name"), joint.get("range")
        if name is None or rng is None:
            continue
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError(f"check_exact_ranges: {copy} has no joint {name} of {original}")
        want = [float(x) for x in rng.split()]
        if degrees and joint.get("type", "hinge") == "hinge":
            want = [math.radians(x) for x in want]
        if np.abs(np.asarray(want) - m.jnt_range[j]).max() > 1e-12:
            raise ValueError(f"check_exact_ranges: {copy} holds rounded joint ranges ({name}: {m.jnt_range[j].tolist()} for {want}); own_margins and "
                             f"own_mujoco need the exact copies, made with: {MAKE_COPIES.format(out=os.path.dirname(copy))}")  # fmt: skip


def patch_envs(local_models: str | None, counter_fix: bool = False, own_margins: bool = False, own_mujoco: bool = False):
    """Build the benchmark's environments from here on in this process: gymnasium's hopper.xml and walker2d.xml from the
    local-coordinate copies in ``local_models`` (None: gymnasium's own, which MuJoCo >= 2.3.4 refuses), dm_control tasks at
    PlaNet's action repeat; a copy that is missing is an error that names the command that makes it; with ``counter_fix``,
    mbrl-lib's dm_control step counter restored on gym's TimeLimit too; with
    ``own_margins``, the gymnasium MuJoCo environments' contact margins as mbrl-lib's own MuJoCo sets them, and with
    ``own_mujoco`` their models as it steps them (the module's docstring; both refused for dm_control tasks). Idempotent;
    ``unpatch_envs`` undoes it."""
    from gymnasium.envs.mujoco import mujoco_env

    import mbrl.third_party.dmc2gym as dmc2gym

    if _PATCHED:
        if (_PATCHED["local_models"], _PATCHED["counter_fix"], _PATCHED["own_margins"], _PATCHED["own_mujoco"]) != (local_models, counter_fix, own_margins, own_mujoco):
            raise RuntimeError(f"patch_envs: already patched with {_PATCHED['local_models']!r}, counter_fix={_PATCHED['counter_fix']}, "
                               f"own_margins={_PATCHED['own_margins']}, own_mujoco={_PATCHED['own_mujoco']}")  # fmt: skip
        return
    if local_models is not None:
        for f in LOCAL.values():
            if not os.path.isfile(os.path.join(local_models, f)):
                raise FileNotFoundError(f"patch_envs: no {f} in {local_models}; make the copies with: {MAKE_COPIES.format(out=local_models)}")
    init, make = mujoco_env.MujocoEnv.__init__, dmc2gym.make

    def mujoco_init(self, model_path, *args, **kwargs):
        original = None
        if model_path in LOCAL:
            if local_models is not None:
                original = os.path.join(os.path.dirname(mujoco_env.__file__), "assets", model_path)
                model_path = os.path.join(os.path.abspath(local_models), LOCAL[model_path])
            elif not reads_global_coordinates():
                raise FileNotFoundError(f"patch_envs: this MuJoCo does not read gymnasium's {model_path} (global coordinates) and no directory of "
                                        f"its local-coordinate copy is given (--local_models); make the copies with: "
                                        f"{MAKE_COPIES.format(out='data/mbrl_models')}")  # fmt: skip
        init(self, model_path, *args, **kwargs)
        if original is not None and (own_margins or own_mujoco):
            check_exact_ranges(self.model, original, model_path)
        if own_mujoco:
            set_own_mujoco(self.model)
        elif own_margins:
            set_own_margins(self.model)

    def dmc_make(domain_name, task_name, **kwargs):
        if domain_name not in REPEAT or "frame_skip" in kwargs:
            raise ValueError(f"patch_envs: dm_control {domain_name}/{task_name} with {kwargs}: not a benchmark task at its action repeat")
        if own_margins or own_mujoco:
            raise ValueError(f"patch_envs: {'own_mujoco' if own_mujoco else 'own_margins'} is for the gymnasium MuJoCo environments, "
                             f"not dm_control's {domain_name}/{task_name}")  # fmt: skip
        return make(domain_name=domain_name, task_name=task_name, frame_skip=REPEAT[domain_name], **kwargs)

    undo = [lambda: setattr(mujoco_env.MujocoEnv, "__init__", init), lambda: setattr(dmc2gym, "make", make)]
    mujoco_env.MujocoEnv.__init__, dmc2gym.make = mujoco_init, dmc_make
    if counter_fix:
        from mbrl.util.dmcontrol import DmcontrolEnvHandler, FreezeDmcontrol

        set_state, leave = DmcontrolEnvHandler.__dict__["set_env_state"], FreezeDmcontrol.__exit__

        def set_env_state(state, env):
            set_state.__func__(state, env)
            env.unwrapped.gym_env._elapsed_steps = state[1]  # gym's TimeLimit inside dmc2gym, whose count truncates the rollouts

        def freeze_exit(self, *args):
            leave(self, *args)
            self._env.unwrapped.gym_env._elapsed_steps = self._elapsed_steps

        DmcontrolEnvHandler.set_env_state, FreezeDmcontrol.__exit__ = staticmethod(set_env_state), freeze_exit
        undo += [lambda: setattr(DmcontrolEnvHandler, "set_env_state", set_state), lambda: setattr(FreezeDmcontrol, "__exit__", leave)]
    _PATCHED.update(local_models=local_models, counter_fix=counter_fix, own_margins=own_margins, own_mujoco=own_mujoco, undo=undo)


def unpatch_envs():
    if _PATCHED:
        for f in _PATCHED.pop("undo"):
            f()
        _PATCHED.clear()


def worker_init(local_models, counter_fix, own_margins, own_mujoco, init, initargs):
    """A pool worker's initializer: the patches, then control_env.py's own ``init``."""
    patch_envs(local_models, counter_fix, own_margins, own_mujoco)
    init(*initargs)


def make_env(spec: MbrlEnv, seed: int):
    """(handler, env) as control_env.py builds its evaluation environment: mbrl-lib's handler for the name, global numpy
    seeded (the PETS environments reset from it), then reset with the seed. Call patch_envs first."""
    import mbrl.util

    if not _PATCHED:
        raise RuntimeError("make_env: patch_envs() first")
    handler = mbrl.util.create_handler_from_str(spec.env)
    env = handler.make_env_from_str(spec.env)
    np.random.seed(seed)
    env.reset(seed=seed)
    return handler, env


def physics(spec: MbrlEnv, env):
    """(MjModel, MjData, physics steps per control step, control steps per environment step) of the environment's simulation."""
    if spec.dm_control:
        dmc = env.unwrapped.gym_env.unwrapped.env._env
        return dmc.physics.model._model, dmc.physics.data._data, int(dmc._n_sub_steps), spec.repeat
    u = env.unwrapped
    return u.model, u.data, int(u.frame_skip), 1


def control_step_seconds(spec: MbrlEnv, env) -> float:
    """The control step: the world model's step and the recordings' (dm_control: its own control step, inside the repeat)."""
    m, _, substeps, _ = physics(spec, env)
    return float(m.opt.timestep) * substeps


def state(spec: MbrlEnv, handler, env) -> dict:
    """What a batched rollout needs of the environment's current state: qpos, qvel, the environment steps left before its time
    limit truncates a rollout (gymnasium's TimeLimit counts as control_env.py's rollouts do), for dm_control the control steps
    left before its own time limit, and the reacher's goal (a hidden attribute the reward reads)."""
    m, d, _, _ = physics(spec, env)
    st = handler.get_current_state(env)
    out = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy()}
    if spec.dm_control:
        tl, dmc = env.unwrapped.gym_env, env.unwrapped.gym_env.unwrapped.env._env
        out["steps_left"] = max(1, int(tl._max_episode_steps) - int(st[1]))
        out["control_left"] = int(math.ceil(dmc._step_limit)) - int(dmc._step_count)
    else:
        out["steps_left"] = max(1, int(env._max_episode_steps) - int(st[1]))
    if "goal" in spec.reads:
        out["goal"] = np.asarray(env.unwrapped.goal, dtype=np.float64).copy()
    return out


# ---------------- fields: what a reward reads of MuJoCo's data ----------------
class Fields:
    """Batched views (B, ...) of MuJoCo data arrays as torch tensors: qpos, qvel, and on request xpos, xmat (B, nbody, 3, 3),
    xipos, site_xpos, subtree_linvel, cfrc_ext. ``get`` raises for an array its source does not give."""

    def __init__(self, **arrays):
        self.arrays = arrays

    def __getattr__(self, name):
        arrays = self.__dict__.get("arrays", {})
        if name not in arrays:
            raise AttributeError(f"Fields: no {name} in this source ({sorted(arrays)})")
        return arrays[name]

    @staticmethod
    def of_mjdata(m, d, names=("qpos", "qvel", "xpos", "xmat", "xipos", "site_xpos", "subtree_linvel", "cfrc_ext")) -> "Fields":
        """A copy of one CPU MjData's arrays, as a batch of one (float64)."""
        out = {}
        for k in names:
            x = np.array(getattr(d, k), dtype=np.float64)
            out[k] = torch.from_numpy(x.reshape(-1, 3, 3) if k == "xmat" else x)[None]
        return Fields(**out)


class Kinematics:
    """MuJoCo's forward kinematics of a tree of hinge and slide joints in torch, batched over qpos (..., nq): the bodies'
    positions and rotations (xpos, xmat (..., nbody, 3, 3)), their centres of mass (xipos) and the sites' positions
    (site_xpos) as mj_kinematics computes them, and with qvel each subtree's centre-of-mass velocity (subtree_linvel, as
    mj_subtreeVel): the derived arrays a reward reads, of the world model's states. Free and ball joints are refused."""

    NAMES = ("xpos", "xmat", "xipos", "site_xpos", "subtree_linvel")

    def __init__(self, m, device="cpu", dtype=torch.float32):
        import mujoco

        if any(m.jnt_type[j] not in (SLIDE, HINGE) for j in range(m.njnt)):
            raise ValueError("Kinematics: the model has free or ball joints (hinges and slides only)")
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        rot = np.zeros((m.nbody, 9))
        for b in range(m.nbody):
            mujoco.mju_quat2Mat(rot[b], m.body_quat[b])
        self.parent = [int(p) for p in m.body_parentid]
        self.joints = [list(range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])) for b in range(m.nbody)]
        self.slide = [bool(m.jnt_type[j] == SLIDE) for j in range(m.njnt)]
        self.qadr, self.dadr, self.jbody = ([int(x) for x in a] for a in (m.jnt_qposadr, m.jnt_dofadr, m.jnt_bodyid))
        self.body_pos, self.body_rot, self.body_ipos = t(m.body_pos), t(rot.reshape(-1, 3, 3)), t(m.body_ipos)
        self.jnt_pos, self.jnt_axis, self.qpos0, self.mass = t(m.jnt_pos), t(m.jnt_axis), t(m.qpos0), t(m.body_mass)
        self.site_body, self.site_pos = torch.as_tensor(m.site_bodyid, dtype=torch.long, device=device), t(m.site_pos)
        sub = np.zeros((m.nbody, m.nbody))  # sub[b, c]: body c is in body b's subtree
        for c in range(m.nbody):
            b = c
            while True:
                sub[b, c] = 1.0
                if b == 0:
                    break
                b = int(m.body_parentid[b])
        self.sub = t(sub)
        self.skew = [t(np.cross(np.eye(3), a)) for a in np.asarray(m.jnt_axis, dtype=np.float64)]  # K x = axis x x

    def __call__(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None, names=NAMES) -> dict:
        from ..core.kinematics import mat_mat, mat_vec

        lead = qpos.shape[:-1]
        eye = torch.eye(3, dtype=qpos.dtype, device=qpos.device)
        xpos, xmat, axes, anchors = [qpos.new_zeros(lead + (3,))], [eye.expand(lead + (3, 3))], {}, {}
        for b in range(1, len(self.parent)):
            p = self.parent[b]
            pos, R = xpos[p] + mat_vec(xmat[p], self.body_pos[b]), mat_mat(xmat[p], self.body_rot[b])
            for j in self.joints[b]:  # as mj_kinematics: the axis and anchor in the frame so far, then the joint's motion
                axes[j], anchors[j] = mat_vec(R, self.jnt_axis[j]), pos + mat_vec(R, self.jnt_pos[j])
                x = (qpos[..., self.qadr[j]] - self.qpos0[self.qadr[j]])[..., None, None]
                if self.slide[j]:
                    pos = pos + axes[j] * x[..., 0]
                else:  # the body turns about its local axis (Rodrigues), about the anchor
                    K = self.skew[j]
                    R = mat_mat(R, eye + torch.sin(x) * K + (1 - torch.cos(x)) * mat_mat(K, K))
                    pos = anchors[j] - mat_vec(R, self.jnt_pos[j])
            xpos.append(pos)
            xmat.append(R)
        xpos, xmat = torch.stack(xpos, -2), torch.stack(xmat, -3)
        out = {"xpos": xpos, "xmat": xmat, "xipos": xpos + mat_vec(xmat, self.body_ipos)}
        out["site_xpos"] = xpos[..., self.site_body, :] + mat_vec(xmat[..., self.site_body, :, :], self.site_pos)
        if "subtree_linvel" in names:  # each body's centre-of-mass velocity, J qvel, then the subtrees' mass-weighted means
            v = torch.zeros_like(out["xipos"])
            for j, a in axes.items():
                w = (a * qvel[..., self.dadr[j]][..., None])[..., None, :]
                lin = w.expand_as(v) if self.slide[j] else torch.cross(w.expand_as(v), out["xipos"] - anchors[j][..., None, :], dim=-1)
                v = v + lin * self.sub[self.jbody[j]][:, None]
            mom = torch.einsum("bc,...cd->...bd", self.sub, v * self.mass[:, None])
            out["subtree_linvel"] = mom / (self.sub @ self.mass).clamp_min(1e-15)[:, None]  # mjMINVAL, as mj_subtreeVel
        return {k: out[k] for k in names}


# ---------------- the rewards and terminations ----------------
def tolerance(x, lower=0.0, upper=0.0, margin=0.0, sigmoid="gaussian", value_at_margin=0.1):
    """dm_control's rewards.tolerance (dm_control/utils/rewards.py) for the sigmoids the benchmark's tasks use."""
    inside = (lower <= x) & (x <= upper)
    if margin == 0:
        return torch.where(inside, 1.0, 0.0).to(x.dtype)
    d = torch.where(x < lower, lower - x, x - upper) / margin
    if sigmoid == "gaussian":
        v = torch.exp(-0.5 * (d * math.sqrt(-2 * math.log(value_at_margin))) ** 2)
    elif sigmoid == "linear":
        s = d * (1 - value_at_margin)
        v = torch.where(s.abs() < 1, 1 - s, torch.zeros_like(s))
    elif sigmoid == "quadratic":
        s = d * math.sqrt(1 - value_at_margin)
        v = torch.where(s.abs() < 1, 1 - s**2, torch.zeros_like(s))
    else:
        raise ValueError(f"tolerance: sigmoid {sigmoid!r} is not one the benchmark's tasks use")
    return torch.where(inside, torch.ones_like(v), v)


def _reacher_tip(q):
    """pets_reacher's hand-written end effector (mbrl/env/pets_reacher.py, get_EE_pos) of qpos[:, :6]."""
    t1, t2, t3, t4, t5, t6 = (q[:, i : i + 1] for i in range(6))
    rot_axis = torch.cat((torch.cos(t2) * torch.cos(t1), torch.cos(t2) * torch.sin(t1), -torch.sin(t2)), 1)
    rot_perp = torch.cat((-torch.sin(t1), torch.cos(t1), torch.zeros_like(t1)), 1)
    end = torch.cat((0.1 * torch.cos(t1) + 0.4 * torch.cos(t1) * torch.cos(t2), 0.1 * torch.sin(t1) + 0.4 * torch.sin(t1) * torch.cos(t2) - 0.188,
                     -0.4 * torch.sin(t2)), 1)  # fmt: skip
    for length, hinge, roll in ((0.321, t4, t3), (0.16828, t6, t5)):
        perp_all = torch.linalg.cross(rot_axis, rot_perp)
        new_axis = torch.cos(hinge) * rot_axis + torch.sin(hinge) * torch.sin(roll) * rot_perp - torch.sin(hinge) * torch.cos(roll) * perp_all
        new_perp = torch.linalg.cross(new_axis, rot_axis)
        new_perp = torch.where(new_perp.norm(dim=1, keepdim=True) < 1e-30, rot_perp, new_perp)
        rot_axis, rot_perp, end = new_axis, new_perp / new_perp.norm(dim=1, keepdim=True), end + length * new_axis
    return end


def _ids(m, kind, *names):
    import mujoco

    out = []
    for n in names:
        i = mujoco.mj_name2id(m, kind, n)
        if i < 0:
            raise ValueError(f"mbrl_envs: the model has no {kind} {n!r}")
        out.append(i)
    return out


class Scorer:
    """An environment's reward and termination of a batch of steps: ``reward(pre, post, ctrl, goal)`` and ``terminated(post)``
    for one environment step (gymnasium environments) or one control step (dm_control, whose environment step sums
    ``repeat`` of them), ctrl (B, nu) the action as the environment received it. Model constants are read once from ``m``
    onto ``device``; ``substeps``: the physics steps of one control step (a gymnasium environment's frame skip, whose dt it is)."""

    def __init__(self, spec: MbrlEnv, m, substeps: int, device="cpu"):
        import mujoco

        self.spec, B, J, S, G = spec, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_GEOM
        self.dt = float(m.opt.timestep) * substeps
        if spec.name == "pets_pusher":
            self.obj, self.tip, self.goal = _ids(m, B, "object", "tips_arm", "goal")
        elif spec.name.startswith("dmc_cartpole"):
            (self.slider,) = (int(m.jnt_qposadr[j]) for j in _ids(m, J, "slider"))
        elif spec.name in ("dmc_cheetah_run", "dmc_walker_walk"):
            (self.torso,) = _ids(m, B, "torso")
        elif spec.name == "dmc_cup_catch":
            (self.ball,), (self.target,), (g,) = _ids(m, B, "ball"), _ids(m, S, "target"), _ids(m, G, "ball")
            self.room = torch.tensor(m.site_size[self.target][[0, 2]] - m.geom_size[g][0], dtype=torch.float64, device=device)
        elif spec.name == "dmc_finger_spin":
            (self.hinge,) = (int(m.jnt_dofadr[j]) for j in _ids(m, J, "hinge"))
        elif spec.name == "ant":
            (self.torso,) = _ids(m, B, "torso")
        if spec.name == "humanoid":
            self.reward = self.terminated = self._refuse

    def _refuse(self, *args, **kwargs):
        raise NotImplementedError(f"{self.spec.name}: no batched reward, it plans on CPU MuJoCo only")

    def reward(self, pre: Fields, post: Fields, ctrl: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        n, a2 = self.spec.name, (ctrl**2).sum(-1)
        if n == "pets_cartpole":  # mbrl/env/pets_cartpole.py: the pole tip against its upright position
            x, th = post.qpos[:, 0], post.qpos[:, 1]
            return torch.exp(-((x - 0.6 * torch.sin(th)) ** 2 + (-0.6 * torch.cos(th) - 0.6) ** 2) / 0.6**2) - 0.01 * a2
        if n == "pets_halfcheetah":  # obs[0], the x velocity over the step, less 0.1 |a|^2 (and 0.0 obs[2]^2)
            return (post.qpos[:, 0] - pre.qpos[:, 0]) / self.dt - 0.0 * post.qpos[:, 2] ** 2 + -0.1 * a2
        if n == "pets_pusher":  # the bodies' positions as the data holds them before the step
            obj = pre.xpos[:, self.obj]
            near, dist = -(obj - pre.xpos[:, self.tip]).abs().sum(-1), -(obj - pre.xpos[:, self.goal]).abs().sum(-1)
            return 1.25 * dist + 0.1 * -a2 + 0.5 * near
        if n == "pets_reacher":
            return -((_reacher_tip(post.qpos) - goal) ** 2).sum(-1) - 0.01 * a2
        if n in ("hopper", "walker2d"):  # forward velocity + the healthy reward (1, terminate_when_unhealthy) - 1e-3 |a|^2
            return ((post.qpos[:, 0] - pre.qpos[:, 0]) / self.dt + 1.0) - 1e-3 * a2
        if n == "inverted_pendulum":
            return torch.ones_like(a2)
        if n == "ant":  # mbrl/env/ant_truncated_obs.py: the torso's x as the data holds it before and after, contact forces after
            forward = (post.xpos[:, self.torso, 0] - pre.xpos[:, self.torso, 0]) / self.dt
            contact = 0.5 * 1e-3 * (post.cfrc_ext.clamp(-1, 1) ** 2).sum((-1, -2))
            return forward - 0.5 * a2 - contact + 1.0
        if n.startswith("dmc_cartpole"):  # cartpole.py, the smooth reward
            upright = (post.xmat[:, 2, 2, 2] + 1) / 2
            centered = (1 + tolerance(post.qpos[:, self.slider], margin=2)) / 2
            small_control = (4 + tolerance(ctrl[:, 0], margin=1, value_at_margin=0, sigmoid="quadratic")) / 5
            small_velocity = (1 + tolerance(post.qvel[:, 1:], margin=5).min(-1).values) / 2
            return upright * small_control * small_velocity * centered
        if n == "dmc_cheetah_run":
            return tolerance(post.subtree_linvel[:, self.torso, 0], 10.0, math.inf, margin=10, value_at_margin=0, sigmoid="linear")
        if n == "dmc_cup_catch":
            gap = (post.site_xpos[:, self.target] - post.xpos[:, self.ball])[:, [0, 2]].abs()
            return (gap < self.room.to(gap.dtype)).all(-1).to(gap.dtype)
        if n == "dmc_finger_spin":
            return (post.qvel[:, self.hinge] <= -15.0).to(post.qvel.dtype)
        if n == "dmc_walker_walk":
            standing = tolerance(post.xpos[:, self.torso, 2], 1.2, math.inf, margin=0.6)
            stand = (3 * standing + (1 + post.xmat[:, self.torso, 2, 2]) / 2) / 4
            move = tolerance(post.subtree_linvel[:, self.torso, 0], 1.0, math.inf, margin=0.5, value_at_margin=0.5, sigmoid="linear")
            return stand * (5 * move + 1) / 6
        raise ValueError(f"Scorer: no reward for {n}")

    def terminated(self, post: Fields) -> torch.Tensor:
        n, q = self.spec.name, post.qpos
        if n == "hopper":  # Hopper-v4's is_healthy: the state (qpos[2:], qvel) within 100, z above 0.7, the angle within 0.2
            s = torch.cat((q[:, 2:], post.qvel), -1)
            return ~(((-100.0 < s) & (s < 100.0)).all(-1) & (0.7 < q[:, 1]) & (q[:, 1] < math.inf) & (-0.2 < q[:, 2]) & (q[:, 2] < 0.2))
        if n == "walker2d":
            return ~((0.8 < q[:, 1]) & (q[:, 1] < 2.0) & (-1.0 < q[:, 2]) & (q[:, 2] < 1.0))
        if n == "inverted_pendulum":
            ob = torch.cat((q, post.qvel), -1)
            return ~torch.isfinite(ob).all(-1) | (ob[:, 1].abs() > 0.2)
        if n == "ant":
            return ~(torch.isfinite(torch.cat((q, post.qvel), -1)).all(-1) & (q[:, 2] >= 0.2) & (q[:, 2] <= 1.0))
        return torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)


@contextlib.contextmanager
def recording_dm_control(spec: MbrlEnv, env, out: list, names=None):
    """While active, each dm_control control step inside the environment's steps appends (MjData fields after it, ``names``
    or all of Fields.of_mjdata's, ctrl, dm_control's reward) to ``out``: the states one dmc2gym step passes through."""
    dmc = env.unwrapped.gym_env.unwrapped.env._env
    step = dmc.step
    m, d = dmc.physics.model._model, dmc.physics.data._data

    def recorded(action):
        ts = step(action)
        if not ts.first():  # dm_control resets on the step after its last one: no control step
            out.append((Fields.of_mjdata(m, d) if names is None else Fields.of_mjdata(m, d, names), np.array(d.ctrl, dtype=np.float64), ts.reward))
        return ts

    dmc.step = recorded
    try:
        yield
    finally:
        del dmc.step
