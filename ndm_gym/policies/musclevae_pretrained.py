"""MuscleVAE's released tracking policy (Feng, Xu and Liu, SIGGRAPH Asia 2023: the pretrained low-level model linked from
the README of wangchek/MuscleVAE_ @ 54c001c) in torch, batched over environments, driving ndm_gym's muscle simulator
(:class:`~ndm_gym.sim.rigid.muscle_sim.MuscleSim`). Their repository has no license: nothing is copied from it, the
inference is re-implemented from its code (references below), and their checkpoint is fetched for validation only by
tools/download_musclevae_pretrained.sh into data/musclevae/pretrained (git-ignored, never redistributed). It is a pickled
torch checkpoint: :func:`load_checkpoint` checks its SHA-256 and opens it with ``torch.load(weights_only=True)``, with
numpy's array reconstruction allowed for its one numpy array (their reset weights, ``balance``).

The policy, as their tracking playground runs it (TS:14-23) with the configuration of the checkpoint's folder (CFG):

* Observation (MU:204-234), 371 values of the body states in the root's frame, in their axes (Y up: their (x, y, z)
  is our (y, z, x), the transpose of musclevae_char.Y_UP_TO_Z_UP): positions relative to the root (23 x 3), rotations
  relative to the root as the first two columns of the rotation matrix, row by row (23 x 6, DQ:550-555), linear and
  angular velocities (23 x 3 each), the bodies' heights (23) and the up direction (3). The state's velocities are
  backward differences over the control step (MU:40-50, CFG env_recompute_velocity, ENV:388), except right after a
  reset, where they are the simulator's (ENV:296). Then the muscle state (ENV:307-309, 391-394; MU:76-85): the body
  parts' f0-weighted MA, MR and MF (15, ndm_gym.robots.muscles.MuscleModel.part_fatigue).
* Normalization (MV:344-364, PU:88-89): the 371 values (x - mean) / (std + 1e-8) with the statistics of their motion
  dataset stored in the checkpoint (MDS:33-52); the muscle state is not normalized; 386 inputs.
* Target (ENV:243-244, 320, 352): the observation of their dataset's reference state in the frame the step tracks, the
  same normalization. Their dataset places the bodies by the BVH joints (the clips' ``body_pos``, ``body_quat_wxyz``,
  ndm_gym.motion.sources.musclevae.load_placement) with velocities by backward differences at 20 Hz, the first frame's
  those of the second (BE:14001-14059, 18795, 18858-18861): :func:`reference_observations`.
* Prior and posterior (MOD:9-87): each an MLP with skip inputs, ELU, three hidden layers of 512 (one input layer and
  CFG encoder_hidden_layer_num 2) and a linear mean, the prior on the observation, the posterior on the target with the
  observation (its first layer) and the target (every later layer); the latent's mean is the prior's mean plus the
  posterior's (MOD:83-87), its standard deviation fixed at 0.3 (CFG encoder_fix_var, MOD:21-23, 41-44). Tracking
  decodes the mean (TS:19-20); their data collection samples it, and with probability 0.4 a step takes the prior's
  sample instead (TC:53-61).
* Decoder (MOD:197-258): a mixture of six experts gated by an MLP (450-64-64-6, ELU, softmax) on the latent and the
  observation; four expert layers (512, 512, 512, 284) on the layer-normalized concatenation of the latent with the
  previous layer's output (the observation for the first), ELU between them and none at the end (CFG env_step_mode
  fatigue_dynamics_constrain has no "hill_type", MOD:209-212). Their collection adds Gaussian noise of std 0.05 to the
  action (MV:72, 401-407; TC:53-61).
* Action (ENV:371, 431-433): the 284 muscles' PD targets (1 + a) l_Tpose, MuscleSim's action as it stands; the muscles
  in their loader's order (sorted by muscleID, LDR:138), the converter's.

:class:`PolicyRunner` runs the policy and the simulator's control step as one CUDA graph: the observation is read from
the simulator's buffers, the action written into its action buffer, the random draws come from the simulator's
generator (registered with the graph), and nothing synchronizes with the host.

References are to wangchek/MuscleVAE_ @ 54c001c: MV = MuscleVAECore/Model/musclevae.py, MOD = MuscleVAECore/Model/
modules.py, MU = MuscleVAECore/Utils/motion_utils.py, MDS = MuscleVAECore/Utils/motion_dataset.py, PU =
MuscleVAECore/Utils/pytorch_utils.py, DQ = MuscleVAECore/Utils/diff_quat.py, ENV = MuscleVAECore/Env/muscle_env.py, TC =
MuscleVAECore/Model/trajectory_collection.py, TS = PlayGround/track_something.py, LDR = Muscle/muscle_scene_loader.py,
BE = ModifyODESrc/VclSimuBackend.pyx; CFG = config.yml of the pretrained model's folder (Data/NNModel/config.yml but for
its gpu index).
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import quat_to_mat
from ..motion.sources import musclevae as clip_source
from ..robots import musclevae_char as mc
from ..robots.muscles import NUM_PARTS

PRETRAINED_DIR = os.path.join(mc.DATA_DIR, "pretrained", "low_level")
CHECKPOINT_PATH = os.path.join(PRETRAINED_DIR, "trained.data")
CHECKPOINT_SHA256 = "29537eb79bd67845972dd072c9f1191587023163c22bcad38a8e90d95f777816"
CHECKPOINT_KEYS = ("self", "balance", "train_iter")
OBS_DIM = 16 * mc.NUM_BODIES + 3  # 371
POLICY_OBS_DIM = OBS_DIM + 3 * NUM_PARTS  # 386: with the muscle state
LATENT_DIM = 64  # CFG latent_size
HIDDEN = 512  # CFG encoder_hidden_layer_size, actor_hidden_layer_size
ENCODER_LAYERS = 3  # one input layer and CFG encoder_hidden_layer_num 2 (MOD:14-17)
DECODER_LAYERS = 4  # CFG actor_hidden_layer_num 3 and the output layer (MOD:215)
NUM_EXPERTS, GATE_HIDDEN = 6, 64  # CFG actor_num_experts, actor_gate_hidden_layer_size
LATENT_STD = 0.3  # CFG encoder_fix_var: the latent's standard deviation (MOD:41-44 with logvar 2 log 0.3)
ACTION_NOISE = 0.05  # their collection's action noise (MV:72, TC:53-61)
PRIOR_PROB = 0.4  # their collection's probability of a step from the prior (TC:59-61)
NORM_EPS = 1e-8  # PU:88-89
LATENT_MODES = ("mean", "sample")


def _their_axes(v: torch.Tensor) -> torch.Tensor:
    """Vectors (..., 3) in our axes to theirs: their (x, y, z) = our (y, z, x)."""
    return torch.cat((v[..., 1:], v[..., :1]), -1)


def observation(pos: torch.Tensor, rot: torch.Tensor, vel: torch.Tensor, avel: torch.Tensor) -> torch.Tensor:
    """Their observation (MU:204-234, 371 values) of body states in our frame: positions (centers of mass) (..., 23, 3),
    rotations (..., 23, 3, 3), linear and angular velocities (..., 23, 3); body 0 is the root. In their axes: a vector v
    in the root's frame is C^T R_root^T v, a relative rotation C^T R_root^T R C, with C = musclevae_char.Y_UP_TO_Z_UP."""
    Rt = rot[..., :1, :, :].transpose(-1, -2)
    lp, lv, la = (_their_axes(mat_vec(Rt, v)) for v in (pos - pos[..., :1, :], vel, avel))
    L = mat_mat(Rt, rot)
    # the first two columns of C^T L C, row by row: (C^T L C)_ij = L_p(i)p(j), p = (1, 2, 0)
    six = torch.stack((L[..., 1, 1], L[..., 1, 2], L[..., 2, 1], L[..., 2, 2], L[..., 0, 1], L[..., 0, 2]), -1)
    up = _their_axes(rot[..., 0, 2, :])  # R_root^T e_z, their up axis in the root's frame
    return torch.cat((lp.flatten(-2), six.flatten(-2), lv.flatten(-2), la.flatten(-2), pos[..., 2], up), -1)


def velocities(pos: torch.Tensor, quat: torch.Tensor, pos0: torch.Tensor, quat0: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Their backward-difference velocities from the poses ``pos0``, ``quat0`` to ``pos``, ``quat`` over ``dt``
    (MU:40-50; BE:14001-14059 on their aligned quaternions): (pos - pos0) / dt, and the vector part of 2 dq q0* with
    dq = (s q - q0) / dt and s = -1 if q . q0 < 0 else 1. Quaternions xyzw, the world frame."""
    s = 1.0 - 2.0 * ((quat * quat0).sum(-1, keepdim=True) < 0.0).to(quat.dtype)
    dq = (s * quat - quat0) / dt
    v0, w0 = -quat0[..., :3], quat0[..., 3:]  # q0*
    dv, dw = dq[..., :3], dq[..., 3:]
    return (pos - pos0) / dt, 2.0 * (dw * v0 + w0 * dv + torch.cross(dv, v0, dim=-1))


def reference_observations(names: tuple[str, ...]) -> torch.Tensor:
    """Their dataset's observations (F, 371) of the named clips concatenated frame by frame, as MuscleSim's Reference
    orders them, in double precision on the CPU: the clips' placement of the bodies with velocities by backward
    differences at 20 Hz, a clip's first frame taking those of its second (BE:12453-12458)."""
    rows = []
    for name in names:
        path = os.path.join(mc.CLIPS_DIR, f"{name}.npz")
        if clip_source.load_clip(path).fps != mc.CONTROL_HZ:
            raise ValueError(f"{path}: expected {mc.CONTROL_HZ} fps")
        pos, quat = (x.double() for x in clip_source.load_placement(path))
        xyzw = torch.cat((quat[..., 1:], quat[..., :1]), -1)
        vel, avel = velocities(pos[1:], xyzw[1:], pos[:-1], xyzw[:-1], 1.0 / mc.CONTROL_HZ)
        vel, avel = torch.cat((vel[:1], vel)), torch.cat((avel[:1], avel))
        rows.append(observation(pos, quat_to_mat(quat), vel, avel))
    return torch.cat(rows)


# ---- the checkpoint
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: str = CHECKPOINT_PATH, expected_sha256: str | None = CHECKPOINT_SHA256) -> dict:
    """Their checkpoint ({"self": the agent's state dict, "balance": their per-frame reset values, "train_iter"}), checked
    against its SHA-256 (None: not checked, for checkpoints written by tests) and opened by torch.load(weights_only=True)
    with numpy's float64 array reconstruction allowed for "balance", under numpy 1's name (theirs) and numpy 2's."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing: run tools/download_musclevae_pretrained.sh")
    if expected_sha256 is not None and (digest := sha256(path)) != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {expected_sha256}")
    reconstruct = np._core.multiarray._reconstruct
    numpy_array = [(reconstruct, "numpy.core.multiarray._reconstruct"), reconstruct, np.ndarray, np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(numpy_array):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or sorted(ck) != sorted(CHECKPOINT_KEYS):
        raise ValueError(f"{path}: expected a dict with keys {CHECKPOINT_KEYS}")
    return ck


def checkpoint_shapes() -> dict[str, tuple[int, ...]]:
    """The policy's tensors in their agent's state dict (MV:44-59) and their shapes."""
    shapes = {"obs_mean": (OBS_DIM,), "obs_std": (OBS_DIM,)}
    for enc, (x, c) in (("prior", (POLICY_OBS_DIM, 0)), ("post", (OBS_DIM, POLICY_OBS_DIM))):
        for i in range(ENCODER_LAYERS):
            shapes[f"encoder.{enc}.fc_layers.{i}.weight"] = (HIDDEN, x + (c if i == 0 else HIDDEN))
            shapes[f"encoder.{enc}.fc_layers.{i}.bias"] = (HIDDEN,)
        shapes[f"encoder.{enc}.mu.weight"], shapes[f"encoder.{enc}.mu.bias"] = (LATENT_DIM, x + HIDDEN), (LATENT_DIM,)
    for i in range(DECODER_LAYERS):
        out = HIDDEN if i < DECODER_LAYERS - 1 else mc.NUM_MUSCLES
        shapes[f"agent.w{i}"] = (NUM_EXPERTS, LATENT_DIM + (POLICY_OBS_DIM if i == 0 else HIDDEN), out)
        shapes[f"agent.b{i}"] = (NUM_EXPERTS, out)
    for i, (n_in, n_out) in zip((0, 2, 4), ((LATENT_DIM + POLICY_OBS_DIM, GATE_HIDDEN), (GATE_HIDDEN, GATE_HIDDEN), (GATE_HIDDEN, NUM_EXPERTS))):
        shapes[f"agent.gate.{i}.weight"], shapes[f"agent.gate.{i}.bias"] = (n_out, n_in), (n_out,)
    return shapes


# their state dict's other entries: the log-variance heads (unused, the variance is fixed), the world model (MV:63), the
# KL weight's schedule (MV:69)
UNUSED_PREFIXES = ("encoder.prior.logvar.", "encoder.post.logvar.", "world_model.", "beta_scheduler.")


class MuscleVAEPolicy(torch.nn.Module):
    """Their tracking policy's inference (module docstring) from their agent's state dict: every tensor of
    :func:`checkpoint_shapes` with its shape, any other entry one of UNUSED_PREFIXES."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        shapes = checkpoint_shapes()
        missing = sorted(set(shapes) - set(state))
        extra = sorted(k for k in set(state) - set(shapes) if not k.startswith(UNUSED_PREFIXES))
        if missing or extra:
            raise ValueError(f"MuscleVAE checkpoint: missing {missing}, unexpected {extra}")
        for k, shape in shapes.items():
            t = state[k]
            if tuple(t.shape) != shape or t.dtype != torch.float32:
                raise ValueError(f"MuscleVAE checkpoint: {k} is {tuple(t.shape)} {t.dtype}, expected {shape} float32")
            self.register_buffer(k.replace(".", "_"), t.detach().clone().contiguous())

    @classmethod
    def load(cls, path: str = CHECKPOINT_PATH, device: str | torch.device = "cpu", expected_sha256: str | None = CHECKPOINT_SHA256) -> MuscleVAEPolicy:
        """The released policy (:func:`load_checkpoint`) on ``device``."""
        return cls(load_checkpoint(path, expected_sha256)["self"]).to(device)

    def _t(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "_"))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Their normalization of observations (..., 371) (MV:344-350, PU:88-89)."""
        return (obs - self.obs_mean) / (self.obs_std + NORM_EPS)

    def policy_input(self, obs: torch.Tensor, muscle_state: torch.Tensor) -> torch.Tensor:
        """The networks' input (..., 386): the normalized observation and the muscle state as it is (MV:352-364)."""
        return torch.cat((self.normalize(obs), muscle_state), -1)

    def _encoder(self, name: str, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """Their Encoder's mean (MOD:24-39): x with c into the first layer, x with the previous layer into the others."""
        h = c
        for i in range(ENCODER_LAYERS):
            h = F.elu(F.linear(x if h is None else torch.cat((x, h), -1), self._t(f"encoder.{name}.fc_layers.{i}.weight"),
                               self._t(f"encoder.{name}.fc_layers.{i}.bias")))  # fmt: skip
        return F.linear(torch.cat((x, h), -1), self._t(f"encoder.{name}.mu.weight"), self._t(f"encoder.{name}.mu.bias"))

    def prior(self, n_obs: torch.Tensor) -> torch.Tensor:
        """The prior's mean (..., 64) of policy inputs (..., 386) (MOD:78-79)."""
        return self._encoder("prior", n_obs, None)

    def posterior(self, n_obs: torch.Tensor, n_target: torch.Tensor) -> torch.Tensor:
        """The posterior's offset (..., 64) from the prior's mean, of policy inputs and normalized targets (..., 371)
        (MOD:80-87: the posterior network takes the target as its input and the observation as its condition)."""
        return self._encoder("post", n_target, n_obs)

    def decode(self, z: torch.Tensor, n_obs: torch.Tensor) -> torch.Tensor:
        """The actions (..., 284) of latents (..., 64) and policy inputs (MOD:240-258). Each expert layer is one matmul of
        the gate-weighted copies of its input, (sum_e g_e x_e) W = [g_1 x, ..., g_6 x] [W_1; ...; W_6]."""
        zc = torch.cat((z, n_obs), -1)
        g = F.softmax(F.linear(F.elu(F.linear(F.elu(F.linear(zc, self.agent_gate_0_weight, self.agent_gate_0_bias)), self.agent_gate_2_weight,
                                              self.agent_gate_2_bias)), self.agent_gate_4_weight, self.agent_gate_4_bias), -1)  # fmt: skip
        h = n_obs
        for i in range(DECODER_LAYERS):
            x = torch.cat((z, h), -1)
            x = F.layer_norm(x, x.shape[-1:])
            w, b = self._t(f"agent.w{i}"), self._t(f"agent.b{i}")
            h = (g.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2) @ w.flatten(0, 1) + g @ b
            if i < DECODER_LAYERS - 1:
                h = F.elu(h)
        return h

    def forward(self, n_obs: torch.Tensor, n_target: torch.Tensor) -> torch.Tensor:
        """The tracking action of their playground (TS:14-23): the posterior's mean decoded."""
        return self.decode(self.prior(n_obs) + self.posterior(n_obs, n_target), n_obs)


class PolicyRunner:
    """The policy driving a :class:`~ndm_gym.sim.rigid.muscle_sim.MuscleSim`, one control step per :meth:`step`.

    ``latent``: "mean", their tracking playground's posterior mean (TS:19-20), or "sample", their collection's sample
    with standard deviation 0.3 (MV:376-377). ``action_noise``: std of Gaussian noise on the action (their collection's
    0.05, ACTION_NOISE). ``prior_prob``: probability that a step decodes the prior's sample instead (their collection's
    0.4, PRIOR_PROB; needs latent "sample"). ``after``: a function of the runner run right after the simulator's step (in
    the graph, e.g. metrics; also once in the capture's warm-up). ``compiled``: the observation and the networks through
    torch.compile (the default on CUDA).

    The step reads the simulator's state (body poses, the poses before its previous step, the velocities of the
    environments reset since then, MuscleSim.just_reset, the fatigue state, the reference frame), writes the action into
    its action buffer and runs its control step; on a simulator with CUDA graphs, policy and step are one graph (the
    first step is a warm-up and every environment is reset after the capture), the random draws from the simulator's
    registered generator; the simulator's own step graph is then unused (construct it with ``capture_step=False``).
    Resets are the simulator's (MuscleSim.reset, or :meth:`reset`). ``tracked`` holds the reference frame each
    environment's last step tracked."""

    def __init__(self, sim, policy: MuscleVAEPolicy, latent: str = "mean", action_noise: float = 0.0, prior_prob: float = 0.0,
                 after: Callable[[PolicyRunner], None] | None = None, compiled: bool | None = None):  # fmt: skip
        if latent not in LATENT_MODES:
            raise ValueError(f"latent must be one of {LATENT_MODES}, got {latent!r}")
        if prior_prob and latent != "sample":
            raise ValueError("steps from the prior sample its latent (TC:59-61): prior_prob needs latent='sample'")
        if not (action_noise >= 0.0 and 0.0 <= prior_prob <= 1.0):
            raise ValueError("action_noise must be >= 0 and prior_prob in [0, 1]")
        self.sim, self.policy = sim, policy.to(sim.device)
        self.latent, self.action_noise, self.prior_prob, self.after = latent, float(action_noise), float(prior_prob), after
        N, dev = sim.N, sim.device
        self.targets = self.policy.normalize(reference_observations(sim.ref.names).to(device=dev, dtype=torch.float32))  # (F, 371)
        if self.targets.shape[0] != sim.ref.clip.shape[0]:
            raise ValueError("the reference observations do not cover the simulator's reference frames")
        self.tracked = torch.zeros(N, dtype=torch.long, device=dev)
        self.eps_latent, self.eps_action = torch.zeros(N, LATENT_DIM, device=dev), torch.zeros(N, mc.NUM_MUSCLES, device=dev)
        self.u_prior = torch.ones(N, device=dev)
        compiled = dev.type == "cuda" if compiled is None else compiled
        self._act = torch.compile(self.act, dynamic=False) if compiled else self.act
        self.graph = None
        if sim.use_graph:
            self.graph = sim.capture(self._body)
            sim.reset()

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy inputs (N, 386) and normalized targets (N, 371) of the simulator's current state."""
        sim = self.sim
        q, qd = sim.views[0].body_q, sim.views[0].body_qd
        vel, avel = velocities(q[..., :3], q[..., 3:], sim.before[..., :3], sim.before[..., 3:], 1.0 / mc.CONTROL_HZ)
        fresh = sim.just_reset[:, None, None]  # reset since the previous step: its poses before that step are another episode's
        vel, avel = torch.where(fresh, qd[..., :3], vel), torch.where(fresh, qd[..., 3:], avel)
        obs = observation(q[..., :3], quat_to_mat(q[..., sim.wxyz]), vel, avel)
        return self.policy.policy_input(obs, sim.part_fatigue()), self.targets[sim.ref_frame]

    def act(self):
        """The policy's action into the simulator's action buffer, from the current state and the drawn noise."""
        n_obs, n_target = self.observe()
        mu_prior = self.policy.prior(n_obs)
        z = mu_prior + self.policy.posterior(n_obs, n_target)
        if self.latent == "sample":
            z = z + LATENT_STD * self.eps_latent
            if self.prior_prob > 0.0:
                z = torch.where((self.u_prior < self.prior_prob).unsqueeze(-1), mu_prior + LATENT_STD * self.eps_latent, z)
        a = self.policy.decode(z, n_obs)
        if self.action_noise > 0.0:
            a = a + self.action_noise * self.eps_action
        self.sim.action.copy_(a)
        self.tracked.copy_(self.sim.ref_frame)

    def _draw(self):
        g, dev = self.sim.gen, self.sim.device
        if self.latent == "sample":
            torch.randn(self.eps_latent.shape, generator=g, device=dev, out=self.eps_latent)
        if self.action_noise > 0.0:
            torch.randn(self.eps_action.shape, generator=g, device=dev, out=self.eps_action)
        if self.prior_prob > 0.0:
            torch.rand(self.u_prior.shape, generator=g, device=dev, out=self.u_prior)

    def _body(self):
        self._draw()
        self._act()
        self.sim.step_body()
        if self.after is not None:
            self.after(self)

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One control step of policy and simulator; returns the simulator's termination and truncation flags (N,)."""
        if self.graph is not None:
            self.graph.replay()
        else:
            self._draw()
            self._act()
            self.sim.step(self.sim.action)
            if self.after is not None:
                self.after(self)
        return self.sim.terminated.clone(), self.sim.truncated.clone()

    def reset(self, mask: torch.Tensor | None = None, frames: torch.Tensor | None = None):
        """The simulator's reset (MuscleSim.reset) of the environments where ``mask`` is true (all if None); the simulator
        marks them (just_reset), so their next observation takes its velocities."""
        self.sim.reset(mask, frames)


__all__ = ["MuscleVAEPolicy", "PolicyRunner", "checkpoint_shapes", "load_checkpoint", "observation", "reference_observations", "velocities"]
