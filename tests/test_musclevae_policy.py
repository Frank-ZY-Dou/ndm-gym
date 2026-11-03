"""MuscleVAE's pretrained tracking policy (ndm_gym.policies.musclevae_pretrained) on the CPU, without their checkpoint,
against independent references written from the paper (Feng, Xu and Liu, SIGGRAPH Asia 2023) and the documented
semantics of their observation: the observation computed in their Y-up frame with rotation matrices and on a hand-built
pose, the backward-difference velocities as the angular velocity of the quaternion path's finite difference, the
normalization, and the networks (the encoders' means bit for bit, the mixture of experts with an explicit loop over the
experts) on random weights; the checkpoint loader on a checkpoint written here; with the converted assets and Newton,
the runner driving two environments of MuscleSim."""

import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ndm_gym.core.rotations import so3_exp
from ndm_gym.policies import musclevae_pretrained as mp
from ndm_gym.robots import musclevae_char as mc

C = torch.tensor(mc.Y_UP_TO_Z_UP, dtype=torch.float64)  # ours = C theirs


def random_rotations(g: np.random.Generator, shape: tuple) -> np.ndarray:
    """Rotation matrices (shape + (3, 3)) of rotation vectors with components of std 2 rad."""
    return so3_exp(torch.tensor(2.0 * g.normal(size=shape + (3,)))).numpy()


def reference_observation(pos: np.ndarray, rot: np.ndarray, vel: np.ndarray, avel: np.ndarray) -> np.ndarray:
    """The observation as the paper and their documentation describe it, computed in their Y-up frame with rotation
    matrices, from positions, velocities and angular velocities (B, n, 3) and rotations (B, n, 3, 3), body 0 the root:
    per body the position relative to the root, the rotation relative to the root as the first two columns of its
    matrix (row by row), the linear and the angular velocity, all in the root's frame; then the bodies' heights (their
    y) and the world's up axis (their y) in the root's frame."""
    B = pos.shape[0]
    root_inv = np.swapaxes(rot[:, :1], -1, -2)  # (B, 1, 3, 3)

    def in_root(v):
        return (root_inv @ v[..., None])[..., 0].reshape(B, -1)

    relative = root_inv @ rot
    up = root_inv[:, 0] @ np.array([0.0, 1.0, 0.0])
    return np.concatenate([in_root(pos - pos[:, :1]), relative[..., :, :2].reshape(B, -1), in_root(vel), in_root(avel), pos[..., 1], up], -1)


def to_ours(pos, rot, vel, avel):
    """States in their frame as ours: v -> C v, R -> C R C^T."""
    t = [torch.tensor(x) for x in (pos, rot, vel, avel)]
    return t[0] @ C.T, C @ t[1] @ C.T, t[2] @ C.T, t[3] @ C.T


def test_observation_against_reference():
    g = np.random.default_rng(0)
    n = mc.NUM_BODIES
    pos, rot = g.normal(size=(16, n, 3)) + [0.0, 1.0, 0.0], random_rotations(g, (16, n))
    vel, avel = g.normal(size=(16, n, 3)), 3 * g.normal(size=(16, n, 3))
    ours = mp.observation(*to_ours(pos, rot, vel, avel))
    assert ours.shape == (16, mp.OBS_DIM) and mp.OBS_DIM == 371
    np.testing.assert_allclose(ours.numpy(), reference_observation(pos, rot, vel, avel), rtol=0, atol=1e-12)
    # any number of bodies, any batch shape
    two = (g.normal(size=(3, 2, 3)), random_rotations(g, (3, 2)), g.normal(size=(3, 2, 3)), g.normal(size=(3, 2, 3)))
    ours2 = mp.observation(*(x.unsqueeze(0) for x in to_ours(*two)))
    np.testing.assert_allclose(ours2[0].numpy(), reference_observation(*two), rtol=0, atol=1e-12)


def test_observation_hand_built():
    """Two bodies in our frame: the root at (1, 2, 0.9) turned 90 degrees about our z (up), the other 0.5 m along our x
    and 0.2 m up, unturned, moving at 1 m/s along x and turning at 2 rad/s about z."""
    c, s = 0.0, 1.0
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    pos = torch.tensor([[1.0, 2.0, 0.9], [1.5, 2.0, 1.1]], dtype=torch.float64)
    rot = torch.stack((Rz, torch.eye(3, dtype=torch.float64)))
    vel = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    avel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    ob = mp.observation(pos, rot, vel, avel)
    assert ob.shape == (35,)
    lp, six, lv, la, h, up = ob[0:6], ob[6:18], ob[18:24], ob[24:30], ob[30:32], ob[32:35]
    # in the root's frame our x is our -y; their axes (x, y, z) = our (y, z, x)
    torch.testing.assert_close(lp, torch.tensor([0.0, 0.0, 0.0, -0.5, 0.2, 0.0], dtype=torch.float64))
    torch.testing.assert_close(lv, torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(la, torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(h, torch.tensor([0.9, 1.1], dtype=torch.float64))
    torch.testing.assert_close(up, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    # the root relative to itself is the identity; the other body is turned by -90 degrees about the up axis (their y):
    # rows of [[cos, sin], [0, 0], [-sin, cos]]... as the first two columns of R_y(-90) = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    torch.testing.assert_close(six[:6], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(six[6:], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float64))


def reference_velocities(pos0, quat0, pos1, quat1, dt):
    """Backward differences over dt from (pos0, quat0) to (pos1, quat1), quaternions xyzw: the linear velocity, and the
    world-frame angular velocity of the quaternion path, w = 2 E(q) dq/dt with E(q) = [q_w I + [q_v]x | -q_v] (from
    dq/dt = 1/2 (w, 0) q), dq/dt the finite difference after turning quat1 into quat0's hemisphere."""
    s = np.where((quat1 * quat0).sum(-1, keepdims=True) < 0.0, -1.0, 1.0)
    qdot = (s * quat1 - quat0) / dt
    v, w = quat0[..., :3], quat0[..., 3]
    vx = np.zeros(v.shape + (3,))
    vx[..., 0, 1], vx[..., 0, 2], vx[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    vx = vx - np.swapaxes(vx, -1, -2)
    E = np.concatenate((w[..., None, None] * np.eye(3) + vx, -v[..., :, None]), -1)  # (..., 3, 4)
    return (pos1 - pos0) / dt, 2.0 * (E @ qdot[..., None])[..., 0]


def random_unit(g, shape):
    q = g.normal(size=shape + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_velocities_against_reference():
    g = np.random.default_rng(1)
    q0 = random_unit(g, (64, 5))
    q1 = q0 + 0.05 * g.normal(size=(64, 5, 4))
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True) * np.where(g.random((64, 5, 1)) < 0.5, -1.0, 1.0)  # either hemisphere
    p0, p1 = g.normal(size=(64, 5, 3)), g.normal(size=(64, 5, 3))
    v_ref, w_ref = reference_velocities(p0, q0, p1, q1, 0.05)
    v, w = mp.velocities(*(torch.tensor(x) for x in (p1, q1, p0, q0)), 0.05)
    np.testing.assert_allclose(v.numpy(), v_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w.numpy(), w_ref, rtol=0, atol=1e-10)
    # either sign of either quaternion gives the same angular velocity
    _, w2 = mp.velocities(*(torch.tensor(x) for x in (p1, -q1, p0, -q0)), 0.05)
    torch.testing.assert_close(w2, w, rtol=0, atol=1e-10)
    # a constant angular velocity about one axis: 2 sin(theta / 2) / dt about it (the quaternion difference's)
    th = 0.3
    qa = np.array([0.0, 0.0, math.sin(th / 2), math.cos(th / 2)])
    _, w3 = mp.velocities(torch.zeros(3, dtype=torch.float64), torch.tensor(qa), torch.zeros(3, dtype=torch.float64), torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
                          0.05)
    torch.testing.assert_close(w3, torch.tensor([0.0, 0.0, 2 * math.sin(th / 2) / 0.05], dtype=torch.float64))


def random_state_dict(seed: int = 0, scale: float = 1.0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {k: scale * torch.randn(shape, generator=g) / math.sqrt(shape[-1] if len(shape) > 1 else 1) for k, shape in mp.checkpoint_shapes().items()}
    sd["obs_std"] = 0.1 + torch.rand(mp.OBS_DIM, generator=g)
    for i in range(mp.DECODER_LAYERS):  # their (experts, in, out) layout: scale by the input size
        sd[f"agent.w{i}"] = sd[f"agent.w{i}"] / math.sqrt(sd[f"agent.w{i}"].shape[1]) * math.sqrt(sd[f"agent.w{i}"].shape[2])
    return sd


def reference_policy(weights: dict, n_obs: torch.Tensor, n_target: torch.Tensor, latent_noise: torch.Tensor | None = None):
    """The policy as the paper describes it, on a state dict of their layout: the learnable prior and the posterior are
    MLPs whose every layer sees their input again next to the previous layer's output (the posterior's first layer the
    observation, its condition), and the posterior's mean is an offset from the prior's; the decoder is a mixture of
    experts, a gating MLP on the latent and the observation weighting the experts, each decoder layer the weighted sum
    of every expert's affine map of the layer-normalized latent and previous output (the observation for the first).
    Returns the prior's and the posterior's means and the action."""

    def skip_mlp(prefix: str, inputs: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        hidden = condition
        for layer in range(mp.ENCODER_LAYERS):
            joined = inputs if hidden is None else torch.cat((inputs, hidden), -1)
            hidden = F.elu(F.linear(joined, weights[f"{prefix}.fc_layers.{layer}.weight"], weights[f"{prefix}.fc_layers.{layer}.bias"]))
        return F.linear(torch.cat((inputs, hidden), -1), weights[f"{prefix}.mu.weight"], weights[f"{prefix}.mu.bias"])

    prior_mean = skip_mlp("encoder.prior", n_obs, None)
    posterior_mean = prior_mean + skip_mlp("encoder.post", n_target, n_obs)
    latent = posterior_mean if latent_noise is None else posterior_mean + latent_noise
    hidden = torch.cat((latent, n_obs), -1)
    for layer in (0, 2):
        hidden = F.elu(F.linear(hidden, weights[f"agent.gate.{layer}.weight"], weights[f"agent.gate.{layer}.bias"]))
    expert_weight = F.softmax(F.linear(hidden, weights["agent.gate.4.weight"], weights["agent.gate.4.bias"]), -1)
    out = n_obs
    for layer in range(mp.DECODER_LAYERS):
        joined = torch.cat((latent, out), -1)
        normed = F.layer_norm(joined, joined.shape[-1:])
        mixed = torch.zeros(normed.shape[0], weights[f"agent.b{layer}"].shape[-1])
        for expert in range(mp.NUM_EXPERTS):
            mixed = mixed + expert_weight[:, expert, None] * (normed @ weights[f"agent.w{layer}"][expert] + weights[f"agent.b{layer}"][expert])
        out = F.elu(mixed) if layer < mp.DECODER_LAYERS - 1 else mixed
    return prior_mean, posterior_mean, out


def test_normalization_and_networks():
    sd = random_state_dict()
    pol = mp.MuscleVAEPolicy(sd)
    g = torch.Generator().manual_seed(2)
    obs, target, ms = torch.randn(8, mp.OBS_DIM, generator=g), torch.randn(8, mp.OBS_DIM, generator=g), torch.rand(8, 15, generator=g)
    n_obs = pol.policy_input(obs, ms)
    assert torch.equal(n_obs[:, : mp.OBS_DIM], (obs - sd["obs_mean"]) / (sd["obs_std"] + 1e-8)) and torch.equal(n_obs[:, mp.OBS_DIM :], ms)
    n_target = pol.normalize(target)
    prior_mean, posterior_mean, action = reference_policy(sd, n_obs, n_target)
    assert torch.equal(pol.prior(n_obs), prior_mean)
    assert torch.equal(pol.prior(n_obs) + pol.posterior(n_obs, n_target), posterior_mean)
    torch.testing.assert_close(pol(n_obs, n_target), action, rtol=1e-5, atol=1e-5)
    assert action.shape == (8, mc.NUM_MUSCLES) and action.abs().max() > 0.1
    noise = mp.LATENT_STD * torch.randn(8, mp.LATENT_DIM, generator=g)
    torch.testing.assert_close(pol.decode(posterior_mean + noise, n_obs), reference_policy(sd, n_obs, n_target, noise)[2], rtol=1e-5, atol=1e-5)
    # the action is the PD target's relative change: (1 + a) l_Tpose (ENV:431-433), MuscleModel.target_length
    muscles = mc.muscle_model() if os.path.isfile(mc.MUSCLES_PATH) else None
    if muscles is not None:
        torch.testing.assert_close(muscles.target_length(action), muscles.rest_length * (1 + action))


def test_load_checkpoint(tmp_path):
    sd = random_state_dict(3)
    full = dict(sd)
    full["encoder.prior.logvar.weight"] = torch.zeros(64, 898)  # entries the policy does not use
    full["world_model.mlp.0.weight"] = torch.zeros(4, 4)
    full["beta_scheduler.cnt"] = torch.tensor(1.0)
    path = os.path.join(tmp_path, "trained.data")
    torch.save({"self": full, "balance": np.linspace(0.0, 1.0, 10), "train_iter": 7}, path)
    ck = mp.load_checkpoint(path, expected_sha256=None)
    assert ck["train_iter"] == 7 and ck["balance"].shape == (10,)
    pol = mp.MuscleVAEPolicy(ck["self"])
    assert torch.equal(pol.agent_w3, sd["agent.w3"]) and torch.equal(pol.obs_std, sd["obs_std"])
    with pytest.raises(ValueError, match="SHA-256"):
        mp.load_checkpoint(path, expected_sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        mp.load_checkpoint(os.path.join(tmp_path, "none.data"))
    bad = dict(sd)
    del bad["agent.b2"]
    with pytest.raises(ValueError, match="missing"):
        mp.MuscleVAEPolicy(bad)
    bad = dict(sd, **{"encoder.extra.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="unexpected"):
        mp.MuscleVAEPolicy(bad)
    bad = dict(sd, **{"agent.w0": torch.zeros(6, 450, 511)})
    with pytest.raises(ValueError, match="agent.w0"):
        mp.MuscleVAEPolicy(bad)
    torch.save({"self": full}, path)
    with pytest.raises(ValueError, match="keys"):
        mp.load_checkpoint(path, expected_sha256=None)


HAVE_DATA = all(os.path.isfile(p) for p in (mc.MJCF_PATH, mc.CHARACTER_PATH, mc.MUSCLES_PATH)) and os.path.isdir(mc.CLIPS_DIR)


@pytest.mark.skipif(not HAVE_DATA, reason="MuscleVAE's assets are optional data: run tools/download_musclevae.sh, then python tools/convert_musclevae.py")
def test_reference_observations():
    """The targets of a clip: its placement's observation with backward-difference velocities, the first frame's those
    of the second; heights are the placement's."""
    from ndm_gym.motion.sources import musclevae as src

    ref = mp.reference_observations(("lafan_walk",))
    pos, quat = (x.double() for x in src.load_placement(os.path.join(mc.CLIPS_DIR, "lafan_walk.npz")))
    assert ref.shape == (pos.shape[0], mp.OBS_DIM) and ref.dtype == torch.float64
    torch.testing.assert_close(ref[:, 345:368], pos[..., 2], rtol=0, atol=0)
    xyzw = torch.cat((quat[..., 1:], quat[..., :1]), -1)
    v, w = mp.velocities(pos[100], xyzw[100], pos[99], xyzw[99], 0.05)
    from ndm_gym.core.rotations import quat_to_mat

    v1, w1 = mp.velocities(pos[1], xyzw[1], pos[0], xyzw[0], 0.05)  # the first frame takes the second's velocities, in its own root frame
    torch.testing.assert_close(ref[0], mp.observation(pos[0], quat_to_mat(quat[0]), v1, w1))
    torch.testing.assert_close(ref[100], mp.observation(pos[100], quat_to_mat(quat[100]), v, w))


@pytest.mark.skipif(not HAVE_DATA, reason="MuscleVAE's assets are optional data")
def test_runner_on_cpu():
    """The runner on two CPU environments of MuscleSim with random weights: the first observation after a reset (the
    simulator's, or the runner's) takes the simulator's velocities, later ones backward differences; the action is the
    policy's of observe()."""
    pytest.importorskip("newton")
    pytest.importorskip("mujoco_warp")
    from ndm_gym.sim.rigid.muscle_sim import MuscleSim

    sim = MuscleSim(2, "cpu", clips=("lafan_walk",), seed=0)
    pol = mp.MuscleVAEPolicy(random_state_dict(4, scale=0.1))
    seen = []
    run = mp.PolicyRunner(sim, pol, after=lambda r: seen.append(r.sim.just_reset.clone()))
    assert run.graph is None and sim.just_reset.all()
    run.reset(frames=torch.tensor([100, 2000]))
    n_obs, n_target = run.observe()
    assert n_obs.shape == (2, mp.POLICY_OBS_DIM) and n_target.shape == (2, mp.OBS_DIM)
    torch.testing.assert_close(n_target, run.targets[torch.tensor([101, 2001])])
    q, qd = sim.views[0].body_q, sim.views[0].body_qd
    obs = mp.observation(q[..., :3], mc_quat_mat(q), qd[..., :3], qd[..., 3:])
    torch.testing.assert_close(n_obs, pol.policy_input(obs, sim.part_fatigue()))
    expected = pol(n_obs, n_target)
    terminated, truncated = run.step()
    torch.testing.assert_close(sim.action, expected)
    assert run.tracked.tolist() == [101, 2001] and sim.ref_frame.tolist() == [102, 2002] and not sim.just_reset.any() and len(seen) == 1
    assert not seen[0].any() and terminated.shape == (2,) and not (terminated | truncated).any()
    # after a step: backward differences over it from the poses before it
    n_obs2, _ = run.observe()
    q, qd, before = sim.views[0].body_q, sim.views[0].body_qd, sim.before
    v, w = mp.velocities(q[..., :3], q[..., 3:], before[..., :3], before[..., 3:], 1.0 / mc.CONTROL_HZ)
    torch.testing.assert_close(n_obs2, pol.policy_input(mp.observation(q[..., :3], mc_quat_mat(q), v, w), sim.part_fatigue()))
    # a reset by the simulator itself, not through the runner: that environment's next observation takes its velocities
    sim.reset(torch.tensor([False, True]))
    assert sim.just_reset.tolist() == [False, True]
    n_obs3, _ = run.observe()
    q, qd = sim.views[0].body_q, sim.views[0].body_qd
    v, w = torch.where(sim.just_reset[:, None, None], qd[..., :3], v), torch.where(sim.just_reset[:, None, None], qd[..., 3:], w)
    torch.testing.assert_close(n_obs3, pol.policy_input(mp.observation(q[..., :3], mc_quat_mat(q), v, w), sim.part_fatigue()))
    torch.testing.assert_close(n_obs3[0], n_obs2[0])
    run.reset(torch.tensor([True, False]))
    assert sim.just_reset.tolist() == [True, True]
    run.step()
    assert not sim.just_reset.any()


def mc_quat_mat(body_q):
    from ndm_gym.core.rotations import quat_to_mat

    return quat_to_mat(body_q[..., [6, 3, 4, 5]])
