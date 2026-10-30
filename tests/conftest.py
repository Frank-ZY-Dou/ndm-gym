"""Shared fixtures. Tests run on CPU and need the converted assets and motions in ``data/`` (scripts/setup.sh)
and the pristine upstream reproduction in ``third_party/supermimic`` (with its own ``data/``)."""

import importlib
import os
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "third_party", "supermimic")
sys.path.insert(0, REPO)
DEV = "cpu"


@pytest.fixture(scope="session")
def upstream():
    """The upstream ``supermimic`` package (@ ee0608f), imported from third_party."""
    assert os.path.isdir(os.path.join(UPSTREAM, "supermimic")), f"missing pristine upstream checkout at {UPSTREAM}"
    sys.path.insert(0, UPSTREAM)
    mods = {n: importlib.import_module(f"supermimic.{n}") for n in ("kinematics", "motion", "mdp", "networks", "buffer", "robot", "rotations")}
    sys.path.remove(UPSTREAM)
    return mods


def with_ndm_fk(k0):
    """Give an upstream kinematics object ndm_gym's FK, so that everything computed from it can be compared exactly.
    ndm_gym writes the FK's 3x3 products elementwise where upstream uses batched matmuls (same arithmetic, other
    summation order); the FK itself is compared with upstream's up to rounding in test_upstream_equivalence."""
    from ndm_gym.core.kinematics import ArticulationKinematics

    k0.forward = types.MethodType(ArticulationKinematics.forward, k0)
    return k0


@pytest.fixture(scope="session")
def kin():
    from ndm_gym.robots import g1

    return g1.kinematics(DEV)


@pytest.fixture(scope="session")
def motion(kin):
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import lafan1

    return MotionLib(lafan1.load_dir(os.path.join(REPO, "data", "lafan1_g1")), kin, DEV)


@pytest.fixture(scope="session")
def mdp(kin, motion):
    from ndm_gym.robots import g1
    from ndm_gym.tasks.tracking import MDP

    return MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=DEV))


def random_states(motion, n: int, seed: int):
    """Reference states near the clips' starts with noise on every component (deterministic)."""
    from ndm_gym.core.rotations import so3_exp

    g = torch.Generator().manual_seed(seed)
    f = motion.clip_start[torch.arange(n) % len(motion.names)] + 40 + torch.randint(0, 50, (n,), generator=g)
    p, R, v, w, q, qd = motion.ref_state(f)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    S = (p + 0.05 * rn(n, 3), so3_exp(0.1 * rn(n, 3)) @ R, v + 0.3 * rn(n, 3), w + 0.3 * rn(n, 3), q + 0.1 * rn(n, 29), qd + 0.5 * rn(n, 29))
    xi = torch.cat((0.3 + 0.9 * torch.rand(n, 1, generator=g), 0.02 * torch.rand(n, 29, generator=g) - 0.01, 0.1 * torch.rand(n, 3, generator=g) - 0.05), -1)
    return f, S, xi
