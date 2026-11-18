"""The fused path of the MJCF world model's kinematics and contact gaps (ndm_gym.worldmodels.mjcf: ``MjcfWorldModel.fused``) on
CPU, for both kinds of model (SuperTrack's accelerations and SuperMimic's increments): off by default; its body poses and
velocities, contact gaps, points, normals, velocities and inputs, and the model's step, against the level-by-level path on
trees with slides, hinges, a ball joint and a free object, above floors and beside objects, in double and single precision;
its poses against MuJoCo's own; a loaded state's geoms' bodies; and the option of the planner, the recorder and the speed
scripts that turns it on; and the compiler's setting of its compiled step, set on the fused path alone. With
MBRL_TEST_DEVICE on a GPU (default cpu), also the fused step as the planner runs it there: captured in a CUDA graph,
uncompiled and compiled, against the level-by-level step."""

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_mjcf_contact_places import GROUNDS, PAIR  # noqa: E402
from test_mjcf_contacts import FOOT, PROBES, PUSHER  # noqa: E402

from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel, contact_spheres, fused_compile  # noqa: E402

DEVICE = os.environ.get("MBRL_TEST_DEVICE", "cpu")

# a base on a slide and a ball joint with an arm on a hinge and a slide, off-axis and turned, beside a free box above a floor
BALL = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <joint name="x" type="slide" axis="1 0 0"/><joint name="b" type="ball" pos="0 0 0.1"/>
      <geom name="base" type="capsule" size="0.05 0.2"/>
      <body name="arm" pos="0.1 0 -0.3" quat="0.9 0.1 0.3 0.2">
        <joint name="h" type="hinge" axis="0 1 1" pos="0.02 0 0.03"/><joint name="s" type="slide" axis="1 0 1"/>
        <geom name="arm" type="capsule" size="0.03 0.1" pos="0 0.05 -0.1" quat="0.7 0.3 0.1 0.2"/>
      </body>
    </body>
    <body name="obj" pos="0.5 0 0.2"><freejoint name="obj"/><geom name="obj" type="box" size="0.1 0.1 0.05"/></body>
  </worldbody>
  <actuator><motor joint="h"/><motor joint="s"/></actuator>
</mujoco>
"""
FLOORED = PROBES.replace("KIND", "box").replace("SIZE", "0.1 0.1 0.05").replace("MARGIN", "")
FLOORED = FLOORED.replace("<worldbody>", '<worldbody><geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -1"/>')
CASES = {
    "probes, a free box and a floor": (FLOORED, dict(static=())),
    "a foot above a floor and a step": (FOOT, dict(static=())),
    "the pusher and its puck on slides": (PUSHER, dict(objects=("object",))),
    "two capsules above a plane": (PAIR.replace("GROUND", GROUNDS["plane"]), dict(static=("floor",))),
    "a ball joint, a free box and a floor": (BALL, dict(static=())),
}


def states(m, n=64, seed=0):
    """Random states: positions of about 0.7 rad or m, unit quaternions, unit-scale velocities."""
    g = torch.Generator().manual_seed(seed)
    q = 0.7 * torch.randn(n, m.nq, generator=g, dtype=torch.float64)
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        span = {0: slice(a + 3, a + 7), 1: slice(a, a + 4)}.get(int(m.jnt_type[j]))
        if span is not None:
            q[:, span] = torch.nn.functional.normalize(q[:, span], dim=-1)
    return q, torch.randn(n, m.nv, generator=g, dtype=torch.float64), torch.randn(n, m.nu, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("double", [True, False])
@pytest.mark.parametrize("kind", MjcfWorldModel.KINDS)
def test_the_fused_path_gives_the_level_by_level_values(case, double, kind):
    xml, how = CASES[case]
    m = mujoco.MjModel.from_xml_string(xml)
    lay = MjcfLayout.from_model(m)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay, 0.02, **how), kind=kind)
    assert MjcfWorldModel.fused is False and wm.fused is False and "fused" not in wm.state_dict()  # off unless turned on, and no state
    wm, tol = (wm.double(), 1e-12) if double else (wm, 2e-5)
    q, v, u = (x if double else x.float() for x in states(m))
    wm.update_norms(q, v, u, q, v)

    def everything():  # the poses and velocities, the gaps, the inputs, the step, and the poses and gaps without velocities
        return wm.body_poses(q, v) + wm.contact_gaps(q, v) + (wm.contact_inputs(q, v),) + wm.step(q, v, u) + wm.body_poses(q) + wm.contact_gaps(q)

    with torch.no_grad():
        level = everything()
        wm.fused = True
        fused = everything()
    assert len(level) == len(fused) == 16
    for a, b in zip(level, fused):
        assert a.shape == b.shape and torch.allclose(a, b, atol=tol, rtol=0)
    d = mujoco.MjData(m)
    for i in (0, 7, 31):  # the fused poses are mj_kinematics', to the constants' single precision
        d.qpos[:], d.qvel[:] = q[i].double().numpy(), v[i].double().numpy()
        mujoco.mj_forward(m, d)
        assert np.abs(fused[0][i].double().numpy() - d.xpos).max() < 1e-6
        assert np.abs(fused[1][i].double().numpy().reshape(-1, 9) - d.xmat).max() < 1e-6


def test_a_loaded_states_geoms_bodies_are_the_fused_paths():
    m = mujoco.MjModel.from_xml_string(PAIR.replace("GROUND", GROUNDS["plane"]))  # two capsules on two bodies
    lay = MjcfLayout.from_model(m)
    cs = contact_spheres(m, lay, 0.02, static=("floor",))
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=cs).double()
    assert wm._cs_bodies == wm.cs_body.tolist() == cs["body"].tolist() and len(set(wm._cs_bodies)) == 2
    swapped = MjcfWorldModel(lay, hidden=(8,), contacts=cs | {"body": cs["body"][::-1].copy(), "count": None}).double()
    assert swapped._cs_bodies == wm._cs_bodies[::-1]
    swapped.load_state_dict(wm.state_dict())  # the state's geoms' bodies stand from then on, on both paths
    assert swapped._cs_bodies == wm._cs_bodies
    q, v, _ = states(m, 16)
    swapped.fused = True
    with torch.no_grad():
        for a, b in zip(wm.contact_gaps(q, v), swapped.contact_gaps(q, v)):
            assert torch.allclose(a, b, atol=1e-12, rtol=0)


@pytest.mark.skipif(not DEVICE.startswith("cuda"), reason="the planner's CUDA graph of the step needs a GPU (MBRL_TEST_DEVICE)")
@pytest.mark.parametrize("case", ["the pusher and its puck on slides", "a ball joint, a free box and a floor"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", MjcfWorldModel.KINDS)
def test_the_fused_step_in_a_cuda_graph(case, compiled, kind):
    xml, how = CASES[case]
    m = mujoco.MjModel.from_xml_string(xml)
    lay = MjcfLayout.from_model(m)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay, 0.02, **how), amp=False, kind=kind)
    q, v, u = (x.float() for x in states(m, 350))
    wm.update_norms(q, v, u, q, v)
    wm = wm.to(DEVICE).eval().requires_grad_(False)
    q, v, u = (x.to(DEVICE) for x in (q, v, u))
    with torch.no_grad():
        level = wm.step(q, v, u) + wm.contact_gaps(q, v)
    wm.fused = True

    def step():
        return wm.step(q, v, u) + wm.contact_gaps(q, v)

    fn = torch.compile(step, dynamic=False) if compiled else step
    with torch.no_grad():
        with fused_compile(wm):  # compiled as the planner compiles it: a graph's independent kernels as one launch
            for _ in range(3):  # compilation and the allocator's warm-up, outside the capture (as sim.mbrl_ndm.NdmRollouts)
                fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):  # no copy to the host inside the step: the capture would refuse it
            out = fn()
        graph.replay()
        torch.cuda.synchronize()
    assert len(out) == len(level) == 6
    for a, b in zip(level, out):
        assert a.shape == b.shape and torch.allclose(a, b, atol=2e-4, rtol=0)


def test_the_fused_paths_compile_setting_is_set_on_it_alone():
    import torch._inductor.config as config

    m = mujoco.MjModel.from_xml_string(FOOT)
    lay = MjcfLayout.from_model(m)
    wm = MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay, 0.02, static=()))
    before = config.combo_kernels
    with fused_compile(wm):  # off the fused path: as it was
        assert config.combo_kernels == before
    wm.fused = True
    with fused_compile(wm):
        assert config.combo_kernels is True
    assert config.combo_kernels == before  # and as it was afterwards


def test_the_options_that_turn_it_on_are_off_by_default():
    from ndm_gym.wmbench import collect_mbrl, plan_mbrl, speed_mbrl, speed_mjcf

    assert plan_mbrl.parse_args(["--env", "pets_pusher", "--out", "o"]).fused is False
    assert plan_mbrl.parse_args(["--env", "pets_pusher", "--out", "o", "--fused"]).fused is True
    assert collect_mbrl.parse_args(["--env", "pets_pusher", "--out", "o"]).fused is False
    assert speed_mbrl.parse_args(["--meta", "m", "--states", "s", "--out", "o"]).fused is False
    assert speed_mjcf.parse_args(["--meta", "m", "--states", "s", "--out", "o"]).fused is False
