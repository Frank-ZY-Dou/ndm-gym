"""RoboKit's simulator with a world model in place of its physics (ndm_gym.sim.robokit_ndm) on Warp's CPU device: a step is
the world model's step, with the derived fields of the state before it; inside RoboKit's own SimMPC (its documentation's
cart) every rollout step is the world model's, a staged plan uses the world model of each step length, a length with no
world model is an error (as in the automatic stage check of a planner given no stages, which also steps 2 and 4 timesteps),
and a timestep changed behind the simulator's back is flagged. Needs RoboKit with MuJoCo Warp
(skipped without)."""

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
pytest.importorskip("robokit")
wp = pytest.importorskip("warp")

from ndm_gym.sim.robokit_ndm import ndm_sim, planning  # noqa: E402
from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel  # noqa: E402

CART = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="x" type="slide" axis="1 0 0" damping="5"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
    <body name="ball" pos="0.5 0 0.3"><freejoint/><geom size="0.05"/></body>
    <geom type="plane" size="2 2 0.1"/>
  </worldbody>
  <actuator><position joint="x" kp="50"/></actuator>
  <sensor><jointpos name="x" joint="x"/></sensor>
</mujoco>
"""


class Counted(torch.nn.Module):
    """A world model whose steps are counted (and their batch sizes kept)."""

    def __init__(self, wm):
        super().__init__()
        self.wm, self.lay, self.calls = wm, wm.lay, []

    def step(self, q, v, u):
        self.calls.append(q.shape[0])
        return self.wm.step(q, v, u)


def world_model(m, dt=None, seed=0):
    lay = MjcfLayout.from_model(m)
    if dt is not None:
        import dataclasses

        lay = dataclasses.replace(lay, dt=dt)
    torch.manual_seed(seed)
    wm = MjcfWorldModel(lay, hidden=(16,))
    for p in wm.net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return wm.eval().requires_grad_(False)


def test_a_step_is_the_world_models_with_the_derived_fields_before_it():
    m = mujoco.MjModel.from_xml_string(CART)
    wm = world_model(m)
    sim = ndm_sim()(m, 3, substeps=1, device="cpu", ndm={0.01: wm}, derived=("kinematics", "sensor_pos"), compile=False)
    rng = np.random.default_rng(0)
    q0 = np.tile(m.qpos0, (3, 1)) + 0.1 * rng.standard_normal((3, m.nq))
    q0[:, 4:8] /= np.linalg.norm(q0[:, 4:8], axis=1, keepdims=True)
    v0, u0 = rng.standard_normal((3, m.nv)), rng.standard_normal((3, m.nu))
    sim.data.qpos.assign(q0.astype(np.float32))
    sim.data.qvel.assign(v0.astype(np.float32))
    sim.data.ctrl.assign(u0.astype(np.float32))
    t0 = sim.data.time.numpy().copy()
    sim.step()
    with torch.no_grad():
        q1, v1 = wm.step(*(torch.tensor(x, dtype=torch.float32) for x in (q0, v0, u0)))
    np.testing.assert_allclose(sim.data.qpos.numpy(), q1.numpy(), atol=1e-6)
    np.testing.assert_allclose(sim.data.qvel.numpy(), v1.numpy(), atol=1e-6)
    np.testing.assert_allclose(sim.data.time.numpy(), t0 + 0.01, atol=1e-7)
    d = mujoco.MjData(m)
    for w in range(3):  # the derived fields and the sensors belong to the state before the step, as MuJoCo's lag
        d.qpos[:] = q0[w]
        mujoco.mj_kinematics(m, d)
        np.testing.assert_allclose(sim.data.xpos.numpy()[w], d.xpos, atol=1e-5)
        np.testing.assert_allclose(sim.data.sensordata.numpy()[w], [q0[w, 0]], atol=1e-6)
    sim.check()
    vel = ndm_sim()(m, 3, substeps=1, device="cpu", ndm={0.01: wm}, derived=("kinematics", "transmission", "actuator_velocity"), compile=False)
    vel.data.qpos.assign(q0.astype(np.float32))
    vel.data.qvel.assign(v0.astype(np.float32))
    vel.step()  # the actuator's velocity of the state before the step: the cart's slide velocity (gear 1)
    np.testing.assert_allclose(vel.data.actuator_velocity.numpy()[:, 0], v0[:, 0], atol=1e-6)
    resized = sim.resized(5)
    assert type(resized) is type(sim) and resized.nworld == 5 and resized.ndm[0.01] is wm


def plan(m, sim, steps=6, stages=None):
    from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
    from robokit.sim import MjWarpSim
    from robokit.terms import SensorTask

    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    kw = {} if stages is None else {"stages": stages}
    config = SimMPCConfig(num_samples=8, horizon_steps=8, node_steps=4, ctrl_steps=2, max_iterations=2, ctrl_std=0.1, **kw)
    ctrl_ref = np.zeros((steps + config.horizon_steps, m.nu))
    config.add(SensorTask(np.full((len(ctrl_ref), m.nsensordata), 0.3), weight={"x": 100.0}))
    with planning():
        mpc = SimMPC(config, sim, ctrl_ref)
        mpc.execute(MjWarpSim(m, 1, data, device="cpu"), steps)
    return mpc


def test_simmpc_rolls_out_on_the_world_model():
    from robokit.helpers.sim_mpc import SimMPCStage

    m = mujoco.MjModel.from_xml_string(CART)
    wm = Counted(world_model(m))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    sim = ndm_sim()(m, 1, data, device="cpu", ndm={0.01: wm}, derived=("kinematics", "sensor_pos"), compile=False)
    mpc = plan(m, sim, stages=[SimMPCStage(2)])  # the stages fixed at the timestep, as an NDM run fixes MuJoCo Warp's choice
    assert type(mpc.sims["rollout"]) is type(sim) and mpc.sims["rollout"].nworld == 7
    assert 7 in wm.calls and len(wm.calls) >= 6 * 8  # every plan's rollouts step the 7 draws on the world model
    for s in mpc.sims.values():
        s.check()
    with pytest.raises(ValueError, match="a step of 0.04 s"):  # no stages: the first plan's automatic check tries 4 and 2 timesteps
        plan(m, ndm_sim()(m, 1, data, device="cpu", ndm={0.01: world_model(m)}, derived=("kinematics", "sensor_pos"), compile=False))


def test_a_staged_plan_uses_each_lengths_world_model_and_refuses_a_missing_one():
    from robokit.helpers.sim_mpc import SimMPCStage

    m = mujoco.MjModel.from_xml_string(CART)
    fine, coarse = Counted(world_model(m)), Counted(world_model(m, dt=0.02, seed=1))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    stages = [SimMPCStage(2, [(0.5, None), (1.0, 0.02)])]  # half the horizon at the timestep, the rest at twice it
    sim = ndm_sim()(m, 1, data, device="cpu", ndm={0.01: fine, 0.02: coarse}, derived=("kinematics", "sensor_pos"), compile=False)
    mpc = plan(m, sim, stages=stages)
    assert fine.calls and coarse.calls and 7 in coarse.calls  # the coarse steps on the coarse world model
    for s in mpc.sims.values():
        s.check()
    only_fine = ndm_sim()(m, 1, data, device="cpu", ndm={0.01: world_model(m)}, derived=("kinematics", "sensor_pos"), compile=False)
    with pytest.raises(ValueError, match="a step of 0.02 s"):
        plan(m, only_fine, stages=stages)


def test_a_hybrid_plan_runs_the_physics_at_the_lengths_it_names():
    from robokit.helpers.sim_mpc import SimMPCStage
    from robokit.sim import MjWarpSim

    m = mujoco.MjModel.from_xml_string(CART)
    phys, hyb = MjWarpSim(m, 2, device="cpu"), ndm_sim()(m, 2, device="cpu", ndm={0.02: world_model(m, dt=0.02)}, physics=(0.01,),
                                                        derived=(), compile=False)  # fmt: skip
    phys.step()
    hyb.step()  # the timestep's length: MuJoCo Warp's own step
    np.testing.assert_array_equal(hyb.data.qpos.numpy(), phys.data.qpos.numpy())
    np.testing.assert_array_equal(hyb.data.qvel.numpy(), phys.data.qvel.numpy())
    coarse = Counted(world_model(m, dt=0.02, seed=1))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    stages = [SimMPCStage(2, [(0.5, None), (1.0, 0.02)])]  # the fine half on the physics, the coarse rest on the world model
    sim = ndm_sim()(m, 1, data, device="cpu", ndm={0.02: coarse}, physics=(0.01,), derived=("kinematics", "sensor_pos"), compile=False)
    mpc = plan(m, sim, stages=stages)
    assert coarse.calls
    for s in mpc.sims.values():
        s.check()
    with pytest.raises(ValueError, match="also to run the physics"):
        ndm_sim()(m, 1, device="cpu", ndm={0.01: world_model(m)}, physics=(0.01,), derived=(), compile=False)


def test_a_timestep_set_behind_its_back_is_flagged():
    m = mujoco.MjModel.from_xml_string(CART)
    sim = ndm_sim()(m, 2, device="cpu", ndm={0.01: world_model(m)}, derived=(), compile=False)
    sim.step()
    sim.check()
    sim.opt.timestep.assign(np.full(sim.opt.timestep.shape, 0.02, dtype=np.float32))  # not through planning()
    sim.step()
    with pytest.raises(RuntimeError, match="timestep"):
        sim.check()
    with pytest.raises(ValueError, match="not MuJoCo Warp stages"):
        ndm_sim()(m, 2, device="cpu", ndm={0.01: world_model(m)}, derived=("kinematix",), compile=False)
    with pytest.raises(ValueError, match="needs a GPU"):
        ndm_sim()(m, 2, device="cpu", ndm={0.01: world_model(m)})
    with pytest.raises(ValueError, match="step others"):  # a 0.01 s model given for 0.02 s
        ndm_sim()(m, 2, device="cpu", compile=False, ndm={0.02: world_model(m)})


def test_plan_robokit_plans_on_the_world_model_and_executes_on_mujoco_warp(tmp_path):
    import json

    from test_robokit_rec import CART as CART_REC
    from test_robokit_rec import FAKE_EXAMPLE

    from ndm_gym.wmbench.plan_robokit import main

    m = mujoco.MjModel.from_xml_string(CART_REC)
    wm = world_model(m)
    ckpt = tmp_path / "cart.pt"
    torch.save({"state": wm.state_dict(), "layout": json.dumps(MjcfLayout.from_model(m).__dict__),
                "args": {"hidden": [16], "act": "silu", "no_ln": False, "wm_param": "accel", "dt": None}}, ckpt)  # fmt: skip
    example = tmp_path / "cart_example.py"
    example.write_text(FAKE_EXAMPLE.format(xml=CART_REC))
    main(["--example", str(example), "--ndm", f"0.01={ckpt}", "--dtype", "fp32", "--derived", "kinematics", "sensor_pos",
          "--stages", "fine", "--no_compile", "--out", str(tmp_path / "ndm.json")])  # fmt: skip
    run = json.loads((tmp_path / "ndm.json").read_text())
    assert run["planners"][0]["sim"] == "NdmSim" and "SimMPCStage" in run["planners"][0]["stages"]
    assert any("cart: pass" in line for line in run["output"])
    main(["--example", str(example), "--out", str(tmp_path / "base.json")])  # the example as it is
    with pytest.raises(ValueError, match="--null plans"):
        main(["--example", str(example), "--null", "--ndm", f"0.01={ckpt}", "--out", str(tmp_path / "x.json")])
    main(["--example", str(example), "--save_reference", str(tmp_path / "ref.npz"), "--out", str(tmp_path / "ref.json")])
    ref = np.load(tmp_path / "ref.npz")  # the cart's SensorTask target, as the example built it; nothing planned
    assert ref["sensor_ref"].ndim == 2 and float(ref["ref_dt"]) == pytest.approx(m.opt.timestep) and len(ref["sensor_name"]) == m.nsensor
    assert (tmp_path / "ref.mjb").exists() and not (tmp_path / "ref.json").exists()
    assert json.loads((tmp_path / "base.json").read_text())["planners"][0]["sim"] == "MjWarpSim"
