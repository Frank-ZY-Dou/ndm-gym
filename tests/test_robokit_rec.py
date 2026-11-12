"""Recording RoboKit's MuJoCo Warp simulator (ndm_gym.sim.robokit_rec) on Warp's CPU device: stepped directly, the rows are
the simulator's states before and after each step, end to end; inside RoboKit's own SimMPC (its documentation's cart, run in
closed loop) the rollouts are recorded and turned into a recording whose resets sit at the rollouts' starts; a ring that
went round is an error. Needs RoboKit with MuJoCo Warp (skipped without)."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
pytest.importorskip("robokit")
wp = pytest.importorskip("warp")

from ndm_gym.sim.robokit_rec import recording, recording_sim  # noqa: E402

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


def test_rows_are_the_simulators_states():
    m = mujoco.MjModel.from_xml_string(CART)
    sim = recording_sim()(m, 4, substeps=2, device="cpu", record=3, capacity=16)
    seen = []
    for t in range(5):
        sim.data.ctrl.assign(np.full((4, 1), 0.1 * t))
        before = (sim.data.qpos.numpy().copy(), sim.data.qvel.numpy().copy())
        sim.step()
        seen.append((before, (sim.data.qpos.numpy().copy(), sim.data.qvel.numpy().copy())))
    rows = sim.drain()
    assert rows["qpos"].shape == (5, 3, m.nq) and rows["ctrl"].shape == (5, 3, 1)
    for t, ((q0, v0), (q1, v1)) in enumerate(seen):
        np.testing.assert_allclose(rows["qpos"][t], q0[:3], atol=1e-6)
        np.testing.assert_allclose(rows["qvel"][t], v0[:3], atol=1e-6)
        np.testing.assert_allclose(rows["qpos1"][t], q1[:3], atol=1e-6)
        np.testing.assert_allclose(rows["qvel1"][t], v1[:3], atol=1e-6)
        np.testing.assert_allclose(rows["ctrl"][t], 0.1 * t, atol=1e-6)
    np.testing.assert_allclose(rows["dt"], 0.02, atol=1e-9)  # two substeps of 10 ms
    rec = recording(rows, 0.02)
    assert not rec["reset"].any() and rec["qpos"].shape == (6, 3, m.nq)
    np.testing.assert_allclose(rec["qpos"][1:], rec["end_qpos"], atol=1e-6)
    assert sim.drain()["qpos"].shape[0] == 0  # nothing new since
    resized = sim.resized(8)
    assert type(resized) is type(sim) and resized.nworld == 8 and resized.record == 3
    for _ in range(17):
        sim.step()
    with pytest.raises(RuntimeError, match="raise capacity"):
        sim.drain()


def test_simmpc_rollouts_are_recorded():
    from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
    from robokit.terms import SensorTask

    m = mujoco.MjModel.from_xml_string(CART)
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    RecordingSim = recording_sim()
    config = SimMPCConfig(num_samples=8, horizon_steps=6, node_steps=3, ctrl_steps=3, max_iterations=2, ctrl_std=0.1)
    ctrl_ref = np.zeros((9 + config.horizon_steps, m.nu))
    config.add(SensorTask(np.full((len(ctrl_ref), m.nsensordata), 0.3), weight={"x": 100.0}))
    planner_sim = RecordingSim(m, 1, data, device="cpu", record=4, capacity=4096)
    mpc = SimMPC(config, planner_sim, ctrl_ref)
    rollout = mpc.sims["rollout"] if hasattr(mpc, "sims") and "rollout" in mpc.sims else None
    assert rollout is not None and type(rollout) is RecordingSim
    drained = []
    (row,), _ = mpc.execute(mujoco_warp_sim(m, data), 9, callback=lambda infos: drained.append(rollout.drain()))
    rows = {k: np.concatenate([d[k] for d in drained]) for k in drained[0]}
    assert len(rows["qpos"]) >= 9  # three plans, each at least one iteration of a 6-step horizon
    rec = recording(rows, m.opt.timestep)
    T = rec["ctrl"].shape[0]
    assert rec["reset"].any() and rec["reset"].sum(0).min() >= 1  # each rollout ends where the next one restarts
    ok = ~rec["reset"]
    np.testing.assert_allclose(rec["qpos"][1:][ok], rec["end_qpos"][ok], atol=1e-6)  # a kept step's successor is its end
    assert T <= len(rows["qpos"]) and row["qpos"].shape == (9, m.nq)  # rows of a longer step (a coarse stage) are left out


def mujoco_warp_sim(m, data):
    from robokit.sim import MjWarpSim

    return MjWarpSim(m, 1, data, device="cpu")


FAKE_EXAMPLE = """
import mujoco, numpy as np, viser
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.sim import MjWarpSim
from robokit.terms import SensorTask

XML = {xml!r}

def main():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    config = SimMPCConfig(num_samples=8, horizon_steps=6, node_steps=3, ctrl_steps=3, max_iterations=2, ctrl_std=0.1)
    ctrl_ref = np.zeros((9 + config.horizon_steps, m.nu))
    config.add(SensorTask(np.full((len(ctrl_ref), m.nsensordata), 0.3), weight={{"x": 100.0}}))
    sim = MjWarpSim(m, 1, d)
    mpc = SimMPC(config, sim, ctrl_ref)
    (rollout,), _ = mpc.execute(sim, 9)
    print("cart: pass")
    viser.ViserServer()
    raise AssertionError("the viewer should have ended the run")

if __name__ == "__main__":
    main()
"""


def test_collect_robokit_on_an_example(tmp_path):
    import json

    from ndm_gym.wmbench.collect_robokit import main

    example = tmp_path / "cart_example.py"
    example.write_text(FAKE_EXAMPLE.format(xml=CART))
    out = tmp_path / "rec" / "cart_s3"
    main(["--example", str(example), "--out", str(out), "--record", "4", "--seed", "3"])
    meta = json.loads((tmp_path / "rec" / "cart_s3.json").read_text())
    assert meta["seed"] == 3 and any("cart: pass" in line for line in meta["output"])
    (pl,) = meta["planners"]
    assert pl["worlds"] == {"rollout": 7, "validation": 1} and pl["num_samples"] == 8 and pl["horizon_steps"] == 6
    assert mujoco.MjModel.from_binary_path(pl["model"]).nq == 8
    rollout = [p for p in meta["files"] if "_rollout_dt" in p]
    assert rollout and all(meta["files"][p]["worlds"] == 4 for p in rollout)
    z = np.load(rollout[0])
    assert z["qpos"].shape[1:] == (4, 8) and z["reset"].any() and float(z["dt"]) == 0.01  # the model's timestep exactly, as fit_mjcf checks
    e = np.load(str(out) + "_executed0.npz")
    assert e["qpos"].shape == (9, 1, 8) and e["ctrl"].shape == (8, 1, 1) and abs(float(e["dt"]) - 0.01) < 1e-9
