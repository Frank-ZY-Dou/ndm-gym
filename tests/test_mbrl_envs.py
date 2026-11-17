"""mbrl-lib's environments for its true-dynamics controller on batched simulators (ndm_gym.sim.mbrl_envs, sim.mbrl_ndm): the
batched rewards and terminations against the environments' own step (check_mbrl_rewards's check), the pusher's named
difference, the refusals, the torch forward kinematics against MuJoCo's, the NDM rollouts' returns,
terminations, truncations, dm_control inner steps and kinematics against direct counts, the run-time fix of mbrl-lib's
dm_control step counter, MuJoCo Warp's rollouts against control_env.py's pool (slow: Warp compiles its CPU kernels), the
recorder's format, short recordings and plans (slow), and the study's simulator comparison on a recording (slow). They need
mbrl-lib at 3f93ccc on the path with gymnasium 0.26 (the study's environment); hopper and walker2d need MBRL_LOCAL_MODELS, the
directory of their local-coordinate copies (tools/convert_mbrl_models.py makes them).
MBRL_TEST_DEVICE (default cpu; cuda:0 on a GPU) is where the MuJoCo Warp and world-model rollouts run: on a GPU through their
captured graphs, and the compiled world model's rollouts are checked against the eager ones."""

import json
import math
import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

mbrl = pytest.importorskip("mbrl")
gymnasium = pytest.importorskip("gymnasium")
if not gymnasium.__version__.startswith("0.26"):
    pytest.skip(f"mbrl-lib needs gymnasium 0.26, not {gymnasium.__version__}", allow_module_level=True)

from ndm_gym.sim.mbrl_envs import (  # noqa: E402
    ENVS, Kinematics, Scorer, contact_margin_rule, make_env, patch_envs, physics, reads_global_coordinates, refusal, set_own_ccd, set_own_invweight,
    set_own_margins, state, unpatch_envs,
)  # fmt: skip

LOCAL = os.environ.get("MBRL_LOCAL_MODELS")
DEVICE = os.environ.get("MBRL_TEST_DEVICE", "cpu")
FAST = ["pets_cartpole", "pets_halfcheetah", "pets_pusher", "pets_reacher", "inverted_pendulum", "hopper", "walker2d", "ant"]
DMC = ["dmc_cartpole_swingup", "dmc_cheetah_run", "dmc_cup_catch", "dmc_finger_spin", "dmc_walker_walk"]


@pytest.fixture(autouse=True)
def patched():
    patch_envs(LOCAL)
    yield
    unpatch_envs()


def _needs(name):
    if name in ("hopper", "walker2d") and not LOCAL:
        pytest.skip("MBRL_LOCAL_MODELS is not set")
    if name.startswith("dmc_"):
        pytest.importorskip("dm_control")


@pytest.mark.parametrize("name", FAST + DMC)
def test_batched_rewards_match_env_step(name):
    _needs(name)
    from ndm_gym.wmbench.check_mbrl_rewards import check, parse_args

    res = check(ENVS[name], parse_args(["--steps", "80", "--perturb", "--seed", "3"]))
    assert res["result"] == "PASS", res
    assert res["termination_mismatches"] == 0 and res["max_abs_err"] < 1e-5
    if name == "pets_pusher":  # the named difference, measured beside the check
        n = res["named_difference"]
        assert n["steps"] > 0 and 0 < n["position_median"] <= n["position_max"] and 0 < n["reward_median"] <= n["reward_max"]


def test_refusals():
    for name in ("ant", "humanoid"):
        assert "cfrc_ext" in refusal(ENVS[name], "ndm") and refusal(ENVS[name], "cpu") is None
    assert "PGS" in refusal(ENVS["humanoid"], "warp") and "1e-4" in refusal(ENVS["ant"], "warp")
    assert all("pets_reacher is not run" in refusal(ENVS["pets_reacher"], s) for s in ("cpu", "warp", "ndm"))
    ok = ["pets_cartpole", "pets_halfcheetah", "pets_pusher", "hopper", "walker2d", "inverted_pendulum"] + DMC
    assert all(refusal(ENVS[n], s) is None for n in ok for s in ("cpu", "warp", "ndm"))  # the NDM plans on these 11
    assert ENVS["pets_pusher"].named and all(not ENVS[n].named for n in ok if n != "pets_pusher")
    assert {n for n, e in ENVS.items() if e.planar} == {"pets_halfcheetah", "hopper", "walker2d", "dmc_cheetah_run", "dmc_walker_walk"}
    _, env = make_env(ENVS["pets_cartpole"], 0)
    with pytest.raises(NotImplementedError, match="CPU MuJoCo only"):
        Scorer(ENVS["humanoid"], physics(ENVS["pets_cartpole"], env)[0], 5).reward(None, None, torch.zeros(1, 1))


@pytest.mark.parametrize("name", ["pets_pusher", "pets_halfcheetah", "dmc_cheetah_run", "dmc_walker_walk", "dmc_cup_catch", "dmc_cartpole_swingup"])
def test_kinematics_is_mujocos(name):
    _needs(name)
    import mujoco

    spec = ENVS[name]
    _, env = make_env(spec, 0)
    m = physics(spec, env)[0]
    rng, d = np.random.default_rng(1), mujoco.MjData(m)
    q, v = np.tile(m.qpos0, (8, 1)) + 0.7 * rng.standard_normal((8, m.nq)), 2.0 * rng.standard_normal((8, m.nv))
    got = Kinematics(m, dtype=torch.float64)(torch.tensor(q), torch.tensor(v))
    for i in range(8):
        d.qpos[:], d.qvel[:] = q[i], v[i]
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        for k, ref in (("xpos", d.xpos), ("xmat", d.xmat.reshape(-1, 3, 3)), ("xipos", d.xipos), ("site_xpos", d.site_xpos),
                       ("subtree_linvel", d.subtree_linvel)):  # fmt: skip
            np.testing.assert_allclose(got[k][i].numpy(), ref, atol=1e-12)
    with pytest.raises(ValueError, match="free or ball"):
        Kinematics(physics(ENVS["ant"], make_env(ENVS["ant"], 0)[1])[0])


class _Stub(torch.nn.Module):
    """A stand-in world model of one step of ``dt``: qpos += dt qvel, qvel += dt ctrl[:, 0] (every degree of freedom)."""

    def __init__(self, dt, nq, nv, nu, still=False):
        super().__init__()
        self.lay, self.still = types.SimpleNamespace(dt=dt, nq=nq, nv=nv, nu=nu), still
        self.in_norm = types.SimpleNamespace(mean=torch.zeros(1, device=DEVICE))

    def step(self, q, v, u):
        return (q, v) if self.still else (q + self.lay.dt * v, v + self.lay.dt * u[:, :1])


def _rollouts(name, still=False, **kw):
    from ndm_gym.sim.mbrl_ndm import NdmRollouts

    spec = ENVS[name]
    handler, env = make_env(spec, 0)
    m, _, substeps, _ = physics(spec, env)
    wm = _Stub(float(m.opt.timestep) * substeps, m.nq, m.nv, m.nu, still)
    return spec, handler, env, m, NdmRollouts(spec, m, substeps, wm, **kw)


def test_ndm_rollouts_count_to_the_termination_and_the_time_limit():
    spec, _, _, m, ro = _rollouts("inverted_pendulum", nworld=3)  # reward 1 per step until |qpos[1]| > 0.2, which ends the rollout
    dt = ro.wm.lay.dt
    st = {"qpos": np.array([0.0, 0.1]), "qvel": np.array([0.0, 0.0]), "steps_left": 1000}
    acts = torch.tensor([0.0, 25.0, -25.0])[:, None, None].expand(3, 20, 1).float()
    got = ro.values(st, acts)
    want = []
    for a in (0.0, 25.0, -25.0):  # the same integration in float64, counted by hand
        q, v, n = 0.1, 0.0, 0
        for _ in range(20):
            q, v, n = q + dt * v, v + dt * a, n + 1
            if abs(q) > 0.2:
                break
        want.append(n)
    assert got.tolist() == want and want[0] == 20 and want[1] < 20
    assert ro.values({**st, "steps_left": 4}, acts).tolist() == [min(4, w) for w in want]  # truncated after the steps left
    from ndm_gym.sim.mbrl_ndm import NdmRollouts

    with pytest.raises(ValueError, match="control step"):
        NdmRollouts(spec, m, 2, _Stub(2 * dt, 2, 2, 1), nworld=3)
    with pytest.raises(ValueError, match="cfrc_ext"):
        NdmRollouts(ENVS["ant"], m, 2, _Stub(dt, 2, 2, 1), nworld=3)


def test_ndm_steps_dm_controls_control_step_repeat_times():
    _needs("dmc_finger_spin")
    spec, handler, env, m, ro = _rollouts("dmc_finger_spin", nworld=2)  # reward 1 per control step at a hinge velocity <= -15
    R, hinge = spec.repeat, int(m.jnt_dofadr[m.joint("hinge").id])
    st = state(spec, handler, env)
    qvel = np.zeros(m.nv)
    qvel[hinge] = -16.0
    st = {**st, "qpos": np.zeros(m.nq), "qvel": qvel}
    acts = torch.tensor([0.0, 20.0])[:, None, None].expand(2, 5, m.nu).float()  # the second world's hinge slows past -15
    got = ro.values(st, acts)
    v, n = -16.0, 0
    for _ in range(5 * R):  # 5 environment steps of R control steps, a reward after each
        v += ro.wm.lay.dt * 20.0
        n += int(v <= -15.0)
    assert got.tolist() == [5 * R, n] and 0 < n < 5 * R
    assert ro.values({**st, "control_left": 2 * R + 1}, acts).tolist()[0] == 2 * R + 1  # dm_control's time limit inside a step


def test_ndm_pusher_reads_the_kinematics_of_its_state():
    """With a world model that holds the state, every step's reward is the first one's, which env.step gives from a restored
    state (set_state's mj_forward: the kinematics are the state's own)."""
    spec, handler, env, m, ro = _rollouts("pets_pusher", still=True, nworld=1)
    st = state(spec, handler, env)
    handler.set_env_state(handler.get_current_state(env), env)
    _, r0, _, _, _ = env.step(np.zeros(m.nu, dtype=np.float32))
    np.testing.assert_allclose(ro.values(st, torch.zeros(1, 5, m.nu)).item(), 5 * r0, rtol=1e-5)


def test_counter_fix_keeps_dm_controls_rollouts_whole():
    _needs("dmc_cheetah_run")
    spec = ENVS["dmc_cheetah_run"]  # R = 4: gym's TimeLimit at 250 steps

    def lengths():
        handler, env = make_env(spec, 0)
        cur, out = handler.get_current_state(env), []
        for _ in range(4):  # as control_env.py's pool worker scores sequences of 100 steps from one state
            handler.set_env_state(cur, env)
            out.append(len(handler.rollout_env(env, env.observation_space.sample(), -1, agent=None, plan=np.zeros((100, 6), np.float32))[1]))
        return out

    assert lengths() == [100, 100, 50, 1]  # as shipped: the worker's counter runs on across rollouts
    unpatch_envs()
    patch_envs(LOCAL, counter_fix=True)
    assert lengths() == [100] * 4


@pytest.mark.parametrize("module", ["plan_mbrl", "collect_mbrl"])
def test_pets_reacher_is_refused(module, tmp_path):
    import importlib

    main = importlib.import_module(f"ndm_gym.wmbench.{module}").main
    with pytest.raises(ValueError, match="pets_reacher is not run"):
        main(["--env", "pets_reacher", "--out", str(tmp_path / "x")])
    with pytest.raises(ValueError, match="counter_fix"):
        main(["--env", "pets_cartpole", "--counter_fix", "--out", str(tmp_path / "x")])


def _margins(*margins):
    import mujoco

    geoms = "".join(f'<body pos="{i} 0 0.1"><freejoint/><geom type="sphere" size="0.1" margin="{g}"/></body>' for i, g in enumerate(margins[1:]))
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom type="plane" size="9 9 1" margin="{margins[0]}"/>{geoms}</worldbody></mujoco>')
    return m, mujoco.MjData(m)


def test_own_margins_are_the_larger_of_two():
    import mujoco

    rule = contact_margin_rule()
    assert rule in ("max", "sum")
    m, d = _margins(0.01, 0.01, 0.01)
    mujoco.mj_forward(m, d)
    assert [c.includemargin for c in d.contact[: d.ncon]] == pytest.approx([0.01 if rule == "max" else 0.02] * 2)  # the installed MuJoCo's own
    set_own_margins(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 2 and [c.includemargin for c in d.contact[: d.ncon]] == pytest.approx([0.01, 0.01])  # the larger of the two geoms'
    assert m.geom_margin.tolist() == pytest.approx([0.01] * 3 if rule == "max" else [0.005] * 3)
    m, _ = _margins(0.0, 0.0)
    set_own_margins(m)  # a model without margins stays as it is
    assert not m.geom_margin.any()
    if rule == "sum":
        with pytest.raises(ValueError, match="margins differ"):
            set_own_margins(_margins(0.0, 0.01)[0])


def _needs_exact():
    """own_margins and own_mujoco refuse copies with rounded joint ranges (those made before tools/convert_mbrl_models.py): the
    tests of the flags on Hopper need the exact ones."""
    import mujoco

    _needs("hopper")
    if abs(mujoco.MjModel.from_xml_path(os.path.join(LOCAL, "hopper_local.xml")).jnt_range[5, 1] - math.radians(45)) > 1e-12:
        pytest.skip(f"MBRL_LOCAL_MODELS={LOCAL} holds copies with rounded joint ranges: tools/convert_mbrl_models.py makes the exact ones")


def test_patch_envs_own_margins_on_hopper():
    _needs_exact()
    spec = ENVS["hopper"]
    before = physics(spec, make_env(spec, 0)[1])[0].geom_margin.copy()
    with pytest.raises(RuntimeError, match="already patched"):
        patch_envs(LOCAL, own_margins=True)
    unpatch_envs()
    patch_envs(LOCAL, own_margins=True)
    after = physics(spec, make_env(spec, 0)[1])[0].geom_margin
    assert before.tolist() == pytest.approx([0.001] * 5) and after.tolist() == pytest.approx([0.001 if contact_margin_rule() == "max" else 0.0005] * 5)
    unpatch_envs()
    patch_envs(LOCAL, counter_fix=True, own_margins=True)
    if "dm_control" in sys.modules or __import__("importlib").util.find_spec("dm_control"):
        with pytest.raises(ValueError, match="own_margins is for the gymnasium"):
            make_env(ENVS["dmc_cheetah_run"], 0)


def test_own_invweight_of_a_slider_body():
    import mujoco

    xml = ('<mujoco><worldbody><body><joint type="slide" axis="1 0 0" armature="0.04"/><joint type="slide" axis="0 1 0" armature="0.04"/>'
           '<geom type="sphere" size="0.05" density="0.00001"/></body><body pos="1 0 0"><joint type="hinge" axis="0 0 1" armature="0.04"/>'
           '<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/></body></worldbody></mujoco>')  # fmt: skip
    m = mujoco.MjModel.from_xml_string(xml)
    mass, dof, body = float(m.body_mass[1]), m.dof_invweight0.copy(), m.body_invweight0.copy()
    if m.body_simple[1] == 2:  # the installed MuJoCo's own: the inverse of the mass, without the armature
        assert dof[:2] == pytest.approx([1 / mass] * 2, rel=1e-9) and dof[0] > 1e6
    set_own_invweight(m)
    assert m.dof_invweight0[:2] == pytest.approx([1 / (0.04 + mass)] * 2, rel=1e-12)  # MuJoCo 2.3.3's: the diagonal of M^-1, armature in M
    assert m.body_invweight0[1] == pytest.approx([2 / (3 * (0.04 + mass)), 0.0], rel=1e-12, abs=1e-300)
    assert np.array_equal(m.dof_invweight0[2:], dof[2:]) and np.array_equal(m.body_invweight0[2], body[2])  # the hinged body stays


def test_patch_envs_own_mujoco_on_the_pusher():
    import mujoco

    spec = ENVS["pets_pusher"]
    before = physics(spec, make_env(spec, 0)[1])[0]
    assert before.geom_margin.tolist() == pytest.approx([0.002] * before.ngeom)
    unpatch_envs()
    patch_envs(LOCAL, own_mujoco=True)
    m = physics(spec, make_env(spec, 0)[1])[0]
    assert m.geom_margin.tolist() == pytest.approx([0.002 if contact_margin_rule() == "max" else 0.001] * m.ngeom)
    # the object's and the goal's slides, as mbrl-lib's own MuJoCo 2.3.3 computes them
    assert m.dof_invweight0[7:].tolist() == pytest.approx([24.999992, 24.999992, 25.0, 25.0], rel=1e-7)
    assert m.body_invweight0[11:, 0].tolist() == pytest.approx([16.666661, 16.666667], rel=1e-7) and not m.body_invweight0[11:, 1].any()
    assert np.array_equal(m.dof_invweight0[:7], before.dof_invweight0[:7]) and np.array_equal(m.body_invweight0[:11], before.body_invweight0[:11])
    bits = mujoco.mjtDisableBit
    if hasattr(bits, "mjDSBL_NATIVECCD"):
        both = int(bits.mjDSBL_NATIVECCD) | int(bits.mjDSBL_MULTICCD)
        assert m.opt.disableflags & both == both and m.opt.ccd_iterations == 50 and not before.opt.disableflags & both
        plain = mujoco.MjModel.from_xml_string("<mujoco/>")
        set_own_ccd(plain)
        assert plain.opt.disableflags == both
    unpatch_envs()
    patch_envs(LOCAL, counter_fix=True, own_mujoco=True)
    if "dm_control" in sys.modules or __import__("importlib").util.find_spec("dm_control"):
        with pytest.raises(ValueError, match="own_mujoco is for the gymnasium"):
            make_env(ENVS["dmc_cheetah_run"], 0)


def test_missing_copies_name_the_command_that_makes_them(tmp_path):
    unpatch_envs()
    with pytest.raises(FileNotFoundError, match=r"no hopper_local.xml in .*tools/convert_mbrl_models.py --out"):
        patch_envs(str(tmp_path))  # a directory without the copies
    if reads_global_coordinates():
        return  # this MuJoCo reads gymnasium's own files: no copy is needed without a directory
    patch_envs(None)
    with pytest.raises(FileNotFoundError, match=r"global coordinates.*tools/convert_mbrl_models.py --out data/mbrl_models"):
        make_env(ENVS["hopper"], 0)
    make_env(ENVS["pets_cartpole"], 0)  # a model that needs no copy still builds


def test_own_flags_refuse_copies_with_rounded_ranges(tmp_path):
    _needs_exact()
    for f in ("hopper_local.xml", "walker2d_local.xml"):  # as mj_saveLastXML writes them: six digits
        text = open(os.path.join(LOCAL, f)).read()
        (tmp_path / f).write_text(re.sub(r'\brange="([^"]*)"', lambda m: 'range="%s"' % " ".join(f"{float(x):.6g}" for x in m.group(1).split()), text))
    spec = ENVS["hopper"]
    unpatch_envs()
    patch_envs(str(tmp_path))  # without the flags they load, as before
    assert physics(spec, make_env(spec, 0)[1])[0].jnt_range[5].tolist() == [-0.785398, 0.785398]
    for flag in ("own_margins", "own_mujoco"):
        unpatch_envs()
        patch_envs(str(tmp_path), **{flag: True})
        with pytest.raises(ValueError, match=r"rounded joint ranges.*tools/convert_mbrl_models.py --out"):
            make_env(spec, 0)
    unpatch_envs()
    patch_envs(LOCAL, own_mujoco=True)  # the exact copies pass
    assert physics(spec, make_env(spec, 0)[1])[0].jnt_range[5].tolist() == [-math.radians(45), math.radians(45)]


def test_recording_joins_segments_with_a_reset_step():
    from ndm_gym.wmbench.collect_mbrl import recording
    from ndm_gym.wmbench.fit_mjcf import starts

    seg = lambda T: (np.random.rand(T + 1, 2, 3), np.random.rand(T + 1, 2, 3), np.random.rand(T, 2, 1))  # noqa: E731
    a, b = seg(4), seg(3)
    rec = recording([a, b], 0.05)
    assert rec["qpos"].shape == (9, 2, 3) and rec["ctrl"].shape == (8, 2, 1) and rec["dt"] == 0.05
    assert rec["reset"][:, 0].tolist() == [False] * 4 + [True] + [False] * 3 and (rec["ctrl"][4] == 0).all()
    assert np.array_equal(rec["qpos"][5:], b[0]) and np.array_equal(rec["ctrl"][5:], b[2])
    windows = {(int(t), int(n)) for t, n in starts({"reset": torch.as_tensor(rec["reset"])}, 2).tolist()}
    assert windows == {(t, n) for t in (0, 1, 2, 5, 6) for n in (0, 1)}  # no window holds the reset step


def test_state_of_an_environment():
    spec = ENVS["pets_reacher"]
    handler, env = make_env(spec, 0)
    env.step(env.action_space.sample())
    st = state(spec, handler, env)
    _, d, _, _ = physics(spec, env)
    assert np.array_equal(st["qpos"], d.qpos) and st["steps_left"] == 999 and np.array_equal(st["goal"], env.unwrapped.goal)


def _speed_files(tmp_path, steps=6):
    """A cart-pole's model, a recording of ``steps`` random steps and their meta, as collect_mbrl writes them; the recorded arrays."""
    import mujoco

    spec = ENVS["pets_cartpole"]
    _, env = make_env(spec, 0)
    m, d, substeps, _ = physics(spec, env)
    rng, q, v, u = np.random.default_rng(0), [d.qpos.copy()], [d.qvel.copy()], []
    for _ in range(steps):
        a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        env.step(a)
        q.append(d.qpos.copy())
        v.append(d.qvel.copy())
        u.append(a.astype(np.float64))
    q, v, u = np.stack(q)[:, None], np.stack(v)[:, None], np.stack(u)[:, None]
    mujoco.mj_saveModel(m, str(tmp_path / "cp_model.mjb"), None)
    np.savez(tmp_path / "cp.npz", qpos=q, qvel=v, ctrl=u, reset=np.zeros(u.shape[:2], bool), dt=np.float64(0.04))
    meta = {"env": spec.name, "model": str(tmp_path / "cp_model.mjb"), "dt": 0.04, "substeps": substeps}
    json.dump(meta, open(tmp_path / "cp.json", "w"))
    return m, substeps, q[:, 0], v[:, 0], u[:, 0], meta


def test_speed_mbrl_cpu_steps_the_environments_control_step(tmp_path):
    import mujoco

    from ndm_gym.wmbench.speed_mbrl import cpu_step, main

    m, substeps, q, v, u, meta = _speed_files(tmp_path)
    q1, v1 = q[:-1].copy(), v[:-1].copy()
    cpu_step(m, mujoco.MjData(m), substeps, q1, v1, u, np.zeros_like(v1))  # every recorded state one control step on
    np.testing.assert_allclose(q1, q[1:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(v1, v[1:], rtol=0, atol=1e-12)
    out = tmp_path / "speed.csv"
    args = ["--meta", str(tmp_path / "cp.json"), "--states", str(tmp_path / "cp.npz"), "--what", "cpu", "--nworld", "1", "3", "--steps", "4",
            "--warmup", "1", "--repeats", "1", "--reset_every", "2", "--out", str(out)]  # fmt: skip
    main(args)
    rows = [x.split(",") for x in open(out).read().splitlines()]
    assert rows[0][:5] == ["what", "model", "nworld", "substeps", "dt"] and [r[:5] for r in rows[1:]] == [
        ["cpu", "cp_model.mjb", "1", str(substeps), "0.04"], ["cpu", "cp_model.mjb", "3", str(substeps), "0.04"]]  # fmt: skip
    assert all(float(r[9]) > 0 and "reset every 2" in r[12] for r in rows[1:])
    json.dump({**meta, "dt": 0.05}, open(tmp_path / "bad.json", "w"))
    with pytest.raises(ValueError, match="control step"):
        main(["--meta", str(tmp_path / "bad.json")] + args[2:])


@pytest.mark.slow
def test_speed_mbrl_warp_row(tmp_path):
    pytest.importorskip("mujoco_warp")
    from ndm_gym.wmbench.speed_mbrl import main

    _speed_files(tmp_path)
    out = tmp_path / "speed.csv"
    main(["--meta", str(tmp_path / "cp.json"), "--states", str(tmp_path / "cp.npz"), "--what", "warp", "--device", DEVICE, "--nworld", "4",
          "--steps", "3", "--warmup", "1", "--repeats", "1", "--out", str(out)])  # fmt: skip
    row = open(out).read().splitlines()[1].split(",")
    assert row[0] == "warp" and row[2] == "4" and float(row[9]) > 0 and "naconmax" in row[12]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["pets_cartpole", "pets_pusher"])
def test_warp_rollouts_match_control_envs_pool(name):
    pytest.importorskip("mujoco_warp")
    from ndm_gym.sim.mbrl_ndm import WarpRollouts

    spec = ENVS[name]
    handler, env = make_env(spec, 0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        env.step(rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32))
    m, _, substeps, _ = physics(spec, env)
    acts = torch.tensor(rng.uniform(env.action_space.low, env.action_space.high, (4, 8, env.action_space.shape[0])), dtype=torch.float32)
    st, cur, ref = state(spec, handler, env), handler.get_current_state(env), []
    for k in range(4):  # as control_env.py's workers score a sequence
        handler.set_env_state(cur, env)
        ref.append(handler.rollout_env(env, env.observation_space.sample(), -1, agent=None, plan=acts[k])[1].sum())
    got = WarpRollouts(spec, m, substeps, 4, device=DEVICE).values(st, acts).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)  # float32 physics against float64
    with pytest.raises(ValueError, match="1e-4"):
        WarpRollouts(ENVS["ant"], m, substeps, 4, device=DEVICE)


@pytest.mark.skipif(not DEVICE.startswith("cuda"), reason="torch.compile of the rollouts needs a GPU (MBRL_TEST_DEVICE)")
@pytest.mark.parametrize("name", ["inverted_pendulum", "dmc_cheetah_run", "pets_pusher"])
def test_compiled_ndm_rollouts_are_the_eager_ones(name):
    _needs(name)
    _, handler, env, m, eager = _rollouts(name, nworld=8)
    compiled = _rollouts(name, nworld=8, compile=True)[4]
    st = state(ENVS[name], handler, env)
    acts = torch.tensor(np.random.default_rng(2).uniform(-1, 1, (8, 6, m.nu)), dtype=torch.float32)
    torch.testing.assert_close(compiled.values(st, acts), eager.values(st, acts), rtol=1e-5, atol=1e-5)


def _run(tmp_path, *a):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-m", *a], capture_output=True, text=True, env=env, cwd=tmp_path)


@pytest.mark.slow
def test_collect_and_plan_run(tmp_path):
    out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", "--env", "pets_cartpole", "--num_steps", "2", "--record", "4", "--episodes", "1",
               "--out", str(tmp_path / "cp"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    for kind in ("rollout", "executed", "random", "noisy"):
        z = np.load(tmp_path / f"cp_{kind}.npz")
        assert z["qpos"].shape[0] == z["ctrl"].shape[0] + 1 and float(z["dt"]) == pytest.approx(0.04)
    out = _run(tmp_path, "ndm_gym.wmbench.plan_mbrl", "--env", "pets_cartpole", "--num_steps", "2", "--out", str(tmp_path / "plan.json"))
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.load(open(tmp_path / "plan.json"))
    assert len(res["steps"]) == 2 and res["population"] == 512 and all(s["plan_s"] > 0 for s in res["steps"]) and res["counter_fix"] is False
    assert res["own_margins"] is False and res["own_mujoco"] is False


@pytest.mark.slow
def test_simulator_comparison_of_an_environment_with_itself(tmp_path):
    out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", "--env", "pets_cartpole", "--num_steps", "8", "--record", "4", "--episodes", "1",
               "--out", str(tmp_path / "cp_s0"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "mbrl_lib", "compare_sim.py")
    files = ["--transitions", str(tmp_path / "c_transitions.npz"), "--sequences", str(tmp_path / "c_sequences.npz")]

    def tool_run(*a):
        return subprocess.run([sys.executable, tool, *a], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=tmp_path)

    out = tool_run("inputs", "--prefix", str(tmp_path / "cp"), "--seeds", "0", "--rollouts", "50", "--rollout_seeds", "0", "--starts", "0:0", "0:4",
                   "--horizon", "4", "--out", str(tmp_path / "c"))  # fmt: skip
    assert out.returncode == 0 and "58 transitions (8 executed), 2 replays" in out.stdout, out.stderr[-2000:]
    for side, extra in (("own", []), ("study", ["--own_mujoco"])):  # the same MuJoCo twice: the second through the study's patches
        out = tool_run("step", "--env", "pets_cartpole", *extra, *files, "--rest", "20", "--out", str(tmp_path / f"{side}.npz"))
        assert out.returncode == 0, out.stderr[-2000:]
    a, b, tr = np.load(tmp_path / "own.npz"), np.load(tmp_path / "study.npz"), np.load(tmp_path / "c_transitions.npz")
    assert a["one_qpos"].shape == (58, 2) and a["seq_qpos"].shape == (2, 4, 2) and a["rest_qpos"].shape == (20, 2)
    np.testing.assert_allclose(a["one_qpos"][tr["executed"]], tr["next_qpos"][tr["executed"]], rtol=0, atol=1e-9)  # the recorded steps again
    out = tool_run("report", "--own", str(tmp_path / "own.npz"), "--study", str(tmp_path / "study.npz"), *files, "--threshold")
    assert out.returncode == 0 and "more than 1e-7 on 0" in out.stdout and out.stdout.rstrip().endswith(": own"), out.stdout[-1500:] + out.stderr[-1500:]
    moved = {k: b[k] for k in b.files}
    moved["one_qpos"] = b["one_qpos"] + 1e-6
    np.savez(tmp_path / "moved.npz", **moved)
    out = tool_run("report", "--own", str(tmp_path / "own.npz"), "--study", str(tmp_path / "moved.npz"), *files, "--threshold")
    assert out.returncode == 1 and out.stdout.rstrip().endswith("NOT own")


@pytest.mark.slow
def test_collect_with_the_planner_on_a_world_model(tmp_path):
    import mujoco

    out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", "--env", "pets_cartpole", "--num_steps", "2", "--record", "4", "--episodes", "1",
               "--out", str(tmp_path / "cp"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    out = _run(tmp_path, "ndm_gym.wmbench.fit_mjcf", "--model", str(tmp_path / "cp_model.mjb"), "--substeps", "2", "--train",
               str(tmp_path / "cp_rollout.npz"), "--test", f"held={tmp_path / 'cp_rollout.npz'}", "--steps", "2", "--batch", "8", "--window", "2",
               "--hidden", "16", "--amp", "none", "--eval_k", "1", "--eval_windows", "8", "--out", str(tmp_path / "fit.csv"), "--save",
               str(tmp_path / "fit.pt"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    ndm = ["--sim", "ndm", "--ndm", str(tmp_path / "fit.pt"), "--dtype", "fp32", "--device", "cpu", "--no_compile"]
    out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", "--env", "pets_cartpole", "--num_steps", "2", "--record", "4", "--episodes", "1",
               *ndm, "--out", str(tmp_path / "cpn"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    meta = json.load(open(tmp_path / "cpn.json"))
    assert meta["sim"] == "ndm" and meta["ndm"] == str(tmp_path / "fit.pt") and meta["num_processes"] == 1 and len(meta["plan_s"]) == 2
    ro, plans = np.load(tmp_path / "cpn_rollout.npz"), np.load(tmp_path / "cpn_plans.npz")
    assert plans["t"].tolist() == [0] and np.array_equal(ro["qpos"][0, 0], plans["qpos"][0])  # every candidate from the plan's start
    m = mujoco.MjModel.from_binary_path(str(tmp_path / "cpn_model.mjb"))
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = ro["qpos"][0, 0], ro["qvel"][0, 0], ro["ctrl"][0, 0]
    mujoco.mj_step(m, d, 2)  # the recorded step is CPU MuJoCo's, not the world model's
    np.testing.assert_allclose(ro["qpos"][1, 0], d.qpos, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ro["qvel"][1, 0], d.qvel, rtol=0, atol=1e-9)
    assert json.load(open(tmp_path / "cp.json"))["sim"] == "cpu"
    with pytest.raises(ValueError, match="--sim ndm needs --ndm"):
        from ndm_gym.wmbench.collect_mbrl import main

        main(["--env", "pets_cartpole", "--sim", "ndm", "--out", str(tmp_path / "x")])


@pytest.mark.slow
def test_collect_until_termination_keeps_nothing_after_it(tmp_path):
    env = ["--env", "inverted_pendulum", "--setting", "pets", "--record", "4", "--record_every", "5", "--episodes", "1"]
    out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", *env, "--num_steps", "2", "--out", str(tmp_path / "p"))
    assert out.returncode == 0, out.stderr[-2000:]
    out = _run(tmp_path, "ndm_gym.wmbench.fit_mjcf", "--model", str(tmp_path / "p_model.mjb"), "--substeps", "2", "--train",
               str(tmp_path / "p_rollout.npz"), "--test", f"held={tmp_path / 'p_rollout.npz'}", "--steps", "2", "--batch", "8", "--window", "2",
               "--hidden", "16", "--amp", "none", "--eval_k", "1", "--eval_windows", "8", "--out", str(tmp_path / "fit.csv"), "--save",
               str(tmp_path / "fit.pt"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    # the planner on a world model of two updates: the pole falls within the 60 steps, and control_env.py steps on
    ndm = ["--sim", "ndm", "--ndm", str(tmp_path / "fit.pt"), "--dtype", "fp32", "--device", "cpu", "--no_compile", "--num_steps", "60"]
    for name, flag in (("all", []), ("cut", ["--until_termination"])):
        out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", *env, *ndm, *flag, "--out", str(tmp_path / name))
        assert out.returncode == 0, out.stderr[-2000:]
    full, cut = json.load(open(tmp_path / "all.json")), json.load(open(tmp_path / "cut.json"))
    n = cut["kept_steps"]
    assert full["kept_steps"] == 60 and not full["until_termination"] and cut["until_termination"] and 0 < n < 60 and n == cut["episode_steps"]
    ex, ex_all, plans = np.load(tmp_path / "cut_executed.npz"), np.load(tmp_path / "all_executed.npz"), np.load(tmp_path / "cut_plans.npz")
    assert ex["ctrl"].shape[0] == n and ex_all["ctrl"].shape[0] == 60 and np.array_equal(ex["qpos"], ex_all["qpos"][: n + 1])
    assert plans["t"].tolist() == [t for t in range(0, 60, 5) if t < n]
    assert abs(ex["qpos"][n, 0, 1]) > 0.2 and (np.abs(ex["qpos"][:n, 0, 1]) <= 0.2).all()  # the kept steps end with the terminating one


@pytest.mark.slow
def test_collect_records_dm_control_at_its_control_step(tmp_path):
    pytest.importorskip("dm_control")
    out = _run(tmp_path, "ndm_gym.wmbench.collect_mbrl", "--env", "dmc_finger_spin", "--num_steps", "2", "--record", "2", "--episodes", "1",
               "--counter_fix", "--num_processes", "2", "--out", str(tmp_path / "f"))  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    meta = json.load(open(tmp_path / "f.json"))
    assert meta["dt"] == pytest.approx(0.02) and meta["substeps"] == 2 and meta["repeat"] == 2 and meta["counter_fix"] is True
    ex, ro = np.load(tmp_path / "f_executed.npz"), np.load(tmp_path / "f_rollout.npz")
    assert ex["ctrl"].shape[0] == 2 * 2 and float(ex["dt"]) == pytest.approx(0.02)  # 2 environment steps of 2 control steps
    assert np.array_equal(ex["ctrl"][0], ex["ctrl"][1]) and ro["ctrl"].shape[0] % (2 * 30 + 1) == 2 * 30  # horizon 30, repeat 2
