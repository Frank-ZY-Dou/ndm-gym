"""HDMI's static props kept out of contact with each other and with the ground (hdmi_sim.exclude_prop_contacts): the
filter pairs on a stand-in Newton builder (needs Isaac Lab's packages); with a GPU, topple_wood_board_and_cross's scene
without a collision candidate between two props or a prop and the ground, no world at the solver's iteration limit over
steps from HDMI's resets, the props' solved accelerations zero and platform0 carrying the robot; and every task's
candidate pairs with the filter equal to those without it minus the prop-prop and prop-ground pairs, which carry no
contact outside topple; wmbench/speed_hoi.py's physics scene placed at the clip's frame 0 (hdmi_sim.place_scene). Each
simulation is a process of this file (``--worker``); marked slow, needs a GPU, Isaac Lab and the converted HDMI clips."""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ndm_gym.tasks.hdmi_tasks import TASKS  # noqa: E402

HAVE_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
HAVE_SIM = HAVE_ISAACLAB and torch.cuda.is_available() and os.path.isdir(os.path.join(REPO, "data", "hdmi", "clips"))
needs_isaaclab = pytest.mark.skipif(not HAVE_ISAACLAB, reason="hdmi_sim imports Isaac Lab's packages")
needs_sim = pytest.mark.skipif(not HAVE_SIM, reason="HDMISim needs a GPU, Isaac Lab and the converted HDMI clips")
TOPPLE = "topple_wood_board_and_cross"


def _immovable(label: str, props: tuple[str, ...]) -> bool:
    """A static shape or a shape of one of the props (bodies labelled by world 0's prim paths)."""
    return label == "static" or any(re.search(f"/{p}(/|$)", label) for p in props)


@needs_isaaclab
def test_prop_filter_pairs():
    """Within each world, every prop against the other props and every colliding static shape; nothing for the
    robot, the object or a static shape that does not collide; refused for a dynamic prop or a world without one."""
    from newton import BodyFlags, ShapeFlags

    from ndm_gym.sim.rigid.hdmi_sim import prop_filter_pairs

    K, D, C = int(BodyFlags.KINEMATIC), int(BodyFlags.DYNAMIC), int(ShapeFlags.COLLIDE_SHAPES)
    bodies = [(w, f"/World/envs/env_{w}/{n}/{n}_link", K if n.startswith("p") else D) for w in range(2) for n in ("robot", "obj", "pa", "pb")]
    shapes = [(-1, C), (-1, 0)]  # the ground and a visual-only static shape
    for b in range(len(bodies)):
        shapes += [(b, C)] + ([(b, C), (b, 0)] if bodies[b][1].endswith("pb_link") else [])  # pb: two colliders, one visual
    builder = SimpleNamespace(body_world=[w for w, _, _ in bodies], body_label=[lab for _, lab, _ in bodies], body_flags=[f for _, _, f in bodies],
                              shape_body=[b for b, _ in shapes], shape_flags=[f for _, f in shapes],
                              body_shapes={b: [s for s, (sb, _) in enumerate(shapes) if sb == b] for b in range(-1, len(bodies))})  # fmt: skip
    patterns = {n: f"/World/envs/env_[^/]+/{n}" for n in ("pa", "pb")}
    got = sorted(tuple(sorted(p)) for p in prop_filter_pairs(builder, patterns))
    want = []
    for w in range(2):
        pa = [s for s, (b, f) in enumerate(shapes) if b >= 0 and bodies[b][0] == w and "/pa/" in bodies[b][1] and f]
        pb = [s for s, (b, f) in enumerate(shapes) if b >= 0 and bodies[b][0] == w and "/pb/" in bodies[b][1] and f]
        want += [tuple(sorted((a, b))) for a in pa for b in pb] + [(0, s) for s in pa + pb]
    assert got == sorted(want) and len(want) == 2 * (2 + 3)
    assert prop_filter_pairs(builder, {}) == []
    with pytest.raises(RuntimeError, match="not a kinematic body"):
        prop_filter_pairs(builder, {"obj": "/World/envs/env_[^/]+/obj"})
    with pytest.raises(RuntimeError, match="no colliding shape"):
        prop_filter_pairs(builder, {**patterns, "pc": "/World/envs/env_[^/]+/pc"})


def place_worker(task: str, out: str):
    """speed_hoi's physics scene of ``task`` (4 envs, four substeps per step) after place_scene, before any step: the
    largest differences of the robot's root pose and joints, the object's root position or hinge angle and the props'
    positions from the clip's frame 0."""
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import instantiate

    import ndm_gym.sim.rigid.hdmi_sim as hs
    from ndm_gym.robots.g1 import G1_JOINT_NAMES

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    cfg, dev = TASKS[task], "cuda:0"
    sim_cfg = hs.make_hdmi_sim(dev, substeps=4)
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        sim = SimulationContext(sim_cfg)
        arrays, side = hs.load_clip(cfg.name)
        scene = instantiate(hs.scene_cfg(cfg, side, 4))
        hs.exclude_prop_contacts(sim, scene, cfg)
        sim.reset()
        hs.check_prop_contacts(sim, scene, cfg)
        hs.place_scene(scene, cfg)
        sim.forward()
        scene.update(0.0)
        ref = lambda k, *i: torch.as_tensor(arrays[k][(0, *i)], dtype=torch.float32, device=dev)  # noqa: E731  frame 0
        rel = lambda asset: asset.data.root_link_pose_w.torch[:, :3] - scene.env_origins  # noqa: E731
        robot, obj = scene["robot"], scene["obj"]
        root = robot.data.root_link_pose_w.torch
        q, q_ref = torch.cat((root[:, 6:7], root[:, 3:6]), -1), ref("keypoint_quat_wxyz", 0)  # wxyz, either sign
        err = {"robot_pos": (rel(robot) - ref("keypoint_pos", 0)).abs().max(),
               "robot_quat": torch.minimum((q - q_ref).abs().amax(-1), (q + q_ref).abs().amax(-1)).max(),
               "joints": (robot.data.joint_pos.torch - ref("joint_pos")[[G1_JOINT_NAMES.index(j) for j in robot.joint_names]]).abs().max(),
               "object_pos": (rel(obj) - ref("object_pos")).abs().max()}  # fmt: skip
        if cfg.object_joint is not None:
            err["hinge"] = (obj.data.joint_pos.torch[:, 0] - ref("object_joint_pos", 0)).abs().max()
        for k, name in enumerate(cfg.extra_objects):
            err[name] = (rel(scene[f"extra_{name}"]) - ref("extra_pos", k)).abs().max()
        torch.save({k: float(v) for k, v in err.items()}, out)


def worker(mode: str, task: str, out: str, fix: bool = True):
    """``pairs``: the scene of ``task`` (64 envs from HDMI's resets) with the prop filter or, ``fix`` False, without
    it: its candidate geom pairs, each geom's body (world 0's label, "static" for the ground) and, over two control
    steps with HDMISim's default targets, the contacts between two props or a prop and the ground. ``topple``: 256 envs
    of topple_wood_board_and_cross, ten control steps with collect_hoi's data policy, terminations and resets from
    HDMI's resets (the worlds at the iteration limit, the props' largest solved acceleration), then every env reset at
    frame 0 and six steps of the reference (the feet's vertical force against the robot's weight). ``place``:
    place_worker."""
    import mujoco_warp as mjw
    import warp as wp
    from isaaclab.app import add_launcher_args, launch_simulation

    import ndm_gym.sim.rigid.hdmi_sim as hs
    from ndm_gym.tasks.hdmi import NUM_ACTIONS, HDMITask
    from ndm_gym.wmbench.collect_hoi import merge_states

    if mode == "place":
        return place_worker(task, out)
    if not fix:
        hs.exclude_prop_contacts = hs.check_prop_contacts = lambda *a, **k: None
    p = argparse.ArgumentParser()
    add_launcher_args(p)
    launcher = p.parse_args([])
    torch.manual_seed(0)
    dev, cfg = "cuda:0", TASKS[task]
    N = 64 if mode == "pairs" else 256
    sim_cfg = hs.make_hdmi_sim(dev)
    with launch_simulation(sim_cfg, launcher), torch.no_grad():
        env = hs.HDMISim(sim_cfg, cfg, N)
        solver, model = env.sim.backend_manager._solver, env.sim.backend_manager.get_model()
        mjm, m, d = solver.mj_model, solver.mjw_model, solver.mjw_data
        body = model.shape_body.numpy()[solver.mjc_geom_to_newton_shape.numpy()[0]]
        geom_body = ["static" if b == -1 else model.body_label[b] for b in body]
        fixed = torch.tensor([_immovable(lab, cfg.extra_objects) for lab in geom_body], device=dev)
        rec = {"pairs": m.nxn_geom_pair_filtered.numpy().tolist(), "geom_body": geom_body}

        def fixed_contacts() -> int:
            n = min(int(d.nacon.numpy()[0]), d.contact.dist.shape[0])
            g = wp.to_torch(d.contact.geom)[:n].long()
            return int((fixed[g[:, 0]] & fixed[g[:, 1]] & (wp.to_torch(d.contact.efc_address)[:n, 0] >= 0)).sum())

        if mode == "pairs":
            prev, rec["contacts"] = env.read(), []
            for _ in range(2):
                x, _ = env.step(torch.zeros(N, NUM_ACTIONS, device=dev), prev)
                prev = x.now
                rec["contacts"].append(fixed_contacts())
            torch.save(rec, out)
            return
        prop_bodies = sorted({int(mjm.geom_bodyid[g]) for g, lab in enumerate(geom_body) if lab != "static" and fixed[g]})
        dofs = [int(mjm.body_dofadr[b]) + i for b in prop_bodies for i in range(int(mjm.body_dofnum[b]))]
        terms = HDMITask(cfg, N, env.joint_limits, env.robot_mass, noise=True, device=dev)
        q0 = env.q0[env.j_act]
        prev = env.read()
        terms.reset(torch.arange(N, device=dev), prev)
        d.overflow.zero_()
        qacc, load = [], []
        for _ in range(10):
            a = (env.clip.joint_pos[env.t.clamp(max=env.L - 1)] - q0) / env.scale
            x, truncated = env.step(a + 0.2 * torch.randn_like(a), prev)
            qacc.append(float(wp.to_torch(d.qacc)[:, dofs].abs().max()))
            done = terms.step(x).terminated | truncated
            prev = x.now
            if done.any():
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                env.reset(ids)
                env.refresh()
                prev = merge_states(done, prev, env.read())
                terms.reset(ids, prev)
        rec["iteration_limit"] = int(((wp.to_torch(d.overflow) & int(mjw.OverflowType.ITERATIONS)) != 0).sum())
        env.reset(torch.arange(N, device=dev), frames=torch.zeros(N, dtype=torch.long, device=dev))
        env.refresh()
        prev = env.read()
        for _ in range(6):
            x, _ = env.step((env.clip.joint_pos[env.t.clamp(max=env.L - 1)] - q0) / env.scale, prev)
            qacc.append(float(wp.to_torch(d.qacc)[:, dofs].abs().max()))
            load.append(float(x.feet_force[..., 2].sum(-1).mean() / (9.81 * env.robot_mass)))
            prev = x.now
        rec |= {"prop_dofs": len(dofs), "prop_qacc": qacc, "feet_load": load}
        torch.save(rec, out)


def run(tmp_path, mode: str, task: str, fix: bool = True) -> dict:
    out = str(tmp_path / f"{mode}_{task}_{int(fix)}.pt")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--mode", mode, "--task", task, "--out", out] + ([] if fix else ["--no_fix"])
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return torch.load(out)


@pytest.mark.slow
@needs_sim
def test_topple_props_out_of_contact(tmp_path):
    r = run(tmp_path, "topple", TOPPLE)
    props = TASKS[TOPPLE].extra_objects
    assert not [p for p in r["pairs"] if all(_immovable(r["geom_body"][g], props) for g in p)]
    assert r["iteration_limit"] == 0
    assert r["prop_dofs"] == 6 * len(props) and max(r["prop_qacc"]) < 1e-3, r["prop_qacc"]  # locked props: no runaway acceleration
    load = r["feet_load"][1:]
    assert sum(load) / len(load) > 0.5, r["feet_load"]  # platform0 carries the robot standing on it at frame 0 (without the filter about 0)


@pytest.mark.slow
@needs_sim
@pytest.mark.parametrize("task", sorted(TASKS))
def test_candidate_pairs_without_prop_pairs(tmp_path, task):
    """The filter removes exactly the candidates between two props or a prop and the ground (none for a task without
    props) and keeps every other pair in its order; without it those pairs carry contacts in topple only."""
    fix, nofix = run(tmp_path, "pairs", task), run(tmp_path, "pairs", task, fix=False)
    props = TASKS[task].extra_objects
    assert fix["geom_body"] == nofix["geom_body"]
    gb = fix["geom_body"]
    prop_pair = lambda p: _immovable(gb[p[0]], props) and _immovable(gb[p[1]], props)  # noqa: E731
    assert fix["pairs"] == [p for p in nofix["pairs"] if not prop_pair(p)]
    assert any(prop_pair(p) for p in nofix["pairs"]) == bool(props)
    assert fix["contacts"] == [0, 0]
    assert (min(nofix["contacts"]) > 0) if task == TOPPLE else nofix["contacts"] == [0, 0], nofix["contacts"]


@pytest.mark.slow
@needs_sim
@pytest.mark.parametrize("task", [TOPPLE, "open_door-feet"])
def test_place_scene(tmp_path, task):
    """speed_hoi's physics scene at the clip's frame 0: the robot, the object (a door's hinge angle) and the props."""
    err = run(tmp_path, "place", task)
    assert set(err) >= {"robot_pos", "robot_quat", "joints", "object_pos"} and max(err.values()) < 1e-5, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode", choices=["pairs", "topple", "place"], required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_fix", action="store_true")
    cli = ap.parse_args()
    worker(cli.mode, cli.task, cli.out, fix=not cli.no_fix)
