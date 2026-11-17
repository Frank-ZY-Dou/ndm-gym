"""fit_mjcf's options on CPU, each on a tiny synthetic set: the contact inputs (``--contacts``, ``--contact_basis``,
``--contact_reach``, ``--contact_cap``, ``--contact_static``, ``--contact_objects``), ``--loss``, ``--init``,
``--cut_medians_from`` and ``--cut_nonfinite``; and that a fit without them is the same fit, bit for bit, as one that names
their default values."""

import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_mjcf_contacts import FOOT, PROBES, PUSHER  # noqa: E402

from ndm_gym.wmbench.fit_mjcf import cut_blowups, cut_nonfinite, main, median_increments, nonfinite_steps, parse_args  # noqa: E402
from ndm_gym.wmbench.speed_mjcf import load_ndm  # noqa: E402
from ndm_gym.worldmodels.mjcf import MjcfLayout, MjcfWorldModel  # noqa: E402

XML = PROBES.replace("KIND", "box").replace("SIZE", "0.1 0.1 0.05").replace("MARGIN", "").replace('"0.005"', '"0.005" gravity="0 0 0"')
FIT = ["--steps", "6", "--batch", "8", "--window", "3", "--hidden", "16", "--amp", "none", "--eval_every", "6", "--eval_k", "1",
       "--eval_windows", "8"]  # fmt: skip


def save(path, qpos, qvel, ctrl, dt):
    """A recording file without resets, as collect_robokit writes one."""
    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    np.savez(path, qpos=f32(qpos), qvel=f32(qvel), ctrl=f32(ctrl), reset=np.zeros(ctrl.shape[:2], dtype=bool), dt=dt)
    return str(path)


def model():
    return mujoco.MjModel.from_xml_string(XML)


def simulated(path, n=6, steps=24, seed=0, kick=0.2):
    """A recording of MuJoCo's steps under random-walk targets of the three slides (no gravity: the object stays where it is)."""
    m, rng = model(), np.random.default_rng(seed)
    d = mujoco.MjData(m)
    qpos, qvel, ctrl = np.zeros((steps + 1, n, m.nq)), np.zeros((steps + 1, n, m.nv)), np.zeros((steps, n, m.nu))
    for i in range(n):
        mujoco.mj_resetData(m, d)
        u = np.zeros(m.nu)
        for t in range(steps):
            qpos[t, i], qvel[t, i] = d.qpos, d.qvel
            u = np.clip(u + kick * 0.05 * rng.standard_normal(m.nu), -0.1, 0.1)
            d.ctrl[:] = ctrl[t, i] = u
            mujoco.mj_step(m, d)
        qpos[steps, i], qvel[steps, i] = d.qpos, d.qvel
    return save(path, qpos, qvel, ctrl, m.opt.timestep)


def drifting(path, speeds, steps=24):
    """A recording whose worlds' slides each move at a constant speed (m per step), the object at rest."""
    m = model()
    n = len(speeds)
    qpos = np.tile(m.qpos0, (steps + 1, n, 1))
    qpos[:, :, :3] += np.arange(steps + 1)[:, None, None] * np.asarray(speeds)[None, :, None]
    qvel = np.zeros((steps + 1, n, m.nv))
    qvel[:, :, :3] = np.asarray(speeds)[None, :, None] / m.opt.timestep
    return save(path, qpos, qvel, np.zeros((steps, n, m.nu)), m.opt.timestep)


@pytest.fixture
def files(tmp_path):
    xml = tmp_path / "probes.xml"
    xml.write_text(XML)
    return SimpleNamespace(dir=tmp_path, xml=str(xml), train=simulated(tmp_path / "train.npz"), test=simulated(tmp_path / "test.npz", n=3, seed=1))


def fit(files, name, *extra, train=None):
    main(["--model", files.xml, "--train", train or files.train, "--test", f"held={files.test}", *FIT, *extra,
          "--out", str(files.dir / f"{name}.csv"), "--save", str(files.dir / f"{name}.pt")])  # fmt: skip
    return torch.load(files.dir / f"{name}.pt", weights_only=False)


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)  # a fit is saved on the device it ran on


def test_defaults_are_the_fit_without_the_options(files):
    a = parse_args(["--model", "m", "--train", "t", "--test", "x=t", "--out", "o"])
    assert (a.contacts, a.contact_basis, a.contact_reach, a.contact_cap, a.contact_spacing) == (False, False, None, 384, 0.0025)
    assert (a.loss, a.init, a.cut_medians_from, a.cut_nonfinite) == ("l1", None, None, False)
    assert (a.data_device, a.obj_w, a.cut_sample) == ("model", 1.0, 1_000_000)
    plain = fit(files, "plain")
    assert not any(k.startswith("cs_") for k in plain["state"]) and plain["args"]["contacts"] is False
    assert same(plain["state"], fit(files, "again")["state"])  # the same fit twice
    named = fit(files, "named", "--loss", "l1", "--data_device", "model", "--obj_w", "1", "--contact_cap", "384", "--contact_spacing", "0.0025",
                "--cut_sample", "1000000")  # fmt: skip
    assert same(plain["state"], named["state"])  # naming the defaults changes nothing, bit for bit


def test_contact_options(files, capsys):
    lay = MjcfLayout.from_model(model())
    load = lambda name: load_ndm(lay, SimpleNamespace(ckpt=str(files.dir / f"{name}.pt")), torch.device("cpu"))  # noqa: E731
    ck = fit(files, "c", "--contacts")
    out = capsys.readouterr().out
    K = ck["state"]["cs_center"].shape[1]
    assert ck["args"]["contacts"] and ck["state"]["cs_center"].shape == (3, K, 3) and not bool(ck["state"]["cs_basis"])
    assert re.search(rf"--contacts: \d+ spheres on 3 geoms \(at most {K} on one\), margin 0 mm", out)
    wm = load("c")
    assert wm.contacts and wm.n_contact == 39 and same(wm.state_dict(), ck["state"])
    assert wm.n_in == MjcfWorldModel(lay, hidden=(16,)).n_in + 39
    fit(files, "cb", "--contacts", "--contact_basis")
    wm = load("cb")
    assert wm.n_contact == 51 and bool(wm.cs_basis)
    near = fit(files, "cr", "--contacts", "--contact_reach", "0.25")  # only the ball comes within 0.25 m of the object (0.18 m)
    assert near["state"]["cs_body"].tolist() == [model().body("ball").id] and near["state"]["cs_center"].shape == (1, 1, 3)
    assert load("cr").n_contact == 13
    with pytest.raises(ValueError, match="--contact_reach 0.001: no robot collision geom comes that near"):
        fit(files, "x", "--contacts", "--contact_reach", "0.001")
    capped = fit(files, "cc", "--contacts", "--contact_cap", "32")
    assert capped["state"]["cs_center"].shape[1] == 81 < K  # the brick's spheres capped: the rod's 81 are the most on one geom


def test_loss_kinds(files):
    lay = MjcfLayout.from_model(model())
    z = np.load(files.train)
    qpos, qvel, ctrl = (torch.tensor(z[k][:4] if k != "ctrl" else z[k][:3]).double() for k in ("qpos", "qvel", "ctrl"))
    torch.manual_seed(0)
    wm = MjcfWorldModel(lay, hidden=(8,)).double()
    wm.update_norms(qpos[:-1], qvel[:-1], ctrl, qpos[1:], qvel[1:])
    with torch.no_grad():
        q, v, errs = qpos[0], qvel[0], []
        for t in range(3):
            q, v = wm.step(q, v, ctrl[t])
            errs.append(wm.error(q, v, qpos[t + 1], qvel[t + 1]))
        e = torch.stack(errs)
        want = {"l1": e.abs().mean(), "l2": e.square().mean(), "huber": torch.where(e.abs() < 1, 0.5 * e.square(), e.abs() - 0.5).mean()}
        assert torch.equal(wm.window_loss(qpos, qvel, ctrl), wm.window_loss(qpos, qvel, ctrl, loss="l1"))  # the default is the absolute error
        for kind, x in want.items():
            assert wm.window_loss(qpos, qvel, ctrl, loss=kind).item() == pytest.approx(x.item(), rel=1e-12)
        assert (e.abs() > 1).any() and (e.abs() < 1).any()  # both branches of the huber loss are read
        with pytest.raises(ValueError, match="'l1', 'l2' or 'huber'"):
            wm.window_loss(qpos, qvel, ctrl, loss="l3")
    for kind in ("l2", "huber"):
        ck = fit(files, kind, "--loss", kind)
        assert ck["args"]["loss"] == kind and len((files.dir / f"{kind}.csv").read_text().strip().splitlines()) == 2
        assert not same(ck["state"], fit(files, "l1")["state"])  # another loss, another fit


def test_init_starts_from_a_fit_and_keeps_its_normalizers(files):
    first = fit(files, "first")
    other = simulated(files.dir / "other.npz", seed=3, kick=1.0)  # wider motions: other normalizers, if they were taken anew
    kept = fit(files, "kept", "--init", str(files.dir / "first.pt"), "--lr", "0", "--steps", "2", "--eval_every", "2", train=other)
    assert same(kept["state"], first["state"])  # no update at a learning rate of zero: the network and the normalizers as loaded
    anew = fit(files, "anew", "--lr", "0", "--steps", "2", "--eval_every", "2", train=other)
    assert not torch.equal(anew["state"]["in_norm.mean"], first["state"]["in_norm.mean"])  # without --init: the new recordings' normalizers
    moved = fit(files, "moved", "--init", str(files.dir / "first.pt"), "--steps", "2", "--eval_every", "2", train=other)
    assert torch.equal(moved["state"]["in_norm.mean"], first["state"]["in_norm.mean"])  # the refit moves the network only
    assert not same(moved["state"], first["state"])


def test_cut_medians_from(files, capsys):
    calm = drifting(files.dir / "calm.npz", [0.001, 0.001, 0.001, 0.001])
    mixed = drifting(files.dir / "mixed.npz", [0.001, 0.1, 0.1, 0.1])  # three worlds in four move a hundred times faster
    share = lambda out, path: float(re.search(rf"--cut 30: ([\d.]+)% of the steps of {re.escape(path)} cut", out).group(1))  # noqa: E731
    fit(files, "own", "--cut", "30", train=mixed)
    assert share(capsys.readouterr().out, mixed) == 0.0  # judged by its own medians, which the fast worlds set: nothing is cut
    fit(files, "ref", "--cut", "30", "--cut_medians_from", calm, train=mixed)
    assert share(capsys.readouterr().out, mixed) == 75.0  # judged by the calm recording's: the three fast worlds are cut


def test_states_that_are_not_finite_are_refused_or_cut(files, capsys):
    bad = files.dir / "bad.npz"
    z = dict(np.load(files.train))
    z["qpos"][10, 1, 0] = np.nan  # world 1's state after step 9: steps 9 and 10 of that world are not finite
    z["ctrl"][5, 0, 2] = np.inf  # and world 0's control at step 5
    np.savez(bad, **z)
    lay = MjcfLayout.from_model(model())
    rec = {k: torch.tensor(z[k]) for k in ("qpos", "qvel", "ctrl", "reset")}
    assert nonfinite_steps(rec).nonzero().tolist() == [[5, 0], [9, 1], [10, 1]]
    with pytest.raises(ValueError, match="kept steps are not finite"):
        cut_blowups(dict(rec), lay, 300.0)
    with pytest.raises(ValueError, match="kept steps are not finite"):
        median_increments([rec], lay)
    with pytest.raises(ValueError, match=rf"not finite: 3 in {re.escape(str(bad))}; --cut_nonfinite cuts"):
        fit(files, "x", train=str(bad))
    with pytest.raises(ValueError, match="--cut_nonfinite"):  # with --cut as well: never dropped without being asked
        fit(files, "x", "--cut", "300", train=str(bad))
    assert cut_nonfinite(rec) == (3, (24 - 5) + (24 - 9))  # each world from its first such step to the recording's end
    assert rec["reset"][5:, 0].all() and rec["reset"][9:, 1].all() and not rec["reset"][:5, 0].any() and not rec["reset"][:, 2:].any()
    assert cut_blowups(rec, lay, 300.0) == 0.0  # the rest is read as before
    capsys.readouterr()
    ck = fit(files, "cut", "--cut_nonfinite", "--cut", "300", train=str(bad))
    assert f"--cut_nonfinite: 3 steps of {bad} are not finite, 34 steps cut" in capsys.readouterr().out
    limits = ("ctrl_lo", "ctrl_hi", "force_lo", "force_hi")  # infinite where an actuator has no limit
    assert all(torch.isfinite(v).all() for k, v in ck["state"].items() if v.is_floating_point() and k not in limits)
    assert float((files.dir / "cut.csv").read_text().strip().splitlines()[-1].split(",")[2]) > 0  # a finite loss


def recorded(xml, path, start, n=6, steps=24, seed=0):
    """A recording of MuJoCo's steps of the model ``xml`` under random-walk controls, each world from qpos0 plus ``start(rng)``."""
    m, rng = mujoco.MjModel.from_xml_string(xml), np.random.default_rng(seed)
    d = mujoco.MjData(m)
    qpos, qvel, ctrl = np.zeros((steps + 1, n, m.nq)), np.zeros((steps + 1, n, m.nv)), np.zeros((steps, n, m.nu))
    for i in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] += start(rng)
        u = np.zeros(m.nu)
        for t in range(steps):
            qpos[t, i], qvel[t, i] = d.qpos, d.qvel
            u = np.clip(u + 0.3 * rng.standard_normal(m.nu), -1, 1)
            d.ctrl[:] = ctrl[t, i] = u
            mujoco.mj_step(m, d)
        qpos[steps, i], qvel[steps, i] = d.qpos, d.qvel
    return save(path, qpos, qvel, ctrl, m.opt.timestep)


def test_contact_static_and_contact_objects(tmp_path, capsys):
    def run(xml, start, name, *extra):
        (tmp_path / f"{name}.xml").write_text(xml)
        train, test = recorded(xml, tmp_path / f"{name}_train.npz", start), recorded(xml, tmp_path / f"{name}_test.npz", start, n=3, seed=1)
        main(["--model", str(tmp_path / f"{name}.xml"), "--train", train, "--test", f"held={test}", *FIT, *extra,
              "--out", str(tmp_path / f"{name}.csv"), "--save", str(tmp_path / f"{name}.pt")])  # fmt: skip
        ck = torch.load(tmp_path / f"{name}.pt", weights_only=False)
        lay = MjcfLayout.from_model(mujoco.MjModel.from_xml_string(xml))
        wm = load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / f"{name}.pt")), torch.device("cpu"))
        assert same(wm.state_dict(), ck["state"])  # the saved fit builds the same model: its spheres, object bodies and static geoms
        return ck, wm

    drop = lambda rng: np.array([0.0, -0.36 + 0.05 * rng.random(), 0.3 * rng.standard_normal()])  # noqa: E731  the foot 0.1-0.15 m up: it lands
    ck, wm = run(FOOT, drop, "floor_all", "--contacts", "--contact_static")
    assert "0 named object bodies, 2 static geoms" in capsys.readouterr().out
    assert ck["state"]["cs_sg"].shape == (2, 11) and "cs_tb" not in ck["state"] and wm.n_targets == 2 and wm.n_contact == 26
    ck, wm = run(FOOT, drop, "floor", "--contacts", "--contact_static", "floor", "--contact_reach", "0.05")
    assert ck["state"]["cs_sg"][:, 0].tolist() == [0.0] and wm.n_contact == 13 and ck["args"]["contact_static"] == ["floor"]
    near = lambda rng: np.concatenate((0.2 * rng.random(2), 0.05 * rng.standard_normal(2)))  # noqa: E731  the tip up to the puck
    ck, wm = run(PUSHER, near, "puck", "--contacts", "--contact_objects", "object", "--contact_basis")
    assert ck["state"]["cs_tb"].tolist() == [2] and "cs_sg" not in ck["state"] and wm.n_contact == 13 + 12 and bool(wm.cs_basis)
    ck, wm = run(PUSHER, near, "puck_table", "--contacts", "--contact_objects", "object", "--contact_static", "table")
    assert wm.n_targets == 2 and wm.n_contact == 26 and "1 named object bodies, 1 static geoms" in capsys.readouterr().out
    q = torch.tensor(np.load(tmp_path / "puck_test.npz")["qpos"][0])
    assert torch.equal(wm.inputs(q, torch.zeros(3, 4), torch.zeros(3, 2))[:, -26:], wm.contact_inputs(q, torch.zeros(3, 4)))
    with pytest.raises(ValueError, match="options of --contacts"):
        run(FOOT, drop, "x", "--contact_static")
    with pytest.raises(ValueError, match="options of --contacts"):
        run(PUSHER, near, "x", "--contact_objects", "object")
    a = parse_args(["--model", "m", "--train", "t", "--test", "x=t", "--out", "o"])
    assert a.contact_objects is None and a.contact_static is None  # by default neither: the free objects alone
