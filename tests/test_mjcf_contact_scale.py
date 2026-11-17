"""The contact columns' scales (ndm_gym.worldmodels.mjcf: ``MjcfWorldModel.update_norms(contact_scale=True)``; fit_mjcf
``--contact_scale``) on CPU: by default the input normalizer leaves the contact inputs as they are; with the option each
column is divided by its spread over the transitions where that exceeds one, against values worked out by hand for a capsule
foot on a floor; the scales add up over the calls; and a fit made with the option holds them in its saved normalizer and
loads without it, while a fit without the option is saved as before."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
mujoco = pytest.importorskip("mujoco")

from test_fit_mjcf_options import FIT, recorded, same  # noqa: E402
from test_mjcf_contacts import FOOT, PROBES  # noqa: E402

from ndm_gym.wmbench.fit_mjcf import main  # noqa: E402
from ndm_gym.wmbench.speed_mjcf import load_ndm  # noqa: E402
from ndm_gym.worldmodels.mjcf import CONTACT_FEATURES, MjcfLayout, MjcfWorldModel, contact_spheres  # noqa: E402

F64 = dict(dtype=torch.float64)
# the level foot (a capsule of radius 40 mm along x, 0.5 m up at rest) on the floor, falling and rising at 2 m/s, and twice 0.46 m up at rest
Q = torch.tensor([[0.0, -0.46, 0.0], [0.0, -0.46, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], **F64)
V = torch.tensor([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], **F64)
U = torch.zeros(4, 1, **F64)
# per column over the four states: the gap in units of 5 mm (0, 0, 4, 4); touch (1, 1, 0, 0); no overlap; the floor's point from
# the body's origin in units of 3 cm, x at an end of the capsule (0.1 m off), y 0, z 40 mm down; the normal (0, 0, 1); the
# velocity there in units of 0.2 m/s (0, 0, -10) and (0, 0, 10), and its normal part: each variance, or one where it is below
VAR = [4.0, 1.0, 1.0, (0.1 / 0.03) ** 2 / 4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 50.0]


def foot():
    m = mujoco.MjModel.from_xml_string(FOOT)
    lay = MjcfLayout.from_model(m)
    return m, lay, MjcfWorldModel(lay, hidden=(8,), contacts=contact_spheres(m, lay, static=("floor",))).double()


def test_the_contact_columns_keep_their_units_by_default():
    m, lay, wm = foot()
    cols = slice(wm.n_in - CONTACT_FEATURES, wm.n_in)
    x = wm.contact_inputs(Q, V)
    assert x.shape == (4, 13) and x[:, 0].tolist() == pytest.approx([0, 0, 4, 4], abs=1e-5)
    assert x[:, 1].tolist() == pytest.approx([1, 1, 0, 0], abs=1e-5)
    assert x[:, 3].abs().tolist() == pytest.approx([0.1 / 0.03, 0.1 / 0.03, 0, 0], abs=1e-5)
    assert x[:, 5].tolist() == pytest.approx([-0.04 / 0.03, -0.04 / 0.03, 0, 0], abs=1e-5)
    assert x[:, 11].tolist() == pytest.approx([-10, 10, 0, 0], abs=1e-4) and x[:, 12].tolist() == pytest.approx([-10, 10, 0, 0], abs=1e-4)
    wm.update_norms(Q, V, U, Q, V)
    assert (wm.in_norm.mean[cols] == 0).all() and (wm.in_norm.var[cols] == 1).all()  # as they are, whatever their spread
    assert torch.allclose(wm.in_norm(wm.inputs(Q, V, U))[:, cols], x / np.sqrt(1 + 1e-4), atol=1e-12)
    box = PROBES.replace("KIND", "box").replace("SIZE", "0.1 0.1 0.05").replace("MARGIN", "")  # three probes and a free object
    plain = MjcfWorldModel(MjcfLayout.from_model(mujoco.MjModel.from_xml_string(box)), hidden=(8,))
    with pytest.raises(ValueError, match="has no contact inputs"):
        plain.update_norms(*(torch.zeros(2, n) for n in (10, 9, 3, 10, 9)), contact_scale=True)


def test_contact_scale_by_hand():
    m, lay, wm = foot()
    cols = slice(wm.n_in - CONTACT_FEATURES, wm.n_in)
    x = wm.contact_inputs(Q, V)
    wm.update_norms(Q, V, U, Q, V, contact_scale=True)
    assert (wm.in_norm.mean[cols] == 0).all()  # divided, not shifted: a geom far from the floor stays at zero
    assert wm.in_norm.var[cols].tolist() == pytest.approx(VAR, abs=1e-4)
    assert (wm.in_norm.var[cols][[1, 2, 4, 5, 6, 7, 8, 9, 10]] == 1).all()  # a spread of one or less: the column as it was, exactly
    scaled = wm.in_norm(wm.inputs(Q, V, U))[:, cols]
    assert torch.allclose(scaled, x / torch.tensor(VAR, **F64).add(1e-4).sqrt(), atol=1e-6)
    assert scaled[:, 11].tolist() == pytest.approx([-10 / np.sqrt(50), 10 / np.sqrt(50), 0, 0], abs=1e-4)  # the velocity at its spread
    rest, both = foot()[2], foot()[2]
    rest.update_norms(Q, V, U, Q, V)
    assert torch.equal(wm.in_norm.mean[: cols.start], rest.in_norm.mean[: cols.start])  # the other columns as without the option
    assert torch.equal(wm.in_norm.var[: cols.start], rest.in_norm.var[: cols.start])
    both.update_norms(Q[:2], V[:2], U[:2], Q[:2], V[:2], contact_scale=True)  # over the calls: the same scales as from all at once
    assert both.in_norm.var[cols].tolist() == pytest.approx([1.0] * 11 + [100.0, 100.0], abs=1e-4)
    both.update_norms(Q[2:], V[2:], U[2:], Q[2:], V[2:], contact_scale=True)
    assert torch.allclose(both.in_norm.var[cols], wm.in_norm.var[cols], atol=1e-12) and (both.in_norm.mean[cols] == 0).all()
    both.update_norms(Q, V, U, Q, V)  # a call without the option leaves the columns as they are again
    assert (both.in_norm.var[cols] == 1).all()


def test_fit_mjcf_contact_scale_is_saved_with_the_fit(tmp_path, capsys):
    drop = lambda rng: np.array([0.0, -0.36 + 0.05 * rng.random(), 0.3 * rng.standard_normal()])  # noqa: E731  the foot 0.1-0.15 m up: it lands
    (tmp_path / "foot.xml").write_text(FOOT)
    train, test = recorded(FOOT, tmp_path / "train.npz", drop), recorded(FOOT, tmp_path / "test.npz", drop, n=3, seed=1)

    def fit(name, *extra):
        main(["--model", str(tmp_path / "foot.xml"), "--train", train, "--test", f"held={test}", *FIT, *extra,
              "--out", str(tmp_path / f"{name}.csv"), "--save", str(tmp_path / f"{name}.pt")])  # fmt: skip
        return str(tmp_path / f"{name}.pt")

    floor = ("--contacts", "--contact_static", "floor")
    a = torch.load(fit("plain", *floor), weights_only=False)
    capsys.readouterr()
    b = torch.load(fit("scaled", *floor, "--contact_scale"), weights_only=False)
    out = capsys.readouterr().out
    cols = slice(-CONTACT_FEATURES, None)
    assert a["args"]["contact_scale"] is False and (a["state"]["in_norm.var"][cols] == 1).all() and (a["state"]["in_norm.mean"][cols] == 0).all()
    assert b["args"]["contact_scale"] is True and a["state"].keys() == b["state"].keys() and set(a) == set(b)  # no key more: the normalizer's
    var = b["state"]["in_norm.var"][cols].cpu()  # a fit is saved on the device it ran on
    assert (b["state"]["in_norm.mean"][cols] == 0).all() and (var >= 1).all() and (var > 1).sum() >= 3
    assert f"--contact_scale: {int((var > 1).sum())} of 13 contact columns divided by their spread, the largest {float(var.max().sqrt()):.3g}" in out
    assert all(torch.equal(a["state"][k][:-CONTACT_FEATURES], b["state"][k][:-CONTACT_FEATURES]) for k in ("in_norm.mean", "in_norm.var"))
    lay = MjcfLayout.from_model(mujoco.MjModel.from_xml_string(FOOT))
    wm = load_ndm(lay, SimpleNamespace(ckpt=str(tmp_path / "scaled.pt")), torch.device("cpu"))  # nothing but the saved fit
    assert same(wm.state_dict(), b["state"])
    rec = np.load(train)
    q, v = (torch.tensor(rec[k][:-1].reshape(-1, 3)) for k in ("qpos", "qvel"))
    want = wm.contact_inputs(q, v).double().var(0, unbiased=False).clamp_min(1.0)  # the spread over the training transitions
    assert torch.allclose(var.double(), want, rtol=1e-4)
    with pytest.raises(ValueError, match="--contact_scale is an option of --contacts"):
        fit("x", "--contact_scale")
    with pytest.raises(ValueError, match="--contact_scale with --init"):
        fit("y", *floor, "--contact_scale", "--init", str(tmp_path / "scaled.pt"))
    again = torch.load(fit("again", *floor, "--init", str(tmp_path / "scaled.pt")), weights_only=False)  # a refit keeps the start's scales
    assert torch.equal(again["state"]["in_norm.var"], b["state"]["in_norm.var"])
