"""Packed transition files of wmbench/collect: every value restored bit for bit."""

import torch

from conftest import DEV


def _bits(x):
    return x.view(torch.int32) if x.dtype == torch.float32 else x


def _recording(T: int = 30, N: int = 5, ds: int = 6, da: int = 4, dx: int = 3, seed: int = 0):
    """A (T >= 8, N) recording as collect writes it (state, action and context sizes ds, da, dx): episodes that end, a
    push halfway, a NaN and a -0.0 in the states."""
    g = torch.Generator().manual_seed(seed)
    done = torch.rand(T, N, generator=g) < 0.1
    s_next = torch.randn(T + 1, N, ds, generator=g)
    s, s1 = s_next[:-1].clone(), s_next[1:].clone()
    s1[done] = torch.randn(int(done.sum()), ds, generator=g)  # an episode ended: the next row starts from a reset state
    s1[T // 2 - 1] = torch.randn(N, ds, generator=g)  # the push before row T // 2 changed every state
    s[3, 2, 1], s1[2, 2, 1] = float("nan"), float("nan")  # equal bits, unequal values
    s[7, 0, 0], s1[6, 0, 0] = 0.0, -0.0  # equal values, unequal bits
    a = torch.randn(T, N, da, generator=g)
    start = torch.cat((torch.ones(1, N, dtype=torch.bool), done[:-1]))
    a_prev = torch.where(start[..., None], 0.0, torch.cat((a[:1], a[:-1])))
    xi = torch.randn(N, dx, generator=g).expand(T, N, dx).clone()
    f = torch.randint(0, 1000, (N,), generator=g).expand(T, N).clone()
    ep = torch.zeros(T, N, dtype=torch.int32)
    for t in range(1, T):
        xi[t] = torch.where(start[t, :, None], torch.randn(N, dx, generator=g), xi[t - 1])
        f[t] = torch.where(start[t], torch.randint(0, 1000, (N,), generator=g), f[t - 1] + 1)
        ep[t] = torch.where(start[t], 0, ep[t - 1] + 1)
    return {"s": s, "s1": s1, "a": a, "a_prev": a_prev, "xi": xi, "r": torch.randn(T, N, generator=g), "f": f, "ep_step": ep, "done": done}


def test_pack_restores_every_value():
    from ndm_gym.wmbench.collect import G1_RULES, pack, unpack

    data = {k: v.to(DEV) for k, v in _recording().items()}
    packed = pack(data, G1_RULES)
    back = unpack(packed, DEV)
    assert back.keys() == data.keys()
    for k, v in data.items():
        assert back[k].dtype == v.dtype and torch.equal(_bits(back[k]), _bits(v)), k
    T, N = data["s"].shape[:2]
    n_starts = int(data["done"][:-1].sum()) + N  # first rows and the rows after an episode end
    for k in ("a_prev", "xi", "f", "ep_step"):
        assert len(packed[k]["idx"]) <= n_starts, k
    assert len(packed["s1"]["idx"]) < T * N // 2  # what the next row repeats is not stored


def test_unpack_reads_full_fields():
    from ndm_gym.wmbench.collect import unpack

    data = _recording(T=8, N=3)
    back = unpack(data, DEV)
    for k, v in data.items():
        assert torch.equal(_bits(back[k]), _bits(v)), k


def test_load_buffer_reads_both_layouts(tmp_path):
    from ndm_gym.buffer import FIELDS
    from ndm_gym.wmbench.collect import G1_RULES, pack
    from ndm_gym.wmbench.fit import load_buffer

    data = _recording(T=20, N=4, ds=FIELDS["s"][0], da=FIELDS["a"][0], dx=FIELDS["xi"][0])
    shapes = {k: (((s,) if isinstance(s, int) else s), dt) for k, (s, dt) in FIELDS.items()}
    assert {k: (tuple(v.shape[2:]), v.dtype) for k, v in data.items()} == shapes  # the replay buffer's fields
    full, packed = tmp_path / "full.pt", tmp_path / "packed.pt"
    torch.save({"meta": {}, "data": data}, full)
    torch.save({"meta": {}, "data": pack(data, G1_RULES)}, packed)
    b_full, _ = load_buffer([str(full)], DEV)
    b_both, _ = load_buffer([str(packed), str(full)], DEV)
    for k in b_full.data:
        assert torch.equal(_bits(b_full.data[k]), _bits(data[k])), k
        assert torch.equal(_bits(b_both.data[k]), _bits(torch.cat((data[k], data[k]), 1))), k
