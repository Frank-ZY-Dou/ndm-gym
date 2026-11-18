"""An MLP world model's whole network as one Triton kernel, for inference (``FusedMLP``): every layer of a tile of samples in one
program, the activations kept on chip from the input to the output.

The networks of ``ndm_gym.core.nets.mlp``: Linear layers, each hidden one followed by an optional LayerNorm and an activation (SiLU,
ELU or ReLU), the hidden layers equally wide (a multiple of 64, at most 512). A network run layer by layer writes every layer's
output to memory and reads it back; once a layer's activations outgrow the L2 cache (fp32, 256 wide: 64 MB at 65,536 samples
against an A100's 40 MB) they go through DRAM and the network's time grows faster than its samples, and with few samples the
kernels' launches and tails dominate. Here a program carries BM samples through all layers in registers: each layer's activations as
chunks of CH (64 or 128) features by BM samples, its products W [out, in] x h [in, BM] chunk by chunk, so that each of the CH / 16
warps holds 16 features of every chunk; only the weights come from L2, once per program and layer. (A form with one product per layer
over a pipelined loop of the inputs, its activations in an L2 slot per program, measured slower at every size; the weights' fetch per
program bounds both.) Where it wins and where it does not (A100 40GB, against the compiled network layer by layer): 3 x 256 wins at
every size measured, 1,024 to 65,536 samples (1.3-1.7x at 1,024, 1.1-1.2x at 4,096, 1.8x at 65,536 in fp32), the network's weights
being a quarter of a megabyte; 3 x 512 in bf16 loses at a planner's 1,228 and 2,047 samples (0.86x, 0.80x): each program streams
the whole network's weights, about 1.1 MB, for 16 to 32 samples, and below about 4,000 samples that fetch outweighs the round trips
the kernel saves. 1,024-wide layers do not fit a program's registers at all.

Precision (``dot``), each as a copy of the network runs it:
  tf32  fp32 throughout, the products' operands rounded to TF32 (to nearest) and accumulated in fp32: the fp32 network with TF32
        products allowed; the first layer, whose few inputs cuBLAS multiplies in full fp32, in full fp32 too (3xTF32 products).
  bf16  the parameters in bf16 and the products accumulated in fp32, every layer's output, the LayerNorm's and the activation's
        rounded to bf16 as torch rounds them for a network with bf16 parameters (``g1_infer.CastNet``); the normalization and the
        activation computed in fp32 in between, as torch computes them.
  ieee  fp32 throughout with exact fp32 products (no tensor cores): for checks.
Forward only; CUDA only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

ACTS = {nn.SiLU: 0, nn.ELU: 1, nn.ReLU: 2}
DOTS = {"tf32": 0, "bf16": 1, "ieee": 2}


@triton.jit
def _act(y, ACT: tl.constexpr):
    if ACT == 0:  # SiLU
        return y / (1.0 + tl.exp(-y))
    elif ACT == 1:  # ELU (alpha 1), with expm1 as torch's (exp(y) - 1 loses the digits of small negative y)
        return tl.where(y > 0, y, libdevice.expm1(tl.minimum(y, 0.0)))
    else:  # ReLU
        return tl.maximum(y, 0.0)


@triton.jit
def _dot(a, b, acc, DOT: tl.constexpr):
    if DOT == 0:
        return tl.dot(a, b, acc, input_precision="tf32")
    elif DOT == 2:
        return tl.dot(a, b, acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _op(v, DOT: tl.constexpr):
    """v as a product's operand: rounded to TF32 to nearest (ties away from zero, as cvt.rna.tf32.f32) in tf32 mode, where the tensor
    cores would otherwise drop the low 13 bits of its fp32 value (a rounding toward zero, biased: every product of a sum errs the same
    way); bf16 in bf16 mode; unchanged in ieee mode."""
    if DOT == 0:
        return ((v.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)
    elif DOT == 1:
        return v.to(tl.bfloat16)
    else:
        return v


@triton.jit
def _norm_act(ys, P, poff, rf, eps, H: tl.constexpr, NC: tl.constexpr, CH: tl.constexpr, LN: tl.constexpr, ACT: tl.constexpr,
              DOT: tl.constexpr):  # fmt: skip
    """A hidden layer's outputs, NC chunks of [CH features, BM samples] (fp32; bf16 in bf16 mode, the layer's output as rounded),
    through the LayerNorm over the features (with LN; its weight and bias at P + poff, P + poff + H) and the activation, in the
    products' operand type."""
    if LN:  # the mean and variance over the features: the chunks summed elementwise first, then one reduction across them each
        t = ys[0].to(tl.float32)
        for n in tl.static_range(1, NC):
            t = t + ys[n].to(tl.float32)
        mean = tl.sum(t, 0) / H
        d = ys[0].to(tl.float32) - mean[None, :]
        q = d * d
        for n in tl.static_range(1, NC):
            d = ys[n].to(tl.float32) - mean[None, :]
            q = q + d * d
        rstd = tl.math.rsqrt(tl.sum(q, 0) / H + eps)
    out = ()
    for n in tl.static_range(NC):
        y = ys[n].to(tl.float32)
        if LN:
            g = tl.load(P + poff + n * CH + rf)
            b = tl.load(P + poff + H + n * CH + rf)
            y = (y - mean[None, :]) * rstd[None, :] * g[:, None] + b[:, None]
            if DOT == 1:  # torch's LayerNorm returns bf16 for a bf16 input
                y = y.to(tl.bfloat16).to(tl.float32)
        out = out + (_op(_act(y, ACT), DOT),)
    return out


@triton.jit
def _mlp_kernel(X, Y, W, P, N, sxn, sxk, syn, eps, IN: tl.constexpr, IN_P: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
                OUT: tl.constexpr, OUT_P: tl.constexpr, LN: tl.constexpr, ACT: tl.constexpr, DOT: tl.constexpr, BM: tl.constexpr,
                CH: tl.constexpr):  # fmt: skip
    """One program: BM samples through every layer, transposed: a layer's activations are NC chunks of [CH features, BM samples] and
    its products W [out, in] x h [in, BM], so that each warp holds its own 16 rows of a chunk (Triton puts the warps of chained
    products along their rows). W: the layers' weights as nn.Linear holds them ([out, in], row-major) one after the other: [H, IN_P],
    (L - 1) x [H, H], [OUT_P, H]; P: per hidden layer its bias, then with LN the LayerNorm's weight and bias (H each), then the head's
    bias (OUT_P)."""
    NC: tl.constexpr = H // CH
    YT: tl.constexpr = tl.bfloat16 if DOT == 1 else tl.float32  # a layer's output as kept until its normalization
    PL: tl.constexpr = 3 * H if LN else H  # the parameters of a hidden layer
    cols = tl.program_id(0) * BM + tl.arange(0, BM)
    cm = cols < N
    rf = tl.arange(0, CH)
    ri = tl.arange(0, IN_P)
    x = tl.load(X + ri[:, None] * sxk + cols[None, :] * sxn, mask=(ri[:, None] < IN) & cm[None, :], other=0.0).to(tl.float32)
    if DOT != 0:  # tf32: the first layer in full fp32 (3xTF32 products), as cuBLAS runs a layer with so few inputs
        x = _op(x, DOT)
    ys = ()
    for n in tl.static_range(NC):  # the first layer: [CH, IN_P] x [IN_P, BM] per chunk
        w0 = tl.load(W + (n * CH + rf)[:, None] * IN_P + ri[None, :])
        if DOT == 0:
            acc = tl.dot(w0, x, tl.zeros((CH, BM), tl.float32), input_precision="tf32x3")
        else:
            acc = _dot(w0, x, tl.zeros((CH, BM), tl.float32), DOT)
        ys = ys + ((acc + tl.load(P + n * CH + rf)[:, None]).to(YT),)
    h = _norm_act(ys, P, H, rf, eps, H, NC, CH, LN, ACT, DOT)
    for layer in tl.static_range(1, L):  # the other hidden layers: [CH, CH] weights at a time
        wo = IN_P * H + (layer - 1) * H * H
        po = layer * PL
        ys = ()
        for n in tl.static_range(NC):
            acc = tl.zeros((CH, BM), tl.float32)
            for k in tl.static_range(NC):
                acc = _dot(tl.load(W + wo + (n * CH + rf)[:, None] * H + (k * CH + rf)[None, :]), h[k], acc, DOT)
            ys = ys + ((acc + tl.load(P + po + n * CH + rf)[:, None]).to(YT),)
        h = _norm_act(ys, P, po + H, rf, eps, H, NC, CH, LN, ACT, DOT)
    ro = tl.arange(0, OUT_P)  # the head: [OUT_P, H] x [H, BM]
    wo = IN_P * H + (L - 1) * H * H
    acc = tl.zeros((OUT_P, BM), tl.float32)
    for k in tl.static_range(NC):
        acc = _dot(tl.load(W + wo + ro[:, None] * H + (k * CH + rf)[None, :]), h[k], acc, DOT)
    y = (acc + tl.load(P + L * PL + ro)[:, None]).to(YT).to(tl.float32)
    tl.store(Y + cols[None, :] * syn + ro[:, None], y, mask=(ro[:, None] < OUT) & cm[None, :])


def _pad(n: int) -> int:
    """n padded for a product's K or M: the next power of two (Triton's ranges), at least 16 (a tensor core's smallest)."""
    return max(16, triton.next_power_of_2(n))


def mlp_layers(net: nn.Sequential) -> tuple[list[tuple[nn.Linear, nn.LayerNorm | None]], nn.Linear, type[nn.Module]]:
    """The hidden layers (Linear, LayerNorm or None) of ``net``, its head and its activation; raises unless ``net`` is an MLP the
    kernel runs (module docstring)."""
    mods = list(net)
    if not mods or not isinstance(mods[-1], nn.Linear):
        raise ValueError("expected an nn.Sequential of Linear layers ending in a Linear head")
    hidden, i, kinds = [], 0, set()
    while i < len(mods) - 1:  # Linear, (LayerNorm), activation, per hidden layer
        lin = mods[i]
        if not isinstance(lin, nn.Linear) or i + 1 >= len(mods) - 1:
            raise ValueError(f"layer {i}: expected Linear, (LayerNorm,) activation, got {[type(m).__name__ for m in mods[i : i + 3]]}")
        norm, j = (mods[i + 1], i + 2) if isinstance(mods[i + 1], nn.LayerNorm) else (None, i + 1)
        if norm is not None and (norm.normalized_shape != (lin.out_features,) or not norm.elementwise_affine or norm.bias is None):
            raise ValueError(f"layer {i + 1}: a LayerNorm over the layer's {lin.out_features} features with weight and bias is supported, not {norm}")
        act = mods[j]
        if type(act) not in ACTS or (isinstance(act, nn.ELU) and act.alpha != 1.0):
            raise ValueError(f"layer {j}: the activations SiLU, ELU (alpha 1) and ReLU are supported, not {act}")
        if lin.bias is None:
            raise ValueError(f"layer {i}: a Linear layer without bias is not supported")
        kinds.add((norm is not None, type(act), None if norm is None else norm.eps))
        hidden.append((lin, norm))
        i = j + 1
    if len(kinds) != 1:
        raise ValueError(f"every hidden layer must have the same normalization and activation, got {kinds}")
    H = hidden[0][0].out_features
    if any(lin.out_features != H for lin, _ in hidden) or H % 64 or H > 512:
        raise ValueError(f"the hidden layers must be equally wide, a multiple of 64 and at most 512: {[lin.out_features for lin, _ in hidden]}")
    if mods[-1].bias is None:
        raise ValueError("a Linear head without bias is not supported")
    return hidden, mods[-1], kinds.pop()[1]


class FusedMLP(nn.Module):
    """``net`` (an ``nn.Sequential`` of ``core.nets.mlp``'s form, on a CUDA device) as one kernel per forward: ``forward(x)`` takes
    [N, in] (any float type, any strides) and returns [N, out] fp32. ``dot``: the precision (module docstring); ``bm``: samples per
    program, ``ch``: feature rows per chunk (64 or 128; a program has a warp per 16), by default ``config``'s. Raises on a network it
    does not run."""

    def __init__(self, net: nn.Sequential, dot: str = "tf32", bm: int | None = None, ch: int | None = None) -> None:
        super().__init__()
        if dot not in DOTS:
            raise ValueError(f"dot must be one of {tuple(DOTS)}, got {dot!r}")
        hidden, head, act = mlp_layers(net)
        if head.weight.device.type != "cuda":
            raise ValueError("FusedMLP runs a Triton kernel: move the network to a CUDA device first")
        H = hidden[0][0].out_features
        self.n_in, self.n_out, self.H, self.L = hidden[0][0].in_features, head.out_features, H, len(hidden)
        self.in_p, self.out_p = _pad(self.n_in), _pad(self.n_out)
        self.ln = hidden[0][1] is not None
        self.eps = float(hidden[0][1].eps) if self.ln else 0.0
        if ch is not None and (ch not in (64, 128) or H % ch):
            raise ValueError(f"ch must be 64 or 128 and divide the hidden width {H}, got {ch}")
        self.act, self.dot, self.bm, self.ch = ACTS[act], dot, bm, ch
        # parameters as the copy holds them: bf16 in bf16 mode (all of them, as CastNet's), fp32 otherwise
        pt = torch.bfloat16 if dot == "bf16" else torch.float32
        dev = head.weight.device
        with torch.no_grad():
            f = lambda t: t.detach().to(pt).float()  # noqa: E731
            w0 = torch.zeros(H, self.in_p, device=dev)
            w0[:, : self.n_in] = f(hidden[0][0].weight)
            wh = torch.zeros(self.out_p, H, device=dev)
            wh[: self.n_out] = f(head.weight)
            ws = [w0.flatten(), *(f(lin.weight).contiguous().flatten() for lin, _ in hidden[1:]), wh.flatten()]
            ps = []
            for lin, norm in hidden:
                ps += [f(lin.bias)] + ([f(norm.weight), f(norm.bias)] if norm is not None else [])
            bh = torch.zeros(self.out_p, device=dev)
            bh[: self.n_out] = f(head.bias)
            w = torch.cat(ws).to(pt).contiguous()
            if dot == "tf32":  # the hidden layers' and the head's weights as the tensor cores' operands, rounded to TF32 to nearest once (as
                # _op rounds the activations); the first layer's stay fp32 (3xTF32 products)
                k = self.in_p * H
                w[k:] = ((w[k:].view(torch.int32) + 0x1000) & -8192).view(torch.float32)
            self.register_buffer("w", w, persistent=False)
            self.register_buffer("p", torch.cat([*ps, bh]).contiguous(), persistent=False)

    def config(self, n: int) -> tuple[int, int]:
        """(samples per program, feature rows per chunk) for n samples (a program has a warp per 16 rows of a chunk): the given ones, or
        the fastest measured for the quadrotor's 3 x 256 network on an A100 40GB (1,024 to 65,536 samples): with few samples, small
        tiles give more programs; with many, larger tiles fetch the weights once for more samples."""
        if self.dot == "ieee":  # the checks' mode: 16 samples (exact fp32 products stage their operands in shared memory), chunks of
            # 128 features where they fit (fewer, larger products compile in a fraction of the time)
            bm, ch = 16, 128
        elif self.dot == "bf16":
            bm, ch = (16, 128) if n <= 2048 else (16, 64) if n <= 8192 else (32, 64) if n <= 32768 else (64, 64)
        else:
            bm, ch = (16, 128) if n <= 2048 else (64, 128) if n <= 8192 else (32, 64) if n <= 32768 else (64, 128)
        ch = ch if self.H % ch == 0 else 64
        return self.bm or bm, self.ch or ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("FusedMLP is inference-only: run it under torch.no_grad() or torch.inference_mode()")
        if x.dim() != 2 or x.shape[1] != self.n_in or x.device != self.w.device:
            raise ValueError(f"expected [N, {self.n_in}] on {self.w.device}, got {tuple(x.shape)} on {x.device}")
        bm, ch = self.config(x.shape[0])
        meta = [self.n_in, self.in_p, self.H, self.L, self.n_out, self.out_p, int(self.ln), self.act, DOTS[self.dot], bm, ch]
        return mlp_op(x, self.w, self.p, meta, self.eps)


@torch.library.custom_op("ndm_gym::mlp_fused", mutates_args=())
def mlp_op(x: torch.Tensor, w: torch.Tensor, p: torch.Tensor, meta: list[int], eps: float) -> torch.Tensor:
    """The kernel as an operator (so that a torch.compile'd step keeps it as one call): x [N, in] (any float type and strides) through
    the network of the weights w and parameters p (FusedMLP's buffers); meta: in, its padding, H, L, out, its padding, LN, the
    activation, the precision, the samples per program and the features per chunk. Returns [N, out] fp32."""
    n_in, in_p, H, L, n_out, out_p, ln, act, dot, bm, ch = meta
    N = x.shape[0]
    y = torch.empty(N, n_out, device=x.device, dtype=torch.float32)
    _mlp_kernel[(triton.cdiv(N, bm),)](x, y, w, p, N, x.stride(0), x.stride(1), y.stride(0), eps, IN=n_in, IN_P=in_p, H=H, L=L, OUT=n_out,
                                       OUT_P=out_p, LN=ln, ACT=act, DOT=dot, BM=bm, CH=ch, num_warps=ch // 16)  # fmt: skip
    return y


@mlp_op.register_fake
def _(x: torch.Tensor, w: torch.Tensor, p: torch.Tensor, meta: list[int], eps: float) -> torch.Tensor:
    return x.new_empty(x.shape[0], meta[4], dtype=torch.float32)


def _tf32(t: torch.Tensor) -> torch.Tensor:
    """fp32 values rounded to TF32 to nearest (ties away from zero, as cvt.rna.tf32.f32), as fp32."""
    return ((t.float().contiguous().view(torch.int32) + 0x1000) & -8192).view(torch.float32)


def _fp64(net: nn.Sequential, x: torch.Tensor, tf32: bool) -> torch.Tensor:
    """net in fp64 on x; with tf32 every Linear layer but the first multiplies operands rounded to TF32 (from their fp32 values)."""
    h = x.double()
    first = True
    for m in net:
        if isinstance(m, nn.Linear):
            w, b = m.weight.double(), m.bias.double()
            if tf32 and not first:
                h, w = _tf32(h).double(), _tf32(m.weight).double()
            h = h @ w.t() + b
            first = False
        elif isinstance(m, nn.LayerNorm):
            h = nn.functional.layer_norm(h, m.normalized_shape, m.weight.double(), m.bias.double(), m.eps)
        else:
            h = m(h)
    return h.float()


def reference(net: nn.Sequential, dot: str):
    """The run of ``net`` a ``FusedMLP`` of precision ``dot`` is held to, a function of the inputs: tf32, the network in fp64 with every
    product's operands but the first layer's rounded to TF32 (what TF32 products do, independent of the library's choice of kernel:
    cuBLAS uses TF32 only for some shapes); bf16, torch with bf16 parameters, the input cast to bf16 and the output to fp32
    (``g1_infer.CastNet``); ieee, torch in fp32 with TF32 off."""
    import copy

    if dot == "tf32":
        return lambda x: _fp64(net, x, True)
    if dot == "bf16":
        net = copy.deepcopy(net).to(torch.bfloat16)

    def run(x: torch.Tensor) -> torch.Tensor:
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return net(x.to(torch.bfloat16)).float() if dot == "bf16" else net(x.float())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    return run


def check(net: nn.Sequential, fm: FusedMLP, x: torch.Tensor) -> dict:
    """Raise unless ``fm`` runs ``net`` as its precision should on the inputs ``x``: against ``net`` with its parameters as ``fm`` holds
    them computed in fp64, ``fm``'s distance at most 2x on average and 4x at the largest that of ``reference(net, fm.dot)``, or, in ieee
    mode, at most 64 fp32 rounding units of the outputs' scale (a sum's order and fp32's exp and division differ from torch's at that
    level). Returns the four distances (max, mean of each)."""
    import copy

    with torch.no_grad():
        held = copy.deepcopy(net)
        if fm.dot == "bf16":
            held = held.to(torch.bfloat16)
        exact = _fp64(held, x.to(torch.bfloat16) if fm.dot == "bf16" else x, False)
        d = (fm(x) - exact).abs()
        r = (reference(net, fm.dot)(x) - exact).abs()
    out = dict(max=float(d.max()), mean=float(d.mean()), ref_max=float(r.max()), ref_mean=float(r.mean()))
    floor = 64 * 2.0**-23 * max(1.0, float(exact.abs().max())) if fm.dot == "ieee" else 0.0
    if not (torch.isfinite(d).all() and (out["max"] <= floor or (out["max"] <= 4 * out["ref_max"] and out["mean"] <= 2 * out["ref_mean"]))):
        raise RuntimeError(f"the fused network ({fm.dot}) differs from the network computed in fp64 by {out['max']:.3g} at most and "
                           f"{out['mean']:.3g} on average, its reference by {out['ref_max']:.3g} and {out['ref_mean']:.3g}, "
                           f"on {torch.cuda.get_device_name(x.device)}")  # fmt: skip
    return out
