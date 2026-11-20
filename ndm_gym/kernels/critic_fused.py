"""SuperMimic's critic minibatch step (algos.supermimic with --critic_compile: the loss ((critic(X[idx]) - Y[idx]) ** 2).mean() on
the inputs normalized once per update, its backward pass, clip_grad_norm_(1.0) and Adam's step) as Triton kernels around cuBLAS's
products, for training (train.py --critic_fused). The critic is Linear(D, N1), ELU, Linear(N1, N2), ELU, Linear(N2, 1).

Per step: the minibatch's rows gathered by a kernel into rows padded to 16 bytes (cuBLAS's aligned products; the first layer's
weights' copy padded alike); then the network forward and backward with the products by cuBLAS, as F.linear and autograd run them
(the first two layers' forward, the second layer's input gradient, the first two layers' weight gradients), and around them three
kernels written by hand: the first ELU; the second ELU, the third layer, the loss against the targets gathered by the minibatch's
indices and the backward pass to the second layer's pre-activation, one program per tile of rows; the first ELU's backward pass.
The row kernels write, per program, the sums the biases', the third layer's weights' and the loss's values need (no atomics: the
same inputs give the same bits). Three small kernels end the step: the per-program sums reduced into the gradients of the biases
and of the third layer and into the loss; every gradient into one flat fp32 buffer with its blocks' sums of squares (and, in its
first program, the step count and Adam's bias corrections); then clip_grad_norm_'s scale and Adam's step over the flat buffer,
which also writes the copies of the parameters the next step's products read. About 45 launches per step become 12 (and cuBLAS's
split-K reductions).

The critic's six parameters are views of one fp32 buffer, their gradients views of another, and Adam's state (exp_avg,
exp_avg_sq, the step count) lives in the optimizer's state as views of two more, so its state_dict is the usual one. Under bf16
autocast every value the default rounds to bf16 is rounded here at the same point (to nearest even, by the bits, so that Triton's
interpreter rounds as CUDA does). Adam follows torch's fused kernel: the moments, step size and denominator in fp64 from the fp32
state, the square root of the second moment in fp32, the gradient scaled by the clipping factor in fp32; beta ** step as a running
product in fp64 (torch takes pow; equal far below fp32 rounding). ELU is x > 0 ? x : expm1(x), expm1 by its Taylor polynomial of
degree 8 on [-0.5, 0] and exp(x) - 1 below (relative error under 1.4e-7; the interpreter has no libdevice); its derivative exp(x)
where x <= 0. Equal to the default up to float rounding: tests/test_critic_fused.py checks the steps' losses, gradients,
parameters and Adam's state against the default's, on the CPU through Triton's interpreter (TRITON_INTERPRET=1) and on a GPU
compiled, fp32 and bf16, also captured in a CUDA graph."""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..core import nets

BM, TAIL_WARPS = 64, 4  # rows per program of the row kernels (each writes one row of per-program sums) and their warps (A100 sweep)
GATHER_ROWS, GATHER_COLS = 16, 128  # the gather's tile
ALIGN = 8  # the first layer's input rows padded to a multiple of 8 values (16 bytes in bf16): cuBLAS's aligned products
BLOCK, FLAT_WARPS = 1024, 4  # elements per program of the flat buffer's kernels


@triton.jit
def _round_bf16(x):
    """fp32 -> the nearest bf16 (ties to even), as fp32: CUDA's conversion, by the bits (finite values)"""
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _lowp(x, LOWP: tl.constexpr):
    """x rounded to the products' dtype: bf16 under autocast, else unchanged"""
    if LOWP:
        x = _round_bf16(x)
    return x


@triton.jit
def _elu(z):
    """ELU (alpha 1): z > 0 ? z : expm1(z); expm1 by its Taylor polynomial on [-0.5, 0], exp(z) - 1 below"""
    x = tl.minimum(z, 0.0)
    p = x * (1.0 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 + x * (1.0 / 5040 + x * (1.0 / 40320))))))))
    return tl.where(z > 0, z, tl.where(x < -0.5, tl.exp(x) - 1.0, p))


@triton.jit
def _elu_back(g, z):
    """ELU's backward pass (torch's elu_backward on its input): z <= 0 ? g * exp(z) : g"""
    return tl.where(z <= 0, g * tl.exp(tl.minimum(z, 0.0)), g)


@triton.jit
def _gather_kernel(X, IDX, XG, M, DP, D: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    """The minibatch's rows: XG[r, :D] = X[IDX[r], :D], XG's rows DP values apart (its padding columns are not written)"""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < M
    src = tl.load(IDX + rows, mask=rm, other=0)
    for c0 in range(0, D, BD):
        cols = c0 + tl.arange(0, BD)
        m = rm[:, None] & (cols < D)[None, :]
        tl.store(XG + rows[:, None] * DP + cols[None, :], tl.load(X + src[:, None] * D + cols[None, :], mask=m), mask=m)


@triton.jit
def _elu_kernel(Z1, H1o, n, LOWP: tl.constexpr, BLOCK: tl.constexpr):
    """The first ELU: H1o = ELU(Z1) (n values, rounded as the default's)"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    z = tl.load(Z1 + i, mask=m, other=0.0).to(tl.float32)
    tl.store(H1o + i, _lowp(_elu(z), LOWP).to(H1o.dtype.element_ty), mask=m)


@triton.jit
def _head_kernel(Z2, Y, IDX, W3, B3, DZ2o, PART, M, scale, N1: tl.constexpr, N2: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
                 LOWP: tl.constexpr):  # fmt: skip
    """Rows [pid * BM, pid * BM + BM) of the minibatch, from the second layer's output Z2 (M, N2) (its bias added): the second ELU, the
    third layer (W3 (N2), B3 (1)), the loss's terms against the targets Y at the rows IDX, and the backward pass to the second layer's
    pre-activation: DZ2o (M, N2); PART's row pid from column N1: the sums over the rows of dZ2 (N2), dv * h2 (N2), dv (1) and the
    squared errors (1); scale: the loss's gradient factor 2 / M"""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < M
    c2 = tl.arange(0, N2)
    z2 = tl.load(Z2 + rows[:, None] * N2 + c2[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    h2 = _lowp(_elu(z2), LOWP)
    w3 = tl.load(W3 + c2).to(tl.float32)
    v = _lowp(tl.sum(h2 * w3[None, :], axis=1) + tl.load(B3).to(tl.float32), LOWP)
    y = tl.load(Y + tl.load(IDX + rows, mask=rm, other=0), mask=rm, other=0.0)
    d = tl.where(rm, v - y, 0.0)
    dv = _lowp(d * scale, LOWP)  # the loss's gradient 2 (v - y) / M, cast back as .float()'s backward pass casts it
    dz2 = tl.where(rm[:, None], _lowp(_elu_back(_lowp(dv[:, None] * w3[None, :], LOWP), z2), LOWP), 0.0)
    tl.store(DZ2o + rows[:, None] * N2 + c2[None, :], dz2.to(DZ2o.dtype.element_ty), mask=rm[:, None])
    part = PART + pid * C
    tl.store(part + N1 + c2, tl.sum(dz2, axis=0))
    tl.store(part + N1 + N2 + c2, tl.sum(dv[:, None] * h2, axis=0))
    tl.store(part + N1 + 2 * N2, tl.sum(dv, axis=0))
    tl.store(part + N1 + 2 * N2 + 1, tl.sum(d * d, axis=0))


@triton.jit
def _elu_back_kernel(DH1, Z1, DZ1o, PART, M, N1: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, LOWP: tl.constexpr):
    """Rows [pid * BM, pid * BM + BM): the first ELU's backward pass, DZ1o = ELU'(Z1) * DH1 (M, N1), and PART's row pid's first N1
    columns: the sums over the rows of dZ1"""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < M
    c1 = tl.arange(0, N1)
    at = rows[:, None] * N1 + c1[None, :]
    z1 = tl.load(Z1 + at, mask=rm[:, None], other=0.0).to(tl.float32)
    dh1 = tl.load(DH1 + at, mask=rm[:, None], other=0.0).to(tl.float32)
    dz1 = tl.where(rm[:, None], _lowp(_elu_back(dh1, z1), LOWP), 0.0)
    tl.store(DZ1o + at, dz1.to(DZ1o.dtype.element_ty), mask=rm[:, None])
    tl.store(PART + pid * C + c1, tl.sum(dz1, axis=0))


@triton.jit
def _parts_kernel(PART, G, LOSS, inv_m, OFF_B1, OFF_B2, OFF_W3, OFF_B3, P: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, C: tl.constexpr,
                  BC: tl.constexpr, BP: tl.constexpr, LOWP: tl.constexpr):  # fmt: skip
    """PART's P rows summed column by column (BC columns per program, in a fixed order) -> the gradients of the first two biases,
    the third layer's weights and its bias (rounded as the default's products round them) at their offsets of the flat buffer G,
    and the loss (the squared errors' sum times inv_m) to LOSS"""
    cols = tl.program_id(0) * BC + tl.arange(0, BC)
    cm = cols < C
    acc = tl.zeros((BC,), dtype=tl.float32)
    for p0 in range(0, P, BP):
        rows = p0 + tl.arange(0, BP)
        acc += tl.sum(tl.load(PART + rows[:, None] * C + cols[None, :], mask=(rows < P)[:, None] & cm[None, :], other=0.0), axis=0)
    off = tl.where(cols < N1, OFF_B1 + cols, tl.where(cols < N1 + N2, OFF_B2 + cols - N1, tl.where(cols < N1 + 2 * N2, OFF_W3 + cols - N1 - N2,
                                                                                                         OFF_B3)))  # fmt: skip
    tl.store(G + off, _lowp(acc, LOWP), mask=cols < N1 + 2 * N2 + 1)
    tl.store(LOSS + cols * 0, acc * inv_m, mask=cols == N1 + 2 * N2 + 1)


@triton.jit
def _gather_sq_kernel(GW1, GW2, G, SQ, STEP, C64, n, N_W1, D, DP, OFF_W2, N_W2, BLOCK: tl.constexpr):
    """The flat gradient buffer G (n): the first two layers' weight gradients GW1 (rows DP values apart, D used) and GW2 (the
    products' dtype) copied in as fp32 at their offsets (0 and OFF_W2), the rest as written; SQ[pid]: the sum of the squares of the
    program's block. The first program
    also counts the step (STEP += 1) and writes Adam's bias corrections for it: C64 = beta1, beta2, eps, beta1 ** step, beta2 ** step,
    1 - beta1 ** step, sqrt(1 - beta2 ** step) (fp64; the powers as running products)"""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    in1 = i < N_W1
    in2 = (i >= OFF_W2) & (i < OFF_W2 + N_W2)
    g1 = tl.load(GW1 + (i // D) * DP + i % D, mask=m & in1, other=0.0).to(tl.float32)
    g2 = tl.load(GW2 + (i - OFF_W2), mask=m & in2, other=0.0).to(tl.float32)
    g0 = tl.load(G + i, mask=m & ~in1 & ~in2, other=0.0)
    g = tl.where(in1, g1, tl.where(in2, g2, g0))
    tl.store(G + i, g, mask=m & (in1 | in2))
    tl.store(SQ + pid, tl.sum(g * g, axis=0))
    if pid == 0:
        tl.store(STEP, tl.load(STEP) + 1.0)
        p1 = tl.load(C64 + 3) * tl.load(C64)
        p2 = tl.load(C64 + 4) * tl.load(C64 + 1)
        tl.store(C64 + 3, p1)
        tl.store(C64 + 4, p2)
        tl.store(C64 + 5, 1.0 - p1)
        tl.store(C64 + 6, tl.sqrt(1.0 - p2))


@triton.jit
def _adam_kernel(P, G, M1, V, PL, W1L, SQ, C64, LR, n, NB, N_W1, D, DP, max_norm, BLOCK: tl.constexpr, NBP: tl.constexpr,
                 LOWP: tl.constexpr):  # fmt: skip
    """clip_grad_norm_(max_norm) and Adam's step (torch's fused kernel) over the flat buffers: the total norm from the NB blocks'
    sums of squares SQ, the gradients G scaled by min(max_norm / (norm + 1e-6), 1) (written back, as clip_grad_norm_ leaves them),
    the moments M1, V and the parameters P updated with the learning rate LR[0] and the bias corrections in C64; then the copies the
    products read: the first layer's weights in W1L (rows DP values apart; bf16 under LOWP, else fp32), under LOWP the others' bf16
    copies in PL (at the parameters' offsets)"""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    b = tl.arange(0, NBP)
    norm = tl.sqrt_rn(tl.sum(tl.load(SQ + b, mask=b < NB, other=0.0), axis=0))
    coef = tl.minimum(tl.div_rn(max_norm, norm + 1e-6), 1.0)
    g = tl.load(G + i, mask=m, other=0.0) * coef
    tl.store(G + i, g, mask=m)
    beta1, beta2, eps = tl.load(C64), tl.load(C64 + 1), tl.load(C64 + 2)
    bc1, bc2s = tl.load(C64 + 5), tl.load(C64 + 6)
    lr = tl.load(LR).to(tl.float64)
    g64 = g.to(tl.float64)
    m1 = (beta1 * tl.load(M1 + i, mask=m, other=0.0).to(tl.float64) + (1.0 - beta1) * g64).to(tl.float32)
    v = (beta2 * tl.load(V + i, mask=m, other=0.0).to(tl.float64) + (1.0 - beta2) * g64 * g64).to(tl.float32)
    denom = tl.sqrt_rn(v).to(tl.float64) / bc2s + eps
    p = (tl.load(P + i, mask=m, other=0.0).to(tl.float64) - (lr / bc1) * m1.to(tl.float64) / denom).to(tl.float32)
    tl.store(M1 + i, m1, mask=m)
    tl.store(V + i, v, mask=m)
    tl.store(P + i, p, mask=m)
    in1 = i < N_W1
    tl.store(W1L + (i // D) * DP + i % D, _lowp(p, LOWP).to(W1L.dtype.element_ty), mask=m & in1)
    if LOWP:
        tl.store(PL + i, _round_bf16(p).to(tl.bfloat16), mask=m & ~in1)


# whether triton.jit made the kernels Triton's interpreter's (TRITON_INTERPRET=1 when this module was imported): then they run on
# CPU tensors too, with fp32 operands in the products (the tests' CPU checks); compiled, on CUDA tensors only
INTERPRETED = not isinstance(_head_kernel, triton.runtime.jit.JITFunction)


class CriticFused:
    """The critic's minibatch steps by the kernels (``step``), for a ``Critic`` (``policies.critic``: Linear, ELU, Linear, ELU, Linear
    to one value) and its Adam optimizer ``opt`` (no weight decay, amsgrad or maximize). Construction makes the critic's parameters
    views of one flat fp32 buffer and their gradients views of another, and keeps Adam's state in the optimizer as views of two more
    (taking over the optimizer's own when it has some: ``attach``, also after its load_state_dict). The products run in
    nets.AMP_DTYPE where its autocast acts (bf16), else in fp32."""

    def __init__(self, critic: nn.Module, opt: torch.optim.Optimizer, max_norm: float = 1.0) -> None:
        net = critic.net
        kinds = [type(m) for m in net]
        if kinds != [nn.Linear, nn.ELU, nn.Linear, nn.ELU, nn.Linear] or any(m.alpha != 1.0 for m in net if isinstance(m, nn.ELU)):
            raise ValueError(f"kernels.critic_fused: a critic of Linear, ELU (alpha 1), Linear, ELU, Linear, got {kinds}")
        l1, l2, l3 = net[0], net[2], net[4]
        self.D, self.N1, self.N2 = l1.in_features, l1.out_features, l2.out_features
        if l3.out_features != 1 or any(k & (k - 1) or not 16 <= k <= 256 for k in (self.N1, self.N2)):
            raise ValueError(f"kernels.critic_fused: hidden widths powers of two in [16, 256] and one output, got {self.N1}, {self.N2}, "
                             f"{l3.out_features}")  # fmt: skip
        if not isinstance(opt, torch.optim.Adam) or len(opt.param_groups) != 1:
            raise ValueError("kernels.critic_fused: an Adam optimizer of the critic's parameters, one group")
        grp = opt.param_groups[0]
        if {id(p) for p in grp["params"]} != {id(p) for p in critic.parameters()}:
            raise ValueError("kernels.critic_fused: an Adam optimizer of the critic's parameters, one group")
        if grp["weight_decay"] or grp["amsgrad"] or grp["maximize"]:
            raise ValueError("kernels.critic_fused: Adam without weight decay, amsgrad or maximize")
        self.critic, self.opt, self.max_norm = critic, opt, float(max_norm)
        self.params = [l1.weight, l1.bias, l2.weight, l2.bias, l3.weight, l3.bias]
        dev = l1.weight.device
        if any(p.dtype != torch.float32 or p.device != dev for p in self.params):
            raise ValueError("kernels.critic_fused: fp32 parameters on one device")
        self.lowp = nets.AMP_DTYPE if nets.AMP_DTYPE is not None and nets._autocasts(dev) else None
        if self.lowp not in (None, torch.bfloat16):
            raise ValueError(f"kernels.critic_fused: bf16 autocast or none, got {self.lowp}")
        sizes = [p.numel() for p in self.params]
        self.offs = [sum(sizes[:k]) for k in range(len(sizes))]
        self.n = sum(sizes)
        self.P = torch.cat([p.detach().reshape(-1) for p in self.params])
        self.G = torch.zeros_like(self.P)
        self.M1, self.V = torch.zeros_like(self.P), torch.zeros_like(self.P)
        self.STEP = torch.zeros((), dtype=torch.float32, device=dev)
        for p, o, k in zip(self.params, self.offs, sizes):
            p.data = self.P[o : o + k].view_as(p)
            p.grad = self.G[o : o + k].view_as(p)
        self.dt = self.lowp or torch.float32  # the products' dtype
        self.DP = -(-self.D // ALIGN) * ALIGN  # the first layer's input rows, padded for cuBLAS's aligned products
        self.W1L = torch.zeros(self.N1, self.DP, dtype=self.dt, device=dev)  # its weights in the products' dtype (zero padding)
        self.PL = torch.empty(self.n, dtype=self.lowp, device=dev) if self.lowp is not None else None  # the others' bf16 copies
        self.NB = triton.cdiv(self.n, BLOCK)
        self.SQ = torch.zeros(self.NB, dtype=torch.float32, device=dev)
        self.C64 = torch.zeros(7, dtype=torch.float64, device=dev)  # beta1, beta2, eps, beta1 ** step, beta2 ** step, the corrections
        self.LOSS = torch.zeros(1, dtype=torch.float32, device=dev)
        self.lr = torch.zeros(1, dtype=torch.float32, device=dev)  # the learning rate when the group's is a number
        self.work: dict[int, tuple] = {}
        self.attach()
        self.refresh()

    def views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        """flat's six parameter views, in the parameters' shapes"""
        return [flat[o : o + p.numel()].view_as(p) for p, o in zip(self.params, self.offs)]

    def attach(self) -> None:
        """Adam's state as views of the flat buffers: the optimizer's own (after its load_state_dict, or steps of its own) copied in,
        the step count the parameters' common one; the running powers of the betas from it"""
        st = [self.opt.state.get(p, {}) for p in self.params]
        if any(st) and not all("exp_avg" in s and "exp_avg_sq" in s and "step" in s for s in st):
            raise ValueError("kernels.critic_fused: Adam's state for some of the critic's parameters only")
        with torch.no_grad():
            if all(st):
                for s, m1, v in zip(st, self.views(self.M1), self.views(self.V)):
                    if s["exp_avg"].data_ptr() != m1.data_ptr():
                        m1.copy_(s["exp_avg"]), v.copy_(s["exp_avg_sq"])
                steps = {float(s["step"]) for s in st}
                if len(steps) != 1:
                    raise ValueError(f"kernels.critic_fused: Adam's step counts differ between the critic's parameters: {sorted(steps)}")
                self.STEP.fill_(steps.pop())
            beta1, beta2 = self.opt.param_groups[0]["betas"]
            step = float(self.STEP)
            self.C64.copy_(torch.tensor([beta1, beta2, self.opt.param_groups[0]["eps"], beta1**step, beta2**step, 0.0, 0.0], dtype=torch.float64))
        for p, m1, v in zip(self.params, self.views(self.M1), self.views(self.V)):
            self.opt.state[p] = {"step": self.STEP, "exp_avg": m1, "exp_avg_sq": v}

    def refresh(self) -> None:
        """The parameters' copies the products read (the first layer's weights padded; bf16 under the autocast), from the
        parameters: after any change of them outside the steps"""
        with torch.no_grad():
            self.W1L[:, : self.D].copy_(self.params[0])
            if self.PL is not None:
                self.PL.copy_(self.P)

    def _work(self, M: int, dt: torch.dtype, dev: torch.device) -> tuple:
        if M not in self.work:
            e = lambda *s, d=dt: torch.empty(*s, dtype=d, device=dev)  # noqa: E731
            self.work[M] = (torch.zeros(M, self.DP, dtype=dt, device=dev), e(M, self.N1), e(M, self.N1), e(M, self.N2), e(M, self.N2),
                            e(M, self.N1), e(M, self.N1), e(triton.cdiv(M, BM), self.N1 + 2 * self.N2 + 2, d=torch.float32), e(self.N1, self.DP),
                            e(self.N2, self.N1))  # fmt: skip
        return self.work[M]

    def step(self, X: torch.Tensor, Y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """One minibatch step on the rows ``idx`` of the normalized inputs X (n, D) (in the products' dtype) and the targets Y (n,):
        the parameters, their gradients (clipped) and Adam's state updated in place; returns the minibatch's loss (a 0-dim view of a
        buffer the next step overwrites)"""
        dt = self.lowp or torch.float32
        if X.dtype != dt or X.dim() != 2 or X.shape[1] != self.D or Y.dtype != torch.float32 or idx.dim() != 1:
            raise ValueError(f"kernels.critic_fused: X (n, {self.D}) {dt}, Y fp32 and idx (M,), got {tuple(X.shape)} {X.dtype}, {Y.dtype}, "
                             f"{tuple(idx.shape)}")  # fmt: skip
        if X.device.type != "cuda" and not INTERPRETED:
            raise RuntimeError("kernels.critic_fused: the kernels run on CUDA tensors (on the CPU only through Triton's interpreter, "
                               "TRITON_INTERPRET=1 when the module is imported)")  # fmt: skip
        M = idx.shape[0]
        XG, Z1, H1, Z2, DZ2, DH1, DZ1, PART, GW1, GW2 = self._work(M, dt, X.device)
        _, b1, W2, b2, W3, b3 = self.views(self.PL if self.PL is not None else self.P)
        grp = self.opt.param_groups[0]
        if torch.is_tensor(grp["lr"]):
            lr = grp["lr"].reshape(1)
        else:
            lr = self.lr
            lr.fill_(grp["lr"])
        lowp = self.lowp is not None
        nprog, C = PART.shape
        with torch.cuda.device(X.device) if X.is_cuda else contextlib.nullcontext():
            _gather_kernel[(triton.cdiv(M, GATHER_ROWS),)](X, idx, XG, M, self.DP, D=self.D, BR=GATHER_ROWS, BD=GATHER_COLS)
            torch.addmm(b1, XG, self.W1L.t(), out=Z1)  # the products as F.linear and autograd run them (under autocast: in bf16)
            _elu_kernel[(triton.cdiv(M * self.N1, BLOCK),)](Z1, H1, M * self.N1, LOWP=lowp, BLOCK=BLOCK, num_warps=FLAT_WARPS)
            torch.addmm(b2, H1, W2.t(), out=Z2)
            _head_kernel[(nprog,)](Z2, Y, idx, W3, b3, DZ2, PART, M, 2.0 / M, N1=self.N1, N2=self.N2, C=C, BM=BM, LOWP=lowp,
                                   num_warps=TAIL_WARPS)  # fmt: skip
            torch.mm(DZ2, W2, out=DH1)
            _elu_back_kernel[(nprog,)](DH1, Z1, DZ1, PART, M, N1=self.N1, C=C, BM=BM, LOWP=lowp, num_warps=TAIL_WARPS)
            torch.mm(DZ1.t(), XG, out=GW1)
            torch.mm(DZ2.t(), H1, out=GW2)
            _parts_kernel[(triton.cdiv(C, 16),)](PART, self.G, self.LOSS, 1.0 / M, self.offs[1], self.offs[3], self.offs[4], self.offs[5],
                                                 P=nprog, N1=self.N1, N2=self.N2, C=C, BC=16, BP=128, LOWP=lowp)  # fmt: skip
            _gather_sq_kernel[(self.NB,)](GW1, GW2, self.G, self.SQ, self.STEP, self.C64, self.n, self.params[0].numel(), self.D, self.DP,
                                          self.offs[2], self.params[2].numel(), BLOCK=BLOCK, num_warps=FLAT_WARPS)  # fmt: skip
            _adam_kernel[(self.NB,)](self.P, self.G, self.M1, self.V, self.PL if lowp else self.P, self.W1L, self.SQ, self.C64, lr, self.n,
                                     self.NB, self.params[0].numel(), self.D, self.DP, self.max_norm, BLOCK=BLOCK,
                                     NBP=triton.next_power_of_2(self.NB), LOWP=lowp, num_warps=FLAT_WARPS)  # fmt: skip
        return self.LOSS[0]


__all__ = ["INTERPRETED", "CriticFused"]
