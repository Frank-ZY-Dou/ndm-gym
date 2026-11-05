# Ported from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658, src/blinkmimic/optim.py) under its MIT License:
#
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Muon on the hidden weight matrices of MLPs, Adam on the rest (``--pi_opt muon``, ``--wm_opt muon``; InstantTrack,
BlinkMimic).

Muon (``torch.optim.Muon``: momentum orthogonalized by Newton-Schulz iterations) updates every Linear layer of the given
MLPs except each MLP's first and last, or the weight matrices given instead (the RigidFormer world model's blocks:
``RigidFormerNet.hidden_matrices``). Its step is rescaled to Adam's update RMS (``adjust_lr_fn="match_rms_adamw"``), so
Adam's learning rate and schedule apply unchanged. Input and output layers, biases and norms stay on Adam.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def hidden_matrices(mlps: list[nn.Module]) -> list[nn.Parameter]:
    """The weights of every Linear layer of each MLP except its first and last."""
    return [lin.weight for m in mlps for lin in [x for x in m.modules() if isinstance(x, nn.Linear)][1:-1]]


class MuonAdam:
    """Muon on the hidden matrices of ``mlps`` (submodules of ``model``), or on the 2-D parameters ``matrices`` of ``model``
    given instead (``mlps`` None), Adam with ``betas`` (and ``adam_kw``, e.g. fused) on the rest of ``model``'s parameters;
    both at the learning rate ``lr``.

    ``capturable`` (train.py --update_graphs; BlinkMimic's --cuda_graph): the learning rates are 0-dim device tensors (set
    with ``set_lr``), Adam is the capturable fused kernel, and the Muon step is done here with torch.optim.Muon's
    arithmetic and the tensor learning rate (torch.optim.Muon reads a tensor learning rate on the host, so a captured step
    would freeze it)."""

    def __init__(self, model: nn.Module, mlps: list[nn.Module] | None, lr: float, betas=(0.9, 0.95), capturable: bool = False,
                 matrices: list[nn.Parameter] | None = None, **adam_kw):  # fmt: skip
        if (mlps is None) == (matrices is None):
            raise ValueError("MuonAdam takes the MLPs whose hidden matrices Muon updates or the matrices themselves, one of the two")
        hidden = {id(p) for p in (hidden_matrices(mlps) if matrices is None else matrices)}
        muon = [p for p in model.parameters() if id(p) in hidden]
        adam = [p for p in model.parameters() if id(p) not in hidden]
        if len(muon) != len(hidden):
            raise ValueError("Muon's matrices must be parameters of the model")
        if not muon:
            raise ValueError("MuonAdam: no matrices for Muon")
        self.capturable = capturable
        if capturable:
            dev = muon[0].device
            self.muon = torch.optim.Muon(muon, lr=torch.tensor(float(lr), device=dev), weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=torch.tensor(float(lr), device=dev), betas=betas, fused=True, capturable=True)
        else:
            self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=lr, betas=betas, **adam_kw)

    @property
    def param_groups(self) -> list[dict]:
        # a property: load_state_dict replaces the optimizers' group dicts
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none), self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None):
        """torch.optim.Muon's step with each group's 0-dim device learning rate; ``ok`` as in ``step``."""
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        for g in self.muon.param_groups:
            if g["weight_decay"] != 0.0:
                raise ValueError("MuonAdam's tensor-lr step has no weight decay")
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.muon.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                buf = st["momentum_buffer"]
                if ok is None:
                    buf.lerp_(p.grad, 1 - g["momentum"])
                else:
                    buf.copy_(torch.where(ok, buf.lerp(p.grad, 1 - g["momentum"]), buf))
                update = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                update = _zeropower_via_newtonschulz(update, g["ns_coefficients"], g["ns_steps"], g["eps"])
                ratio = _adjust_lr(1.0, g["adjust_lr_fn"], p.shape)
                d = update.to(p.dtype) * (g["lr"] * -ratio)  # fp32 product (a bf16 update times a 0-dim tensor stays bf16)
                p.add_(d if ok is None else torch.where(ok, d, 0.0))

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None):
        """``ok`` (train.py --no_sync): a 0-dim bool device tensor; where it is false the whole step is skipped without a
        host read (parameters, momenta, Adam's moments and step counts unchanged). The gradients must be finite (zeroed
        first)."""
        if ok is not None:
            if self.capturable:
                self._muon_step_tensor_lr(ok)
            else:  # torch.optim.Muon's own step (unchanged arithmetic), undone where ok is false
                ps = [p for g in self.muon.param_groups for p in g["params"] if p.grad is not None]
                for p in ps:
                    self.muon.state[p].setdefault("momentum_buffer", torch.zeros_like(p.grad, memory_format=torch.preserve_format))
                bufs = [self.muon.state[p]["momentum_buffer"] for p in ps]
                keep = [t.clone() for t in ps + bufs]
                self.muon.step()
                for t, k in zip(ps + bufs, keep):
                    t.copy_(torch.where(ok, t, k))
            if not self.adam.defaults.get("fused"):
                raise ValueError("a gated MuonAdam step needs the fused Adam (--fused_adam)")
            self.adam.found_inf = (~ok).float()  # the fused kernel skips the step and undoes its step count
            try:
                self.adam.step()
            finally:
                self.adam.found_inf = None
            return
        if self.capturable:
            self._muon_step_tensor_lr()
        else:
            self.muon.step()
        self.adam.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd: dict):
        lrs = [g["lr"] for g in self.param_groups]
        self.muon.load_state_dict(sd["muon"]), self.adam.load_state_dict(sd["adam"])
        if self.capturable:  # keep the device learning-rate tensors (the training loop fills them)
            for g, lr in zip(self.param_groups, lrs):
                lr.fill_(float(g["lr"]))
                g["lr"] = lr
