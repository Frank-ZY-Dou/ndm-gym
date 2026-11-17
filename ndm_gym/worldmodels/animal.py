"""World model of an animal body (robots.animals) for animal motion tracking: MjcfWorldModel (worldmodels.mjcf) over one control
step, on the body built without its actuators' activation filters, whose controls are then the activations.

A body whose actuators filter their controls (the rat: na > 0) carries the activations in its state. Over a control step the
filter is a closed form of the activation at its start and the control held (robots.animals.activation_filter): the
activation it ends at is computed, not predicted, and the forces of its physics steps see the mean activation. The network
reads the mean activation as the actuators' control and, beside it, the activation the step starts from, which with the mean
fixes the forces' course over the step (the mean alone leaves it open)."""

from __future__ import annotations

import torch

from .mjcf import MjcfLayout, MjcfWorldModel


class AnimalWorldModel(MjcfWorldModel):
    """``MjcfWorldModel`` whose control input is (the actuators' controls with the filtered ones' mean activations, the start
    activations) for ``n_act`` > 0: (..., nu + n_act)."""

    def __init__(self, lay: MjcfLayout, n_act: int = 0, **kw):
        if n_act and kw.get("backbone", "mlp") != "mlp":
            raise ValueError("AnimalWorldModel: the start activation is an input of the MLP backbone only")
        object.__setattr__(self, "n_act", int(n_act))
        super().__init__(lay, **kw)

    def inputs(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        if not self.n_act:
            return super().inputs(qpos, qvel, ctrl)
        nu = self.lay.nu
        if ctrl.shape[-1] == nu:  # MjcfWorldModel's own call for the input width: no start activation
            ctrl = torch.cat((ctrl, ctrl.new_zeros(ctrl.shape[:-1] + (self.n_act,))), -1)
        return torch.cat((super().inputs(qpos, qvel, ctrl[..., :nu]), ctrl[..., nu:]), -1)
