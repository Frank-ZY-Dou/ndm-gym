"""Deploy a trusted local Door development checkpoint in native CPU physics.

The model and replay reference are explicit inputs and must match the saved
algorithm contract. Checkpoints use pickle: load only files you trust.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .motion.adroit_door import DoorReference
from .sim.rigid.adroit_door import DoorPhysics
from .tasks.adroit_door import DoorTask
from .train_door import DOOR_METHODS, evaluate
from .worldmodels.mjcf import MjcfLayout


def checkpoint_options(protocol, state):
    """Recover model-method constructor flags before strict state validation."""
    options = {}
    if protocol.get("method") in ("supertrack", "supermimic"):
        for key, default in (("task_weights", False), ("upstream_policy_optimizer", False),
                             ("correction_reg", 0.), ("correction_imag_noise", 0.)):
            options[key] = state.get(key, default)
    if protocol.get("method") == "supermimic":
        options.update({k: protocol.get(k, 1) for k in ("critic_epochs", "critic_minibatches")})
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "model", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=None,
                        help="ARRz radians; default uses saved evaluation offsets")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint["protocol"]
    offsets = protocol["offsets"] if args.offsets is None else args.offsets
    if not offsets or not np.isfinite(offsets).all():
        raise ValueError("Offsets must be nonempty and finite")
    import mujoco

    torch.set_num_threads(1)
    reference = DoorReference.load(args.reference)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    task = DoorTask(reference, model.actuator_ctrlrange)
    layout = replace(MjcfLayout.from_model(model), dt=task.dt)
    actions = torch.as_tensor(reference.actions, dtype=torch.float32).clamp(-1, 1)
    options = checkpoint_options(protocol, checkpoint["algo"])
    algo = DOOR_METHODS[protocol.get("method", "residual")](task, layout, actions, hidden=(32, 32), horizon=protocol["horizon"], **options)
    algo.load_state_dict(checkpoint["algo"])
    args.output.mkdir(parents=True, exist_ok=False)
    metrics, rollouts = evaluate(algo, DoorPhysics(model, reference), reference, offsets)
    np.savez_compressed(args.output / "physical-rollouts.npz", **rollouts)
    result = {"evaluations": metrics, "checkpoint_sha256": hashlib.sha256(
        args.checkpoint.read_bytes()).hexdigest(), "offsets": offsets,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "learned_policy": protocol["policy_updates"] + protocol.get("supervised_updates", 0) > 0, "three_seed_evidence": False,
        "mujoco": mujoco.__version__, "device": "CPU", "threads": 1}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
