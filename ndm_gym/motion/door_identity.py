"""Byte identity of native Door model XML and reference; no simulator imports.

This verifies the supplied files, not transitive MJCF includes or engine
compatibility. Historical checkpoints without digests need their original source
and cannot certify a new evaluation against arbitrary replacement inputs.
"""
import hashlib
from pathlib import Path


def input_identity(model, reference):
    return {name + "_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (("model", model), ("reference", reference))}


def verify_input_identity(protocol, model, reference):
    identity = input_identity(model, reference)
    for key, digest in identity.items():
        if key not in protocol:
            raise ValueError(f"Checkpoint lacks {key}; use original historical source, not unverified evaluation")
        if protocol[key] != digest:
            raise ValueError(f"Checkpoint input identity mismatch: {key}")
    return identity
