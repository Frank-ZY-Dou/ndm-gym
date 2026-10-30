"""Interacted objects of the human-object-interaction tasks: shape, nominal physical properties
used in evaluation, and the ranges training draws from.

``largebox``: OMOMO's large box (OmniRetarget/Holosoma ``largebox.obj``, 0.471 x 0.459 x 0.408 m, centered on its
origin within 6 mm), modeled as a cuboid; training ranges follow Holosoma's randomization of that box (mass 1-4 kg,
sliding friction 0.1-0.6), evaluation uses their midpoints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    size: tuple[float, float, float]  # cuboid extents (m), centered on the object's frame
    mass: float  # kg, evaluation
    friction: float  # evaluation
    mass_range: tuple[float, float]  # training
    friction_range: tuple[float, float]  # training


OBJECTS = {
    "largebox": ObjectSpec("largebox", (0.471, 0.459, 0.408), mass=2.5, friction=0.35, mass_range=(1.0, 4.0), friction_range=(0.1, 0.6)),
}


def get(name: str) -> ObjectSpec:
    if name not in OBJECTS:
        raise KeyError(f"unknown object {name!r}; known: {sorted(OBJECTS)}")
    return OBJECTS[name]
