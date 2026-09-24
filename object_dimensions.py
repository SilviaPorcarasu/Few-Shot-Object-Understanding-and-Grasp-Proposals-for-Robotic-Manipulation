"""
Real object dimensions used as optional priors during grasp scoring.

These values do NOT replace the measured point cloud geometry.
They act as a sanity-check prior:
  - the object's overall size can stabilise scale-dependent metrics
  - the local closing diameter can be compared against plausible
    cross-sections for that specific object

All values are in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObjectDimensions:
    """
    Real dimensions for one known object class.

    bounding_box_xyz_m:
      Approximate full object extents (length, width, height) in meters.

    grasp_diameter_ranges_m:
      Plausible local closing diameters for the gripper.
      Example:
        - a cup can be grasped either on the body (~65-85 mm)
          or on the handle (~10-25 mm)
        - a screwdriver can be grasped on the shaft (~5 mm)
          or on the handle (~20 mm)
    """

    canonical_name: str
    bounding_box_xyz_m: tuple[float, float, float]
    grasp_diameter_ranges_m: tuple[tuple[float, float], ...]
    notes: str = ""

    @property
    def bounding_box_diagonal_m(self) -> float:
        x, y, z = self.bounding_box_xyz_m
        return math.sqrt(x * x + y * y + z * z)

    def as_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "bounding_box_xyz_m": [float(v) for v in self.bounding_box_xyz_m],
            "bounding_box_diagonal_m": float(self.bounding_box_diagonal_m),
            "grasp_diameter_ranges_m": [
                [float(lo), float(hi)]
                for (lo, hi) in self.grasp_diameter_ranges_m
            ],
            "notes": self.notes,
        }


#  Real object sizes provided for the current dataset

OBJECT_DIMENSIONS = {
    "cup": ObjectDimensions(
        canonical_name="cup",
        bounding_box_xyz_m=(0.085, 0.085, 0.120),
        grasp_diameter_ranges_m=(
            (0.060, 0.090),   # cup body / base region
            (0.010, 0.025),   # handle pinch
        ),
        notes="Body diameter 85 mm, base diameter 65 mm, height 120 mm.",
    ),
    "roll_of_tape": ObjectDimensions(
        canonical_name="roll_of_tape",
        bounding_box_xyz_m=(0.100, 0.100, 0.050),
        grasp_diameter_ranges_m=(
            (0.045, 0.055),   # tape height / thickness
            (0.090, 0.105),   # outer diameter
        ),
        notes="Outer diameter 100 mm, height 50 mm.",
    ),
    "screwdriver": ObjectDimensions(
        canonical_name="screwdriver",
        bounding_box_xyz_m=(0.205, 0.020, 0.020),
        grasp_diameter_ranges_m=(
            (0.004, 0.008),   # shaft
            (0.018, 0.022),   # handle base
        ),
        notes="Length 205 mm, handle diameter 20 mm, shaft diameter 5 mm.",
    ),
    "computer_mouse": ObjectDimensions(
        canonical_name="computer_mouse",
        bounding_box_xyz_m=(0.100, 0.060, 0.030),
        grasp_diameter_ranges_m=(
            (0.025, 0.035),   # height-like closing
            (0.055, 0.065),   # width-like closing
        ),
        notes="Length 100 mm, width 60 mm, height 30 mm.",
    ),
}


ALIASES = {
    "blue_cup": "cup",
    "cup": "cup",
    "mug": "cup",
    "roll_of_tape": "roll_of_tape",
    "tape": "roll_of_tape",
    "tape_roll": "roll_of_tape",
    "phillips_screwdriver": "screwdriver",
    "philips_screwdriver": "screwdriver",
    "screwdriver": "screwdriver",
    "black_computer_mouse": "computer_mouse",
    "white_computer_mouse": "computer_mouse",
    "computer_mouse": "computer_mouse",
    "mouse": "computer_mouse",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def get_object_dimensions(object_name: str) -> ObjectDimensions | None:
    """
    Return the real-dimension prior for an object label, if known.

    The matcher is intentionally simple:
      1. exact alias match
      2. substring alias match (useful for stems like rgb_..._blue_cup)
    """
    norm = _normalize_name(object_name)

    canonical = ALIASES.get(norm)
    if canonical is not None:
        return OBJECT_DIMENSIONS[canonical]

    for alias, canonical_name in ALIASES.items():
        if alias in norm:
            return OBJECT_DIMENSIONS[canonical_name]

    return None
