"""
Robotiq 3-Finger Gripper — Inverse Kinematics
==============================================
Given the contact points (F1, F2, F3) from a grasp JSON, computes:
  - Palm pose  (position + orientation matrix)
  - Per-finger proximal joint angle θ1 (main actuated joint)
  - Coupled medial/distal angles (θ2, θ3) via the underactuation law

Robotiq 3F (S model) geometry  [from official CAD / datasheet]:
  L1  = 70  mm   proximal phalanx
  L2  = 40  mm   medial  phalanx
  L3  = 26  mm   distal  phalanx  (fingertip to medial-distal joint)

  Finger base offset from palm centre: r_base = 43 mm
  In BASIC mode:
    F1 base angle  =   0°  (thumb, opposing side)
    F2 base angle  = 120°
    F3 base angle  = 240°
  In WIDE mode:   base angles shift by ±17°
  In PINCH mode:  base angles shift inward ~10°

Underactuation coupling (Robotiq simplified law):
    θ2 = 0.7 * θ1         (medial follows proximal)
    θ3 = 0.5 * θ1         (distal follows proximal)
"""

import json
import math
import sys
import numpy as np
from pathlib import Path


# ── Robotiq 3F geometry constants (metres) ──────────────────────────────────
L1 = 0.070   # proximal phalanx
L2 = 0.040   # medial phalanx
L3 = 0.026   # distal phalanx

# Effective reach per finger: fingertip is at L1+L2+L3 from base when fully open
L_EFF = L1 + L2 + L3   # 136 mm total

# Finger base radius from palm centre (BASIC mode)
R_BASE = 0.043

# Base angles per mode (degrees, relative to palm X-axis)
BASE_ANGLES_DEG = {
    "basic":   [0.0,  120.0, 240.0],
    "wide":    [0.0,  137.0, 223.0],
    "pinch":   [10.0, 110.0, 250.0],
    "scissor": [0.0,  120.0, 240.0],
}

# Underactuation coupling ratios
COUPLING_MEDIAL  = 0.70   # θ2 = k2 * θ1
COUPLING_DISTAL  = 0.50   # θ3 = k3 * θ1


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def palm_frame_from_grasp(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (origin, R) where R is a 3x3 rotation matrix:
      R[:, 0]  = X axis  = tangent_long  (finger spread direction)
      R[:, 1]  = Y axis  = tangent_perp
      R[:, 2]  = Z axis  = approach direction (palm Z points INTO the object)
    """
    origin = np.array(candidate["patch_centroid_xyz_m"])
    x_axis = normalize(candidate["tangent_1_xyz_unit"])
    z_axis = normalize(candidate["approach_direction_xyz_unit"])
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))   # reorthogonalise
    R = np.column_stack([x_axis, y_axis, z_axis])
    return origin, R


def world_to_palm(point_world, origin, R):
    """Transform a world-frame point into the palm frame."""
    return R.T @ (np.asarray(point_world) - origin)


def finger_base_position_palm(finger_idx: int, mode: str) -> np.ndarray:
    """
    Position of finger base joint in palm frame.
    Bases lie in the XY plane of the palm (at Z=0 of the palm frame).
    """
    angles = BASE_ANGLES_DEG.get(mode.lower(), BASE_ANGLES_DEG["basic"])
    phi = math.radians(angles[finger_idx])
    return np.array([R_BASE * math.cos(phi),
                     R_BASE * math.sin(phi),
                     0.0])


def ik_2link_planar(p_local: np.ndarray) -> tuple[float, float] | None:
    """
    2-link planar IK for a single finger, modelled as L1+L2 (proximal+medial)
    reaching toward a target point expressed in the finger's local plane.

    p_local: 2D vector [along, across] in the finger's flexion plane.
    Returns (θ1, θ2) in radians, or None if unreachable.
    """
    x, y = float(p_local[0]), float(p_local[1])
    d = math.hypot(x, y)

    # Reachability check
    if d > L1 + L2 + 1e-4:
        return None   # too far
    if d < abs(L1 - L2) - 1e-4:
        return None   # too close

    # Standard 2-link IK (elbow-down solution)
    cos_theta2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    theta2 = math.acos(cos_theta2)                      # elbow-down
    theta2 = -theta2                                     # closing motion

    alpha = math.atan2(y, x)
    beta  = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta

    return theta1, theta2


def finger_joint_angles(
    fingertip_world: np.ndarray,
    finger_idx: int,
    mode: str,
    palm_origin: np.ndarray,
    palm_R: np.ndarray,
) -> dict:
    """
    Full IK for one Robotiq 3F finger.
    Returns a dict with θ1 (proximal), θ2 (medial), θ3 (distal) in degrees.
    """
    # --- 1. fingertip in palm frame ---
    tip_palm = world_to_palm(fingertip_world, palm_origin, palm_R)

    # --- 2. finger base in palm frame ---
    base_palm = finger_base_position_palm(finger_idx, mode)

    # --- 3. vector from base to fingertip in palm frame ---
    vec = tip_palm - base_palm          # 3D, palm coordinates

    # --- 4. project into the finger's flexion plane ---
    # The finger flexes along its local Z (palm Z) and Y axes.
    # Radial component (in XY palm plane, away from palm Z):
    radial = math.hypot(vec[0], vec[1])
    axial  = vec[2]                     # along palm approach axis

    p_local = np.array([radial, axial])

    # --- 5. 2-link IK ---
    result = ik_2link_planar(p_local)
    if result is None:
        return {"reachable": False,
                "tip_distance_m": float(np.linalg.norm(vec))}

    theta1_rad, theta2_rad = result

    # --- 6. coupled angles (underactuation) ---
    theta1_deg = math.degrees(theta1_rad)
    theta2_deg = math.degrees(theta2_rad)
    theta3_deg = COUPLING_DISTAL / COUPLING_MEDIAL * theta2_deg

    return {
        "reachable":    True,
        "theta1_deg":   round(theta1_deg, 2),   # proximal
        "theta2_deg":   round(theta2_deg, 2),   # medial  (coupled)
        "theta3_deg":   round(theta3_deg, 2),   # distal  (coupled)
        "tip_palm_m":   tip_palm.tolist(),
        "base_palm_m":  base_palm.tolist(),
    }


def solve_ik(scores_path: str) -> dict:
    data = json.loads(Path(scores_path).read_text())
    candidate = data["best_candidate"]
    mode = candidate["gripper_mode"]

    palm_origin, palm_R = palm_frame_from_grasp(candidate)

    contacts = candidate["contacts"]
    finger_names = ["F1 (thumb)", "F2", "F3"]

    results = {
        "mode": mode,
        "palm_position_m":    palm_origin.tolist(),
        "palm_rotation_matrix": palm_R.tolist(),
        "palm_euler_xyz_deg": [
            round(math.degrees(math.atan2( palm_R[2,1], palm_R[2,2])), 2),
            round(math.degrees(math.atan2(-palm_R[2,0], math.hypot(palm_R[2,1], palm_R[2,2]))), 2),
            round(math.degrees(math.atan2( palm_R[1,0], palm_R[0,0])), 2),
        ],
        "fingers": {}
    }

    for i, (name, contact) in enumerate(zip(finger_names, contacts)):
        tip = np.array(contact["point_xyz_m"])
        angles = finger_joint_angles(tip, i, mode, palm_origin, palm_R)
        results["fingers"][name] = angles

    return results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path(__file__).resolve().parent.parent / "T2_lotion_flex.json")

    result = solve_ik(path)

    print(f"\nGripper mode : {result['mode'].upper()}")
    print(f"Palm position: {[round(v*100,2) for v in result['palm_position_m']]} cm")
    print(f"Palm Euler   : {result['palm_euler_xyz_deg']} deg (XYZ)")
    print()
    for fname, angles in result["fingers"].items():
        if not angles["reachable"]:
            print(f"  {fname}: UNREACHABLE (dist={angles['tip_distance_m']*100:.1f}cm)")
        else:
            print(f"  {fname}:")
            print(f"    θ1 proximal = {angles['theta1_deg']:+7.2f}°")
            print(f"    θ2 medial   = {angles['theta2_deg']:+7.2f}°  (coupled)")
            print(f"    θ3 distal   = {angles['theta3_deg']:+7.2f}°  (coupled)")
    print()


if __name__ == "__main__":
    main()
