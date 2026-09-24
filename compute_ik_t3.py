from pathlib import Path
import json, math, numpy as np

L1, L2, L3 = 0.070, 0.040, 0.026
R_BASE = 0.043
BASE_ANGLES_DEG = {"wide": [0.0, 137.0, 223.0]}
COUPLING_MEDIAL = 0.70
COUPLING_DISTAL = 0.50

def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def palm_frame(candidate):
    origin = np.array(candidate["patch_centroid_xyz_m"])
    x = normalize(candidate["tangent_1_xyz_unit"])
    z = normalize(candidate["approach_direction_xyz_unit"])
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    R = np.column_stack([x, y, z])
    return origin, R

def world_to_palm(p, origin, R):
    return R.T @ (np.asarray(p) - origin)

def finger_base(idx, mode):
    phi = math.radians(BASE_ANGLES_DEG[mode][idx])
    return np.array([R_BASE*math.cos(phi), R_BASE*math.sin(phi), 0.0])

def ik_2link(p):
    x, y = float(p[0]), float(p[1])
    d = math.hypot(x, y)
    if d > L1+L2+1e-4 or d < abs(L1-L2)-1e-4:
        return None
    cos_t2 = (d**2 - L1**2 - L2**2) / (2*L1*L2)
    cos_t2 = float(np.clip(cos_t2, -1.0, 1.0))
    t2 = -math.acos(cos_t2)
    alpha = math.atan2(y, x)
    beta = math.atan2(L2*math.sin(t2), L1+L2*math.cos(t2))
    t1 = alpha - beta
    return t1, t2

data = json.load(open((Path(__file__).resolve().parent / 'T3_lotion_heatmap_flex.json')))
c = data["best_candidate"]
mode = c["gripper_mode"]
origin, R = palm_frame(c)

euler = [
    round(math.degrees(math.atan2(R[2,1], R[2,2])), 2),
    round(math.degrees(math.atan2(-R[2,0], math.hypot(R[2,1], R[2,2]))), 2),
    round(math.degrees(math.atan2(R[1,0], R[0,0])), 2),
]

fingers = {}
names = ["F1 (thumb)", "F2", "F3"]
for i, (name, contact) in enumerate(zip(names, c["contacts"])):
    tip = np.array(contact["point_xyz_m"])
    tip_palm = world_to_palm(tip, origin, R)
    base_palm = finger_base(i, mode)
    vec = tip_palm - base_palm
    radial = math.hypot(vec[0], vec[1])
    axial = vec[2]
    res = ik_2link(np.array([radial, axial]))
    if res is None:
        fingers[name] = {"reachable": False}
    else:
        t1, t2 = res
        t1d = math.degrees(t1)
        t2d = math.degrees(t2)
        t3d = COUPLING_DISTAL / COUPLING_MEDIAL * t2d
        fingers[name] = {
            "reachable": True,
            "theta1_deg": round(t1d, 2),
            "theta2_deg": round(t2d, 2),
            "theta3_deg": round(t3d, 2),
        }

result = {
    "mode": mode,
    "palm_position_m": origin.tolist(),
    "palm_euler_xyz_deg": euler,
    "fingers": fingers,
}

print(f"\nGripper mode : {mode.upper()}")
print(f"Palm position: {[round(v*100,2) for v in origin.tolist()]} cm")
print(f"Palm Euler   : {euler} deg (XYZ)")
print()
for name, a in fingers.items():
    if not a["reachable"]:
        print(f"  {name}: UNREACHABLE")
    else:
        print(f"  {name}:")
        print(f"    theta1 proximal = {a['theta1_deg']:+7.2f} deg")
        print(f"    theta2 medial   = {a['theta2_deg']:+7.2f} deg  (coupled)")
        print(f"    theta3 distal   = {a['theta3_deg']:+7.2f} deg  (coupled)")
print()

out = (Path(__file__).resolve().parent / 'T3_lotion_heatmap_flex_ik.json')
json.dump(result, open(out, 'w'), indent=2)
print(f"Salvat: {out}")
