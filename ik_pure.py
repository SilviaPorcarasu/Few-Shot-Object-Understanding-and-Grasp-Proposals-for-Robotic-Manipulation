from pathlib import Path
import json, math

L1, L2 = 0.070, 0.040
R_BASE = 0.043
WIDE_ANGLES = [0.0, 137.0, 223.0]
COUPLING_MEDIAL = 0.70
COUPLING_DISTAL = 0.50

def dot(a, b): return sum(x*y for x,y in zip(a,b))
def norm(v): return math.sqrt(dot(v,v))
def scale(v, s): return [x*s for x in v]
def add(a, b): return [x+y for x,y in zip(a,b)]
def sub(a, b): return [x-y for x,y in zip(a,b)]
def normalize(v): n=norm(v); return [x/n for x in v]
def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def matvecT(R, v):
    return [dot(R[0], v), dot(R[1], v), dot(R[2], v)]

data = json.load(open((Path(__file__).resolve().parent / 'T3_lotion_heatmap_flex.json')))
c = data["best_candidate"]

origin = c["patch_centroid_xyz_m"]
x = normalize(c["tangent_1_xyz_unit"])
z = normalize(c["approach_direction_xyz_unit"])
y = normalize(cross(z, x))
x = normalize(cross(y, z))
R = [x, y, z]

euler = [
    round(math.degrees(math.atan2(R[2][1], R[2][2])), 2),
    round(math.degrees(math.atan2(-R[2][0], math.hypot(R[2][1], R[2][2]))), 2),
    round(math.degrees(math.atan2(R[1][0], R[0][0])), 2),
]

names = ["F1 (thumb)", "F2", "F3"]
fingers = {}
for i, (name, contact) in enumerate(zip(names, c["contacts"])):
    tip = contact["point_xyz_m"]
    diff = sub(tip, origin)
    tip_palm = [dot(R[j], diff) for j in range(3)]
    phi = math.radians(WIDE_ANGLES[i])
    base = [R_BASE*math.cos(phi), R_BASE*math.sin(phi), 0.0]
    vec = sub(tip_palm, base)
    radial = math.hypot(vec[0], vec[1])
    axial = vec[2]
    d = math.hypot(radial, axial)
    if d > L1+L2+1e-4 or d < abs(L1-L2)-1e-4:
        fingers[name] = {"reachable": False}
        continue
    cos_t2 = (d**2 - L1**2 - L2**2) / (2*L1*L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2 = -math.acos(cos_t2)
    alpha = math.atan2(axial, radial)
    beta = math.atan2(L2*math.sin(t2), L1+L2*math.cos(t2))
    t1 = alpha - beta
    t1d = round(math.degrees(t1), 2)
    t2d = round(math.degrees(t2), 2)
    t3d = round(COUPLING_DISTAL/COUPLING_MEDIAL*t2d, 2)
    fingers[name] = {"reachable": True, "theta1_deg": t1d, "theta2_deg": t2d, "theta3_deg": t3d}

result = {"mode": c["gripper_mode"], "palm_position_m": origin, "palm_euler_xyz_deg": euler, "fingers": fingers}

print(f"\nGripper mode : {c['gripper_mode'].upper()}")
print(f"Palm position: {[round(v*100,2) for v in origin]} cm")
print(f"Palm Euler   : {euler} deg")
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
