from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2


OUT_DIR = Path("outputs_symmetric_rotation180/rgb_1778075546_roll_of_tape")
PATCH_ID = 1

RGB_PATH = Path("data/rgb_1778075546.png")


def load_patch_npz(path):
    data = np.load(path, allow_pickle=True)

    orig = data["original_points_xyz"].astype(np.float32)
    sym = data["symmetric_points_xyz"].astype(np.float32)

    orig = orig[np.isfinite(orig).all(axis=1)]
    sym = sym[np.isfinite(sym).all(axis=1)]

    orig_centroid = data["centroid_original_xyz"].astype(np.float32)
    sym_centroid = data["centroid_symmetric_xyz"].astype(np.float32)

    return orig, sym, orig_centroid, sym_centroid


def remove_outliers_z(pts, z_percentile_low=1, z_percentile_high=99):
    if len(pts) <= 1:
        return pts

    z = pts[:, 2]
    lo, hi = np.percentile(z, [z_percentile_low, z_percentile_high])
    return pts[(z >= lo) & (z <= hi)]


patch_file = OUT_DIR / f"patch_{PATCH_ID:02d}.npz"

if not patch_file.exists():
    raise RuntimeError(f"Nu găsesc patch npz: {patch_file}")

orig, sym, orig_centroid, sym_centroid = load_patch_npz(patch_file)

orig = remove_outliers_z(orig)
sym = remove_outliers_z(sym)

print("patch file:", patch_file)
print("original points:", orig.shape)
print("symmetric points:", sym.shape)
print("centroid original:", orig_centroid)
print("centroid symmetric:", sym_centroid)
print("centroid distance:", np.linalg.norm(sym_centroid - orig_centroid))


# =========================================================
# 1. 3D plot
# =========================================================

fig = plt.figure(figsize=(11, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    orig[:, 0],
    orig[:, 1],
    orig[:, 2],
    s=12,
    c="limegreen",
    alpha=1.0,
    label="original patch",
)

ax.scatter(
    sym[:, 0],
    sym[:, 1],
    sym[:, 2],
    s=12,
    c="cyan",
    alpha=0.9,
    label="symmetric patch",
)

ax.scatter(
    orig_centroid[0],
    orig_centroid[1],
    orig_centroid[2],
    s=250,
    c="red",
    marker="o",
    edgecolors="black",
    linewidths=1.5,
    label="original centroid",
)

ax.scatter(
    sym_centroid[0],
    sym_centroid[1],
    sym_centroid[2],
    s=320,
    c="magenta",
    marker="X",
    edgecolors="black",
    linewidths=1.5,
    label="symmetric centroid",
)

all_pts = np.vstack([
    orig,
    sym,
    orig_centroid.reshape(1, 3),
    sym_centroid.reshape(1, 3),
])

center = all_pts.mean(axis=0)
max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2

if max_range < 1e-4:
    max_range = 0.01

padding = max_range * 0.15
max_range = max_range + padding

ax.set_xlim(center[0] - max_range, center[0] + max_range)
ax.set_ylim(center[1] - max_range, center[1] + max_range)
ax.set_zlim(center[2] - max_range, center[2] + max_range)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    f"{OUT_DIR.name}: original patch + symmetric patch",
    fontsize=15,
)

ax.legend(fontsize=10)

# View aproximativ din perspectiva camerei
ax.view_init(elev=90, azim=-90)

plt.tight_layout()
plt.show()


# =========================================================
# 2. 2D overlay debug
# =========================================================

debug_img_path = OUT_DIR / "debug_mask_patch_semantic_model.png"

if not debug_img_path.exists():
    raise RuntimeError(f"Nu găsesc debug overlay: {debug_img_path}")

rgb = cv2.imread(str(RGB_PATH))
if rgb is None:
    raise RuntimeError(f"Nu găsesc RGB: {RGB_PATH}")

rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

debug_img = cv2.imread(str(debug_img_path))
debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].imshow(rgb)
ax[0].set_title("RGB original")
ax[0].axis("off")

ax[1].imshow(debug_img)
ax[1].set_title("Mask + patch 2D overlay")
ax[1].axis("off")

plt.tight_layout()
plt.show()