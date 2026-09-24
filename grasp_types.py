"""
Data structures passed between pipeline stages.

Each stage produces one type and hands it to the next:
  select_patches    -> Patch
  compute_frames    -> LocalFrame
  generate_contacts -> Contact, GraspCandidate
  score_grasps      -> ScoredGrasp
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Patch:
    """
    A small sub-zone of the grasp patch.

    select_patches splits the grasp zone into smaller patches,
    each centred on a seed point. Contains the indices of the
    cloud points that belong to this patch.
    """
    patch_id: int                   # unique identifier
    seed_index: int                 # index of the seed point in the cloud
    seed_point: np.ndarray          # (3,) XYZ coordinates of the seed
    point_indices: np.ndarray       # (K,) indices into the cloud
    mean_heatmap: float = 0.0       # average ML affordance over the patch
                                    # points (0 if heatmap unavailable)
    visible_edge_score: float = 0.0   # optional generic prior from the
                                      # 2D patch location on the observed
                                      # object silhouette/boundary


@dataclass
class LocalFrame:
    """
    Local coordinate system built on a patch via PCA.

    PCA (Principal Component Analysis) finds the 3 main directions:
    - tangent_1, tangent_2: two directions lying in the surface plane
    - normal: direction perpendicular to the surface, pointing outward

    The gripper approaches from the direction opposite to the normal.
    """
    patch: Patch
    centroid: np.ndarray            # (3,) geometric centre of the patch points
    tangent_1: np.ndarray           # (3,) principal direction 1 (in-plane)
    tangent_2: np.ndarray           # (3,) principal direction 2 (in-plane)
    normal: np.ndarray              # (3,) surface normal (pointing outward)
    pca_eigenvalues: np.ndarray     # (3,) how spread the points are along
                                    #       each direction (descending)
    local_elongation: float = 0.0   # 1 - lambda2/lambda1 in the patch
                                    # plane: high = locally elongated
                                    # (handle, neck, ridge), low = locally
                                    # isotropic (flat body, sphere, cube)


@dataclass
class Contact:
    """
    One finger contact on the object surface.

    The Robotiq 3F gripper has 3 fingers, so each grasp has 3 contacts.
    """
    name: str                       # "contact_1", "contact_2", "contact_3"
    angle_deg: float                # placement angle on the tangent plane (degrees)
    point: np.ndarray               # (3,) 3D position of the contact
    normal: np.ndarray              # (3,) surface normal (pointing outward)
    visible: bool                   # True = real point from the cloud
    distance_to_visible_m: float    # distance to the nearest real cloud point
    pixel: tuple[int, int] | None = None   # (u, v) in the RGB image, if available
    heatmap_value: float = 0.0      # 0..1 affordance score from the ML heatmap


@dataclass
class GraspCandidate:
    """
    A candidate grasp: 3 contacts placed on a local frame.

    approach_direction = the direction the gripper comes from (opposite of normal).
    gripper_mode       = which Robotiq 3F operation mode produced this
                         candidate (basic / wide / pinch / scissor).
    finger_spread_m    = longitudinal B-C spread used during placement.
    """
    frame: LocalFrame
    approach_direction: np.ndarray  # (3,) approach direction
    contacts: list[Contact] = field(default_factory=list)
    gripper_mode: str = "basic"
    finger_spread_m: float = 0.050
    rotation_deg: float = 0.0

    @property
    def visible_count(self) -> int:
        """How many contacts land on real (not estimated) points."""
        return sum(1 for c in self.contacts if c.visible)


@dataclass
class ScoredGrasp:
    """
    A candidate with all quality metrics computed.

    valid = True only if ALL hard constraints are satisfied.
    final_score = weighted average of soft metrics (0..1, higher = better).
    """
    candidate: GraspCandidate
    valid: bool
    final_score: float

    # Soft metrics (each between 0 and 1)
    patch_planarity: float          # how flat the surface is
    triangle_score: float           # how equilateral the triangle is
    triangle_area_m2: float         # triangle area (m^2)
    alignment_score: float          # approach perpendicular to object's
                                    # principal axis (1 = perpendicular for
                                    # elongated objects; 1 always for
                                    # near-isotropic objects)
    feature_score: float            # local elongation: bonus for handles,
                                    # necks, ridges, rims (any protrusion)
    distance_to_com_m: float        # actual distance to centroid (m, diag.)
    visible_fraction: float         # fraction of visible contacts
    required_opening_m: float       # how wide the gripper must open
    heatmap_score: float            # average ML affordance over the 3 contacts
    thickness_score: float          # local cross-section is in gripper sweet-spot
    local_diameter_m: float         # measured local diameter (m)

    # Hard constraints (each must be True)
    opening_feasible: bool          # fits inside the gripper opening?
    contacts_reachable: bool        # fingers can reach the contacts?
    palm_collision_free: bool       # palm does not hit the object?
    fingers_collision_free: bool    # fingers do not pass through the object?
    approach_clear: bool            # approach path is obstacle-free?
    support_clear: bool             # grasp stays clear of the support plane?
    top_entry_like: bool            # contacts sit in the upper rim/top band,
                                    # so the grasp behaves like an entry from
                                    # above into an opening/cavity

    # Wrench-space quality (from grasp_wrench.py)
    opposition_score: float = 0.0   # normal-opposition score [0,1], friction-free
    wrench_score: float = 0.0       # friction-cone wrench score [0,1], 0 if disabled

    # Derived classification (default last so it has a default value)
    grip_type: str = "fingertip"    # "encompassing" or "fingertip" — see
                                    # Robotiq 3F manual section 1, fig 1.5
    object_flexible: bool = False   # True if the object is deformable/squeezable
    flex_score: float = 0.0         # compression-adjusted thickness score (0 for rigid)
