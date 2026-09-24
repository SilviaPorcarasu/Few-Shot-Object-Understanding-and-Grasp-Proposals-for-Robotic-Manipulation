from __future__ import annotations

from typing import Dict, Iterable, Literal, TypedDict

SymmetryLabel = Literal["symmetric", "asymmetric"]


class ObjectInfo(TypedDict):
    id: int
    name: str              # clean internal/canonical name returned by the detector
    symmetry: SymmetryLabel
    flexible: bool         # True for deformable / flexible objects; False for rigid objects
    sam_prompt: str        # short text prompt for SAM3
    detector_labels: list[str]  # simple zero-shot labels for OWL-ViT


# Clean object catalog.
#
# Design rules:
# - name: stable canonical name used by the pipeline/results; no dataset ids.
# - detector_labels: general, visual object names for zero-shot detection; no opaque ids.
# - sam_prompt: short prompt used by SAM3. Prefer simple nouns/noun phrases.
# - symmetry: preserved as the pipeline routing signal.
# - flexible: object-level deformability flag propagated to grasping metadata.
#
# Important: detector labels are intentionally object-level labels, not dataset labels.
# OWL-ViT/SAM generally performs better with "computer mouse", "bottle", "toy elephant"
# than with labels such as "sum37_secret_repair" or "072-a_toy_airplane".
OBJECT_CATALOG: Dict[str, ObjectInfo] = {
    # YCB-style objects
    "003_cracker_box": {
        "id": 0,
        "name": "cracker box",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "cracker box",
        "detector_labels": ["cracker box", "food box", "rectangular box"],
    },
    "004_sugar_box": {
        "id": 1,
        "name": "sugar box",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "sugar box",
        "detector_labels": ["sugar box", "food box", "rectangular box"],
    },
    "005_tomato_soup_can": {
        "id": 2,
        "name": "tomato soup can",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "soup can",
        "detector_labels": ["tomato soup can", "soup can", "cylindrical can"],
    },
    "006_mustard_bottle": {
        "id": 3,
        "name": "mustard bottle",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "mustard bottle",
        "detector_labels": ["mustard bottle", "yellow bottle", "squeeze bottle"],
    },
    "010_potted_meat_can": {
        "id": 4,
        "name": "potted meat can",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "potted meat can",
        "detector_labels": ["potted meat can", "rectangular can", "metal food can"],
    },
    "011_banana": {
        "id": 5,
        "name": "banana",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "banana",
        "detector_labels": ["banana"],
    },
    "024_bowl": {
        "id": 6,
        "name": "bowl",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "bowl",
        "detector_labels": ["bowl"],
    },
    "025_mug": {
        "id": 7,
        "name": "mug",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "mug",
        "detector_labels": ["mug", "coffee mug", "cup with handle"],
    },
    "035_power_drill": {
        "id": 8,
        "name": "power drill",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "power drill",
        "detector_labels": ["power drill", "drill", "electric drill"],
    },
    "037_scissors": {
        "id": 9,
        "name": "scissors",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "scissors",
        "detector_labels": ["scissors"],
    },
    "001_chips_can": {
        "id": 10,
        "name": "chips can",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "chips can",
        "detector_labels": ["chips can", "snack can", "cylindrical can"],
    },
    "012_strawberry": {
        "id": 11,
        "name": "strawberry",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "strawberry",
        "detector_labels": ["strawberry"],
    },
    "013_apple": {
        "id": 12,
        "name": "apple",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "apple",
        "detector_labels": ["apple"],
    },
    "014_lemon": {
        "id": 13,
        "name": "lemon",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "lemon",
        "detector_labels": ["lemon"],
    },
    "015_peach": {
        "id": 14,
        "name": "peach",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "peach",
        "detector_labels": ["peach"],
    },
    "016_pear": {
        "id": 15,
        "name": "pear",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "pear",
        "detector_labels": ["pear"],
    },
    "017_orange": {
        "id": 16,
        "name": "orange",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "orange",
        "detector_labels": ["orange fruit", "orange"],
    },
    "018_plum": {
        "id": 17,
        "name": "plum",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "plum",
        "detector_labels": ["plum"],
    },
    "032_knife": {
        "id": 18,
        "name": "knife",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "knife",
        "detector_labels": ["knife", "kitchen knife"],
    },
    "043_phillips_screwdriver": {
        "id": 19,
        "name": "phillips screwdriver",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "screwdriver",
        "detector_labels": ["screwdriver", "phillips screwdriver", "blue screwdriver"],
    },
    "044_flat_screwdriver": {
        "id": 20,
        "name": "flat screwdriver",
        "symmetry": "asymmetric", "flexible": False,
        "sam_prompt": "screwdriver",
        "detector_labels": ["screwdriver", "flat screwdriver", "red screwdriver"],
    },
    "057_racquetball": {
        "id": 21,
        "name": "racquetball",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "ball",
        "detector_labels": ["ball", "blue ball", "racquetball"],
    },
    "065-b_cups": {
        "id": 22,
        "name": "blue cup",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "cup",
        "detector_labels": ["blue cup", "plastic cup", "cup"],
    },
    "065-d_cups": {
        "id": 23,
        "name": "yellow cup",
        "symmetry": "symmetric", "flexible": False,
        "sam_prompt": "cup",
        "detector_labels": ["yellow cup", "plastic cup", "cup"],
    },

    # Toy airplanes / aircraft parts
    "072-a_toy_airplane": {"id": 24, "name": "toy airplane", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane", "detector_labels": ["toy airplane", "small airplane toy"]},
    "072-c_toy_airplane": {"id": 25, "name": "toy airplane", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane", "detector_labels": ["toy airplane", "small airplane toy"]},
    "072-d_toy_airplane": {"id": 26, "name": "toy airplane part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane part", "detector_labels": ["toy airplane part", "airplane toy part"]},
    "072-f_toy_airplane": {"id": 27, "name": "toy airplane part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane part", "detector_labels": ["toy airplane part", "airplane toy part"]},
    "072-h_toy_airplane": {"id": 28, "name": "toy airplane propeller", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane propeller", "detector_labels": ["toy airplane propeller", "toy propeller"]},
    "072-i_toy_airplane": {"id": 29, "name": "toy airplane part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane part", "detector_labels": ["toy airplane part", "airplane toy part"]},
    "072-j_toy_airplane": {"id": 30, "name": "toy airplane part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane part", "detector_labels": ["toy airplane part", "airplane toy part"]},
    "072-k_toy_airplane": {"id": 31, "name": "toy airplane part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toy airplane part", "detector_labels": ["toy airplane part", "airplane toy part"]},

    "038_padlock": {"id": 32, "name": "padlock", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "padlock", "detector_labels": ["padlock", "lock"]},
    "dragon": {"id": 33, "name": "dragon toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "dragon toy", "detector_labels": ["dragon toy", "toy dragon", "dragon figurine"]},

    # Toiletry / grocery products as generic visual categories
    "sum37_secret_repair": {"id": 34, "name": "cosmetic tube", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "cosmetic tube", "detector_labels": ["cosmetic tube", "skin care tube", "small tube"]},
    "jvr_cleansing_foam": {"id": 35, "name": "cleansing foam tube", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "cleansing foam tube", "detector_labels": ["cleansing foam tube", "face wash tube", "cosmetic tube"]},
    "dabao_wash_soup": {"id": 36, "name": "soap bottle", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "soap bottle", "detector_labels": ["soap bottle", "wash bottle", "rectangular bottle"]},
    "nzskincare_mouth_rinse": {"id": 37, "name": "mouth rinse bottle", "symmetry": "symmetric", "flexible": True, "sam_prompt": "mouth rinse bottle", "detector_labels": ["mouth rinse bottle", "clear bottle", "small bottle"]},
    "dabao_sod": {"id": 38, "name": "lotion bottle", "symmetry": "symmetric", "flexible": True, "sam_prompt": "lotion bottle", "detector_labels": ["lotion bottle", "white bottle", "bottle with red cap"]},
    "soap_box": {"id": 39, "name": "soap box", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "soap box", "detector_labels": ["soap box", "small rectangular box", "box"]},
    "kispa_cleanser": {"id": 40, "name": "cleanser bottle", "symmetry": "symmetric", "flexible": True, "sam_prompt": "cleanser bottle", "detector_labels": ["cleanser bottle", "pump bottle", "white bottle"]},
    "darlie_toothpaste": {"id": 41, "name": "toothpaste tube", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "toothpaste tube", "detector_labels": ["toothpaste tube", "tube"]},
    "nivea_men_oil_control": {"id": 42, "name": "face wash tube", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "face wash tube", "detector_labels": ["face wash tube", "cosmetic tube", "black tube"]},
    "baoke_marker": {"id": 43, "name": "marker pen", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "marker pen", "detector_labels": ["marker pen", "pen", "brown marker"]},
    "hosjam": {"id": 44, "name": "pump bottle", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "pump bottle", "detector_labels": ["pump bottle", "soap dispenser", "green bottle"]},
    "pitcher_cap": {"id": 45, "name": "pitcher cap", "symmetry": "symmetric", "flexible": False, "sam_prompt": "cap", "detector_labels": ["cap", "plastic cap", "pitcher cap"]},
    "dish": {"id": 46, "name": "dish", "symmetry": "symmetric", "flexible": False, "sam_prompt": "dish", "detector_labels": ["dish", "plate"]},
    "white_mouse": {"id": 47, "name": "white computer mouse", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "computer mouse", "detector_labels": ["white computer mouse", "computer mouse", "mouse"]},

    # Animal toys
    "camel": {"id": 48, "name": "camel toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "camel toy", "detector_labels": ["camel toy", "toy camel", "camel figurine"]},
    "deer": {"id": 49, "name": "deer toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "deer toy", "detector_labels": ["deer toy", "toy deer", "deer figurine"]},
    "zebra": {"id": 50, "name": "zebra toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "zebra toy", "detector_labels": ["zebra toy", "toy zebra", "zebra figurine"]},
    "large_elephant": {"id": 51, "name": "large elephant toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "elephant toy", "detector_labels": ["large elephant toy", "toy elephant", "elephant figurine"]},
    "rhinocero": {"id": 52, "name": "rhinoceros toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "rhinoceros toy", "detector_labels": ["rhinoceros toy", "rhino toy", "toy rhinoceros"]},
    "small_elephant": {"id": 53, "name": "small elephant toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "elephant toy", "detector_labels": ["small elephant toy", "toy elephant", "elephant figurine"]},
    "monkey": {"id": 54, "name": "monkey toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "monkey toy", "detector_labels": ["monkey toy", "toy monkey", "monkey figurine"]},
    "giraffe": {"id": 55, "name": "giraffe toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "giraffe toy", "detector_labels": ["giraffe toy", "toy giraffe", "giraffe figurine"]},
    "gorilla": {"id": 56, "name": "gorilla toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "gorilla toy", "detector_labels": ["gorilla toy", "toy gorilla", "gorilla figurine"]},

    # More household / product objects
    "weiquan": {"id": 57, "name": "drink can", "symmetry": "symmetric", "flexible": False, "sam_prompt": "drink can", "detector_labels": ["drink can", "yellow can", "cylindrical can"]},
    "darlie_box": {"id": 58, "name": "toothpaste box", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "toothpaste box", "detector_labels": ["toothpaste box", "blue box", "rectangular box"]},
    "soap": {"id": 59, "name": "bar of soap", "symmetry": "symmetric", "flexible": False, "sam_prompt": "bar of soap", "detector_labels": ["bar of soap", "soap"]},
    "black_mouse": {"id": 60, "name": "black computer mouse", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "computer mouse", "detector_labels": ["black computer mouse", "computer mouse", "mouse"]},
    "dabao_facewash": {"id": 61, "name": "face wash bottle", "symmetry": "symmetric", "flexible": False, "sam_prompt": "face wash bottle", "detector_labels": ["face wash bottle", "white bottle", "bottle"]},
    "pantene": {"id": 62, "name": "shampoo tube", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "shampoo tube", "detector_labels": ["shampoo tube", "cosmetic tube", "tube"]},
    "head_shoulders_supreme": {"id": 63, "name": "shampoo tube", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "shampoo tube", "detector_labels": ["shampoo tube", "cosmetic tube", "tube"]},
    "thera_med": {"id": 64, "name": "toothpaste bottle", "symmetry": "symmetric", "flexible": False, "sam_prompt": "toothpaste bottle", "detector_labels": ["toothpaste bottle", "toothpaste tube", "tube"]},
    "dove": {"id": 65, "name": "shampoo bottle", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "shampoo bottle", "detector_labels": ["shampoo bottle", "white bottle", "dove bottle"]},
    "head_shoulders_care": {"id": 66, "name": "shampoo bottle", "symmetry": "asymmetric", "flexible": True, "sam_prompt": "shampoo bottle", "detector_labels": ["shampoo bottle", "white bottle", "head and shoulders bottle"]},
    "lion": {"id": 67, "name": "lion toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "lion toy", "detector_labels": ["lion toy", "toy lion", "lion figurine"]},
    "coconut_juice_box": {"id": 68, "name": "juice box", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "juice box", "detector_labels": ["juice box", "drink carton", "rectangular carton"]},
    "hippo": {"id": 69, "name": "hippo toy", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "hippo toy", "detector_labels": ["hippo toy", "toy hippo", "hippopotamus figurine"]},
    "tape": {"id": 70, "name": "roll of tape", "symmetry": "symmetric", "flexible": False, "sam_prompt": "roll of tape", "detector_labels": ["roll of tape", "tape roll", "tape"]},
    "rubiks_cube": {"id": 71, "name": "rubiks cube", "symmetry": "symmetric", "flexible": False, "sam_prompt": "rubiks cube", "detector_labels": ["rubiks cube", "cube puzzle", "colored cube"]},
    "peeler_cover": {"id": 72, "name": "peeler cover", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "peeler cover", "detector_labels": ["peeler cover", "plastic cover", "kitchen tool cover"]},
    "peeler": {"id": 73, "name": "peeler", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "peeler", "detector_labels": ["peeler", "vegetable peeler", "kitchen peeler"]},
    "ice_cube_mould": {"id": 74, "name": "ice cube tray", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "ice cube tray", "detector_labels": ["ice cube tray", "ice cube mould", "plastic tray"]},

    # Mechanical / 3D printed objects
    "bar_clamp": {"id": 75, "name": "bar clamp", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "bar clamp", "detector_labels": ["bar clamp", "clamp", "tool clamp"]},
    "climbing_hold": {"id": 76, "name": "climbing hold", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "climbing hold", "detector_labels": ["climbing hold", "rock climbing hold", "plastic hold"]},
    "endstop_holder": {"id": 77, "name": "endstop holder", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "mechanical bracket", "detector_labels": ["mechanical bracket", "plastic bracket", "endstop holder"]},
    "gearbox": {"id": 78, "name": "gearbox", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "gearbox", "detector_labels": ["gearbox", "gear housing", "mechanical housing"]},
    "mount1": {"id": 79, "name": "mount bracket", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "mount bracket", "detector_labels": ["mount bracket", "mechanical mount", "plastic bracket"]},
    "mount2": {"id": 80, "name": "mount bracket", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "mount bracket", "detector_labels": ["mount bracket", "mechanical mount", "plastic bracket"]},
    "nozzle": {"id": 81, "name": "nozzle", "symmetry": "symmetric", "flexible": False, "sam_prompt": "nozzle", "detector_labels": ["nozzle", "plastic nozzle", "cone nozzle"]},
    "part1": {"id": 82, "name": "mechanical part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "mechanical part", "detector_labels": ["mechanical part", "plastic part", "bracket"]},
    "part3": {"id": 83, "name": "mechanical part", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "mechanical part", "detector_labels": ["mechanical part", "plastic part", "bracket"]},
    "pawn": {"id": 84, "name": "chess pawn", "symmetry": "symmetric", "flexible": False, "sam_prompt": "chess pawn", "detector_labels": ["chess pawn", "pawn"]},
    "pipe_connector": {"id": 85, "name": "pipe connector", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "pipe connector", "detector_labels": ["pipe connector", "pipe fitting", "connector"]},
    "wrench": {"id": 88, "name": "wrench", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "wrench", "detector_labels": ["wrench", "spanner", "open-end wrench", "adjustable wrench"]},
    "turbine_housing": {"id": 86, "name": "turbine housing", "symmetry": "asymmetric", "flexible": False, "sam_prompt": "turbine housing", "detector_labels": ["turbine housing", "mechanical housing", "housing"]},
    "vase": {"id": 87, "name": "vase", "symmetry": "symmetric", "flexible": False, "sam_prompt": "vase", "detector_labels": ["vase"]},
    "hand_cream_tube": {"id": 89, "name": "hand cream tube", "symmetry": "symmetric", "flexible": True, "sam_prompt": "hand cream tube", "detector_labels": ["hand cream tube", "white hand cream tube", "cream tube", "white tube with black label", "cosmetic cream tube"]},
}


def slugify_object_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def get_object_key(name_or_key: str) -> str:
    """Resolve a catalog key, canonical name, SAM prompt, or detector label to a key."""
    if name_or_key in OBJECT_CATALOG:
        return name_or_key

    normalized = _normalize_label(name_or_key)
    slug = slugify_object_name(name_or_key)
    if slug in OBJECT_CATALOG:
        return slug

    for key, info in OBJECT_CATALOG.items():
        aliases = [info["name"], info["sam_prompt"], *info["detector_labels"]]
        if any(_normalize_label(alias) == normalized for alias in aliases):
            return key

    raise KeyError(f"Unknown object label: {name_or_key!r}")


def catalog_name_for_object(key_or_name: str) -> str:
    return OBJECT_CATALOG[get_object_key(key_or_name)]["name"]


def sam_prompt_for_object(key_or_name: str) -> str:
    return OBJECT_CATALOG[get_object_key(key_or_name)]["sam_prompt"]


def symmetry_for_name(name: str) -> SymmetryLabel:
    return OBJECT_CATALOG[get_object_key(name)]["symmetry"]


def flexible_for_name(name: str) -> bool:
    """Return whether the catalog marks this object as flexible/deformable."""
    return bool(OBJECT_CATALOG[get_object_key(name)].get("flexible", False))


def detector_queries() -> list[tuple[str, str]]:
    """Return globally unique (object_key, detector_label) pairs for zero-shot detection."""
    pairs: list[tuple[str, str]] = []
    seen_labels: set[str] = set()

    for key, info in OBJECT_CATALOG.items():
        # Put the canonical name and SAM prompt first; then aliases.
        labels = [info["name"], info["sam_prompt"], *info["detector_labels"]]
        for label in labels:
            label = _normalize_label(label)
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            pairs.append((key, label))

    return pairs


def candidate_detector_labels() -> list[str]:
    """Human-readable labels passed to the zero-shot detector."""
    return [label for _, label in detector_queries()]


def split_by_symmetry(names: Iterable[str]) -> dict[SymmetryLabel, list[str]]:
    """Group object names by symmetry while avoiding duplicates."""
    out: dict[SymmetryLabel, list[str]] = {"symmetric": [], "asymmetric": []}
    seen: set[str] = set()

    for name in names:
        key = get_object_key(name)
        canonical = OBJECT_CATALOG[key]["name"]
        if canonical in seen:
            continue
        seen.add(canonical)
        out[OBJECT_CATALOG[key]["symmetry"]].append(canonical)

    return out
