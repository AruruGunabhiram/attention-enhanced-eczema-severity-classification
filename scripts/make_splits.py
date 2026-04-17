from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
DATA_ROOT = Path("data/raw/dermnet")
HAM_ROOT  = Path("data/raw/ham10000")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def gather(folder: Path, label: str):
    rows = []
    if not folder.exists():
        print(f"  [MISSING] {folder}")
        return rows
    for p in folder.rglob("*"):
        if p.suffix.lower() in EXTS:
            rows.append({"filepath": str(p), "label": label})
    print(f"  {folder.name}: {len(rows)} images → '{label}'")
    return rows

def save_split(df: pd.DataFrame, prefix: str, holdout: float = 0.2):
    train_df, temp_df = train_test_split(
        df, test_size=holdout, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )
    train_df.to_csv(SPLIT_DIR / f"{prefix}_train.csv", index=False)
    val_df.to_csv(SPLIT_DIR   / f"{prefix}_val.csv",   index=False)
    test_df.to_csv(SPLIT_DIR  / f"{prefix}_test.csv",  index=False)
    print(f"  Saved {prefix}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  Class counts:\n{df['label'].value_counts().to_string()}\n")

# ===========================================================================
# STAGE 1: skin vs not_skin
# All dermnet images = skin
# HAM10000 images    = skin
# NOT-SKIN: use non-medical image folders if available, else skip with warning
# ===========================================================================
print("=" * 60)
print("STAGE 1: skin vs not_skin")
print("=" * 60)

skin_rows = []
for split in ("train", "test"):
    folder = DATA_ROOT / split
    if folder.exists():
        for p in folder.rglob("*"):
            if p.suffix.lower() in EXTS:
                skin_rows.append({"filepath": str(p), "label": "skin"})

# Also include ham10000 as skin if it has images
if HAM_ROOT.exists():
    for p in HAM_ROOT.rglob("*"):
        if p.suffix.lower() in EXTS:
            skin_rows.append({"filepath": str(p), "label": "skin"})
    print(f"  ham10000: included as skin images")

print(f"  Total skin images: {len(skin_rows)}")

# not_skin: generate from CIFAR-10 classes 0-7 (non-skin objects) if folder is absent/empty
NOT_SKIN_DIR = Path("data/raw/not_skin")
NOT_SKIN_COUNT = 3000
# CIFAR-10 classes 0-7: airplane, automobile, bird, cat, deer, dog, frog, horse
CIFAR_NON_SKIN_CLASSES = list(range(8))

existing_not_skin = list(NOT_SKIN_DIR.rglob("*")) if NOT_SKIN_DIR.exists() else []
existing_images = [p for p in existing_not_skin if p.suffix.lower() in EXTS]

if not existing_images:
    print(f"  Generating {NOT_SKIN_COUNT} not_skin images from CIFAR-10 → {NOT_SKIN_DIR}")
    import tensorflow as tf  # noqa: E402  (import here to keep top-level imports light)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).flatten()

    mask = np.isin(y_all, CIFAR_NON_SKIN_CLASSES)
    x_non_skin = x_all[mask]

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(x_non_skin), size=min(NOT_SKIN_COUNT, len(x_non_skin)), replace=False)

    NOT_SKIN_DIR.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        img = Image.fromarray(x_non_skin[idx]).resize((224, 224), Image.BILINEAR)
        img.save(NOT_SKIN_DIR / f"not_skin_{i:04d}.jpg")
        if (i + 1) % 500 == 0:
            print(f"    Saved {i + 1}/{len(indices)} images...")
    print(f"  Done — {len(indices)} not_skin images saved to {NOT_SKIN_DIR}")
else:
    print(f"  Found {len(existing_images)} existing not_skin images in {NOT_SKIN_DIR}, skipping generation.")

not_skin_rows = []
for p in NOT_SKIN_DIR.rglob("*"):
    if p.suffix.lower() in EXTS:
        not_skin_rows.append({"filepath": str(p), "label": "not_skin"})

if not not_skin_rows:
    print("  [WARNING] No not_skin images found after generation attempt. Stage 1 SKIPPED.")
else:
    # Balance classes
    n = min(len(skin_rows), len(not_skin_rows))
    skin_df     = pd.DataFrame(skin_rows).sample(n=n, random_state=SEED)
    not_skin_df = pd.DataFrame(not_skin_rows).sample(n=n, random_state=SEED)
    stage1_df   = pd.concat([skin_df, not_skin_df], ignore_index=True)
    save_split(stage1_df, "stage1")

# ===========================================================================
# STAGE 2: eczema vs other_skin (balanced)
# Positive: Atopic Dermatitis Photos + Eczema Photos
# Negative: 4 other skin disease folders
# ===========================================================================
print("=" * 60)
print("STAGE 2: eczema vs other_skin")
print("=" * 60)

eczema_rows = []
for split in ("train", "test"):
    eczema_rows += gather(DATA_ROOT / split / "Atopic Dermatitis Photos", "eczema")
    eczema_rows += gather(DATA_ROOT / split / "Eczema Photos",            "eczema")

negative_folders = [
    "Acne and Rosacea Photos",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Warts Molluscum and other Viral Infections",
]
other_rows = []
for folder_name in negative_folders:
    for split in ("train", "test"):
        other_rows += gather(DATA_ROOT / split / folder_name, "other_skin")

eczema_df = pd.DataFrame(eczema_rows)
other_df  = pd.DataFrame(other_rows)

print(f"\n  Before balancing: eczema={len(eczema_df)}, other_skin={len(other_df)}")
n_eczema = len(eczema_df)
if len(other_df) > n_eczema:
    other_df = other_df.sample(n=n_eczema, random_state=SEED)
print(f"  After balancing:  eczema={len(eczema_df)}, other_skin={len(other_df)}\n")

stage2_df = pd.concat([eczema_df, other_df], ignore_index=True)
save_split(stage2_df, "stage2")

# ===========================================================================
# STAGE 3: severity — mild / moderate / severe
# Derived from filename keywords in "Eczema Photos":
#   eczema-acute-*    → mild
#   eczema-subacute-* → moderate
#   eczema-chronic-*  → severe
# Also use Atopic Dermatitis Photos — labeled by filename if possible
# ===========================================================================
print("=" * 60)
print("STAGE 3: severity (mild / moderate / severe)")
print("=" * 60)

SEVERITY_MAP = (
    # ── severe: chronic/lichenified/widespread presentations ──────────────────
    ("chronic",           "severe"),
    ("lichen-simplex",    "severe"),
    ("prurigo",           "severe"),
    ("hyperkerato",       "severe"),
    ("stasis",            "severe"),
    ("fissure",           "severe"),
    ("heels-dry",         "severe"),
    ("trunk-generalized", "severe"),   # widespread/generalized trunk = severe
    ("generalized",       "severe"),
    ("erythroderma",      "severe"),
    ("paraesthetica",     "severe"),   # notalgia paraesthetica = chronic neurogenic
    # ── moderate: subacute/localised chronic body-site presentations ──────────
    ("subacute",          "moderate"),
    ("dyshidro",          "moderate"),
    ("pompholyx",         "moderate"),
    ("nummular",          "moderate"),
    ("asteatotic",        "moderate"),
    ("desquamation",      "moderate"),
    ("excoriat",          "moderate"),
    ("impetigin",         "moderate"),
    ("fingertip",         "moderate"),  # fingertip eczema = chronic localised
    ("hand",              "moderate"),  # hand eczema = chronic contact/endogenous
    ("foot",              "moderate"),  # foot eczema
    ("exfoliativa",       "moderate"),  # keratolysis exfoliativa = moderate scaling
    ("areola",            "moderate"),
    ("axillae",           "moderate"),
    ("ears",              "moderate"),
    ("trunk",             "moderate"),  # non-generalised trunk
    ("arms",              "moderate"),
    ("leg",               "moderate"),
    ("vulva",             "moderate"),
    ("scrotum",           "moderate"),
    ("anal",              "moderate"),
    ("reaction",          "moderate"),  # contact reaction (often subacute)
    ("disease",           "moderate"),
    # ── mild: acute / early / facial / superficial presentations ─────────────
    ("acute",             "mild"),
    ("contact",           "mild"),
    ("diaper",            "mild"),
    ("dermatitis",        "mild"),
    ("face",              "mild"),      # facial eczema typically mild/atopic
    ("lids",              "mild"),      # eyelid eczema = mild atopic/contact
    ("superficial",       "mild"),
    ("psychogenic",       "mild"),      # neurotic/factitial = mild presentation
    ("factitial",         "mild"),
    ("fiberglass",        "mild"),      # irritant contact = mild
)

severity_rows = []
skipped = 0

for split in ("train", "test"):
    folder = DATA_ROOT / split / "Eczema Photos"
    if not folder.exists():
        print(f"  [MISSING] {folder}")
        continue
    for p in folder.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        name = p.stem.lower()
        label = None
        for keyword, severity in SEVERITY_MAP:
            if keyword in name:
                label = severity
                break
        if label:
            severity_rows.append({"filepath": str(p), "label": label})
        else:
            skipped += 1

print(f"  Labeled: {len(severity_rows)} images")
print(f"  Skipped (no keyword match): {skipped} images")

if not severity_rows:
    print("  [ERROR] No severity images found — check folder name")
else:
    stage3_df = pd.DataFrame(severity_rows)
    print(f"\n  Label distribution:\n{stage3_df['label'].value_counts().to_string()}\n")
    save_split(stage3_df, "stage3", holdout=0.3)

print("=" * 60)
print("DONE")
print("=" * 60)