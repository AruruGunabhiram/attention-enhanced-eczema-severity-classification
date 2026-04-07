from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_ROOT = Path("data/raw/dermnet")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def gather_from_class_folder(folder: Path, label: str):
    rows = []
    if not folder.exists():
        print(f"Missing folder: {folder}")
        return rows
    for p in folder.rglob("*"):
        if p.suffix.lower() in EXTS:
            rows.append({"filepath": str(p), "label": label})
    print(f"{folder}: {len(rows)} images -> {label}")
    return rows

def save_split(df: pd.DataFrame, prefix: str):
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )

    train_df.to_csv(SPLIT_DIR / f"{prefix}_train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / f"{prefix}_val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / f"{prefix}_test.csv", index=False)

    print(f"Saved {prefix}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Stage 2 only for now: eczema vs other_skin
rows = []
rows += gather_from_class_folder(DATA_ROOT / "train" / "Atopic Dermatitis Photos", "eczema")
rows += gather_from_class_folder(DATA_ROOT / "test" / "Atopic Dermatitis Photos", "eczema")

# pick a few other skin disease folders as negative class
negative_folders = [
    "Acne and Rosacea Photos",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Warts Molluscum and other Viral Infections",
]

for folder_name in negative_folders:
    rows += gather_from_class_folder(DATA_ROOT / "train" / folder_name, "other_skin")
    rows += gather_from_class_folder(DATA_ROOT / "test" / folder_name, "other_skin")

df = pd.DataFrame(rows)
print(df["label"].value_counts())

save_split(df, "stage2")

print("Done.")