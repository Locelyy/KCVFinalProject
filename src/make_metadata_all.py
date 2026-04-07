from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/raw")

MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

CLASS_TO_IDX = {
    "adenosis": 0,
    "ductal_carcinoma": 1,
    "fibroadenoma": 2,
    "lobular_carcinoma": 3,
    "mucinous_carcinoma": 4,
    "papillary_carcinoma": 5,
    "phyllodes_tumor": 6,
    "tubular_adenoma": 7,
}

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def main():
    rows = []

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    for magnification in MAGNIFICATIONS:
        mag_dir = DATA_ROOT / magnification

        if not mag_dir.exists():
            print(f"Warning: magnification folder not found -> {mag_dir}")
            continue

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = mag_dir / class_name

            if not class_dir.exists():
                print(f"Warning: class folder not found -> {class_dir}")
                continue

            for image_path in class_dir.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                    rows.append({
                        "image_path": str(image_path).replace("\\", "/"),
                        "class_name": class_name,
                        "class_idx": class_idx,
                        "magnification": magnification
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No images found. Check folder structure and file extensions.")

    output_path = Path("data/metadata_all.csv")
    df.to_csv(output_path, index=False)

    print(f"Metadata saved to: {output_path}")
    print(f"Total images: {len(df)}")

    print("\nCounts by magnification:")
    print(df["magnification"].value_counts().sort_index())

    print("\nCounts by class:")
    print(df["class_name"].value_counts().sort_index())

    print("\nCounts by magnification and class:")
    print(pd.crosstab(df["magnification"], df["class_name"]))

if __name__ == "__main__":
    main()