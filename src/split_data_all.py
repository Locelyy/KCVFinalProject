from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def show_distribution(name, df):
    print(f"\n{name} total: {len(df)}")
    print("\nBy magnification:")
    print(df["magnification"].value_counts().sort_index())
    print("\nBy class:")
    print(df["class_name"].value_counts().sort_index())

def main():
    metadata_path = Path("data/metadata_all.csv")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # stratify by combined label: class + magnification
    df["stratify_label"] = df["class_name"] + "_" + df["magnification"]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["stratify_label"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["stratify_label"],
        random_state=RANDOM_STATE
    )

    train_df = train_df.drop(columns=["stratify_label"])
    val_df = val_df.drop(columns=["stratify_label"])
    test_df = test_df.drop(columns=["stratify_label"])

    train_df.to_csv("data/train_all.csv", index=False)
    val_df.to_csv("data/val_all.csv", index=False)
    test_df.to_csv("data/test_all.csv", index=False)

    print("Split complete.")
    show_distribution("Train", train_df)
    show_distribution("Validation", val_df)
    show_distribution("Test", test_df)

if __name__ == "__main__":
    main()