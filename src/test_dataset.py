from dataset import BreakHisDataset, get_train_transforms

def main():
    dataset = BreakHisDataset(
        csv_file="data/train.csv",
        transform=get_train_transforms()
    )

    print(f"Total training images: {len(dataset)}")

    image, label = dataset[0]

    print(f"Image tensor shape: {image.shape}")
    print(f"Label: {label}")

if __name__ == "__main__":
    main()