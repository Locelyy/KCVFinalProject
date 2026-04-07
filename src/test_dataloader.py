from torch.utils.data import DataLoader
from dataset import BreakHisDataset, get_train_transforms

def main():
    dataset = BreakHisDataset(
        csv_file="data/train.csv",
        transform=get_train_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    images, labels = next(iter(dataloader))

    print(f"Batch image shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Labels: {labels}")

if __name__ == "__main__":
    main()