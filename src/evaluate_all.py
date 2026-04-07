import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import BreakHisDataset, get_val_test_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma"
]

NUM_CLASSES = 8
BATCH_SIZE = 8


def main():
    print(f"Using device: {DEVICE}")

    test_dataset = BreakHisDataset(
        csv_file="data/test_all.csv",
        transform=get_val_test_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load("models/best_efficientnet_v2_all_mag.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)

    print(f"\nTest Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()