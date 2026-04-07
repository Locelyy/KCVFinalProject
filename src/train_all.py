from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models

from dataset import BreakHisDataset, get_train_transforms, get_val_test_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 8
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

# CutMix hyperparameters
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0

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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        r = np.random.rand(1)
        if CUTMIX_ALPHA > 0 and r < CUTMIX_PROB:
            lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
            rand_index = torch.randperm(images.size()[0]).to(DEVICE)
            
            target_a = labels
            target_b = labels[rand_index]
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            outputs = model(images)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == target_a).sum().item() * lam + (preds == target_b).sum().item() * (1. - lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

        loss = loss / ACCUMULATION_STEPS
        loss.backward()

        if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(loader)):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUMULATION_STEPS * images.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    print(f"Using device: {DEVICE}")

    train_dataset = BreakHisDataset(
        csv_file="data/train_all.csv",
        transform=get_train_transforms()
    )
    val_dataset = BreakHisDataset(
        csv_file="data/val_all.csv",
        transform=get_val_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    class_counts = train_dataset.df["class_idx"].value_counts().sort_index()
    # class_counts = (
    # train_dataset.df["class_idx"]
    # .value_counts()
    # .reindex(range(NUM_CLASSES), fill_value=0)
    # .sort_index()
    # )
    total_samples = len(train_dataset.df)
    num_classes = len(class_counts)

    weights = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

    print("\nClass counts:")
    print(class_counts)
    print("\nClass weights:")
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    warmup_epochs = 5
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("New best model found and saved in memory.")

    Path("models").mkdir(exist_ok=True)
    torch.save(best_model_wts, "models/best_efficientnet_v2_all_mag.pth")

    print(f"\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved to: models/best_efficientnet_v2_all_mag.pth")


if __name__ == "__main__":
    main()