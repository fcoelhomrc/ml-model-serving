import logging
import os
import tempfile
from pathlib import Path

import certifi
import mlflow
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml_model_serving import logger
from ml_model_serving.main import NUM_CLASSES, BeanClassifier, build_transform

log = logging.getLogger(__name__)

LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 32


class _BeansDataset(Dataset):
    def __init__(self, hf_split, transform: transforms.Compose):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.transform(item["image"].convert("RGB")), item["labels"]


def main():
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    logger.setup()

    ds = load_dataset("AI-Lab-Makerere/beans")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        _BeansDataset(ds["train"], train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        _BeansDataset(ds["validation"], build_transform()),
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeanClassifier(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("bean-classification")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model": "MobileNetV2",
            "num_classes": NUM_CLASSES,
        })

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    preds = model(images.to(device)).argmax(dim=-1).cpu()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            acc = accuracy_score(all_labels, all_preds)
            bal_acc = balanced_accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_accuracy": acc,
                "val_balanced_accuracy": bal_acc,
                "val_f1_macro": f1,
            }, step=epoch)

            log.info(
                f"Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}"
                f"  acc={acc:.4f}  bal_acc={bal_acc:.4f}  f1={f1:.4f}"
            )

        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "model.pth"
            torch.save(model.state_dict(), weights_path)
            mlflow.log_artifact(str(weights_path))

        log.info(f"Run ID: {run.info.run_id}")
