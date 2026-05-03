import logging
import tempfile
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import logger
from main import INPUT_DIM, OUTPUT_DIM, _MLP

log = logging.getLogger(__name__)

LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 32
N_SAMPLES = 500


def main():
    logger.setup()

    X = torch.randn(N_SAMPLES, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (N_SAMPLES,))
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("ml-model-serving")

    with mlflow.start_run() as run:
        mlflow.log_params({"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "n_samples": N_SAMPLES})

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            log.info(f"Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "model.pth"
            torch.save(model.state_dict(), weights_path)
            mlflow.log_artifact(str(weights_path))

        log.info(f"Run ID: {run.info.run_id}")
