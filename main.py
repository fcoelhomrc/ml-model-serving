import logging
import os
import tempfile
from contextlib import asynccontextmanager

import boto3
import torch
import torch.nn as nn
from botocore.exceptions import ClientError
from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger(__name__)

INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 4

device: torch.device
model: nn.Module


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _MLP().to(device)

    bucket = os.getenv("MODEL_BUCKET")
    if bucket:
        try:
            s3 = boto3.client("s3")
            with tempfile.TemporaryDirectory() as tmp:
                local_path = f"{tmp}/model.pth"
                s3.download_file(bucket, "models/production/model.pth", local_path)
                state_dict = torch.load(local_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                log.info("Loaded production model from S3")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                log.warning("No production model in S3, using random weights")
            else:
                raise
    else:
        log.warning("MODEL_BUCKET not set, using random weights")

    model.eval()
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    logits: list[float]
    predicted_class: int


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return PredictResponse(
        logits=logits.squeeze(0).tolist(),
        predicted_class=int(logits.argmax(dim=-1).item()),
    )
