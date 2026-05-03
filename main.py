from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

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
