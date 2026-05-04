import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import boto3
import torch
import torch.nn as nn
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

log = logging.getLogger(__name__)

NUM_CLASSES = 3
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device: torch.device
model: nn.Module
transform: transforms.Compose


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class BeanClassifier(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(1280, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model, transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeanClassifier().to(device)
    transform = build_transform()

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


class PredictResponse(BaseModel):
    predicted_class: int
    label: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    log.info(f"Received image: {x.shape}")
    with torch.no_grad():
        logits = model(x)
    log.info(f"Computed logits: {logits.shape}")
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    predicted = int(logits.argmax(dim=-1).item())
    return PredictResponse(
        predicted_class=predicted,
        label=CLASS_NAMES[predicted],
        probabilities={name: round(p, 4) for name, p in zip(CLASS_NAMES, probs)},
    )
