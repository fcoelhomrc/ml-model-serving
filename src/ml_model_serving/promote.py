import argparse
import logging
import os
import tempfile

import boto3
import mlflow

from ml_model_serving import logger

log = logging.getLogger(__name__)


def main():
    logger.setup()

    parser = argparse.ArgumentParser(description="Promote a training run to production on S3")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to promote")
    args = parser.parse_args()

    bucket = os.environ["MODEL_BUCKET"]

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmp:
        local_path = client.download_artifacts(args.run_id, "model.pth", tmp)

        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, "models/production/model.pth")

    log.info(f"Promoted run {args.run_id} → s3://{bucket}/models/production/model.pth")
