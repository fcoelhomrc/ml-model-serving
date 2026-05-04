# ml-model-serving

Learning project. Goal is to get familiar with the tools around deploying ML models (Docker, CI/CD, Ansible, AWS), focusing on serving / MLOps tooling.

The actual model is a MobileNetV2 fine-tuned to classify bean leaf images (healthy / angular leaf spot / bean rust). It's a simple enough task that I can focus on the infra side.

![](example.png)

## How it fits together

```
train locally (GPU)
    → MLflow tracks the run + saves model.pth
    → promote --run-id <id>  pushes it to S3

git push --tags v0.x
    → GitHub Actions builds the Docker image → GHCR
    → Ansible SSHes into EC2, pulls the image, restarts the container
    → container loads model.pth from S3 on startup
```

Dev environment is Nix + uv. Two shells: `nix develop` for CUDA (training), `nix develop .#minimal` for CPU-only work.

## Running things

**Train:**
```bash
uv run --extra training train
```
Set `MLFLOW_TRACKING_URI` if you want a remote MLflow server, otherwise it writes to `mlruns/` locally.

**Promote a run to S3:**
```bash
MODEL_BUCKET=<bucket> uv run promote --run-id <mlflow-run-id>
```

**Serve locally:**
```bash
MODEL_BUCKET=<bucket> uv run --extra inference uvicorn ml_model_serving.main:app --port 8080
# or
docker build -t ml-model-serving . && docker run -p 8080:8080 -e MODEL_BUCKET=<bucket> ml-model-serving
```
`MODEL_BUCKET` is optional — it'll just start with random weights and warn you.

**Provision a fresh EC2 (one-time):**
```bash
ansible-playbook -i inventory.aws_ec2.yml playbook-provision.yml
```

**Deploy manually:**
```bash
ansible-playbook -i inventory.aws_ec2.yml playbook-deploy.yml -e "model_bucket=<bucket>"
```
CI does this automatically on version tags.

## API

```
GET  /health   → { status, device }
POST /predict  → image upload → { predicted_class, label, probabilities }
```

## GitHub Actions secrets needed

`SSH_PRIVATE_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MODEL_BUCKET`

The EC2 instance needs the tag `Name: ml-model-serving` in `eu-north-1` for the dynamic inventory to find it.
