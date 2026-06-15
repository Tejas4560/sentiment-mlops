# Sentiment Analysis MLOps Pipeline

## Problem Statement

Build a production-ready sentiment analysis system that:
- Accepts a sentence as input via a public HTTP endpoint
- Returns whether the sentiment is POSITIVE or NEGATIVE with a confidence score
- Is deployed on AWS, monitored via CloudWatch, and automatically redeployed via Azure DevOps CI/CD whenever code changes

This is not just a model — it is a full MLOps pipeline covering model serving, containerization,
cloud infrastructure, observability, and automated deployment.

---

## Why This Project

Sentiment analysis is simple enough that it doesn't get in the way of learning the infrastructure.
The ML part (a pretrained HuggingFace model) is solved in Phase 1. Every phase after that is pure
MLOps — the skills that actually show up in DevOps/MLOps job descriptions.

By the end you will have hands-on experience with:
- HuggingFace Transformers + PyTorch inference
- FastAPI + Docker containerization
- AWS (IAM, ECR, Lambda, API Gateway, CloudWatch, SageMaker Model Registry)
- Azure DevOps CI/CD pipelines with OIDC authentication to AWS
- Terraform Infrastructure as Code

---

## Architecture

```
Developer pushes code
        |
        v
  Azure DevOps Pipeline
        |
        |-- builds Docker image
        |-- pushes to AWS ECR
        |-- updates Lambda function
        |
        v
  AWS API Gateway  <-- public HTTPS endpoint
        |
        v
  AWS Lambda       <-- runs Docker container (CPU inference)
        |
        |-- loads DistilBERT from /saved_model/
        |-- tokenizes input text
        |-- runs forward pass
        |-- returns label + confidence
        |
        v
  AWS CloudWatch   <-- logs every request + custom metrics
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Model | DistilBERT (HuggingFace) | Pretrained, fast on CPU, no GPU needed |
| Serving | FastAPI | Lightweight, async, auto-generates API docs |
| Container | Docker | Same image runs locally and on Lambda |
| Registry | AWS ECR | Native Lambda container support |
| Compute | AWS Lambda | ~$0 cost at portfolio scale, no idle billing |
| Model Registry | SageMaker Model Registry | MLOps story without SageMaker endpoint cost |
| Monitoring | AWS CloudWatch + EMF | Free structured logging + custom metrics |
| CI/CD | Azure DevOps | Industry-standard, shows cross-cloud skills |
| Auth | OIDC federation | No long-lived AWS keys stored in Azure DevOps |
| IaC | Terraform | Reproducible infrastructure, no click-ops |

---

## Phases

### Phase 1 — Local Foundations (current)
**Goal:** A working inference function that takes text in, returns a prediction.

Steps:
1. Install dependencies (transformers, torch, numpy)
2. Run `model/download_model.py` — downloads DistilBERT (~250MB) and saves to `model/saved_model/`
3. Run `model/predict.py` — verifies inference works locally

Output: `model/saved_model/` folder (the artifact every later phase depends on)

Concepts covered: tokenizers, model artifacts, lazy singleton loading, CPU inference

---

### Phase 2 — Containerize
**Goal:** The inference function wrapped in an HTTP API, running inside Docker.

Steps:
1. Write `app/main.py` — FastAPI app with a `/predict` POST endpoint
2. Write `Dockerfile` — packages the app + saved_model into a container image
3. `docker build` and `docker run` locally
4. Test with `curl` — same request format Lambda will use

Output: A Docker image that runs the full API locally

Concepts covered: REST APIs, request/response schemas, Docker layers, why containers solve "works on my machine"

---

### Phase 3 — AWS Deployment
**Goal:** The container running on AWS, accessible via a public URL.

Steps:
1. Create IAM role for Lambda with the right permissions
2. Create ECR repository and push the Docker image
3. Create Lambda function using the ECR image
4. Wire API Gateway in front of Lambda
5. Test the live endpoint with `curl`

Output: A public `https://xxxx.execute-api.region.amazonaws.com/predict` URL

Concepts covered: IAM roles and policies, ECR, Lambda container images, API Gateway proxy integration, cold starts

---

### Phase 4 — Observability
**Goal:** Every request logged, key metrics visible in CloudWatch dashboards.

Steps:
1. Add structured JSON logging to the FastAPI app
2. Emit custom CloudWatch metrics using EMF (Embedded Metric Format)
   - Request latency
   - Prediction confidence distribution
   - Request count by label (POSITIVE vs NEGATIVE)
3. Create a CloudWatch dashboard

Output: A CloudWatch dashboard showing real-time model behavior

Concepts covered: structured logging vs print statements, what EMF is and why it costs nothing, what metrics matter for ML systems

---

### Phase 5 — Azure DevOps CI/CD
**Goal:** Pushing code automatically rebuilds and redeploys everything. No manual steps.

Steps:
1. Set up Azure DevOps project and connect to the GitHub repo
2. Configure OIDC trust between Azure DevOps and AWS (no stored access keys)
3. Write `azure-pipelines.yml` — triggers on code push:
   - builds Docker image
   - pushes to ECR
   - updates Lambda with new image
4. Write a second pipeline for scheduled model retraining

Output: Two pipelines — one for deploy, one for retrain

Concepts covered: CI/CD triggers, YAML pipelines, OIDC federation, why OIDC beats long-lived keys, pipeline secrets

---

### Phase 6 — Infrastructure as Code
**Goal:** Every AWS resource defined in Terraform. One command recreates the entire stack.

Steps:
1. Write Terraform for: ECR repo, IAM roles, Lambda function, API Gateway
2. `terraform plan` — preview changes
3. `terraform apply` — provision everything
4. Destroy and recreate to prove it works

Output: A `terraform/` folder that is the single source of truth for all infrastructure

Concepts covered: why IaC matters, Terraform state, plan vs apply, resource dependencies

---

## Repository Structure (end state)

```
sentiment-mlops/
├── PROJECT.md                  ← this file
├── model/
│   ├── requirements.txt        ← pinned model dependencies
│   ├── download_model.py       ← one-time: download + save artifact
│   ├── predict.py              ← core inference function (imported by app)
│   └── saved_model/            ← model weights + tokenizer config
├── app/
│   ├── main.py                 ← FastAPI app
│   └── requirements.txt        ← app dependencies (fastapi, uvicorn)
├── Dockerfile                  ← container definition
├── azure-pipelines.yml         ← CI/CD pipeline (deploy)
├── azure-pipelines-retrain.yml ← CI/CD pipeline (retrain)
└── terraform/
    ├── main.tf
    ├── variables.tf
    └── outputs.tf
```

---

## Key Decisions and Why

**Lambda over SageMaker endpoint**
SageMaker real-time endpoints cost ~$50/month idle. Lambda costs ~$0 at portfolio traffic.
We still use SageMaker Model Registry to register the model artifact — that gives the MLOps story
(versioned models, approval workflows) without the endpoint bill.

**OIDC over long-lived AWS keys in Azure DevOps**
Storing AWS_ACCESS_KEY_ID in Azure DevOps secrets is the naive approach and a security risk —
if the secret leaks, it's valid until manually rotated. OIDC means Azure DevOps proves its identity
to AWS using a short-lived token issued per pipeline run. No stored credentials, no rotation burden.

**CPU-only torch wheel in Docker**
The full torch wheel is 797MB and includes CUDA. Lambda has no GPU. We use the CPU-only
wheel (~250MB) in the Dockerfile to keep the image lean and cold-start fast.

**Artifact saved to disk, not downloaded at runtime**
The model is saved to `saved_model/` and baked into the Docker image. This means:
- No internet access needed at inference time (Lambda can be VPC-isolated)
- Cold start is fast (load from disk, not network)
- Reproducible — the exact model version is locked to the image
```
