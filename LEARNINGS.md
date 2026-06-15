# Sentiment MLOps — Learning Log

Running notes of everything covered. Updated at the end of each phase.

---

## Phase 1 — Local Foundations ✓

### What we built
- Downloaded a pretrained DistilBERT sentiment model from HuggingFace Hub
- Saved the model artifact to disk (`model/saved_model/`)
- Wrote an inference function that takes text in, returns label + confidence
- Verified it works locally with test sentences

---

### Concepts Learned

#### HuggingFace Transformers
HuggingFace is a library that gives you access to thousands of pretrained models.
Every model on HuggingFace has two parts you always need together:

| Part | What it does |
|---|---|
| **Tokenizer** | Converts raw text → numbers the model understands |
| **Model** | Takes those numbers → outputs class scores (logits) |

`AutoTokenizer` and `AutoModelForSequenceClassification` are "auto" classes —
you give them a model name string, they figure out the right architecture automatically.

#### Transfer Learning
Instead of training a model from scratch (needs weeks of GPU time + millions of labeled examples),
we use a model already trained on massive data and just run inference directly.
`distilbert-base-uncased-finetuned-sst-2-english` is already fine-tuned on sentiment data —
no training needed at all.

#### What DistilBERT is
- A compressed version of BERT (Bidirectional Encoder Representations from Transformers)
- 66M parameters vs BERT's 110M — 40% smaller, 60% faster, 97% of the accuracy
- "distil" = distilled (knowledge distillation — a small model trained to mimic a large one)
- "base-uncased" = medium size, lowercase text
- "finetuned-sst-2-english" = already fine-tuned on Stanford Sentiment Treebank

#### CPU vs GPU for inference
- PyTorch runs the model math on CPU by default
- DistilBERT is small enough that CPU is fine (~100-400ms per request)
- Lambda has no GPU anyway — so CPU-only is the right choice
- CPU-only torch wheel = ~250MB vs full CUDA wheel = ~797MB (we used CPU-only)

#### Model Artifact
After `download_model.py` runs, `model/saved_model/` contains:

| File | What it is |
|---|---|
| `model.safetensors` | The actual model weights (~250MB) |
| `config.json` | Model architecture configuration |
| `vocab.txt` | The 30,522 tokens DistilBERT understands |
| `tokenizer.json` | Tokenization rules |
| `label_map.json` | Maps `{0: NEGATIVE, 1: POSITIVE}` |

This folder is the **artifact** — the thing CI/CD will later package into a Docker image.
Baking it into the image means no internet download at inference time.

#### Lazy Singleton Loading (in `predict.py`)
```python
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:          # only load if not already loaded
        _tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
        _model.eval()
```
The `if _tokenizer is None` guard means the model loads exactly once — on the first request.
Every subsequent request reuses the already-loaded model.
In Lambda this means: load once per container cold start, not once per request.
Critical for performance — loading the model takes ~2s, inference takes ~200ms.

#### Inference pipeline (what happens inside `predict()`)
```
text (string)
  → tokenizer → token IDs (list of numbers)
  → model forward pass → logits (raw scores, e.g. [-3.2, 4.1])
  → softmax → probabilities (e.g. [0.02, 0.98])
  → argmax → predicted class (1)
  → label_map → "POSITIVE"
  → confidence = 0.98
```

`torch.no_grad()` wraps the forward pass — tells PyTorch not to track gradients
(we're not training, so we don't need them — saves memory and speeds up inference).

---

### Files Created

| File | Purpose |
|---|---|
| `model/requirements.txt` | Pinned dependencies: transformers, torch (cpu), numpy |
| `model/download_model.py` | One-time script: downloads + saves model artifact to disk |
| `model/predict.py` | Core inference function — imported by every later layer |
| `model/saved_model/` | The artifact folder (created when download_model.py runs) |
| `PROJECT.md` | Full project plan, architecture, all 6 phases |
| `LEARNINGS.md` | This file |

---

### Commands Used

```bash
# Install dependencies (CPU-only torch, much smaller than CUDA version)
.venv/bin/pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.44.2 numpy==1.26.4

# Download and save the model artifact
.venv/bin/python model/download_model.py

# Run inference on test sentences
.venv/bin/python model/predict.py

# Test interactively (load model once, run many sentences)
.venv/bin/python
>>> from model.predict import predict
>>> predict("I loved this!")
>>> predict("Terrible experience.")
>>> exit()
```

---

### Key Decisions Made

- **CPU-only torch** — Lambda has no GPU. Smaller wheel, faster Docker builds.
- **Save model to disk** — not downloaded at runtime. Faster cold starts, no internet dependency.
- **Separate download_model.py from predict.py** — download is a one-time setup step,
  predict is runtime. Different jobs, different files.
- **predict() returns a dict** — so FastAPI (Phase 2) can return it directly as JSON with no conversion.

---

## Phase 2 — Containerize ✓

### What we built
- A FastAPI app with `POST /predict` and `GET /health` endpoints
- A Dockerfile that packages app + model artifact into one container image
- Tested locally with `docker run` and verified with `curl`

---

### Concepts Learned

#### Why Docker
Without Docker: model runs on your laptop because you have Python, torch, transformers installed.
On Lambda: blank environment — nothing pre-installed.
Docker packages everything (Python, deps, model weights) into a **container image** that runs
identically everywhere. "Works on my machine" becomes "works everywhere."

#### Docker Layers
Dockerfile builds top to bottom. Each `RUN`/`COPY` line is a **layer** that gets cached.
If a layer hasn't changed, Docker reuses the cache — no re-downloading.

Rule: put things that change rarely (dependencies) before things that change often (your code).
That's why we `COPY requirements.txt` + `pip install` BEFORE `COPY app/` and `COPY model/`.

```
FROM python:3.12-slim          ← base layer (cached after first pull)
COPY requirements.txt          ← only changes when deps change
RUN pip install                ← cached as long as requirements.txt unchanged
COPY model/ app/               ← changes when code changes — cache breaks here
CMD ["uvicorn", ...]           ← what runs when the container starts
```

#### FastAPI
Lightweight Python web framework. Two key concepts:

**Pydantic models** — define the exact shape of JSON in/out:
```python
class PredictRequest(BaseModel):
    text: str        # FastAPI validates this automatically
                     # wrong type → 422 error before your code runs
```

**Route decorators** — map HTTP methods + paths to functions:
```python
@app.post("/predict")          # POST /predict → this function
def predict_sentiment(request: PredictRequest):
    ...
```

#### Port mapping in docker run
```bash
docker run -p 8000:8000 sentiment-api
#              ↑      ↑
#         host port  container port
```
The container has its own network. `-p 8000:8000` punches a hole:
requests to `localhost:8000` on your machine forward to port 8000 inside the container.

#### Health endpoint
`GET /health` returns `{"status": "ok"}`. Lambda and load balancers ping this to confirm
the container is alive before routing real traffic to it. Always include one.

---

### Files Created

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app — thin layer that calls `model/predict.py` |
| `app/requirements.txt` | App deps: fastapi, uvicorn, pydantic |
| `Dockerfile` | Container recipe — base image, deps, code, startup command |

---

### Commands Used

```bash
# Build the container image (run from project root)
docker build -t sentiment-api .

# Run it locally, forward port 8000
docker run -p 8000:8000 sentiment-api

# Test health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an amazing project!"}'
# → {"text":"...","label":"POSITIVE","confidence":0.9999}
```

---

### Key Decisions Made

- **python:3.12-slim base image** — minimal Linux + Python, no extras. Keeps image small.
- **CPU-only torch in Dockerfile** — Lambda has no GPU. Explicit `--index-url` targets PyTorch's CPU wheel server.
- **Thin API layer** — `main.py` only handles HTTP concerns. All ML logic stays in `model/predict.py`. If we swap FastAPI for something else later, the model code is untouched.
- **`sys.path.append`** — lets `app/main.py` import from `model/predict.py` without making it a Python package. Simple, works inside the container's `/app` working directory.

---

## Phase 3 — AWS Deployment ✓

### What we built
- Pushed Docker image to AWS ECR (private container registry)
- Deployed Lambda function using the ECR image
- Wired API Gateway in front of Lambda for a public HTTPS URL
- Live endpoint: `https://cn7dhw7va3.execute-api.us-east-1.amazonaws.com/predict`

---

### Concepts Learned

#### IAM (Identity and Access Management)
AWS's permission system. Every identity (human, app, pipeline) gets only the permissions it needs.
- **Root user** — never use for day-to-day work. Unlimited power, no guardrails.
- **IAM User** — `sentiment-mlops-dev` with only ECR + Lambda + API Gateway permissions
- **Access keys** — temporary credentials the CLI uses to authenticate as that IAM user
- **IAM Role** — like a user but for AWS services (Lambda uses a role to call CloudWatch etc.)

```bash
aws sts get-caller-identity   # confirms which identity the CLI is using
```

#### ECR (Elastic Container Registry)
AWS's private Docker registry. Lambda needs to pull images from somewhere it trusts — ECR is that place.

```
Your laptop → docker push → ECR repository → Lambda pulls from here
```

Three steps to push an image:
```bash
# 1. Authenticate Docker to ECR (gets a 12-hour temp password)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  797855612851.dkr.ecr.us-east-1.amazonaws.com

# 2. Tag local image with full ECR address
docker tag sentiment-api:latest \
  797855612851.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:latest

# 3. Push (only uploads changed layers — fast on subsequent pushes)
docker push 797855612851.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:latest
```

#### Lambda
Serverless compute — runs your container on demand. You pay only for milliseconds used. No idle billing.

| Term | What it means |
|---|---|
| **Cold start** | First request spins up container + loads model (~40-50s). Subsequent requests reuse the warm container (~1s). |
| **Init phase** | Time to start container and import Python modules. `import torch` alone takes ~10s. |
| **Execution phase** | Time for your handler to run. Model loads lazily here on first call. |
| **Timeout** | Set to 3 minutes. Must be longer than cold start time. |
| **Memory** | Set to 2048MB. Also determines CPU share — more memory = faster CPU. |

**Why the direct handler, not Mangum:**
Mangum wraps FastAPI as a Lambda handler but has complex event format detection that kept failing.
Our direct `handler()` function is simpler — reads `event["body"]`, calls `predict()`, returns a dict.
Lambda expects exactly this: a function that takes `(event, context)` and returns a dict.

#### Lambda response format
Lambda must return a dict in this shape for API Gateway to understand it:
```python
{
    "statusCode": 200,
    "headers": {"Content-Type": "application/json"},
    "body": json.dumps(result)   # must be a STRING, not a dict
}
```

#### API Gateway (HTTP API)
Sits in front of Lambda and gives it a public HTTPS URL.
```
Internet → API Gateway → Lambda → handler() → response
```
Route `/{proxy+}` with method `ANY` means all paths and methods forward to Lambda.
API Gateway unpacks the response dict and sends the body back to the caller as HTTP.

#### Deploying a new image to Lambda
When you push a new image with the same `:latest` tag, Lambda doesn't auto-update.
You must explicitly tell it to use the new image:
```bash
aws lambda update-function-code \
  --function-name sentiment-api \
  --image-uri 797855612851.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:latest \
  --region us-east-1

# Wait for update to fully apply before testing
aws lambda wait function-updated --function-name sentiment-api --region us-east-1
```

---

### Files Changed
| File | Change |
|---|---|
| `app/main.py` | Added direct Lambda `handler()` function — replaces Mangum |
| `app/requirements.txt` | Removed mangum |
| `Dockerfile` | Changed base image to `public.ecr.aws/lambda/python:3.12` (includes Lambda Runtime Client) |

---

### Commands Used
```bash
# Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 797855612851.dkr.ecr.us-east-1.amazonaws.com
docker tag sentiment-api:latest 797855612851.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:latest
docker push 797855612851.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:latest

# Deploy to Lambda
aws lambda update-function-code --function-name sentiment-api --image-uri 797855612851.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:latest --region us-east-1
aws lambda wait function-updated --function-name sentiment-api --region us-east-1

# Test the live endpoint
curl -X POST https://cn7dhw7va3.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an amazing project!"}'
# → {"text": "...", "label": "POSITIVE", "confidence": 0.9999}
```

---

### Key Decisions Made
- **Direct Lambda handler over Mangum** — simpler, no event format inference issues, easier to debug
- **Timeout set to 3 minutes** — cold start (init + model load) takes ~50s, needed headroom
- **Memory set to 2048MB** — model needs ~400MB, extra memory also gives more CPU for faster inference
- **HTTP API over REST API in API Gateway** — simpler, cheaper, lower latency for our use case

---

## Phase 4 — Observability ✓

### What we built
- Structured JSON logging on every request (queryable in CloudWatch Logs Insights)
- EMF custom metrics: Latency, Confidence, PredictionCount — visible in CloudWatch Metrics
- Fixed a critical bug: model loading moved from invocation to init phase

---

### Concepts Learned

#### Why Observability Matters for ML
Without it you're blind to:
- How many requests/min are hitting the API
- Whether model confidence is dropping (data drift)
- Label distribution shifting (POSITIVE/NEGATIVE ratio changing)
- Silent failures and timeouts

#### Structured Logging vs Print Statements
```python
# Bad — unqueryable plain text
print("prediction done, label=POSITIVE")

# Good — every field is queryable in CloudWatch Logs Insights
logger.info(json.dumps({
    "request_id": "abc123",
    "label": "POSITIVE",
    "confidence": 0.9999,
    "latency_ms": 245.3,
}))
```
With structured logs you can run queries like:
`filter confidence < 0.7 | stats count() by label`

#### EMF (Embedded Metric Format)
The trick that gives you CloudWatch custom metrics for free.
Write a specially structured JSON line to stdout → CloudWatch automatically extracts it as a metric.
No extra API calls, no extra SDK, no extra cost beyond log storage.

```python
def emit_metric(metric_name, value, unit, dimensions):
    emf_log = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": "SentimentAPI",          # appears under Custom namespaces
                "Dimensions": [list(dimensions.keys())],
                "Metrics": [{"Name": metric_name, "Unit": unit}],
            }],
        },
        metric_name: value,   # the actual metric value
        **dimensions,         # dimension values (e.g. {"Label": "POSITIVE"})
    }
    print(json.dumps(emf_log))  # just a print! CloudWatch does the rest
```

We emit 3 metrics per request:
| Metric | Unit | What it tells you |
|---|---|---|
| `Latency` | Milliseconds | How fast inference is running |
| `Confidence` | Percent | Model certainty — drop = data drift |
| `PredictionCount` | Count (by Label) | POSITIVE vs NEGATIVE ratio over time |

#### Critical Bug Fixed: Init Phase vs Invocation Phase
Lambda has two separate phases:

| Phase | What runs | API Gateway timeout applies? |
|---|---|---|
| **Init** | Container start + Python imports + module-level code | NO |
| **Invocation** | Your handler() function | YES (29s hard limit for HTTP API) |

Original code used lazy loading — model loaded on first invocation (~44s) → API Gateway 29s timeout → 503.

Fix: load model at module level (outside handler), so it loads during init phase:
```python
# predict.py — runs during Lambda init, not during invocation
_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
_model.eval()
```

Result:
- Init phase: ~50s (torch import + model load) → API Gateway doesn't care
- Invocation: ~1-2s (just inference) → well under 29s limit ✓

#### API Gateway HTTP API Hard Limit
HTTP API type has a 29-second maximum integration timeout. Cannot be changed.
This is why front-loading model loading into init is mandatory for heavy ML models on Lambda.

---

### Files Changed
| File | Change |
|---|---|
| `app/main.py` | Added structured JSON logging + EMF metric emission |
| `model/predict.py` | Moved model loading from lazy (first call) to eager (module import / init phase) |

---

### Commands Used
```bash
# Send test requests to generate CloudWatch data
curl -X POST https://cn7dhw7va3.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an amazing project!"}'

# View metrics: CloudWatch → Metrics → All metrics → Custom namespaces → SentimentAPI
# View logs:    Lambda → Monitor → View CloudWatch logs → latest stream
```

---

### Key Decisions Made
- **EMF over CloudWatch PutMetricData API** — zero extra cost, works from a print statement, no SDK changes
- **Structured JSON logs** — plain text logs can't be queried; JSON logs can
- **3 metrics chosen deliberately** — Latency (performance), Confidence (model health), PredictionCount by Label (distribution shift)

---

## Phase 5 — Azure DevOps CI/CD

*Coming after Phase 4*

---

## Phase 6 — Terraform IaC

*Coming after Phase 5*
