from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import json
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.predict import predict

# Structured JSON logger — every log line is a parseable JSON object.
# CloudWatch can filter/query JSON logs; plain text logs cannot be queried.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float


def emit_metric(metric_name: str, value: float, unit: str, dimensions: dict):
    """
    Emit a CloudWatch custom metric using EMF (Embedded Metric Format).

    EMF works by printing a specially structured JSON line to stdout.
    CloudWatch Logs picks it up and automatically creates a metric — no
    extra API calls, no extra cost beyond the log storage.

    Units CloudWatch accepts: "Milliseconds", "Count", "Percent", "None"
    """
    emf_log = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "SentimentAPI",
                    "Dimensions": [list(dimensions.keys())],
                    "Metrics": [{"Name": metric_name, "Unit": unit}],
                }
            ],
        },
        metric_name: value,
        **dimensions,
    }
    print(json.dumps(emf_log))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text field cannot be empty")
    return predict(request.text)


def handler(event, context):
    """
    Lambda entry point. Adds structured logging and CloudWatch metrics
    around every request.
    """
    request_id = context.aws_request_id if context else "local"
    path = event.get("path", "/predict")
    method = event.get("httpMethod", "POST")

    # ── health check ──────────────────────────────────────────────────────────
    if path == "/health" and method == "GET":
        logger.info(json.dumps({"request_id": request_id, "path": path, "status": 200}))
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "ok"}),
        }

    # ── predict ───────────────────────────────────────────────────────────────
    start_time = time.time()
    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        text = body.get("text", "")
        if not text.strip():
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "text field cannot be empty"}),
            }

        result = predict(text)
        latency_ms = (time.time() - start_time) * 1000

        # Structured log — every field is queryable in CloudWatch Logs Insights
        logger.info(json.dumps({
            "request_id": request_id,
            "path": path,
            "status": 200,
            "label": result["label"],
            "confidence": result["confidence"],
            "latency_ms": round(latency_ms, 2),
            "text_length": len(text),
        }))

        # EMF metrics — become graphable in CloudWatch Metrics automatically
        emit_metric("Latency", latency_ms, "Milliseconds", {"Service": "SentimentAPI"})
        emit_metric("Confidence", result["confidence"] * 100, "Percent", {"Service": "SentimentAPI"})
        emit_metric("PredictionCount", 1, "Count", {"Label": result["label"]})

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(json.dumps({
            "request_id": request_id,
            "path": path,
            "status": 500,
            "error": str(e),
            "latency_ms": round(latency_ms, 2),
        }))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
