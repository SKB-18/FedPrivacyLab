# FedPrivacyLab — Federated Inference Docker Stack

## Overview

The inference stack consists of four services:

| Service | Port | Description |
|---|---|---|
| `inference-coordinator` | 8001 | FastAPI coordinator — model registry + metadata aggregator |
| `client-1/2/3` | — | Containerized clients — local inference + metadata logging |
| `inference-dashboard` | 8502 | Streamlit dashboard — live inference metrics |

**Privacy guarantee**: All predictions are made on-device inside each client container.
Only aggregate counts and timing reach the coordinator. Raw predictions, input features,
and user identities are never transmitted.

## Quick Start

```bash
# Build and start all services
docker compose -f docker-compose.inference.yml up --build

# Check coordinator health
curl http://localhost:8001/health

# View inference stats
curl http://localhost:8001/inference_stats

# Open dashboard
open http://localhost:8502
```

## Service Details

### inference-coordinator

FastAPI app running `fedprivacylab/inference/coordinator.py`.

Endpoints:
- `POST /register_client` — client registers, gets model version + ε budget
- `POST /get_model` — returns model descriptor (TFLite path or stub)
- `POST /client_inference_log` — client logs metadata (counts + latency only)
- `GET /inference_stats` — aggregate stats across all clients
- `GET /model_accuracy_drift` — validation accuracy vs model version
- `GET /clients` — active client registry

### client-1, client-2, client-3

Each runs `docker/inference_test_script.py`:
1. Registers with coordinator
2. Downloads and caches global model locally
3. Every `BATCH_INTERVAL_SECONDS`: makes `NUM_PREDICTIONS_PER_BATCH` local predictions
4. Logs metadata (count + latency) to coordinator
5. Prints: `[client_01] Batch #5: Made 5 inferences in 43ms, epsilon_remaining: 0.8821`

### Environment Variables (clients)

| Variable | Default | Description |
|---|---|---|
| `COORDINATOR_URL` | `http://inference-coordinator:8001` | Coordinator base URL |
| `CLIENT_ID` | `client_01` | Unique client identifier |
| `NUM_PREDICTIONS_PER_BATCH` | `5` | Predictions per batch |
| `BATCH_INTERVAL_SECONDS` | `10` | Seconds between batches |
| `FEATURE_DIM` | `8` | Input feature dimension |

## Running Individual Services

```bash
# Coordinator only
docker compose -f docker-compose.inference.yml up inference-coordinator

# All clients (after coordinator is healthy)
docker compose -f docker-compose.inference.yml up client-1 client-2 client-3

# Logs from a specific client
docker compose -f docker-compose.inference.yml logs -f client-1
```

## Teardown

```bash
docker compose -f docker-compose.inference.yml down -v
```
