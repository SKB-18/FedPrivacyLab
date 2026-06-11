# FedPrivacyLab API Documentation

## Overview

FedPrivacyLab exposes a complete RESTful API built with FastAPI. All endpoints use JSON for request/response bodies.

**Base URL:** `http://localhost:8000/api/v1`  
**Documentation:** `http://localhost:8000/docs` (Swagger UI)  
**Alternative Docs:** `http://localhost:8000/redoc` (ReDoc)

---

## Authentication

Currently, no authentication is required. For production deployment, implement JWT token-based auth.

**Future:** `Authorization: Bearer <token>`

---

## Experiment Lifecycle APIs

### POST /experiments/start

Create and initialize a new experiment.

**Request:**
```json
{
  "name": "string",                    // Experiment name
  "mode": "string",                    // Mode: centralized_baseline, federated_analytics, fedavg, fedavg_secureagg, fedavg_secureagg_dp
  "dataset": "string",                 // Dataset: emnist, telemetry, real_hdfs_loghub
  "num_clients": 100,                  // Number of clients
  "rounds": 10,                        // Number of rounds
  "clients_per_round": 50,             // Clients selected per round
  "local_epochs": 3,                   // Local training epochs
  "dropout_rate": 0.1,                 // Client dropout probability
  "dp_enabled": true,                  // Enable differential privacy
  "secure_agg_enabled": true,          // Enable secure aggregation
  "clipping_norm": 1.0,                // L2 clipping norm
  "noise_multiplier": 0.5,             // Gaussian noise scale
  "epsilon": 1.0,                      // DP privacy budget
  "delta": 1e-5                        // DP failure probability
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "name": "my_experiment",
  "mode": "fedavg_secureagg_dp",
  "dataset": "emnist",
  "num_clients": 100,
  "rounds": 10,
  "clients_per_round": 50,
  "local_epochs": 3,
  "dp_enabled": true,
  "secure_agg_enabled": true,
  "dropout_rate": 0.1,
  "clipping_norm": 1.0,
  "noise_multiplier": 0.5,
  "epsilon": 1.0,
  "delta": 1e-5,
  "status": "initialized",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Errors:**
- `400 Bad Request` — Invalid parameters
- `422 Unprocessable Entity` — Validation error (invalid enum, type mismatch)

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test",
    "mode": "fedavg_secureagg_dp",
    "num_clients": 100,
    "rounds": 5,
    "clients_per_round": 50,
    "local_epochs": 2,
    "dp_enabled": true,
    "epsilon": 1.0,
    "secure_agg_enabled": true
  }'
```

---

### GET /experiments/{experiment_id}

Retrieve experiment details.

**Path Parameters:**
- `experiment_id` (int) — Experiment ID from /experiments/start response

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "test",
  "mode": "fedavg_secureagg_dp",
  "status": "initialized",
  ...
}
```

**Errors:**
- `404 Not Found` — Experiment doesn't exist

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/experiments/1
```

---

### POST /experiments/{experiment_id}/run-round

Execute a single round of federated learning.

**Path Parameters:**
- `experiment_id` (int) — Experiment ID

**Response (200 OK):**
```json
{
  "experiment_id": 1,
  "round_number": 1,
  "status": "completed",
  "metrics": {
    "selected_clients": 50,
    "completed_clients": 45,
    "dropped_clients": 5,
    "train_loss": 0.234,
    "eval_loss": 0.198,
    "accuracy": 0.956,
    "f1_score": 0.923,
    "precision_score": 0.945,
    "recall_score": 0.912,
    "roc_auc": 0.987
  }
}
```

**Errors:**
- `404 Not Found` — Experiment not found
- `400 Bad Request` — Experiment already completed

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/experiments/1/run-round
```

**Time:** 20-60 seconds per round (depends on num_clients, local_epochs)

---

### POST /experiments/{experiment_id}/run-all

Execute all remaining rounds.

**Path Parameters:**
- `experiment_id` (int) — Experiment ID

**Response (200 OK):**
```json
{
  "experiment_id": 1,
  "rounds_completed": 5,
  "status": "completed",
  "all_metrics": [
    {
      "round_number": 1,
      "accuracy": 0.956,
      ...
    },
    {
      "round_number": 2,
      "accuracy": 0.962,
      ...
    },
    ...
  ]
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/experiments/1/run-all
```

**Time:** Depends on num_rounds (typically 2-5 minutes for 10 rounds)

---

## Metrics APIs

### GET /experiments/{experiment_id}/metrics

Retrieve metrics for a specific round.

**Path Parameters:**
- `experiment_id` (int) — Experiment ID
- `round` (int, optional) — Round number (defaults to latest)

**Query Parameters:**
```
?round=1  # Fetch round 1 metrics
```

**Response (200 OK):**
```json
{
  "round_number": 1,
  "selected_clients": 50,
  "completed_clients": 45,
  "dropped_clients": 5,
  "train_loss": 0.234,
  "eval_loss": 0.198,
  "accuracy": 0.956,
  "roc_auc": 0.987,
  "f1_score": 0.923,
  "precision_score": 0.945,
  "recall_score": 0.912
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/experiments/1/metrics?round=1
```

---

## Analytics APIs

### GET /analytics/results/{experiment_id}

Retrieve analytics results (federated measurement).

**Path Parameters:**
- `experiment_id` (int) — Experiment ID

**Query Parameters:**
```
?round=1&metric=count  # Filter by round and metric
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "round_number": 1,
      "metric_name": "count",
      "feature": "events",
      "true_value": 1000,
      "federated_value": 980,
      "dp_noisy_value": 985,
      "epsilon": 0.1,
      "absolute_error": 20,
      "relative_error": 0.02,
      "suppressed": false
    }
  ]
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/analytics/results/1?round=1
```

---

## Privacy APIs

### GET /privacy/risk-report/{experiment_id}

Retrieve privacy risk assessment.

**Path Parameters:**
- `experiment_id` (int) — Experiment ID

**Response (200 OK):**
```json
{
  "experiment_id": 1,
  "privacy_mode": "fedavg_secureagg_dp",
  "clipping_norm": 1.0,
  "noise_multiplier": 0.5,
  "epsilon": 1.0,
  "delta": 1e-5,
  "secure_agg_enabled": true,
  "contribution_bound": 10,
  "risk_level": "low",
  "threat_model": {
    "model_inversion": "mitigated by DP + secure agg",
    "membership_inference": "mitigated by DP + clipping",
    "gradient_leakage": "mitigated by clipping + secure agg"
  },
  "notes": "Strong privacy guarantees with ε=1.0"
}
```

**Risk Levels:**
- `low` — ε ≤ 1.0, secure agg enabled, DP enabled
- `medium` — ε ≤ 5.0 or secure agg enabled
- `high` — ε > 5.0 and secure agg disabled

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/privacy/risk-report/1
```

### GET /privacy/accounting/{experiment_id}

Track privacy budget consumption.

**Response (200 OK):**
```json
{
  "experiment_id": 1,
  "target_epsilon": 1.0,
  "spent_epsilon": 0.5,
  "remaining_epsilon": 0.5,
  "budget_exceeded": false,
  "rounds": 5,
  "history": [
    {
      "round": 1,
      "epsilon_per_round": 0.1,
      "epsilon_cumulative": 0.1,
      "remaining": 0.9
    },
    {
      "round": 2,
      "epsilon_per_round": 0.1,
      "epsilon_cumulative": 0.2,
      "remaining": 0.8
    }
  ]
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/privacy/accounting/1
```

---

## Learning APIs

### GET /learning/convergence/{experiment_id}

Retrieve convergence metrics (training progress).

**Response (200 OK):**
```json
{
  "experiment_id": 1,
  "convergence_data": [
    {
      "round": 1,
      "global_accuracy": 0.856,
      "global_loss": 0.456,
      "communication_cost": 1.2
    },
    {
      "round": 2,
      "global_accuracy": 0.879,
      "global_loss": 0.398,
      "communication_cost": 1.2
    },
    {
      "round": 3,
      "global_accuracy": 0.894,
      "global_loss": 0.321,
      "communication_cost": 1.2
    }
  ]
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/learning/convergence/1
```

---

## Utility & Simulation APIs

### POST /simulate/clients

Preview client distribution without starting an experiment.

**Request:**
```json
{
  "num_clients": 100,
  "min_records": 10,
  "max_records": 100,
  "random_seed": 42,
  "non_iid_severity": 0.5
}
```

**Response (200 OK):**
```json
{
  "num_clients": 100,
  "total_records": 5234,
  "profile_distribution": {
    "heavy_user": 30,
    "medium_user": 50,
    "light_user": 20
  },
  "sample_client": {
    "client_id": "client_0",
    "profile": "medium_user",
    "num_records": 52
  }
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/simulate/clients \
  -H "Content-Type: application/json" \
  -d '{
    "num_clients": 100,
    "min_records": 10,
    "max_records": 100,
    "non_iid_severity": 0.5
  }'
```

---

### POST /simulate/non-iid

Preview non-IID distribution.

**Request:**
```json
{
  "num_clients": 100,
  "severity": 0.5,
  "random_seed": 42
}
```

**Response (200 OK):**
```json
{
  "severity": 0.5,
  "profile_distribution": {
    "profile_0": 15,
    "profile_1": 22,
    "profile_2": 18,
    ...
  }
}
```

---

### GET /datasets/real-hdfs/summary

Get summary of real HDFS data.

**Response (200 OK):**
```json
{
  "dataset": "real_hdfs_loghub",
  "total_records": 575061,
  "time_period": "2010-01-20 to 2010-11-02",
  "components": [
    "Balancer",
    "BlockManager",
    "BlockReceiver",
    "DataNode",
    "FSEditLog",
    "FSImage",
    "FSNamesystem",
    "IPC",
    "NameNode",
    "Replicator"
  ],
  "log_types": [
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR"
  ]
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/datasets/real-hdfs/summary
```

---

## Health Check

### GET /health

Check API health status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**Example cURL:**
```bash
curl http://localhost:8000/api/v1/health
```

---

## Error Responses

All errors return JSON with status code and message.

**Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**

| Code | Meaning |
|------|---------|
| 200 | OK — Request succeeded |
| 201 | Created — Resource created |
| 400 | Bad Request — Invalid parameters |
| 404 | Not Found — Resource doesn't exist |
| 422 | Unprocessable Entity — Validation error |
| 500 | Internal Server Error — Server error |

**Example Error:**
```bash
curl http://localhost:8000/api/v1/experiments/999

# Response (404):
{"detail": "Experiment not found"}
```

---

## Request/Response Best Practices

### Use Query Parameters for Filtering

```bash
# Good
curl http://localhost:8000/api/v1/experiments/1/metrics?round=1

# Avoid
curl http://localhost:8000/api/v1/experiments/1/metrics/round/1
```

### Send JSON with Correct Content-Type

```bash
curl -X POST ... \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Parse JSON Response

```bash
# Pretty print
curl http://localhost:8000/api/v1/experiments/1 | jq .

# Extract field
curl http://localhost:8000/api/v1/experiments/1 | jq '.accuracy'
```

---

## Rate Limiting

Currently, no rate limiting is enforced. For production, implement:

```
- 100 requests/minute per IP
- 10 concurrent requests max
- Exponential backoff for retries
```

---

## Webhook/Event Streaming

Not implemented in v1.0. Future versions may support:
- WebSocket for real-time metrics
- Server-Sent Events (SSE) for notifications
- Webhook callbacks on experiment completion

---

## Client Examples

### Python Client

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# Create experiment
response = requests.post(
    f"{BASE_URL}/experiments/start",
    json={
        "name": "test",
        "mode": "fedavg_secureagg_dp",
        "num_clients": 100,
        "rounds": 5,
        "clients_per_round": 50,
        "local_epochs": 2,
        "dp_enabled": True,
        "epsilon": 1.0,
        "secure_agg_enabled": True
    }
)
experiment_id = response.json()["id"]

# Run single round
response = requests.post(
    f"{BASE_URL}/experiments/{experiment_id}/run-round"
)
metrics = response.json()["metrics"]
print(f"Accuracy: {metrics['accuracy']:.3f}")

# Get privacy report
response = requests.get(f"{BASE_URL}/privacy/risk-report/{experiment_id}")
report = response.json()
print(f"Risk Level: {report['risk_level']}")
```

### JavaScript/Node.js Client

```javascript
const BASE_URL = "http://localhost:8000/api/v1";

async function runExperiment() {
  // Create experiment
  const startResponse = await fetch(`${BASE_URL}/experiments/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "test",
      mode: "fedavg_secureagg_dp",
      num_clients: 100,
      rounds: 5,
      clients_per_round: 50,
      local_epochs: 2,
      dp_enabled: true,
      epsilon: 1.0,
      secure_agg_enabled: true
    })
  });
  
  const experiment = await startResponse.json();
  const experimentId = experiment.id;
  
  // Run all rounds
  const runResponse = await fetch(
    `${BASE_URL}/experiments/${experimentId}/run-all`,
    { method: "POST" }
  );
  
  const result = await runResponse.json();
  console.log(`Completed ${result.rounds_completed} rounds`);
}

runExperiment().catch(console.error);
```

### Bash Script

```bash
#!/bin/bash

BASE_URL="http://localhost:8000/api/v1"

# Create experiment
RESPONSE=$(curl -s -X POST "$BASE_URL/experiments/start" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "bash_test",
    "mode": "fedavg_secureagg_dp",
    "num_clients": 100,
    "rounds": 5,
    "clients_per_round": 50,
    "local_epochs": 2,
    "dp_enabled": true,
    "epsilon": 1.0
  }')

EXPERIMENT_ID=$(echo $RESPONSE | jq '.id')
echo "Created experiment: $EXPERIMENT_ID"

# Run all rounds
curl -s -X POST "$BASE_URL/experiments/$EXPERIMENT_ID/run-all" | jq .

# Get results
curl -s "$BASE_URL/experiments/$EXPERIMENT_ID/metrics" | jq '.accuracy'
```

---

## WebSocket Support (Future)

```javascript
// Future: Real-time metrics streaming
const ws = new WebSocket('ws://localhost:8000/ws/experiments/1');

ws.onmessage = (event) => {
  const metrics = JSON.parse(event.data);
  console.log(`Round ${metrics.round}: accuracy=${metrics.accuracy}`);
};
```

---

**For more examples, see COMPREHENSIVE_README.md**
