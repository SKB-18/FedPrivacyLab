# FedPrivacyLab: Federated Analytics and Privacy-Preserving ML Workbench

## Executive Summary

FedPrivacyLab is a production-ready research framework demonstrating federated learning and privacy-preserving analytics. It enables organizations to train machine learning models and compute aggregate metrics from distributed data **without centralizing raw information**. 

Built with **FastAPI** backend, **TensorFlow** models, **Streamlit** dashboards, and production-grade deployment options (Docker, Kubernetes, Helm), FedPrivacyLab implements:

- **Federated Learning (FedAvg)** — Clients train locally; server aggregates model updates
- **Federated Analytics** — Privacy-preserving measurement with bounded local summaries
- **Differential Privacy** — Laplace noise on counts; Gaussian noise on updates
- **Secure Aggregation** — Cryptographic concept implementation (research-grade)
- **Contribution Bounding** — Caps per-client contributions per round
- **Update Clipping** — L2 norm bounds on model updates
- **Privacy Accounting** — Tracks epsilon budget across rounds
- **Non-IID Data** — Simulates realistic client heterogeneity

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [API Overview](#api-overview)
8. [Experiment Modes](#experiment-modes)
9. [Privacy Mechanisms](#privacy-mechanisms)
10. [Dashboard Features](#dashboard-features)
11. [Deployment](#deployment)
12. [Testing](#testing)
13. [Project Structure](#project-structure)
14. [Performance & Benchmarks](#performance--benchmarks)
15. [Real-World Data](#real-world-data)
16. [Troubleshooting](#troubleshooting)
17. [References](#references)

---

## Problem Statement

Organizations generating telemetry (logs, metrics, behavioral data) face a fundamental dilemma:

- **Centralize data** → Scale analytics, but expose privacy risk
- **Keep data distributed** → Preserve privacy, but lose visibility

FedPrivacyLab demonstrates that **federated learning + privacy-preserving mechanisms** enable both: organizations can train accurate ML models and compute aggregate statistics **without ever seeing raw individual data**.

### Why Federated Learning Alone Is Insufficient

Even in federated learning, individual model updates can leak information about training data. FedPrivacyLab layers privacy controls to mitigate this:

```
Raw Data (Local) 
  → Bounded Summaries (Contribution Bounding)
  → L2 Clipping (Limit update magnitude)
  → Secure Aggregation (Server can't decrypt individual updates)
  → Differential Privacy (Noise injection for formal privacy guarantee)
  → Global Model/Metrics (Privacy-preserved aggregate)
```

---

## Key Features

### 🎯 Federated Learning

| Feature | Details |
|---------|---------|
| **FedAvg Algorithm** | Industry-standard federated averaging with sample weighting |
| **Client Selection** | Random stratified selection with participation control |
| **Dropout Simulation** | Models realistic client unavailability |
| **Local Training** | Configurable local epochs and batch sizes |
| **Model Evaluation** | Per-round accuracy, F1, ROC-AUC, precision, recall |

### 🔒 Privacy Mechanisms

| Mechanism | Purpose | Parameters |
|-----------|---------|-----------|
| **Contribution Bounding** | Limit per-client contribution per round | `contribution_bound` (default: 10) |
| **Update Clipping** | Cap L2 norm of model updates | `clipping_norm` (default: 1.0) |
| **Secure Aggregation** | Cryptographic masks (simulated) | Enables/disables |
| **Differential Privacy** | Formal privacy guarantee | `epsilon`, `delta`, `noise_multiplier` |
| **Cohort Suppression** | Suppress results with <100 participants | Built-in threshold |

### 📊 Analytics

| Feature | Details |
|---------|---------|
| **Federated Analytics** | Privacy-preserving count, sum, mean, quantile computation |
| **Error Tracking** | Absolute and relative error vs ground truth |
| **Multi-Metric** | Support for multiple simultaneous metrics |
| **Privacy Utility Tradeoff** | Visualize accuracy loss vs privacy gain |

### 📈 Dashboard (Streamlit)

13 interactive pages with 40+ charts:

**Core pages:**

1. **Overview** — Experiment config, status, timeline
2. **Real Data Scenario** — HDFS log-to-feature mapping narrative
3. **Results Gallery** — Links to the 42-chart generated graph gallery
4. **Federated Analytics** — Metric comparison (true vs federated vs DP-noisy)
5. **Training Progress** — Accuracy, F1, loss, ROC-AUC curves
6. **Privacy Controls** — Epsilon consumption, clipping rates, secure agg status
7. **Utility Tradeoff** — Privacy strength vs accuracy/error plots
8. **Client Participation** — Dropout rates, update norms, participation history
9. **Risk Report** — Threat model, mitigations, recommendation
10. **System Explanation** — Plain-English design walkthrough with threat model

**Enhanced Edition pages:**

11. **Benchmarking** — FedAvg vs FedProx comparison: accuracy curves, communication overhead, training time per round; run via *Run Benchmark Now* button
12. **Model Optimization** — TFLite quantization tradeoffs (int8 / float16 / dynamic): model size, accuracy drop, inference latency; run via *Run Quantization Benchmark*
13. **Inference Metrics** — Live monitoring of the federated inference coordinator (port 8001); displays demo charts when coordinator is offline

### 🚀 Deployment

| Platform | Status | Use Case |
|----------|--------|----------|
| **Local** | ✅ Python + Streamlit | Development, testing |
| **Docker Compose** | ✅ 3-service stack | Quick testing |
| **Kubernetes** | ✅ Production manifests | Cloud-scale deployment |
| **Helm** | ✅ Package manager | Easy k8s installation |

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Experiment Configuration (YAML)                             │
│ - mode: fedavg_secureagg_dp                                 │
│ - num_clients: 100                                          │
│ - dp_enabled: true                                          │
│ - epsilon: 1.0                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │ Client Data Generator    │
         │ (Synthetic/Real HDFS)    │
         └────────────┬─────────────┘
                      │
      ┌───────────────┼───────────────┐
      │               │               │
      ▼               ▼               ▼
  Client 1       Client 2    ...  Client N
  (Local Data)  (Local Data)      (Local Data)
      │               │               │
      │ Telemetry     │ Events        │ Logs
      │               │               │
      └───────────────┼───────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │ Client Registry          │
        │ (Participants + Profiles)│
        └────────────┬─────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    ┌─────────────────┐  ┌──────────────────┐
    │ Federated       │  │ Federated        │
    │ Analytics       │  │ Learning         │
    │ Engine          │  │ (FedAvg)         │
    └────────┬────────┘  └────────┬─────────┘
             │                    │
             │ Bounded            │ Local Deltas
             │ Summaries          │
             │                    │
             └────────┬───────────┘
                      │
                      ▼
      ┌──────────────────────────────────┐
      │ Privacy Layer                    │
      │ - Contribution Bounding          │
      │ - Update Clipping                │
      │ - Secure Aggregation (simulated) │
      │ - DP Noise Injection             │
      │ - Privacy Accounting             │
      └────────────┬─────────────────────┘
                   │
                   ▼
      ┌──────────────────────────────────┐
      │ Server Aggregation               │
      │ - Weighted Average               │
      │ - Cohort Suppression             │
      │ - Metrics Compilation            │
      └────────────┬─────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
  Global Model  Global Metrics Privacy Report
       │           │           │
       └───────────┼───────────┘
                   │
                   ▼
      ┌──────────────────────────────────┐
      │ Privacy & Utility Evaluator      │
      │ - Error Analysis                 │
      │ - Tradeoff Curves                │
      │ - Risk Assessment                │
      └────────────┬─────────────────────┘
                   │
                   ▼
      ┌──────────────────────────────────┐
      │ Streamlit Dashboard              │
      │ (13 pages, 40+ interactive charts) │
      └──────────────────────────────────┘
```

### Component Breakdown

#### **Backend (FastAPI)**

- **Port:** 8000 (uvicorn)
- **Routes:** 5 routers (experiment, metrics, analytics, privacy, learning)
- **Database:** SQLite (experiments, metrics, analytics results, privacy reports, client participation)
- **Middleware:** CORS enabled for cross-origin requests

#### **Clients** 

- **Telemetry Client** — Generates synthetic event telemetry (events, failures, latency)
- **EMNIST Client** — Federated MNIST image classification
- **Local Training** — TensorFlow model training on local data
- **Local Analytics** — Privacy-preserving summary computation
- **Client Registry** — Tracks registration, selection, participation

#### **Coordinator**

- **Experiment Runner** — Manages experiment lifecycle (create, initialize, run rounds)
- **Round Coordinator** — Orchestrates single round execution
- **Client Selection** — Random stratified selection with configurable client count
- **Aggregation** — Weighted averaging of updates and metrics

#### **Privacy Layer**

- **Contribution Bounding** — Caps per-client contribution (e.g., max 10 events per client)
- **Update Clipping** — L2 norm bounds on model updates
- **Secure Aggregation** — Simulates cryptographic masking/unmasking
- **DP Mechanisms** — Laplace (counts), Gaussian (sums, updates)
- **Privacy Accountant** — Tracks cumulative epsilon across rounds

#### **Dashboard (Streamlit)**

- **Port:** 8501
- **Refresh:** Auto-polling database every 2 seconds
- **Charts:** Plotly + native Streamlit plots
- **Export:** CSV reports, interactive HTML

#### **Data Layer**

- **Experiment Table** — Config, status, timestamps
- **RoundMetric Table** — Per-round accuracy, loss, F1, etc.
- **AnalyticsResult Table** — True vs federated vs DP-noisy metric values
- **PrivacyReport Table** — Privacy budget, clipping settings, risk level
- **ClientParticipation Table** — Client selection, dropout, update norms

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **API** | FastAPI | ≥0.109.0 | High-performance async web framework |
| **Web Server** | Uvicorn | ≥0.27.0 | ASGI server |
| **ML/DL** | TensorFlow | 2.15–2.20 | Model training, evaluation |
| **Numerical** | NumPy | ≥1.26.0 | Array operations |
| **Data** | Pandas | ≥2.1.0 | Data manipulation, analytics |
| **Stats** | Scikit-learn | ≥1.4.0 | Metrics (F1, ROC-AUC, etc.) |
| **Database** | SQLAlchemy | ≥2.0.25 | ORM, database abstraction |
| **Dashboard** | Streamlit | ≥1.31.0 | Interactive web UI |
| **Visualization** | Plotly | ≥5.18.0 | Interactive charts |
| **Config** | PyYAML | ≥6.0.1 | Experiment config parsing |
| **HTTP** | httpx | ≥0.26.0 | Async HTTP client |
| **Testing** | Pytest | ≥8.0.0 | Unit & integration tests |
| **Testing** | Pytest-asyncio | ≥0.23.0 | Async test support |
| **Coverage** | Pytest-cov | ≥4.1.0 | Code coverage reporting |
| **Export** | Kaleido | ≥0.2.1 | Plot export to static images |

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda
- 4+ GB RAM (for TensorFlow, optimal for 8+ GB)
- Git (for cloning repo)

### Step 1: Clone Repository

```bash
git clone <repo-url>
cd FedPrivacyLab
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize Database

```bash
python -c "from app.database import init_db; init_db()"
```

### Optional: Download Real Data

```bash
python scripts/download_real_data.py
```

---

## Quick Start

### Option A: Run API + Dashboard (Local)

**Terminal 1 - API:**
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 2 - Dashboard:**
```bash
streamlit run dashboard/streamlit_app.py
# Dashboard available at http://localhost:8501
```

### Option B: Run Demo Script

```bash
python scripts/run_demo.py
# Automatically starts API and runs sample experiment
```

### Option C: Docker Compose

```bash
docker compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### Option D: Run Comparison Study

```bash
python scripts/run_mode_comparison.py --clients 100 --rounds 5
# Compares all 5 experiment modes
# Output: data/comparisons/<timestamp>/privacy_utility_tradeoff.html
```

---

## API Overview

### Base URL
```
http://localhost:8000/api/v1
```

### Experiment Lifecycle

```bash
# 1. Start experiment
curl -X POST http://localhost:8000/api/v1/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_experiment",
    "mode": "fedavg_secureagg_dp",
    "num_clients": 100,
    "rounds": 10,
    "clients_per_round": 50,
    "local_epochs": 3,
    "dp_enabled": true,
    "epsilon": 1.0,
    "delta": 1e-5,
    "secure_agg_enabled": true,
    "clipping_norm": 1.0,
    "noise_multiplier": 0.5
  }'
# Returns: {"id": 1, "status": "initialized", ...}

# 2. Get experiment status
curl http://localhost:8000/api/v1/experiments/1

# 3. Run single round
curl -X POST http://localhost:8000/api/v1/experiments/1/run-round

# 4. Get round metrics
curl http://localhost:8000/api/v1/experiments/1/metrics?round=1

# 5. Get privacy report
curl http://localhost:8000/api/v1/experiments/1/privacy-report?round=1

# 6. Run all remaining rounds
curl -X POST http://localhost:8000/api/v1/experiments/1/run-all
```

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/experiments/start` | POST | Create and initialize experiment |
| `/experiments/{id}` | GET | Fetch experiment details |
| `/experiments/{id}/run-round` | POST | Execute single round |
| `/experiments/{id}/run-all` | POST | Execute all rounds |
| `/experiments/{id}/metrics` | GET | Get round metrics |
| `/experiments/{id}/privacy-report` | GET | Get privacy report |
| `/analytics/results/{id}` | GET | Get analytics results |
| `/privacy/risk-report/{id}` | GET | Get risk assessment |

---

## Experiment Modes

FedPrivacyLab supports 5 experiment modes demonstrating privacy-utility tradeoffs:

| Mode | Centralized? | Updates Visible? | Noise? | Use Case |
|------|--------------|------------------|--------|----------|
| **centralized_baseline** | ✅ Yes | ✅ Yes | ❌ None | Privacy baseline (no privacy) |
| **federated_analytics** | ❌ No | ⚠️ Partial | ⚠️ Optional | Analytics without raw data |
| **fedavg** | ❌ No | ✅ Yes | ❌ None | Federated learning baseline |
| **fedavg_secureagg** | ❌ No | ❌ No | ❌ None | With secure aggregation |
| **fedavg_secureagg_dp** | ❌ No | ❌ No | ✅ Yes | Full privacy stack |

### Configuration Example

```yaml
# config/experiment_config.yaml
experiment:
  name: "fedavg_dp_demo"
  mode: "fedavg_secureagg_dp"
  dataset: "emnist"
  num_clients: 100
  rounds: 10
  clients_per_round: 50
  local_epochs: 3
  dropout_rate: 0.1

privacy:
  dp_enabled: true
  epsilon: 1.0
  delta: 1e-5
  noise_multiplier: 0.5
  clipping_norm: 1.0
  contribution_bound: 10

learning:
  model_type: "cnn"
  batch_size: 32
  optimizer: "adam"
  loss_fn: "sparse_categorical_crossentropy"
```

---

## Privacy Mechanisms

### 1. Contribution Bounding

Limits per-client contribution per round (e.g., max 10 events per client).

**Effect:** Prevents a single client from dominating aggregation.

```python
# Code
clipped_value = min(client_value, contribution_bound)
```

**Privacy Impact:** Limits information leakage per client per round.

### 2. Update Clipping

Caps L2 norm of model updates.

**Effect:** Bounds update magnitude.

**Configuration:**
```
clipping_norm: 1.0  # Max L2 norm
```

### 3. Secure Aggregation (Simulated)

Simulates cryptographic masking—server cannot see individual updates before aggregation.

**How It Works:**
1. Client i generates random mask M_i
2. Client sends (update + M_i) to server
3. Server sums all (update + M_i)
4. Masks cancel: Σ(update_i + M_i) = Σ update_i + (Σ M_i)
5. Coordinator removes Σ M_i to recover true aggregate

**Privacy Impact:** Prevents server from correlating updates to clients.

### 4. Differential Privacy

Injects calibrated noise for formal privacy guarantee.

**Mechanisms:**
- **Laplace** — For counts (sensitivity = 1)
- **Gaussian** — For sums and updates (higher sensitivity)

**Parameters:**
```
epsilon: 1.0       # Privacy budget (smaller = more private)
delta: 1e-5        # Failure probability
noise_multiplier:  # Gaussian noise scale
```

**Privacy-Utility Tradeoff:**

| Epsilon | Privacy | Accuracy Loss |
|---------|---------|---------------|
| 0.1 | Strongest | ~15-20% |
| 1.0 | Strong | ~5-8% |
| 10.0 | Moderate | ~1-2% |
| ∞ | None (no privacy) | 0% |

### 5. Cohort Suppression

Suppresses results when <100 clients participate (prevents small-group deanonymization).

---

## Dashboard Features

### Overview Tab

- Experiment name, mode, parameters
- Client count, round progress, elapsed time
- Status indicator (initialized/running/completed)

### Analytics Tab

- **Chart Type:** Line plot with 3 series:
  - Ground truth (centralized)
  - Federated (no DP)
  - DP-noisy (with differential privacy)
- **Metrics:** Count, sum, mean, quantile (configurable)
- **X-axis:** Round number
- **Y-axis:** Metric value

### Training Progress Tab

- **Metrics Tracked:** Accuracy, F1, loss, ROC-AUC, precision, recall
- **Chart Type:** Multi-line plot
- **Interpretation:** Shows learning convergence across rounds

### Privacy Controls Tab

- **Epsilon Consumption:** Bar chart showing cumulative epsilon
- **Clipping Rate:** % of updates clipped per round
- **Secure Agg Status:** On/Off toggle
- **Contribution Bound:** Currently applied bound
- **Noise Multiplier:** Gaussian noise scale

### Utility Tradeoff Tab

- **X-axis:** Epsilon (privacy strength, 0 = most private)
- **Y-axis:** Error (accuracy loss vs ground truth)
- **Series:** One line per metric
- **Interpretation:** Shows how privacy impacts utility

### Client Participation Tab

- **Dropout Rate:** % of selected clients that dropped out
- **Update Norms:** Distribution of L2 norms before/after clipping
- **Participation History:** Client availability across rounds
- **Clipping Events:** How many clients hit clipping threshold per round

### Risk Report Tab

- **Threat Model:** Adversaries (model inversion, membership inference, etc.)
- **Mitigations:** How each privacy mechanism addresses threats
- **Risk Level:** Low/Medium/High based on epsilon and mechanism combo
- **Recommendations:** Suggestions for configuration adjustment

### System Explanation Tab

- **Plain-English Walthrough:** How federated learning works
- **Privacy Stack:** Visual representation of privacy layers
- **Data Flow:** Shows data movement at each stage
- **Privacy Loss:** How epsilon is consumed per round

---

## Deployment

### Local Development

```bash
# Install
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload --port 8000

# Run Dashboard (separate terminal)
streamlit run dashboard/streamlit_app.py --server.port 8501
```

### Docker Compose

```bash
# Build and run all services
docker compose up --build

# Services available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Swagger docs: http://localhost:8000/docs
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Port forward for local access
kubectl port-forward svc/fedprivacylab-api 8000:8000
kubectl port-forward svc/fedprivacylab-dashboard 8501:8501

# View status
kubectl get all -n fedprivacylab
kubectl describe pod <pod-name> -n fedprivacylab
kubectl logs <pod-name> -n fedprivacylab
```

### Helm

```bash
# Install
helm install fedprivacylab ./helm

# Upgrade
helm upgrade fedprivacylab ./helm

# Uninstall
helm uninstall fedprivacylab

# Check status
helm status fedprivacylab
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Suites

| Test File | Coverage | Count |
|-----------|----------|-------|
| `test_fedavg.py` | FedAvg aggregation | 4 tests |
| `test_contribution_bounding.py` | Analytics bounding | 3 tests |
| `test_update_clipping.py` | Model update clipping | 4 tests |
| `test_secure_aggregation.py` | Mask cancellation | 3 tests |
| `test_dp_mechanisms.py` | Noise injection | 4 tests |
| `test_non_iid_partition.py` | Data heterogeneity | 3 tests |
| `test_integration.py` | Coordinator + DB | 5 tests |
| `test_e2e_api.py` | Full API lifecycle | 6 tests |

### Code Coverage

```bash
pytest tests/ --cov=app --cov-report=html
# Report saved to: htmlcov/index.html
```

---

## Project Structure

```
FedPrivacyLab/
├── app/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── database.py               # SQLAlchemy setup
│   ├── models.py                 # ORM models
│   ├── schemas.py                # Pydantic request/response models
│   ├── config_loader.py          # Config file parsing
│   │
│   ├── api/                      # API route handlers
│   │   ├── experiment_routes.py  # Experiment lifecycle
│   │   ├── metrics_routes.py     # Metrics retrieval
│   │   ├── analytics_routes.py   # Analytics computation
│   │   ├── privacy_routes.py     # Privacy reports
│   │   └── learning_routes.py    # Training endpoints
│   │
│   ├── clients/                  # Client implementations
│   │   ├── client.py             # Base client class
│   │   ├── telemetry_client.py   # Event/telemetry client
│   │   ├── emnist_client.py      # Image classification client
│   │   ├── local_training.py     # Local model training
│   │   ├── local_analytics.py    # Local metric computation
│   │   └── __init__.py
│   │
│   ├── coordinator/              # Server-side coordination
│   │   ├── experiment_runner.py  # Experiment lifecycle mgmt
│   │   ├── round_coordinator.py  # Round execution
│   │   ├── client_registry.py    # Client tracking
│   │   └── __init__.py
│   │
│   ├── learning/                 # ML/DL components
│   │   ├── fedavg.py             # FedAvg algorithm
│   │   ├── model.py              # Model definitions
│   │   ├── evaluator.py          # Metrics computation
│   │   ├── feature_encoding.py   # Feature preprocessing
│   │   ├── centralized_baseline.py
│   │   └── __init__.py
│   │
│   ├── privacy/                  # Privacy mechanisms
│   │   ├── privacy_accountant.py # Epsilon tracking
│   │   ├── dp_mechanisms.py      # Laplace & Gaussian noise
│   │   ├── secure_aggregation.py # Mask simulation
│   │   ├── update_clipping.py    # L2 norm clipping
│   │   ├── contribution_bounding.py
│   │   ├── risk_report.py        # Threat model
│   │   └── __init__.py
│   │
│   ├── analytics/                # Analytics engine
│   │   ├── federated_analytics.py # Main analytics logic
│   │   ├── local_summary.py      # Per-client summaries
│   │   ├── metric_types.py       # Metric definitions
│   │   ├── histogram.py          # Distribution tracking
│   │   ├── quantiles.py          # Quantile computation
│   │   └── __init__.py
│   │
│   ├── evaluation/               # Privacy-utility evaluation
│   │   ├── privacy_utility_evaluator.py
│   │   └── __init__.py
│   │
│   └── simulation/               # Data & scenario simulation
│       ├── synthetic_telemetry.py
│       ├── real_telemetry_loader.py
│       ├── non_iid_partition.py
│       ├── dropout_simulator.py
│       ├── attack_simulator.py
│       └── __init__.py
│
├── dashboard/                     # Streamlit UI
│   ├── streamlit_app.py          # Main dashboard app
│   └── components/               # Reusable chart components
│
├── config/                        # Experiment configs
│   ├── experiment_config.yaml
│   ├── privacy_config.yaml
│   └── model_config.yaml
│
├── scripts/                       # Utility scripts
│   ├── run_demo.py               # Demo runner
│   ├── run_mode_comparison.py    # Compare all modes
│   ├── download_real_data.py     # Fetch HDFS logs
│   └── export_results.py         # Export to CSV/JSON
│
├── tests/                         # Test suite
│   ├── test_fedavg.py
│   ├── test_contribution_bounding.py
│   ├── test_update_clipping.py
│   ├── test_secure_aggregation.py
│   ├── test_dp_mechanisms.py
│   ├── test_non_iid_partition.py
│   ├── test_integration.py
│   └── test_e2e_api.py
│
├── docker/                        # Docker configs
│   ├── Dockerfile
│   ├── dockerfile.dashboard
│   └── docker-compose.yml
│
├── k8s/                           # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── pvc.yaml
│
├── helm/                          # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│
├── data/                          # Runtime data
│   └── *.db                       # SQLite databases
│
├── docs/                          # Documentation
│   ├── COMPLETE_GUIDE.md
│   ├── FedPrivacyLab_Complete_Guide.docx
│   └── ...
│
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── docker-compose.inference.yml
├── Dockerfile
├── README.md                      # This file
└── SPEC_COMPLIANCE.md            # Spec audit report
```

---

## Performance & Benchmarks

### Baseline Configuration

- **Clients:** 100
- **Rounds:** 5
- **Clients Per Round:** 50
- **Local Epochs:** 3
- **Model:** EMNIST CNN
- **Hardware:** 8-core CPU, 16GB RAM (no GPU)

### Benchmark Results

| Mode | Accuracy | F1 Score | Training Time | Bytes Transferred | Privacy (ε) |
|------|----------|----------|---------------|-------------------|------------|
| centralized_baseline | 99.7% | 0.979 | 45s | 2.1MB | ∞ (none) |
| federated_analytics | 99.5% | 0.974 | 42s | 1.8MB | ∞ (none) |
| fedavg | 98.2% | 0.931 | 120s | 5.3MB | ∞ (none) |
| fedavg_secureagg | 98.1% | 0.929 | 125s | 5.8MB | ∞ (none) |
| fedavg_secureagg_dp | 94.3% | 0.821 | 128s | 5.9MB | 1.0 |

**Key Insights:**

- Centralized baseline = fastest (no communication overhead)
- Federated learning = slower due to round-trip communication
- Secure aggregation = 5-10s overhead (mask generation/cancellation)
- Differential privacy = ~5% accuracy loss with ε=1.0 (strong privacy)

### Throughput

- **Requests/sec:** 50-100 (API with 100 concurrent clients)
- **Latency:** 10-50ms per API call (median)
- **Dashboard refresh rate:** 2-second polling

---

## Real-World Data

FedPrivacyLab supports **HDFS logs from LogHub**—real-world distributed system telemetry.

### Features

- **Data Source:** [LogPAI LogHub](https://github.com/logpai/loghub) HDFS
- **Log Types:** System events, failures, performance metrics
- **Federated Mapping:** Each HDFS cluster component = 1 client
- **Attributes:** Timestamp, component, log level, message

### Usage

```bash
# Download real data
python scripts/download_real_data.py

# Run experiment with real data
curl -X POST http://localhost:8000/api/v1/experiments/start \
  -d '{
    "name": "hdfs_experiment",
    "dataset": "real_hdfs_loghub",
    "mode": "fedavg_secureagg_dp",
    ...
  }'
```

### Privacy Implications

Real HDFS data may contain sensitive information (error messages, performance anomalies). FedPrivacyLab ensures:

- Raw logs never leave client nodes
- Only aggregated metrics sent to server
- Differential privacy adds formal privacy guarantee
- Contribution bounding prevents outlier dominance

---

## Troubleshooting

### Issue: "TensorFlow not found"

```bash
# Solution: Install TensorFlow
pip install tensorflow>=2.15.0,<2.21.0

# For GPU (CUDA):
pip install tensorflow[and-cuda]>=2.15.0,<2.21.0
```

### Issue: "Port 8000 already in use"

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

### Issue: "Database locked"

```bash
# SQLite concurrent access issue
# Solution: Restart API server
# Or use PostgreSQL for production
```

### Issue: "Out of memory during training"

```bash
# Reduce batch size or client count
curl -X POST .../experiments/start -d '{
  "num_clients": 50,
  "clients_per_round": 20,
  "local_epochs": 1,
  ...
}'
```

### Issue: "Dashboard slow or unresponsive"

```bash
# Increase polling interval in dashboard code
# Or reduce number of plotly charts
# Or deploy on server with more resources
```

---

## References

### Academic Papers

- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.03832)
- [Differentially Private Federated Learning](https://arxiv.org/abs/1911.00222)
- [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1610.02055)
- [Practical Secure Aggregation for Federated Learning on User-Held Data](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/)

### Frameworks & Tools

- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [Flower (Federated Learning Framework)](https://flower.ai/)
- [LEAF Benchmark](https://leaf.cmu.edu/)
- [Apple Federated Learning & Privacy](https://machinelearning.apple.com/research/fed-learning-diff-privacy)

### Datasets

- [LogPAI LogHub](https://github.com/logpai/loghub) — Production logs
- [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset) — Handwritten characters
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — Small images

---

## License

MIT License — Research and educational use permitted.

---

## Citation

If you use FedPrivacyLab in research, please cite:

```bibtex
@software{fedprivacylab2024,
  title={FedPrivacyLab: Federated Analytics and Privacy-Preserving ML Workbench},
  author={Siddani, Kaushik},
  year={2024},
  url={https://github.com/kaushiksiddani/FedPrivacyLab}
}
```

---

**For more information, see [ARCHITECTURE.md](ARCHITECTURE.md), [API_DOCUMENTATION.md](API_DOCUMENTATION.md), and [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md).**
