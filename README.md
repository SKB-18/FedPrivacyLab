# FedPrivacyLab

**Federated Analytics and Privacy-Preserving ML Workbench**

Graduate-level research project simulating decentralized clients that run aggregate analytics and federated model training **without centralizing raw telemetry**. Implements FedAvg, client selection, dropout simulation, update clipping, differential privacy noise, secure aggregation simulation, and Streamlit dashboards for privacy–utility tradeoff analysis.

**Full documentation:** [docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md) — architecture, datasets, privacy mechanisms, API, graphs, results, and testing (38 tests).  
**Word version:** [docs/FedPrivacyLab_Complete_Guide.docx](docs/FedPrivacyLab_Complete_Guide.docx)

## Problem Statement

Organizations need useful aggregate metrics and ML models from distributed user data, but collecting raw logs centrally creates privacy risk. Federated learning keeps data local, yet **FL alone is not automatically private**—individual updates can still leak information.

FedPrivacyLab demonstrates the full engineering stack: federated analytics (privacy-preserving measurement), federated learning (FedAvg), and layered privacy controls (bounding, clipping, secure aggregation simulation, DP noise).

## Real-Life Data (LogHub HDFS)

Production **Hadoop HDFS logs** from [LogPAI LogHub](https://github.com/logpai/loghub) are supported (`dataset: real_hdfs_loghub`). Each HDFS component is a federated client; events map to latency, failure, and engagement fields. See `SPEC_COMPLIANCE.md` for the full spec audit.

```bash
python scripts/download_real_data.py
```

## Architecture

```
Experiment Config (YAML)
        │
        ▼
Client Data Generator ──► Synthetic Telemetry Clients
        │
        ▼
Client Registry
        │
        ▼
Round Coordinator
   ├── Federated Analytics Engine ──► Bounded Local Summaries
   └── Federated Learning Engine ──► Local Training
              │
              ▼
   Contribution Bounding / Update Clipping
              │
              ▼
   Secure Aggregation Simulator
              │
              ▼
   Differential Privacy Engine
              │
              ▼
   Server Aggregator ──► Global Metrics / Global Model
              │
              ▼
   Privacy & Utility Evaluator ──► Streamlit Dashboard
```

## Experiment Modes

| Mode | Raw Data Centralized? | Individual Updates Visible? | DP Noise? |
|------|----------------------|------------------------------|-----------|
| `centralized_baseline` | Yes | N/A | No |
| `federated_analytics` | No | Local summaries may be visible | Optional |
| `fedavg` | No | Yes | No |
| `fedavg_secureagg` | No | No | No |
| `fedavg_secureagg_dp` | No | No | Yes |

## Privacy Mechanisms

1. **Contribution bounding** — caps events, failures, and latency sums per client per round  
2. **Update clipping** — L2 norm bound on model updates  
3. **Secure aggregation (simulated)** — additive masks cancel in aggregate  
4. **Differential privacy** — Laplace noise on counts; Gaussian noise on sums/updates  
5. **Cohort suppression** — suppresses reports when &lt; 100 clients participate  

## Enhanced Edition (`fedprivacylab/` package)

Three additional capabilities beyond the core spec:

| Feature | How to use |
|---------|-----------|
| **FedAvg vs FedProx benchmarking** | Click *Run Benchmark Now* on the Benchmarking page, or `python -m fedprivacylab.benchmarking.suite` |
| **TFLite model quantization** | Click *Run Quantization Benchmark* on the Model Optimization page (int8 / float16 / dynamic) |
| **Federated inference coordinator** | `uvicorn fedprivacylab.inference.coordinator:app --port 8001`; monitor live in the Inference Metrics page |

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Install

```bash
cd FedPrivacyLab
pip install -r requirements.txt
```

### Run API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Run Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

### Start an Experiment

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/start \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"demo\",\"mode\":\"fedavg_secureagg_dp\",\"num_clients\":100,\"rounds\":5,\"clients_per_round\":40,\"local_epochs\":2,\"dp_enabled\":true,\"secure_agg_enabled\":true}"

curl -X POST http://127.0.0.1:8000/api/v1/experiments/1/run-round
curl http://127.0.0.1:8000/api/v1/experiments/1/metrics
curl http://127.0.0.1:8000/api/v1/experiments/1/privacy-report
```

Or run all rounds:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/1/run-all
```

### Demo Script

```bash
python scripts/run_demo.py
```

### Compare All Experiment Modes (recommended)

Runs all five modes with the same seed and exports CSV + interactive HTML charts:

```bash
python scripts/run_mode_comparison.py --clients 150 --rounds 4
```

Outputs land in `data/comparisons/<timestamp>/` including `privacy_utility_tradeoff.html`.

### Docker

```bash
docker compose up --build
```

API: http://localhost:8000 — Dashboard: http://localhost:8501

## Testing

```bash
pytest tests/ -v
```

| Suite | Coverage |
|-------|----------|
| `test_fedavg.py` | FedAvg unit tests |
| `test_contribution_bounding.py` | Analytics bounding |
| `test_update_clipping.py` | L2 clipping |
| `test_secure_aggregation.py` | Mask cancellation |
| `test_dp_mechanisms.py` | DP noise |
| `test_non_iid_partition.py` | Non-IID profiles |
| `test_integration.py` | Coordinator + DB |
| `test_e2e_api.py` | Full API lifecycle |

## Dashboard Pages (13)

**Core**

1. **Overview** — experiment config and status  
2. **Real Data Scenario** — HDFS log-to-feature mapping narrative  
3. **Results Gallery** — links to the 42-chart generated graph gallery  
4. **Federated Analytics** — true vs federated vs DP-noisy metrics  
5. **Training Progress** — accuracy, ROC-AUC, loss, F1 by round  
6. **Privacy Controls** — epsilon, clipping, secure agg settings  
7. **Utility Tradeoff** — privacy strength vs accuracy/error  
8. **Client Participation** — dropout, update norms, clipping rate  
9. **Risk Report** — threat model and mitigations  
10. **System Explanation** — plain-English design walkthrough  

**Enhanced Edition**

11. **Benchmarking** — FedAvg vs FedProx: accuracy curves, communication overhead, training time; click *Run Benchmark Now* or generate via `python -m fedprivacylab.benchmarking.suite`  
12. **Model Optimization** — TFLite quantization (int8 / float16 / dynamic): size reduction, accuracy drop, latency; click *Run Quantization Benchmark*  
13. **Inference Metrics** — live monitoring of the federated inference coordinator (port 8001); shows demo charts when coordinator is offline; start with `uvicorn fedprivacylab.inference.coordinator:app --port 8001`

## Project Structure

```
FedPrivacyLab/
  app/              # FastAPI backend, coordinator, clients, privacy, learning
  config/           # experiment_config.yaml, privacy_config.yaml
  dashboard/        # Streamlit visual analytics (13 pages)
  fedprivacylab/    # Enhanced Edition package (benchmarking, inference, quantization)
  tests/            # unit, integration, e2e (38 tests)
  scripts/          # workflow scripts
  docs/             # complete guide (MD + DOCX)
  data/             # SQLite DB, graphs, comparisons (created at runtime, gitignored)
```

## Limitations

- Secure aggregation is **simulated**, not production cryptography  
- Privacy accounting uses simplified composition  
- Synthetic telemetry may not reflect real user behavior  
- Demonstrates design patterns, not production-scale guarantees  

## Interview Explanation (30 seconds)

> FedPrivacyLab is a federated analytics and federated learning workbench. Clients keep raw telemetry local. In analytics mode, they send bounded summaries; in learning mode, they train locally and send clipped model updates. I implemented FedAvg, client selection, dropout simulation, secure aggregation as a concept, and DP noise—with a dashboard comparing privacy strength against utility loss.

## References

- [Apple FL with DP](https://machinelearning.apple.com/research/fed-learning-diff-privacy)
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [Flower Framework](https://flower.ai/docs/framework/index.html)
- [LEAF Benchmark](https://leaf.cmu.edu/)
- [Google Practical Secure Aggregation](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/)

## License

MIT — research and educational use.
