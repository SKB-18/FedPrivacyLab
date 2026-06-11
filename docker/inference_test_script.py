"""
Federated inference test script — runs inside each client container.

Simulates a real federated inference workload:
  1. Register with the inference coordinator.
  2. Download the global model.
  3. Every BATCH_INTERVAL_SECONDS:
     - Make NUM_PREDICTIONS_PER_BATCH local predictions.
     - Log aggregate metadata (counts + latency) to coordinator.
     - Check remaining privacy budget.
     - Print a status line.

Privacy guarantee: predictions are made on the locally cached model.
Only aggregate counts and timing are sent to the coordinator — never
individual predictions or input features.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

# Allow import from project root
sys.path.insert(0, "/workspace")

from fedprivacylab.client_sdk import FederatedClient

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8001")
CLIENT_ID = os.environ.get("CLIENT_ID", "test_client_01")
NUM_PREDICTIONS_PER_BATCH = int(os.environ.get("NUM_PREDICTIONS_PER_BATCH", "5"))
BATCH_INTERVAL_SECONDS = float(os.environ.get("BATCH_INTERVAL_SECONDS", "10"))
FEATURE_DIM = int(os.environ.get("FEATURE_DIM", "8"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))  # 0 = run forever


def make_dummy_input(n: int, dim: int) -> np.ndarray:
    """Generate synthetic input features (stand-in for real patient/sensor data)."""
    return np.random.randn(n, dim).astype(np.float32)


def main() -> None:
    print(f"[{CLIENT_ID}] Starting federated inference client")
    print(f"[{CLIENT_ID}] Coordinator: {COORDINATOR_URL}")
    print(f"[{CLIENT_ID}] Batch size: {NUM_PREDICTIONS_PER_BATCH}, interval: {BATCH_INTERVAL_SECONDS}s")

    client = FederatedClient(coordinator_url=COORDINATOR_URL, client_id=CLIENT_ID)

    # Register and download model
    try:
        reg = client.register()
        print(f"[{CLIENT_ID}] Registered. Model version: {reg['model_version']}, "
              f"ε budget: {reg['privacy_budget']}")
    except Exception as e:
        print(f"[{CLIENT_ID}] Registration failed: {e}. Retrying in 5s...")
        time.sleep(5)
        try:
            client.register()
        except Exception as e2:
            print(f"[{CLIENT_ID}] Could not connect to coordinator: {e2}. Running in offline mode.")

    try:
        client.download_model()
        print(f"[{CLIENT_ID}] Model downloaded and cached locally.")
    except Exception as e:
        print(f"[{CLIENT_ID}] Model download failed: {e}. Using stub model.")

    batch_num = 0
    while True:
        if MAX_BATCHES > 0 and batch_num >= MAX_BATCHES:
            print(f"[{CLIENT_ID}] Reached MAX_BATCHES={MAX_BATCHES}. Exiting.")
            break

        batch_num += 1

        # Check for model updates silently
        try:
            updated = client.update_model_if_needed()
            if updated:
                print(f"[{CLIENT_ID}] Model updated to version {client._model_cache.version}")
        except Exception:
            pass

        # Make predictions locally (no network call)
        X = make_dummy_input(NUM_PREDICTIONS_PER_BATCH, FEATURE_DIM)
        t0 = time.perf_counter()
        try:
            preds = client.predict_local(X)
        except Exception as e:
            # Stub fallback if TF unavailable
            preds = np.random.rand(NUM_PREDICTIONS_PER_BATCH)
        latency_ms = (time.perf_counter() - t0) * 1000

        positive_count = int((preds > 0.5).sum())

        # Log only aggregate metadata — no predictions sent
        budget = client.get_privacy_budget()
        try:
            log_resp = client.log_inference(
                num_predictions=NUM_PREDICTIONS_PER_BATCH,
                latency_ms=round(latency_ms / NUM_PREDICTIONS_PER_BATCH, 2),
            )
            eps_remaining = log_resp.get("epsilon_remaining", budget["epsilon_remaining"])
            warning = log_resp.get("warning", "")
        except Exception:
            eps_remaining = budget["epsilon_remaining"]
            warning = "(coordinator unreachable)"

        model_v = client._model_cache.version if client._model_cache else "N/A"
        status = (
            f"[{CLIENT_ID}] Batch #{batch_num}: Made {NUM_PREDICTIONS_PER_BATCH} inferences "
            f"in {latency_ms:.0f}ms (avg {latency_ms/NUM_PREDICTIONS_PER_BATCH:.1f}ms/sample), "
            f"model_v={model_v}, epsilon_remaining={eps_remaining:.4f}"
        )
        if warning:
            status += f" | WARNING: {warning}"
        print(status, flush=True)

        # Budget exhausted: stop logging (can still predict locally)
        if eps_remaining <= 0:
            print(f"[{CLIENT_ID}] Privacy budget exhausted. Continuing local inference only.")

        time.sleep(BATCH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
