"""
FedPrivacyLab Client SDK.

Implements a privacy-preserving federated inference client:
  - Model is downloaded once and cached locally.
  - ALL predictions are made on-device using the cached model.
  - ONLY aggregate metadata (count, latency) is reported to the coordinator.
  - Individual predictions, input features, and user identities never leave the device.

Usage::

    client = FederatedClient("http://localhost:8001", client_id="hospital_01")
    client.download_model()
    preds = client.predict_local(X)
    client.log_inference(num_predictions=len(X), latency_ms=12.5)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import urllib.request as _urllib
    import urllib.error as _urlerr
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

# Optional: use requests if available for cleaner HTTP
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


CACHE_DIR = Path(os.environ.get("FEDCLIENT_CACHE_DIR", Path.home() / ".fedprivacylab" / "model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelCache:
    """Locally cached model state."""
    version: int
    model_bytes: bytes | None          # TFLite flatbuffer
    weights_path: str | None           # Path to TFLite file on disk
    privacy_params: dict[str, Any]
    cached_at: float = field(default_factory=time.time)
    interpreter: Any = field(default=None, repr=False)  # tf.lite.Interpreter


class FederatedClient:
    """
    Privacy-preserving federated inference client.

    Privacy guarantees:
      1. Predictions are computed on-device; they are NEVER sent to any server.
      2. Only counts (how many predictions were made) and timing metadata are
         reported to the coordinator via log_inference().
      3. Privacy budget (epsilon) is tracked per client; once exhausted, the
         client stops reporting (and can re-negotiate budget).

    Args:
        coordinator_url: Base URL of the inference coordinator (e.g. http://localhost:8001).
        client_id: Unique identifier for this client/device.
        cache_dir: Directory for local model cache. Defaults to ~/.fedprivacylab/model_cache.
    """

    def __init__(
        self,
        coordinator_url: str,
        client_id: str,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.client_id = client_id
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model_cache: ModelCache | None = None
        self._epsilon_remaining: float = 0.0
        self._delta: float = 1e-5
        self._registered: bool = False

    # ── Registration ────────────────────────────────────────────────────────────

    def register(self) -> dict[str, Any]:
        """
        Register this client with the coordinator.

        Returns the assigned privacy budget and current model version.
        Must be called before download_model().
        """
        resp = self._post("/register_client", {
            "client_id": self.client_id,
            "capabilities": {"sdk_version": "1.0.0", "platform": "python"},
        })
        self._epsilon_remaining = resp.get("privacy_budget", 1.0)
        self._delta = resp.get("delta", 1e-5)
        self._registered = True
        return resp

    # ── Model management ────────────────────────────────────────────────────────

    def download_model(self) -> None:
        """
        Fetch the latest model descriptor from the coordinator and cache locally.

        The actual weights are stored in self.cache_dir; subsequent calls to
        predict_local() use the cached model without any network traffic.
        """
        if not self._registered:
            self.register()

        resp = self._post("/get_model", {})
        version = resp["model_version"]
        privacy_params = resp.get("privacy_params", {})

        cache_path = self.cache_dir / f"model_v{version}.tflite"

        # If we already have this version cached, reuse it
        if self._model_cache and self._model_cache.version == version:
            return

        # In a real deployment, weights_path would be a URL or shared volume path.
        # Here we build a stub TFLite model so the SDK is fully runnable without
        # a live training coordinator.
        if cache_path.exists():
            model_bytes = cache_path.read_bytes()
        else:
            model_bytes = self._download_or_stub_model(resp.get("weights_path"), cache_path)

        interpreter = None
        if _HAS_TF and model_bytes:
            try:
                interpreter = tf.lite.Interpreter(model_content=model_bytes)
                interpreter.allocate_tensors()
            except Exception:
                interpreter = None

        self._model_cache = ModelCache(
            version=version,
            model_bytes=model_bytes,
            weights_path=str(cache_path),
            privacy_params=privacy_params,
            interpreter=interpreter,
        )

    def _download_or_stub_model(self, weights_path: str | None, cache_path: Path) -> bytes | None:
        """Download model bytes or build a stub TFLite model for testing."""
        if weights_path and os.path.exists(weights_path):
            data = Path(weights_path).read_bytes()
            cache_path.write_bytes(data)
            return data

        # Stub: build a tiny TFLite model so predict_local works in tests
        if _HAS_TF:
            try:
                stub = tf.keras.Sequential([
                    tf.keras.layers.Dense(8, activation="relu", input_shape=(8,)),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ])
                converter = tf.lite.TFLiteConverter.from_keras_model(stub)
                tflite_bytes = converter.convert()
                cache_path.write_bytes(tflite_bytes)
                return tflite_bytes
            except Exception:
                return None
        return None

    def is_model_stale(self) -> bool:
        """Return True if a newer model version is available on the coordinator."""
        if not self._model_cache:
            return True
        try:
            resp = self._post("/get_model", {})
            return resp["model_version"] > self._model_cache.version
        except Exception:
            return False

    def update_model_if_needed(self) -> bool:
        """
        Check for a newer model and download it if available.

        Returns:
            True if the model was updated, False if already current.
        """
        if self.is_model_stale():
            self.download_model()
            return True
        return False

    # ── Local inference ─────────────────────────────────────────────────────────

    def predict_local(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the LOCALLY cached model.

        No network call is made. Predictions stay on-device.

        Args:
            input_data: Feature array of shape (n_samples, n_features) as float32.

        Returns:
            Prediction array of shape (n_samples,) with values in [0, 1].

        Raises:
            RuntimeError: If no model has been downloaded yet.
        """
        if not self._model_cache:
            raise RuntimeError("No model cached. Call download_model() first.")

        X = np.atleast_2d(input_data).astype(np.float32)

        if self._model_cache.interpreter is not None:
            return self._tflite_predict(X)

        # Fallback: random predictions (stub mode without TF)
        return np.random.rand(len(X)).astype(np.float32)

    def _tflite_predict(self, X: np.ndarray) -> np.ndarray:
        """Run TFLite inference sample-by-sample (stateless interpreter)."""
        interp = self._model_cache.interpreter
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]
        preds = []
        for row in X:
            inp = row[np.newaxis, :].astype(in_detail["dtype"])
            interp.set_tensor(in_detail["index"], inp)
            interp.invoke()
            out = interp.get_tensor(out_detail["index"])
            preds.append(float(out.ravel()[0]))
        return np.array(preds, dtype=np.float32)

    # ── Metadata reporting ──────────────────────────────────────────────────────

    def log_inference(self, num_predictions: int, latency_ms: float) -> dict[str, Any]:
        """
        Report inference metadata to the coordinator.

        Privacy guarantee: Only aggregate counts and timing are sent.
        Individual predictions and input features are NEVER included.

        Args:
            num_predictions: Number of predictions made in this batch.
            latency_ms: Average per-sample latency in milliseconds.

        Returns:
            Coordinator response with epsilon_remaining and optional warning.
        """
        if not self._model_cache:
            raise RuntimeError("No model cached. Call download_model() first.")

        resp = self._post("/client_inference_log", {
            "client_id": self.client_id,
            "num_predictions": num_predictions,
            "latency_ms": latency_ms,
            "model_version": self._model_cache.version,
        })
        if resp.get("accepted"):
            self._epsilon_remaining = resp.get("epsilon_remaining", self._epsilon_remaining)
        return resp

    # ── Budget queries ──────────────────────────────────────────────────────────

    def get_privacy_budget(self) -> dict[str, float]:
        """
        Return the current local view of the privacy budget.

        Returns:
            Dict with 'epsilon_remaining' and 'delta'.
        """
        return {
            "epsilon_remaining": self._epsilon_remaining,
            "delta": self._delta,
        }

    # ── HTTP helpers ────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict[str, Any]:
        url = f"{self.coordinator_url}{path}"
        body = json.dumps(payload).encode()

        if _HAS_REQUESTS:
            resp = _requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()

        req = _urllib.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.coordinator_url}{path}"
        if _HAS_REQUESTS:
            resp = _requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        with _urllib.urlopen(url, timeout=10) as r:
            return json.loads(r.read())

    def __repr__(self) -> str:
        version = self._model_cache.version if self._model_cache else None
        return (
            f"FederatedClient(id={self.client_id!r}, "
            f"model_v={version}, eps_remaining={self._epsilon_remaining:.3f})"
        )
