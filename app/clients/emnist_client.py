"""Optional Federated EMNIST client (spec Dataset B) — lightweight subset loader."""

from app.clients.client import BaseClient

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore


class EMNISTClient(BaseClient):
    """
    Federated EMNIST-style client using TF MNIST as a stand-in subset for MVP demos.
    Full TFF EMNIST integration can replace this loader post-MVP.
    """

    _data_cache: dict[str, tuple] = {}

    def __init__(self, client_id: str, client_index: int = 0, num_clients: int = 10):
        super().__init__(client_id)
        self.client_index = client_index
        self.num_clients = num_clients

    def _load_partition(self):
        if not tf:
            raise ImportError("TensorFlow required for EMNISTClient")
        key = f"{self.num_clients}"
        if key not in EMNISTClient._data_cache:
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            n = len(x_train)
            per = n // self.num_clients
            parts = []
            for i in range(self.num_clients):
                s, e = i * per, (i + 1) * per if i < self.num_clients - 1 else n
                parts.append((x_train[s:e], y_train[s:e]))
            EMNISTClient._data_cache[key] = parts
        x, y = EMNISTClient._data_cache[key][self.client_index]
        return x, y

    def get_local_records(self) -> list[dict]:
        """Map image subset to telemetry-like records for unified pipeline."""
        x, y = self._load_partition()
        records = []
        for i in range(min(100, len(x))):
            flat = x[i].flatten()
            latency = float(flat.mean())
            failure = int(y[i] % 2)
            records.append(
                {
                    "feature": "emnist_digit",
                    "latency_ms": latency,
                    "failure": failure,
                    "clicked_recommendation": int((y[i] > 4)),
                    "session_length_sec": 120.0,
                    "device_tier": "mid",
                    "network_type": "wifi",
                    "app_version": "1.3",
                    "locale": "en_US",
                    "locale_bucket": "en_US",
                    "label_poor_experience": int(y[i] >= 5),
                }
            )
        return records

    def num_examples(self) -> int:
        return len(self.get_local_records())
