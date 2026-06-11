FROM python:3.9-slim

WORKDIR /workspace

# System deps for numpy/scipy (needed by TF's transitive deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the SDK and the test script
COPY fedprivacylab/client_sdk.py /workspace/fedprivacylab/client_sdk.py
COPY fedprivacylab/__init__.py   /workspace/fedprivacylab/__init__.py
COPY docker/inference_test_script.py /workspace/inference_test_script.py

# Create model cache directory
RUN mkdir -p /root/.fedprivacylab/model_cache

ENV PYTHONPATH=/workspace
ENV COORDINATOR_URL=http://inference-coordinator:8001
ENV CLIENT_ID=client_01
ENV NUM_PREDICTIONS_PER_BATCH=5
ENV BATCH_INTERVAL_SECONDS=10
ENV FEATURE_DIM=8
ENV FEDCLIENT_CACHE_DIR=/root/.fedprivacylab/model_cache

ENTRYPOINT ["python", "/workspace/inference_test_script.py"]
