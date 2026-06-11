"""
FedPrivacyLab Extended API router.

Adds dataset discovery endpoints on top of the existing app/ API.
Mount this router in app/main.py or run as a standalone FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from fedprivacylab.datasets import load_dataset, dataset_summary, FederatedDatasetInfo

router = APIRouter(prefix="/api/v1/fedprivacylab", tags=["fedprivacylab-extended"])

AVAILABLE_DATASETS = {
    "mimic": {
        "name": "MIMIC-III Mortality Prediction",
        "description": (
            "Synthetic ICU mortality dataset mirroring MIMIC-III schema. "
            "5000+ admissions split across 10 hospital clients (non-IID by severity)."
        ),
        "default_clients": 10,
        "num_classes": 2,
        "feature_dim": 20,
        "task": "binary_classification",
    },
    "cifar10": {
        "name": "CIFAR-10 Image Classification",
        "description": (
            "CIFAR-10 (60k images, 10 classes) split across 20 clients with "
            "2-3 classes each — severely non-IID."
        ),
        "default_clients": 20,
        "num_classes": 10,
        "feature_dim": 3072,
        "task": "multiclass_classification",
    },
}


@router.get("/datasets")
async def list_datasets() -> dict:
    """List all available federated datasets with metadata."""
    return {
        "datasets": [
            {"dataset_id": k, **v}
            for k, v in AVAILABLE_DATASETS.items()
        ]
    }


@router.get("/dataset-info/{dataset_name}")
async def get_dataset_info(dataset_name: str) -> dict:
    """
    Return detailed statistics for a federated dataset.

    Path param:
        dataset_name: 'mimic' or 'cifar10'

    Note: For CIFAR-10, TensorFlow must be installed (downloads ~170MB on first call).
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found. Available: {list(AVAILABLE_DATASETS.keys())}",
        )

    try:
        # Load a small sample to compute real statistics
        if dataset_name == "mimic":
            _, info = load_dataset(dataset_name, num_clients=10, n_records=1000)
        else:
            _, info = load_dataset(dataset_name, num_clients=5)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "dataset_id": dataset_name,
        **AVAILABLE_DATASETS[dataset_name],
        "statistics": dataset_summary(info),
    }
