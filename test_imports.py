import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
results = []
imports = [
    ("fedprivacylab.datasets", ["load_dataset", "load_mimic_subset"]),
    ("fedprivacylab.benchmarking.suite", ["benchmark_fedavg", "benchmark_fedprox", "benchmark_comparison"]),
    ("fedprivacylab.quantization", ["quantize_model", "evaluate_quantized_model", "get_model_size"]),
    ("fedprivacylab.inference.coordinator", ["app"]),
    ("fedprivacylab.inference.privacy_accounting", ["PrivacyBudgetManager", "infer_privacy_cost"]),
    ("fedprivacylab.client_sdk", ["FederatedClient"]),
]
for mod, names in imports:
    try:
        m = __import__(mod, fromlist=names)
        for n in names:
            getattr(m, n)
        results.append(f"IMPORT_SUCCESS: {mod}")
        print(f"OK: {mod}")
    except Exception as e:
        results.append(f"IMPORT_FAILED: {mod} -- {e}")
        print(f"FAIL: {mod} -- {e}")
print("\n".join(results))
