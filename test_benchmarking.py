import sys; sys.path.insert(0, '.')
from fedprivacylab.benchmarking.suite import benchmark_fedavg, benchmark_fedprox, benchmark_comparison
import os

print("Running FedAvg (3 rounds, 2 clients)...")
fa = benchmark_fedavg(num_rounds=3, num_clients=2)
for r in fa:
    print(f"  Round {r.round}: acc={r.accuracy:.4f} f1={r.f1_score:.4f} t={r.training_time_seconds:.2f}s eps={r.epsilon:.4f}")

print("Running FedProx (3 rounds, 2 clients, mu=0.01)...")
fp = benchmark_fedprox(num_rounds=3, num_clients=2, mu=0.01)
for r in fp:
    print(f"  Round {r.round}: acc={r.accuracy:.4f} f1={r.f1_score:.4f} t={r.training_time_seconds:.2f}s eps={r.epsilon:.4f}")

print("Running comparison + saving CSV...")
csv = benchmark_comparison(num_rounds=3, num_clients=2)
print(f"CSV saved: {csv}")
import pandas as pd
df = pd.read_csv(csv)
print(f"CSV rows: {len(df)}, columns: {list(df.columns)}")
