import sys; sys.path.insert(0, '.')
from fedprivacylab.inference.privacy_accounting import PrivacyBudgetManager, infer_privacy_cost

cost = infer_privacy_cost(10)
print(f"Cost for 10 inferences: {cost:.6f}")

mgr = PrivacyBudgetManager(total_epsilon=1.0, total_delta=1e-5)
mgr.allocate_budget('client_1')
r1 = mgr.log_inference_query('client_1', 10)
r2 = mgr.log_inference_query('client_1', 50)
print(f"After 2 batches: remaining={r2['epsilon_remaining']:.6f}, accepted={r2['accepted']}")
print(f"Budget state: {mgr.get_client_budget('client_1')}")
print(f"Exceeded: {mgr.check_epsilon_exceeded('client_1')}")
print(f"Summary: {mgr.summary()}")
