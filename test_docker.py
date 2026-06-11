import yaml, sys

print("=== docker-compose.inference.yml ===")
with open('docker-compose.inference.yml') as f:
    dc = yaml.safe_load(f)
services = list(dc.get('services', {}).keys())
print(f"Services: {services}")
required = ['inference-coordinator', 'client-1', 'client-2', 'client-3', 'inference-dashboard']
found = 0
for svc in required:
    status = 'FOUND' if svc in services else 'MISSING'
    if svc in services:
        found += 1
    print(f"  {status}: {svc}")
print(f"Total: {found}/{len(required)}")

print("\n=== docker-compose.yml ===")
with open('docker-compose.yml') as f:
    dc2 = yaml.safe_load(f)
print(f"Services: {list(dc2.get('services', {}).keys())}")

print("\n=== docker/client.dockerfile ===")
with open('docker/client.dockerfile') as f:
    content = f.read()
print(f"FROM python:3.9-slim present: {'FROM python:3.9-slim' in content}")
print(f"inference_test_script.py present: {'inference_test_script.py' in content}")
