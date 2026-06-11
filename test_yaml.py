import yaml, glob, os

yaml_files = (
    glob.glob("k8s/*.yaml") +
    glob.glob("k8s/*.md") +  # skip md
    glob.glob("helm/*.yaml") +
    glob.glob("helm/templates/*.yaml") +
    glob.glob("helm/templates/*.tpl") +
    ["docker-compose.inference.yml", "docker-compose.yml"]
)
# filter only yaml/yml
yaml_files = [f for f in yaml_files if f.endswith(('.yaml', '.yml', '.tpl'))]

passed = 0
failed = 0
for f in sorted(yaml_files):
    try:
        with open(f) as fh:
            yaml.safe_load(fh)
        print(f"  PASS: {f}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {f} -- {e}")
        failed += 1

print(f"\nYAML: {passed} passed, {failed} failed out of {passed+failed} files")

# Check unique ports in docker-compose.inference.yml
print("\n=== Port uniqueness check ===")
with open('docker-compose.inference.yml') as f:
    dc = yaml.safe_load(f)
ports = []
for svc, cfg in dc.get('services', {}).items():
    for p in (cfg.get('ports') or []):
        host_port = str(p).split(':')[0]
        ports.append((host_port, svc))
seen = {}
for port, svc in ports:
    if port in seen:
        print(f"  DUPLICATE port {port}: {seen[port]} and {svc}")
    else:
        seen[port] = svc
        print(f"  UNIQUE port {port}: {svc}")
