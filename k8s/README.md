# FedPrivacyLab Kubernetes Deployment

## Prerequisites
- Kubernetes cluster (v1.24+)
- kubectl configured
- Container image built and pushed: `docker build -t fedprivacylab:latest .`

## Deploy

```bash
# Create namespace and storage
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/data-pvc.yaml

# Deploy coordinator API (port 8000, ClusterIP)
kubectl apply -f k8s/coordinator-deployment.yaml
kubectl apply -f k8s/coordinator-service.yaml

# Deploy Streamlit dashboard (port 8501, LoadBalancer)
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/streamlit-service.yaml
```

## Verify

```bash
kubectl -n fedprivacylab get pods
kubectl -n fedprivacylab get svc
# Wait for LoadBalancer EXTERNAL-IP, then open http://<EXTERNAL-IP>
```

## Resource Limits
Both services are capped at **500m CPU / 512Mi memory** (requests: 250m / 256Mi).

## Teardown

```bash
kubectl delete namespace fedprivacylab
```
