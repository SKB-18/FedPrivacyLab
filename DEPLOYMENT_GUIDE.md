# FedPrivacyLab: Deployment Guide

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Helm Deployment](#helm-deployment)
5. [Production Checklist](#production-checklist)
6. [Troubleshooting](#troubleshooting)
7. [Scaling](#scaling)

---

## Local Development

### Prerequisites

- **Python:** 3.11 or higher
- **pip:** Latest version
- **Git:** For cloning repo
- **RAM:** 4+ GB recommended (8+ GB for GPU)
- **Disk:** 2+ GB free space

### Step-by-Step Installation

#### Step 1: Clone Repository

```bash
git clone <repository-url>
cd FedPrivacyLab
```

#### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# Verify activation
which python  # Should show path to venv/bin/python
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Installation Time:** 5-10 minutes (depends on internet speed, TensorFlow download)

**Troubleshooting:**
- If TensorFlow fails: `pip install tensorflow>=2.15.0,<2.21.0 --upgrade`
- If numpy conflict: `pip install numpy>=1.26.0 --upgrade`
- On Windows, might need Visual C++ build tools

#### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
python -c "from app.main import app; print('App imports successfully')"
```

#### Step 5: Initialize Database

```bash
python -c "from app.database import init_db; init_db(); print('Database initialized')"
```

**Output:** Creates `data/fedprivacylab.db` (SQLite)

### Running Locally

#### Terminal 1: Start API Server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Access:**
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### Terminal 2: Start Dashboard

```bash
streamlit run dashboard/streamlit_app.py --server.port 8501
```

**Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Access:**
- Dashboard: http://localhost:8501

#### Terminal 3: Run Experiments (Optional)

```bash
python scripts/run_demo.py
# or
curl -X POST http://localhost:8000/api/v1/experiments/start \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","mode":"fedavg_secureagg_dp","num_clients":50,"rounds":3}'
```

### Development Workflow

```
Edit code
    ↓
uvicorn --reload (auto-restarts on code changes)
    ↓
Test via http://localhost:8000/docs
    ↓
Check dashboard at http://localhost:8501
    ↓
Run tests: pytest tests/ -v
    ↓
Commit: git add . && git commit -m "..."
```

---

## Docker Deployment

### Prerequisites

- **Docker:** 20.10+ installed and running
- **Docker Compose:** 1.29+ installed
- **Disk:** 2+ GB free (for images)

### Quick Start

```bash
# Build and run all services
docker compose up --build

# Services available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Database: sqlite (in container volume)
```

**Components:**
- API container (FastAPI)
- Dashboard container (Streamlit)
- Network bridge for inter-container communication
- Shared volume for database

### Docker Compose Files

#### docker-compose.yml (Training + Inference)

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/fedprivacylab.db
    volumes:
      - ./data:/app/data
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  dashboard:
    build:
      context: .
      dockerfile: docker/dockerfile.dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    depends_on:
      - api
    command: streamlit run dashboard/streamlit_app.py --server.port 8501
```

#### docker-compose.inference.yml (Inference Only)

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODE=inference_only
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Building Images

**Build API Image:**
```bash
docker build -f Dockerfile -t fedprivacylab-api:latest .
```

**Build Dashboard Image:**
```bash
docker build -f docker/dockerfile.dashboard -t fedprivacylab-dashboard:latest .
```

**Check Images:**
```bash
docker images | grep fedprivacylab
```

### Running Containers

**Run API Container:**
```bash
docker run -d \
  --name fedprivacylab-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  fedprivacylab-api:latest

# Check logs
docker logs -f fedprivacylab-api
```

**Run Dashboard Container:**
```bash
docker run -d \
  --name fedprivacylab-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  --link fedprivacylab-api:api \
  fedprivacylab-dashboard:latest

# Check logs
docker logs -f fedprivacylab-dashboard
```

### Managing Containers

```bash
# View running containers
docker ps

# Stop container
docker stop fedprivacylab-api

# Restart container
docker restart fedprivacylab-api

# Remove container
docker rm fedprivacylab-api

# View container logs
docker logs --tail 100 fedprivacylab-api

# Execute command in running container
docker exec -it fedprivacylab-api bash
```

### Volume Management

**Persistent Database:**
```bash
# Create named volume
docker volume create fedprivacylab-data

# Use in container
docker run -v fedprivacylab-data:/app/data ...

# List volumes
docker volume ls

# Inspect volume
docker volume inspect fedprivacylab-data
```

### Network Configuration

**Custom Network:**
```bash
# Create network
docker network create fedprivacylab-net

# Run containers on network
docker run --network fedprivacylab-net ...

# Containers can communicate using service names
```

---

## Kubernetes Deployment

### Prerequisites

- **kubectl:** 1.24+ installed and configured
- **Kubernetes Cluster:** 1.24+ (EKS, GKE, AKS, Minikube)
- **Storage:** PersistentVolume support
- **Resources:** 2+ GB RAM minimum per node

### Cluster Setup

**Local Testing (Minikube):**
```bash
# Start cluster
minikube start --cpus=4 --memory=8192 --disk-size=20g

# Enable Docker daemon
eval $(minikube docker-env)

# Enable ingress addon
minikube addons enable ingress
```

**Cloud Cluster (EKS, GKE, AKS):**
```bash
# EKS
aws eks create-cluster --name fedprivacylab ...

# GKE
gcloud container clusters create fedprivacylab ...

# AKS
az aks create --resource-group mygroup --name fedprivacylab ...
```

### Deploy with kubectl

#### Step 1: Create Namespace

```bash
kubectl create namespace fedprivacylab

# Verify
kubectl get namespaces | grep fedprivacylab
```

#### Step 2: Apply ConfigMaps

```bash
kubectl create configmap fedprivacylab-config \
  --from-file=config/experiment_config.yaml \
  --from-file=config/privacy_config.yaml \
  -n fedprivacylab

# Verify
kubectl get configmaps -n fedprivacylab
```

#### Step 3: Apply Manifests

```bash
# Deployment
kubectl apply -f k8s/deployment.yaml -n fedprivacylab

# Services
kubectl apply -f k8s/service.yaml -n fedprivacylab

# PersistentVolumeClaim
kubectl apply -f k8s/pvc.yaml -n fedprivacylab

# Ingress (if enabled)
kubectl apply -f k8s/ingress.yaml -n fedprivacylab
```

#### Step 4: Verify Deployment

```bash
# Check pods
kubectl get pods -n fedprivacylab
# Expected output:
# NAME                               READY   STATUS    RESTARTS   AGE
# fedprivacylab-api-xxxxx           1/1     Running   0          10s
# fedprivacylab-dashboard-xxxxx     1/1     Running   0          10s

# Check services
kubectl get svc -n fedprivacylab
# Expected output:
# NAME                    TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)
# fedprivacylab-api      ClusterIP  10.0.1.100      <none>        8000/TCP
# fedprivacylab-dashboard ClusterIP  10.0.1.101      <none>        8501/TCP

# Check PVCs
kubectl get pvc -n fedprivacylab
```

### Access Services

**Port Forward:**
```bash
# API
kubectl port-forward svc/fedprivacylab-api 8000:8000 -n fedprivacylab

# Dashboard
kubectl port-forward svc/fedprivacylab-dashboard 8501:8501 -n fedprivacylab

# Access at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
```

**Ingress (if configured):**
```bash
# Get ingress IP
kubectl get ingress -n fedprivacylab

# Access via ingress IP (e.g., 10.0.0.50)
# http://10.0.0.50/api
# http://10.0.0.50/dashboard
```

### Scaling

**Horizontal Scaling (More Pods):**
```bash
# Scale API to 3 replicas
kubectl scale deployment fedprivacylab-api --replicas=3 -n fedprivacylab

# Scale dashboard to 2 replicas
kubectl scale deployment fedprivacylab-dashboard --replicas=2 -n fedprivacylab

# Check status
kubectl get pods -n fedprivacylab
```

**Autoscaling:**
```bash
# Create HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment fedprivacylab-api \
  --min=2 --max=10 --cpu-percent=80 -n fedprivacylab

# Check HPA
kubectl get hpa -n fedprivacylab

# Delete HPA
kubectl delete hpa fedprivacylab-api -n fedprivacylab
```

### Monitoring & Debugging

**View Logs:**
```bash
# Pod logs
kubectl logs <pod-name> -n fedprivacylab

# Follow logs
kubectl logs -f <pod-name> -n fedprivacylab

# View all pod logs (last 10 lines)
kubectl logs deployment/fedprivacylab-api -n fedprivacylab --tail=10
```

**Describe Resources:**
```bash
# Pod details
kubectl describe pod <pod-name> -n fedprivacylab

# Deployment details
kubectl describe deployment fedprivacylab-api -n fedprivacylab
```

**Execute Commands:**
```bash
# Run shell in pod
kubectl exec -it <pod-name> -n fedprivacylab -- bash

# Run Python command
kubectl exec <pod-name> -n fedprivacylab -- python -c "import tensorflow as tf; print(tf.__version__)"
```

**Check Resource Usage:**
```bash
# Node resources
kubectl top nodes

# Pod resources
kubectl top pods -n fedprivacylab
```

---

## Helm Deployment

### Prerequisites

- **Helm:** 3.10+ installed
- **Kubernetes:** 1.24+ running
- **Chart Values:** Customized for your environment

### Install Chart

```bash
# Add repository (if using Helm Hub)
# helm repo add fedprivacylab https://...
# helm repo update

# Install release
helm install fedprivacylab ./helm -n fedprivacylab --create-namespace

# Verify installation
helm status fedprivacylab -n fedprivacylab

# Check installed resources
helm list -n fedprivacylab
```

### Customize Values

**Override Values on Install:**
```bash
helm install fedprivacylab ./helm \
  --set replicaCount=3 \
  --set image.tag=v1.2.0 \
  --set resources.limits.cpu=2000m \
  -n fedprivacylab --create-namespace
```

**Use Custom Values File:**
```bash
# Copy and edit values
cp helm/values.yaml my-values.yaml
vim my-values.yaml  # Edit as needed

# Install with custom values
helm install fedprivacylab ./helm -f my-values.yaml -n fedprivacylab --create-namespace
```

**values.yaml Structure:**
```yaml
replicaCount: 2

image:
  repository: fedprivacylab-api
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

persistence:
  enabled: true
  size: 10Gi
  mountPath: /app/data
```

### Upgrade Release

```bash
# Upgrade to new version
helm upgrade fedprivacylab ./helm \
  --set image.tag=v1.3.0 \
  -n fedprivacylab

# Verify upgrade
helm status fedprivacylab -n fedprivacylab

# Check rollout status
kubectl rollout status deployment/fedprivacylab-api -n fedprivacylab
```

### Rollback Release

```bash
# View release history
helm history fedprivacylab -n fedprivacylab

# Rollback to previous release
helm rollback fedprivacylab -n fedprivacylab

# Rollback to specific revision
helm rollback fedprivacylab 1 -n fedprivacylab  # Revision 1
```

### Uninstall Release

```bash
# Remove release
helm uninstall fedprivacylab -n fedprivacylab

# Delete namespace (optional)
kubectl delete namespace fedprivacylab
```

---

## Production Checklist

- [ ] **Database:** Use PostgreSQL (not SQLite) for production
- [ ] **Environment Variables:** Configure via ConfigMaps/Secrets
- [ ] **Secret Management:** Use Kubernetes Secrets for API keys, certs
- [ ] **CORS:** Restrict origins (not `["*"]`)
- [ ] **Authentication:** Implement token-based auth (JWT)
- [ ] **TLS/HTTPS:** Enable SSL certificates (Let's Encrypt)
- [ ] **Monitoring:** Deploy Prometheus + Grafana
- [ ] **Logging:** Central logging (ELK, Splunk, GCP Logging)
- [ ] **Resource Limits:** Set CPU/memory requests and limits
- [ ] **Health Checks:** Enable liveness and readiness probes
- [ ] **Backup:** Configure PersistentVolume backups
- [ ] **CI/CD:** Set up GitHub Actions or GitLab CI
- [ ] **Tests:** Run full test suite (pytest) before deploy
- [ ] **Documentation:** Update deployment docs for your environment
- [ ] **Load Testing:** Test with expected concurrent users
- [ ] **Security Scan:** Run container image vulnerability scan

### Production Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/fedprivacylab

# API Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=https://myapp.com,https://dashboard.myapp.com

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Monitoring
SENTRY_DSN=https://...@sentry.io/...
PROMETHEUS_ENABLED=true
```

### Production Secrets (Kubernetes)

```bash
# Create secrets
kubectl create secret generic fedprivacylab-secrets \
  --from-literal=database-password=secure-password \
  --from-literal=secret-key=secret-key-here \
  -n fedprivacylab

# Use in deployment (patch values.yaml):
env:
  - name: DATABASE_PASSWORD
    valueFrom:
      secretKeyRef:
        name: fedprivacylab-secrets
        key: database-password
```

---

## Troubleshooting

### Common Issues

#### Issue: "Port 8000 already in use"

**Solution 1: Use different port**
```bash
uvicorn app.main:app --port 8001
```

**Solution 2: Kill process using port**
```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Solution 3: Docker - check port mapping**
```bash
docker ps | grep 8000
docker stop <container-id>
```

#### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow>=2.15.0,<2.21.0
pip install -r requirements.txt --upgrade
```

#### Issue: "Database locked" (SQLite)

**Cause:** Concurrent access to SQLite  
**Solution 1:** Use single worker
```bash
# Use --workers=1 (default)
uvicorn app.main:app --workers=1
```

**Solution 2:** Migrate to PostgreSQL (production)
```bash
DATABASE_URL=postgresql://user:password@localhost/fedprivacylab
```

#### Issue: "Out of memory" during training

**Solution 1: Reduce batch size**
```bash
curl -X POST .../experiments/start -d '{
  "batch_size": 16,  # Reduce from 32
  ...
}'
```

**Solution 2: Reduce clients per round**
```bash
curl -X POST .../experiments/start -d '{
  "clients_per_round": 20,  # Reduce from 50
  ...
}'
```

**Solution 3: Add memory to container/node**
```bash
# Docker
docker run -m 4g ...  # 4 GB memory limit

# Kubernetes
resources:
  limits:
    memory: 4Gi
```

#### Issue: "Streamlit not responding"

**Solution 1: Restart dashboard**
```bash
# Kill streamlit process
pkill -f streamlit

# Restart
streamlit run dashboard/streamlit_app.py
```

**Solution 2: Clear cache**
```bash
rm -rf ~/.streamlit/cache
streamlit run dashboard/streamlit_app.py
```

**Solution 3: Check logs**
```bash
streamlit run dashboard/streamlit_app.py --logger.level=debug
```

#### Issue: "Kubernetes pod in CrashLoopBackOff"

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n fedprivacylab
kubectl logs <pod-name> -n fedprivacylab
```

**Common Causes & Solutions:**
1. **Image pull failed** → Check image name, registry
2. **Memory limit too low** → Increase limits in deployment
3. **Missing ConfigMap** → Create ConfigMap before pod
4. **Invalid environment variables** → Check Secrets/ConfigMaps

---

## Scaling

### Vertical Scaling (More Resources)

**For Single Machine:**
```
CPU: 8 cores → 16 cores
RAM: 16GB → 32GB
SSD: 100GB → 500GB

Impact: Faster training, more concurrent experiments
```

### Horizontal Scaling (More Machines)

**Kubernetes Deployment:**
```bash
# Scale API replicas
kubectl scale deployment fedprivacylab-api --replicas=5 -n fedprivacylab

# Auto-scale based on CPU
kubectl autoscale deployment fedprivacylab-api \
  --min=2 --max=20 --cpu-percent=70 -n fedprivacylab
```

**Load Balancing:**
```yaml
# Service with load balancer
apiVersion: v1
kind: Service
metadata:
  name: fedprivacylab-api-lb
spec:
  type: LoadBalancer  # or ClusterIP + Ingress
  selector:
    app: fedprivacylab-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Database Scaling

**SQLite → PostgreSQL:**
```bash
# Install PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15

# Update connection string
export DATABASE_URL=postgresql://postgres:password@localhost:5432/fedprivacylab

# Run migrations
alembic upgrade head
```

**PostgreSQL Replication:**
```sql
-- Primary-replica setup for high availability
-- Requires manual configuration
-- See PostgreSQL docs for streaming replication
```

---

**For additional help, see COMPREHENSIVE_README.md or open an issue on GitHub.**
