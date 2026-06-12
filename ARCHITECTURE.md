# FedPrivacyLab System Architecture

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Core Components](#core-components)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Privacy Mechanisms (Detailed)](#privacy-mechanisms-detailed)
5. [Experiment Modes](#experiment-modes)
6. [Database Schema](#database-schema)
7. [API Architecture](#api-architecture)
8. [Client Architecture](#client-architecture)
9. [Coordinator Architecture](#coordinator-architecture)
10. [Privacy Accounting](#privacy-accounting)
11. [Failure Modes & Mitigations](#failure-modes--mitigations)
12. [Scalability Considerations](#scalability-considerations)

---

## Architectural Overview

FedPrivacyLab follows a **master-worker** federated learning architecture with layered privacy controls:

```
┌──────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT LAYER                            │
│  Configuration → Execution → Metrics Collection → Evaluation     │
└──────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌──────────────────┐        ┌──────────────────┐
        │   COORDINATOR    │        │   CLIENTS (N)    │
        │   (Master)       │◄─────►│  (Workers)       │
        │                  │        │                  │
        │ - Aggregation    │        │ - Local Training │
        │ - Selection      │        │ - Analytics      │
        │ - Evaluation     │        │ - Dropout Sim    │
        └──────────────────┘        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │   PRIVACY LAYER  │
        │                  │
        │ - DP Mechanisms  │
        │ - Secure Agg     │
        │ - Clipping       │
        │ - Bounding       │
        │ - Accounting     │
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │   DATABASE       │
        │                  │
        │ - Experiments    │
        │ - Metrics        │
        │ - Analytics      │
        │ - Privacy Logs   │
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │   DASHBOARD      │
        │   (Streamlit)    │
        │                  │
        │ 13 Pages, 40+ Charts
        └──────────────────┘
```

---

## Core Components

### 1. Coordinator (Server)

**Responsibility:** Orchestrates federated learning rounds

**Key Classes:**
- `ExperimentRunner` — Manages experiment lifecycle
- `RoundCoordinator` — Executes single rounds
- `ClientRegistry` — Tracks client participation
- `AggregationEngine` — Weighted average of updates

**Process per Round:**

1. **Client Selection** → Select K clients from N randomly
2. **Model Distribution** → Send global model to selected clients
3. **Local Training** → Wait for clients to complete training
4. **Update Collection** → Receive model deltas from clients
5. **Privacy Processing** → Apply clipping, secure agg, DP noise
6. **Aggregation** → Weighted average of updates
7. **Model Update** → θ_new = θ_old + aggregate(deltas)
8. **Metrics** → Evaluate on test set
9. **Database Storage** → Log metrics and privacy budget

**Code Structure:**

```python
class ExperimentRunner:
    def create_experiment(request: ExperimentStartRequest) -> Experiment:
        # Initialize experiment in DB
        pass
    
    def initialize_clients(experiment, request):
        # Generate/load client data
        # Assign client profiles (non-IID)
        # Register clients
        pass
    
    def run_single_round(experiment_id: int) -> dict:
        # 1. Select clients
        # 2. Distribute model
        # 3. Collect updates
        # 4. Apply privacy
        # 5. Aggregate
        # 6. Evaluate
        # 7. Log metrics
        return metrics
    
    def run_full_experiment(experiment_id: int) -> dict:
        # Loop: run_single_round() for each round
        pass
```

### 2. Clients (Workers)

**Responsibility:** Train locally, compute analytics, send updates

**Client Types:**

| Type | Data | Task |
|------|------|------|
| **TelemetryClient** | Events (count, sum, latency) | Analytics |
| **EMNISTClient** | MNIST/EMNIST images | Image classification |
| **AnalyticsClient** | Generic telemetry | Aggregation |
| **TrainingClient** | Structured data | Model training |

**Process per Client per Round:**

```python
class Client:
    def __init__(self, client_id, model, local_data):
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
    
    def receive_model(global_model):
        # Copy global weights to local model
        self.model.set_weights(global_model)
    
    def local_training(epochs: int, batch_size: int):
        # Multi-epoch training on local data
        for epoch in range(epochs):
            for batch in self.local_data:
                # Forward pass, compute loss
                # Backward pass, update weights
                loss = self.model.train_on_batch(batch)
        
        # Return delta (model - initial_weights)
        delta = self.model.get_weights() - initial_weights
        return delta
    
    def dropout_simulation(dropout_rate: float) -> bool:
        # Random dropout (client unavailable)
        if random() < dropout_rate:
            return False  # Dropout
        return True  # Participate
```

### 3. Privacy Layer

**Responsibility:** Enforce privacy guarantees

**Components:**

```python
class PrivacyLayer:
    def __init__(self, config: PrivacyConfig):
        self.clipping_norm = config.clipping_norm
        self.secure_agg_enabled = config.secure_agg_enabled
        self.dp_enabled = config.dp_enabled
        self.privacy_accountant = PrivacyAccountant(config.epsilon, config.delta)
    
    def process_update(update: np.ndarray, client_id: str) -> np.ndarray:
        # 1. Contribution bounding (analytics)
        update = self.bound_contribution(update)
        
        # 2. Clipping (learning)
        update = self.clip_update(update)
        
        # 3. Secure aggregation (add mask)
        if self.secure_agg_enabled:
            mask = self.generate_mask(update.shape)
            update_with_mask = update + mask
            return update_with_mask, mask
        
        return update
    
    def aggregate(updates_with_privacy: list):
        # 1. Sum all updates
        aggregate = sum(updates_with_privacy)
        
        # 2. Cancel masks (if secure agg)
        if self.secure_agg_enabled:
            aggregate -= sum(all_masks)  # Σ(u_i + m_i) - Σm_i = Σu_i
        
        # 3. Average by client count
        aggregate /= len(updates_with_privacy)
        
        # 4. Add DP noise
        if self.dp_enabled:
            noise = np.random.normal(0, self.noise_scale, aggregate.shape)
            aggregate += noise
            self.privacy_accountant.charge(self.epsilon_per_round)
        
        return aggregate
```

### 4. Analytics Engine

**Responsibility:** Compute privacy-preserving metrics

**Supported Metrics:**

- **Count** — Number of records (Laplace DP)
- **Sum** — Total of numeric field (Gaussian DP)
- **Mean** — Average value (derived from sum/count)
- **Quantile** — Percentile value
- **Histogram** — Distribution over buckets

**Process:**

```python
class FederatedAnalytics:
    def compute_metric(metric_name: str, feature: str):
        # 1. Gather local summaries from clients
        local_summaries = []
        for client in selected_clients:
            summary = client.compute_local_summary(feature)
            # Example: {"count": 100, "sum": 5000, "max": 150}
            local_summaries.append(summary)
        
        # 2. Bound contributions per client
        bounded_summaries = []
        for summary in local_summaries:
            bounded = self.contribution_bounding.bound(summary)
            bounded_summaries.append(bounded)
        
        # 3. Aggregate across clients
        aggregate = self.aggregate_summaries(bounded_summaries)
        # aggregate = {"count": 4850, "sum": 48500}  # bounded
        
        # 4. Add DP noise (if enabled)
        if self.dp_enabled:
            aggregate = self.add_dp_noise(aggregate)
        
        # 5. Return sanitized result
        return aggregate
```

---

## Data Flow Diagrams

### Federated Learning Flow (One Round)

```
Round Start
    │
    ▼
┌─────────────────────────────────────────────┐
│ Coordinator: Client Selection                │
│ - Select K from N clients                   │
│ - Randomly or stratified                    │
└────────────┬────────────────────────────────┘
             │
             ├──────────────────────────────┐
             │                              │
             ▼                              ▼
        Selected Clients              Unselected Clients
        (K clients)                   (N-K clients)
             │                        - Wait (idle)
             │
             ├─ Client 1               
             │    │
             │    ▼
             │  1. Receive global model
             │  2. Local training (E epochs)
             │  3. Compute delta = local_model - global_model
             │  4. Apply contribution bounding (analytics)
             │  5. Apply clipping (learning)
             │  6. Secure aggregation mask (if enabled)
             │  7. Dropout simulation (may drop out)
             │  8. Send delta to coordinator
             │    │
             │    ▼
             │
             ├─ Client 2, 3, ..., K (same as Client 1)
             │
             └─────────────────┬──────────────────┘
                               │
                   ┌───────────▼────────────┐
                   │ Coordinator: Aggregation
                   │                        │
                   │ 1. Collect deltas      │
                   │    (may have dropouts) │
                   │                        │
                   │ 2. Privacy processing: │
                   │    - Cancel masks (SA) │
                   │    - Add DP noise      │
                   │    - Track epsilon     │
                   │                        │
                   │ 3. Weighted average:   │
                   │    Δ = Σ(n_i / N * Δ_i)
                   │                        │
                   │ 4. Update model:       │
                   │    θ' = θ + Δ          │
                   │                        │
                   │ 5. Evaluate on test    │
                   │    - Accuracy, F1, etc│
                   │                        │
                   │ 6. Log metrics to DB   │
                   └───────────┬────────────┘
                               │
                               ▼
                          End of Round
```

### Analytics Flow (Privacy-Preserving Measurement)

```
Analytics Request (e.g., "count events per client")
    │
    ▼
┌──────────────────────────────────────┐
│ Coordinator: Client Selection         │
│ - Select reporting clients            │
└────────────┬─────────────────────────┘
             │
             ├─ Client 1: Local Summary
             │  │
             │  ▼
             │ 1. Scan local data for metric
             │ 2. Compute local aggregate
             │    Example: count=150, sum=5000
             │ 3. Apply contribution bound
             │    Capped to max=100 events
             │    Result: count=100, sum=3333
             │ 4. Send to coordinator
             │
             ├─ Client 2, 3, ..., K (same)
             │
             └─────────────────┬─────────────┘
                               │
                   ┌───────────▼────────────┐
                   │ Coordinator: Aggregation
                   │                        │
                   │ Bounded Summaries:     │
                   │ - From K clients       │
                   │ - Each capped to bound │
                   │                        │
                   │ Aggregate = Sum all    │
                   │            = 3333*K    │
                   │            = 33330     │
                   │                        │
                   │ Add DP Noise (Laplace)│
                   │ noise ~ Lap(0, b)     │
                   │ b = 1/epsilon          │
                   │ result += noise        │
                   │                        │
                   │ ε consumption: +0.1    │
                   │ (epsilon remaining)    │
                   │                        │
                   │ Log to DB:             │
                   │ - True value           │
                   │ - Federated value      │
                   │ - DP-noisy value       │
                   │ - Error metrics        │
                   └───────────┬────────────┘
                               │
                               ▼
                        Analytics Result
```

---

## Privacy Mechanisms (Detailed)

### 1. Contribution Bounding

**Purpose:** Limit per-client influence per round

**How It Works:**

```python
class ContributionBounding:
    def __init__(self, bound: int = 10):
        self.bound = bound
    
    def bound(self, value: float) -> float:
        # Clamp value to [0, bound]
        return min(max(value, 0), self.bound)
    
    def bound_vector(self, vector: np.ndarray) -> np.ndarray:
        # Element-wise bounding
        return np.minimum(np.maximum(vector, 0), self.bound)
```

**Example (Events):**
```
Client reports: 500 events
Bound: 100 events/round
Result: 100 events (capped)
```

**Privacy Impact:**
- Prevents single client from dominating result
- Formal privacy: Sensitivity reduced from ∞ to bound
- Accuracy impact: Minimal if clients diverse

### 2. Update Clipping (L2 Norm)

**Purpose:** Bound model update magnitude (learning)

**How It Works:**

```python
class UpdateClipping:
    def __init__(self, clipping_norm: float = 1.0):
        self.clipping_norm = clipping_norm
    
    def clip(self, update: np.ndarray) -> np.ndarray:
        # Compute L2 norm
        norm = np.linalg.norm(update, ord=2)
        
        # Scale to norm if exceeds threshold
        if norm > self.clipping_norm:
            update = update * (self.clipping_norm / norm)
        
        return update
```

**Example:**
```
Update vector: [0.5, 1.2, 0.3]  (from client)
L2 norm: sqrt(0.5^2 + 1.2^2 + 0.3^2) = 1.31
Clipping norm: 1.0
Exceeds? Yes (1.31 > 1.0)
Scale factor: 1.0 / 1.31 = 0.763
Clipped: [0.382, 0.916, 0.229]
```

**Privacy Impact:**
- Bounds sensitivity for DP composition
- Gradient clipping = standard DP-SGD technique
- Accuracy impact: ~2-3% if norm set appropriately

### 3. Secure Aggregation (Simulated)

**Purpose:** Hide individual updates from server until aggregated

**Cryptographic Concept (Simplified):**

```python
class SecureAggregation:
    def __init__(self):
        self.masks = {}
    
    def generate_mask(self, shape: tuple) -> np.ndarray:
        # Random mask sampled from normal distribution
        mask = np.random.normal(0, 1, shape)
        return mask
    
    def send_with_mask(client_id: str, update: np.ndarray) -> np.ndarray:
        # Client generates mask and adds to update
        mask = self.generate_mask(update.shape)
        masked_update = update + mask
        
        # Store mask for later removal
        self.masks[client_id] = mask
        
        return masked_update
    
    def aggregate_and_unmask(masked_updates: list[np.ndarray]) -> np.ndarray:
        # 1. Sum all masked updates
        sum_masked = np.sum(masked_updates, axis=0)
        
        # 2. Subtract all masks
        sum_masks = np.sum(list(self.masks.values()), axis=0)
        
        # 3. Result = sum(u_i + m_i) - sum(m_i) = sum(u_i)
        aggregate = sum_masked - sum_masks
        
        # 4. Average
        result = aggregate / len(masked_updates)
        
        return result
```

**Security Property:**
- **Without dropout:** Server learns nothing about individual updates (mathematically masked)
- **With dropout:** Client can infer non-dropping clients' updates (known limitation)
- **Note:** This is a simplified simulation, not cryptographically secure

**Privacy Impact:**
- Prevents server from seeing per-client updates
- Doesn't add DP (formal privacy) — only hides updates from server
- No accuracy impact (pure aggregation)

### 4. Differential Privacy (DP)

**Purpose:** Formal privacy guarantee against all adversaries

**Mechanisms:**

#### Laplace Mechanism (Counts)
```python
class LaplaceMechanism:
    def __init__(self, epsilon: float, sensitivity: float):
        self.epsilon = epsilon
        self.sensitivity = sensitivity  # How much query changes if one record added/removed
    
    def add_noise(self, query_result: float) -> float:
        # Scale = sensitivity / epsilon
        scale = self.sensitivity / self.epsilon
        
        # Add Laplace noise: noise ~ Laplace(0, scale)
        noise = np.random.laplace(0, scale)
        
        return query_result + noise
```

**Example:**
```
Query: Count events
Sensitivity: 1 (bounding ensures max change = 1)
Result: 1000
Epsilon: 1.0

Scale = 1 / 1.0 = 1.0
Noise ~ Laplace(0, 1.0)
Noisy result = 1000 + noise
E.g., 1000 + 0.34 = 1000.34
```

#### Gaussian Mechanism (Sums/Updates)
```python
class GaussianMechanism:
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
    
    def add_noise(self, query_result: np.ndarray) -> np.ndarray:
        # Noise scale: sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        denominator = self.epsilon
        numerator = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))
        scale = numerator / denominator
        
        noise = np.random.normal(0, scale, query_result.shape)
        return query_result + noise
```

**Privacy-Utility Tradeoff:**

| Epsilon | Interpretation | Accuracy Impact |
|---------|---|---|
| 0.1 | Very strong privacy | 15-20% loss |
| 0.5 | Strong privacy | 10-15% loss |
| 1.0 | Moderate privacy | 5-8% loss |
| 5.0 | Weak privacy | 1-2% loss |
| ∞ | No privacy | 0% loss |

### 5. Privacy Accounting

**Purpose:** Track cumulative privacy loss across rounds

```python
class PrivacyAccountant:
    def __init__(self, target_epsilon: float, target_delta: float):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.spent_epsilon = 0.0
        self.rounds = 0
    
    def charge(self, epsilon_per_round: float) -> None:
        # Basic composition: epsilon_total = sum(epsilon_per_round)
        self.spent_epsilon += epsilon_per_round
        self.rounds += 1
    
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self.spent_epsilon)
    
    def budget_exceeded(self) -> bool:
        return self.spent_epsilon > self.target_epsilon
```

**Composition Rules:**
- **Sequential:** ε_total = sum(ε_i) [basic, loose]
- **Parallel:** ε_total ≈ ε (when each query on disjoint data)
- **Advanced:** Renyi composition gives tighter bounds

---

## Experiment Modes

FedPrivacyLab supports 5 modes to study privacy-utility tradeoffs:

### Mode 1: Centralized Baseline

```python
mode = "centralized_baseline"
# All data collected on server
# No privacy
# Baseline accuracy (100%)
# Used to measure privacy cost
```

**What happens:**
1. All client data centralized on server
2. Global model trained on full dataset
3. No privacy mechanisms applied
4. Reference for accuracy comparison

### Mode 2: Federated Analytics

```python
mode = "federated_analytics"
# Clients send bounded summaries (not raw data)
# Raw telemetry never centralized
# Optional DP on summaries
# For measurement without model training
```

**What happens:**
1. Clients compute local summaries
2. Summaries bounded (contribution bounding)
3. Summaries aggregated on server
4. Optional DP noise added
5. No model training

### Mode 3: FedAvg

```python
mode = "fedavg"
# Classic federated averaging
# Model updates sent to server (not encrypted)
# No differential privacy
# Baseline for FL privacy cost
```

**What happens:**
1. Clients train locally
2. Model deltas sent to server (visible)
3. Server aggregates with weighted average
4. No privacy mechanisms
5. Reference for FL accuracy

### Mode 4: FedAvg + Secure Aggregation

```python
mode = "fedavg_secureagg"
# Model updates encrypted before aggregation
# Server cannot see individual updates
# No differential privacy
# Baseline for secure aggregation
```

**What happens:**
1. Clients train locally
2. Model deltas masked with random noise
3. Masked updates sent to server
4. Server aggregates without seeing individual updates
5. Masks cancel in aggregate (Σ(u + m) - Σm = Σu)

### Mode 5: FedAvg + SecureAgg + DP (Full Stack)

```python
mode = "fedavg_secureagg_dp"
# Combines all privacy mechanisms
# Formal DP guarantee
# Highest privacy, may have accuracy loss
# Production recommendation
```

**What happens:**
1. Clients train locally
2. Updates clipped (L2 norm bound)
3. Updates masked (secure agg)
4. DP noise added to aggregate
5. Privacy budget tracked

**Configuration:**
```yaml
privacy:
  dp_enabled: true
  epsilon: 1.0          # Privacy budget
  delta: 1e-5           # Failure probability
  noise_multiplier: 0.5 # Gaussian noise scale
  clipping_norm: 1.0    # L2 clip threshold
```

---

## Database Schema

### Experiments Table

```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,  -- {centralized, federated_analytics, fedavg, ...}
    dataset VARCHAR NOT NULL,  -- {emnist, telemetry, real_hdfs_loghub}
    num_clients INTEGER,
    rounds INTEGER,
    clients_per_round INTEGER,
    local_epochs INTEGER,
    dp_enabled BOOLEAN DEFAULT FALSE,
    secure_agg_enabled BOOLEAN DEFAULT FALSE,
    dropout_rate FLOAT,
    clipping_norm FLOAT,
    noise_multiplier FLOAT,
    epsilon FLOAT,
    delta FLOAT,
    status VARCHAR DEFAULT 'created',  -- {created, running, completed}
    created_at DATETIME DEFAULT NOW()
);
```

### RoundMetrics Table

```sql
CREATE TABLE round_metrics (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL FOREIGN KEY,
    round_number INTEGER NOT NULL,
    selected_clients INTEGER,
    completed_clients INTEGER,
    dropped_clients INTEGER,
    train_loss FLOAT,
    eval_loss FLOAT,
    accuracy FLOAT,
    roc_auc FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    created_at DATETIME DEFAULT NOW(),
    INDEX(experiment_id, round_number)
);
```

### AnalyticsResults Table

```sql
CREATE TABLE analytics_results (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL FOREIGN KEY,
    round_number INTEGER NOT NULL,
    metric_name VARCHAR NOT NULL,  -- {count, sum, mean, quantile}
    feature VARCHAR,  -- Which field (e.g., "latency")
    true_value FLOAT,  -- Ground truth (centralized)
    federated_value FLOAT,  -- Federated (no DP)
    dp_noisy_value FLOAT,  -- With DP
    epsilon FLOAT,  -- Privacy budget spent
    absolute_error FLOAT,  -- |federated - true|
    relative_error FLOAT,  -- |federated - true| / |true|
    suppressed BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT NOW(),
    INDEX(experiment_id, round_number)
);
```

### PrivacyReports Table

```sql
CREATE TABLE privacy_reports (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL FOREIGN KEY,
    round_number INTEGER,
    privacy_mode VARCHAR,
    clipping_norm FLOAT,
    noise_multiplier FLOAT,
    epsilon FLOAT,
    delta FLOAT,
    secure_agg_enabled BOOLEAN,
    contribution_bound INTEGER,
    risk_level VARCHAR,  -- {low, medium, high}
    notes TEXT,
    created_at DATETIME DEFAULT NOW()
);
```

### ClientParticipation Table

```sql
CREATE TABLE client_participation (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL FOREIGN KEY,
    round_number INTEGER NOT NULL,
    client_id_hash VARCHAR NOT NULL,  -- Hashed client ID
    selected BOOLEAN,
    completed BOOLEAN,
    num_examples INTEGER,
    update_norm FLOAT,  -- L2 norm before clipping
    clipped BOOLEAN,  -- Whether update was clipped
    created_at DATETIME DEFAULT NOW(),
    INDEX(experiment_id, round_number, client_id_hash)
);
```

---

## API Architecture

### Request/Response Pattern

```
Client Request
    │
    ▼
┌──────────────────────────────┐
│ FastAPI Routing              │
│ - Method dispatch            │
│ - Path parameter extraction  │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Request Validation           │
│ - Pydantic schema check      │
│ - Type conversion            │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Dependency Injection         │
│ - Database session           │
│ - Authentication (future)    │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Route Handler                │
│ - Business logic             │
│ - Database queries           │
│ - Computation                │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Response Serialization       │
│ - Pydantic model to JSON     │
└────────────┬─────────────────┘
             │
             ▼
JSON Response (200/400/500)
```

### Key API Patterns

**POST /experiments/start**
- Request: `ExperimentStartRequest` (pydantic)
- Handler: `start_experiment(request, db)`
- Response: `ExperimentResponse` (pydantic)

**GET /experiments/{id}**
- Path param: `experiment_id: int`
- Handler: Query DB, return experiment
- Response: `ExperimentResponse`

**POST /experiments/{id}/run-round**
- Path param: `experiment_id: int`
- Handler: `ExperimentRunner.run_single_round(id)`
- Response: `RunRoundResponse` with metrics

---

## Client Architecture

### Client Base Class

```python
class Client(ABC):
    def __init__(self, client_id: str, profile: str, model, data):
        self.client_id = client_id
        self.profile = profile  # e.g., "heavy_user", "light_user"
        self.model = model  # TensorFlow model
        self.data = data  # Client's local data
    
    @abstractmethod
    def compute_local_summary(self) -> dict:
        """Compute privacy-preserving local summary"""
        pass
    
    @abstractmethod
    def local_training(self, global_model, epochs: int):
        """Train on local data and return delta"""
        pass
    
    @abstractmethod
    def dropout_simulation(self, rate: float) -> bool:
        """Simulate client dropout (unavailability)"""
        pass
```

### Telemetry Client (Analytics)

```python
class TelemetryClient(Client):
    def __init__(self, client_id, profile, telemetry_data):
        self.client_id = client_id
        self.profile = profile
        self.data = telemetry_data  # List of (timestamp, event_type, value)
    
    def compute_local_summary(self):
        # Compute: count, sum, mean, quantile of each feature
        count = len(self.data)
        values = [v for _, _, v in self.data]
        
        return {
            "count": count,
            "sum": sum(values),
            "mean": sum(values) / count if count > 0 else 0,
            "quantiles": np.quantile(values, [0.25, 0.5, 0.75])
        }
    
    def local_training(self, global_model, epochs: int):
        # Not used in analytics mode
        raise NotImplementedError("Analytics clients don't train models")
```

### Training Client (EMNIST)

```python
class EMNISTClient(Client):
    def __init__(self, client_id, profile, local_dataset, model):
        self.client_id = client_id
        self.profile = profile
        self.local_dataset = local_dataset
        self.model = model
        self.initial_weights = None
    
    def local_training(self, global_model, epochs: int):
        # 1. Copy global weights
        self.model.set_weights(global_model)
        self.initial_weights = self.model.get_weights()
        
        # 2. Train locally
        for epoch in range(epochs):
            self.model.fit(self.local_dataset, epochs=1, verbose=0)
        
        # 3. Compute delta
        final_weights = self.model.get_weights()
        delta = [f - i for f, i in zip(final_weights, self.initial_weights)]
        
        return np.array(delta, dtype=object)
    
    def dropout_simulation(self, rate: float) -> bool:
        # Random Bernoulli with dropout probability
        return np.random.random() > rate
```

---

## Coordinator Architecture

### ExperimentRunner (Main Orchestrator)

```python
class ExperimentRunner:
    def __init__(self, db: Session):
        self.db = db
        self.round_coordinator = RoundCoordinator(db)
    
    def create_experiment(self, request: ExperimentStartRequest) -> Experiment:
        exp = Experiment(
            name=request.name,
            mode=request.mode,
            dataset=request.dataset,
            num_clients=request.num_clients,
            rounds=request.rounds,
            # ... other fields
        )
        self.db.add(exp)
        self.db.commit()
        return exp
    
    def initialize_clients(self, experiment: Experiment, request):
        # 1. Load dataset
        dataset = load_dataset(request.dataset)
        
        # 2. Assign non-IID partitions
        client_data_partitions = assign_non_iid_partitions(
            dataset,
            experiment.num_clients,
            severity=request.non_iid_severity
        )
        
        # 3. Create clients
        clients = []
        for i, partition in enumerate(client_data_partitions):
            client = create_client(
                client_id=f"client_{i}",
                profile="mixed",
                data=partition,
                model_type=request.model_type
            )
            clients.append(client)
        
        # 4. Register in registry
        registry = ClientRegistry()
        for client in clients:
            registry.register(client)
        
        experiment.status = "initialized"
        self.db.commit()
    
    def run_single_round(self, experiment_id: int) -> dict:
        experiment = self.db.query(Experiment).get(experiment_id)
        round_num = experiment.next_round
        
        # Delegate to RoundCoordinator
        metrics = self.round_coordinator.execute_round(
            experiment=experiment,
            round_number=round_num
        )
        
        # Save metrics to DB
        round_metric = RoundMetric(
            experiment_id=experiment_id,
            round_number=round_num,
            **metrics
        )
        self.db.add(round_metric)
        self.db.commit()
        
        experiment.next_round += 1
        if experiment.next_round > experiment.rounds:
            experiment.status = "completed"
        else:
            experiment.status = "running"
        self.db.commit()
        
        return metrics
    
    def run_full_experiment(self, experiment_id: int):
        experiment = self.db.query(Experiment).get(experiment_id)
        all_metrics = []
        
        for round_num in range(experiment.next_round, experiment.rounds + 1):
            metrics = self.run_single_round(experiment_id)
            all_metrics.append(metrics)
        
        return {
            "experiment_id": experiment_id,
            "rounds_completed": experiment.rounds,
            "all_metrics": all_metrics
        }
```

### RoundCoordinator (Per-Round Execution)

```python
class RoundCoordinator:
    def execute_round(self, experiment: Experiment, round_number: int) -> dict:
        # Phase 1: Selection
        selected_clients = self.select_clients(experiment)
        
        # Phase 2: Distribute
        global_model = self.get_global_model(experiment)
        for client in selected_clients:
            client.receive_model(global_model)
        
        # Phase 3: Local Computation
        updates = []
        for client in selected_clients:
            # Check dropout
            if not client.dropout_simulation(experiment.dropout_rate):
                continue  # Client dropped out
            
            if experiment.mode in ["fedavg", "fedavg_secureagg", "fedavg_secureagg_dp"]:
                update = client.local_training(experiment.local_epochs)
            else:  # Analytics mode
                update = client.compute_local_summary()
            
            updates.append(update)
        
        # Phase 4: Privacy Processing
        if experiment.mode == "fedavg_secureagg_dp":
            # Clipping
            updates = [self.clip_update(u, experiment.clipping_norm) for u in updates]
            
            # Secure Aggregation
            masked_updates, masks = zip(*[self.add_mask(u) for u in updates])
            
            # Aggregation
            aggregate = np.mean(masked_updates, axis=0) - np.mean(masks, axis=0)
            
            # DP Noise
            aggregate = self.add_dp_noise(aggregate, experiment.epsilon)
        else:
            # Simple average
            aggregate = np.mean(updates, axis=0)
        
        # Phase 5: Model Update
        if experiment.mode != "federated_analytics":
            new_weights = self.get_global_weights(experiment) + aggregate
            self.save_global_model(experiment, new_weights)
        
        # Phase 6: Evaluation
        metrics = self.evaluate(experiment, round_number)
        
        return metrics
    
    def select_clients(self, experiment: Experiment) -> list[Client]:
        registry = ClientRegistry()
        all_clients = registry.get_all_clients()
        
        # Random selection
        selected = np.random.choice(
            all_clients,
            size=min(experiment.clients_per_round, len(all_clients)),
            replace=False
        )
        
        return selected.tolist()
```

---

## Privacy Accounting

### Privacy Budget Tracking

```python
class PrivacyAccountant:
    def __init__(self, epsilon: float, delta: float):
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.spent_epsilon = 0.0
        self.history = []
    
    def charge(self, epsilon_used: float):
        """Add to privacy cost"""
        self.spent_epsilon += epsilon_used
        self.history.append({
            "round": len(self.history),
            "epsilon_used": epsilon_used,
            "epsilon_cumulative": self.spent_epsilon,
            "remaining": self.target_epsilon - self.spent_epsilon
        })
    
    def report(self):
        return {
            "spent": self.spent_epsilon,
            "remaining": self.target_epsilon - self.spent_epsilon,
            "exceeded": self.spent_epsilon > self.target_epsilon,
            "history": self.history
        }
```

### Composition Rules

**Basic Sequential Composition:**
```
ε_total = Σ ε_i  (per round)
```

**Parallel Composition:**
```
If each query operates on disjoint data:
ε_total ≈ ε (single query)
```

**Advanced Composition (Rényi):**
```
Tighter bounds using information theory
ε_total ≈ sqrt(2k * ln(1/δ) * α * σ^2) / ε
where k = rounds, α = order, σ = noise scale
```

---

## Failure Modes & Mitigations

| Failure Mode | Impact | Mitigation |
|--------------|--------|-----------|
| **Client Dropout** | Reduced aggregation | Select more clients per round |
| **Database Locked** | API unresponsive | Use PostgreSQL for concurrency |
| **Memory Overflow** | Training crash | Reduce batch size, client count |
| **Network Partition** | Updates lost | Implement client-side queueing |
| **Malicious Client** | Model poisoning | Byzantine-robust aggregation |
| **Privacy Budget Exhausted** | ε < 0 | Cutoff queries, increase ε |
| **Model Divergence** | Accuracy drops | Reduce learning rate, epochs |

---

## Scalability Considerations

### Vertical Scaling (Single Machine)

| Resource | Bottleneck | Limit |
|----------|-----------|-------|
| **RAM** | Model size + client data | ~100 GB (TensorFlow) |
| **CPU** | Training loops | 32+ cores |
| **Disk** | Database, checkpoints | ~1 TB SSD |

**Optimization:**
- Reduce batch size
- Reduce clients_per_round
- Use quantization (int8 models)

### Horizontal Scaling (Distributed)

| Component | Scaling Strategy |
|-----------|------------------|
| **API Server** | Load balance (nginx) |
| **Database** | Sharding by experiment_id |
| **Dashboard** | Multiple instances, cached results |
| **Clients** | Distributed simulator or real edge devices |

**Deployment:**
- Kubernetes StatefulSet for coordinator
- Horizontal Pod Autoscaling for API
- Redis for session caching

---

**For implementation details, see [CODE_REFERENCE.md](CODE_REFERENCE.md). For deployment, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).**
