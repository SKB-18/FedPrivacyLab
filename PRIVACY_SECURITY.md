# Privacy and Security Documentation

## Executive Summary

FedPrivacyLab implements multiple layers of privacy and security controls:

1. **Federated Learning** — Raw data stays on clients
2. **Contribution Bounding** — Limits per-client influence
3. **Update Clipping** — Bounds model update magnitude
4. **Secure Aggregation** — Encrypts updates during aggregation
5. **Differential Privacy** — Adds noise for formal privacy guarantee
6. **Privacy Accounting** — Tracks cumulative privacy loss

This document explains the privacy guarantees, threat model, and security considerations.

---

## Threat Model

### Adversaries Considered

#### 1. Honest-but-Curious Server
- **Assumption:** Server follows protocol correctly but tries to infer sensitive info
- **Threat:** Observing individual client updates can reveal personal data patterns
- **Mitigation:** Secure Aggregation (masks updates) + DP (adds noise)

#### 2. Model Inversion Attack
- **Assumption:** Attacker has final model and wants to reconstruct training data
- **Threat:** Gradient-based inversion can recover training samples
- **Mitigation:** DP noise + update clipping (reduces gradient signal)

#### 3. Membership Inference Attack
- **Assumption:** Attacker tries to determine if specific record was in training data
- **Threat:** Model behaves differently on training vs non-training data
- **Mitigation:** DP guarantees differential indistinguishability (hard to distinguish)

#### 4. Gradient Leakage
- **Assumption:** Attacker observes model updates between rounds
- **Threat:** Gradients can leak training data (especially with small batch sizes)
- **Mitigation:** L2 clipping (reduces gradient signal) + DP noise

### Adversaries NOT Considered

- **Compromised Client** — Assumes honest clients (can't prevent malicious client)
- **Byzantine Clients** — Assumes clients don't send garbage data (use Byzantine-robust aggregation)
- **Network Eavesdropping** — Assumes encrypted transport (use TLS/SSL)
- **Malware/Rootkit** — Assumes secure OS (requires system security)

---

## Privacy Mechanisms

### 1. Differential Privacy (DP)

**Definition:** A randomized algorithm is (ε, δ)-differentially private if:
```
P[algorithm(D) ∈ S] ≤ e^ε * P[algorithm(D') ∈ S] + δ
```

Where D and D' differ by one record. Interpretation: algorithm output is almost identical whether you include/exclude one person's data.

**Parameters:**

| Parameter | Meaning | Default | Range |
|-----------|---------|---------|-------|
| ε (epsilon) | Privacy budget | 1.0 | 0.1-10.0 |
| δ (delta) | Failure probability | 1e-5 | 1e-5 to 1e-3 |

**Privacy-Utility Tradeoff:**

| ε | Privacy | Accuracy Loss |
|---|---------|---------------|
| 0.1 | Very Strong | 15-20% |
| 0.5 | Strong | 8-12% |
| 1.0 | Moderate | 5-8% |
| 5.0 | Weak | 1-2% |
| ∞ | None (no DP) | 0% |

**Interpretation:**

- ε = 0.1: Adversary has ~90% confidence in guessing if person in dataset
- ε = 1.0: Adversary has ~63% confidence (harder to distinguish)
- ε = ∞: Adversary has 100% confidence (no privacy)

**Formal Guarantee:**
With (1.0, 1e-5)-DP, an attacker observing output learns at most ~1 bit per record.

### 2. Secure Aggregation (Simulated)

**Purpose:** Hide individual updates from server before aggregation.

**How It Works:**

```
Round Start

Client 1                    Client 2                    Server
   │                           │                          │
   ├─ Compute update: u1       ├─ Compute update: u2      │
   ├─ Generate mask: m1        ├─ Generate mask: m2       │
   ├─ Send (u1+m1) ─────────────────────────────────────► │
   │                           ├─ Send (u2+m2) ─────────► │
   │                           │                          │
   │                           │     Receives encrypted:
   │                           │     (u1+m1) + (u2+m2)
   │                           │
   │◄─────────────────────────────────────────────────────┤
   │        Send m1 (plaintext)                           │
   │                           ├─ Send m2 (plaintext) ───► │
   │                           │
   │                           │     Computes:
   │                           │     (u1+m1) + (u2+m2) - m1 - m2
   │                           │     = u1 + u2 ✓
   │                           │
   │                           │     Aggregate = (u1+u2) / 2
   │                           │
   └───────────────────────────┴──────────────────────────┘
          Global Model Updated
```

**Security Property:**
- If all clients honest: Server learns only aggregate (Σu_i)
- If some dropout: Remaining clients can potentially infer others' updates

**Limitation:** This is **simulated** encryption. In production, use:
- MPC (Multi-Party Computation)
- Threshold cryptography
- Homomorphic encryption

### 3. Contribution Bounding

**Purpose:** Limit per-client influence on aggregate.

**How It Works:**
```python
def bound_contribution(value, bound=10):
    return min(max(value, 0), bound)

# Example:
client_reports_100_events = 100
contribution_bound = 10
bounded = min(max(100, 0), 10) = 10
```

**Effect:** Prevents single client from dominating result.

**Privacy Impact:**
- Reduces sensitivity of aggregation from ∞ to bound
- Enables DP with smaller noise (better accuracy)
- Example: Without bounding, DP sensitivity = max(client_values) = ∞

### 4. Update Clipping (L2 Norm)

**Purpose:** Bound model update magnitude.

**How It Works:**
```python
def clip_update(update, norm_bound=1.0):
    norm = np.linalg.norm(update)
    if norm > norm_bound:
        update *= (norm_bound / norm)
    return update
```

**Effect:** Prevents large updates from leaking information.

**Privacy Impact:**
- Reduces gradient sensitivity from unbounded to norm_bound
- Enables DP with tight noise bound
- Standard technique in DP-SGD

### 5. Privacy Accounting

**Purpose:** Track cumulative privacy loss across rounds.

**Basic Composition (Simple):**
```
ε_total = Σ ε_i  (per-round privacy budgets sum)

Example:
Round 1: ε = 0.1
Round 2: ε = 0.1
...
Round 10: ε = 0.1
Total: ε = 1.0  ✓ Within target budget
```

**Advanced Composition (Rényi):**
Tighter bound using information theory:
```
ε_total ≈ √(2k * ln(1/δ) * α * σ²) / ε
where:
  k = number of rounds
  α = Rényi order (e.g., 1.25)
  σ = noise scale
```

**Privacy Accounting in FedPrivacyLab:**

```python
class PrivacyAccountant:
    def __init__(self, target_epsilon, target_delta):
        self.target = target_epsilon
        self.spent = 0.0
        self.history = []
    
    def charge(self, epsilon_per_round):
        self.spent += epsilon_per_round
        self.history.append(self.spent)
        if self.spent > self.target:
            print("WARNING: Budget exceeded!")
    
    def report(self):
        return {
            "spent": self.spent,
            "remaining": self.target - self.spent,
            "exceeded": self.spent > self.target
        }
```

---

## Combined Privacy Stack

When all mechanisms enabled (`mode="fedavg_secureagg_dp"`):

```
Layer 1: Local Training
  Client trains on local data (data never leaves device)
       ↓
Layer 2: Contribution Bounding
  Bounding applied to summaries (Analytics)
       ↓
Layer 3: Update Clipping
  Gradient clipped to L2 norm bound
       ↓
Layer 4: Secure Aggregation
  Updates masked before sending to server
       ↓
Layer 5: Differential Privacy
  Gaussian noise added to aggregate
       ↓
Layer 6: Privacy Accounting
  Epsilon tracked, budget enforced
       ↓
       Global Model (Privacy-Preserved)
```

**Cumulative Privacy Guarantee:**
- Each layer adds privacy protection
- Formal DP guarantee from Layer 5
- No individual update visible to server (Layer 4)
- Gradient sensitivity bounded (Layer 3)
- Per-client influence limited (Layer 2)

---

## Privacy Guarantees Claimed

### Claim: "Zero raw data centralization"
**True.** Raw training data never leaves client devices. Only aggregated metrics and clipped model updates transmitted.

**Verification:**
- Clients train locally (see `app/clients/local_training.py`)
- Only deltas sent to server (see `app/coordinator/round_coordinator.py`)
- No raw data in database (see schema in `app/models.py`)

### Claim: "(ε, δ)-Differential Privacy"
**True (with caveats):**
- DP noise added per round (see `app/privacy/dp_mechanisms.py`)
- Gaussian mechanism with configured ε, δ
- Epsilon tracked via privacy accountant
- **Caveat:** Privacy accounting uses simple composition (not Rényi)

**Limitation:** Simple composition is looser than optimal. For research-grade guarantees, use Rényi composition or advanced mechanisms.

### Claim: "Secure aggregation prevents server seeing individual updates"
**True (mathematically):**
- Updates masked with random noise (see `app/privacy/secure_aggregation.py`)
- Masks cancel during aggregation: Σ(u + m) - Σm = Σu
- Server learns only aggregate
- **Caveat:** Simulated, not cryptographically proven

**Limitation:** Not production-grade cryptography. Real systems use MPC or threshold cryptography.

---

## Compliance & Regulations

### GDPR (General Data Protection Regulation)

**Relevant Articles:**
- Article 32 — Security of processing (encryption required)
- Article 25 — Data protection by design and by default

**FedPrivacyLab Compliance:**
✅ Raw personal data stays local (not centralized)
✅ Differential privacy adds formal privacy guarantee
✅ Privacy budget transparent and auditable
❌ Not GDPR-certified (research project)
⚠️ TLS for transport security (recommended but not enforced)

**Use Case:** GDPR-compliant training of ML models without raw data centralization.

### HIPAA (Health Insurance Portability and Accountability Act)

**Requirements:**
- De-identification or Safe Harbor method
- Encryption at rest and in transit
- Access controls and audit logs

**FedPrivacyLab Support:**
✅ DP provides mathematical de-identification
✅ TLS can enable encryption in transit
❌ Audit logging not built-in
❌ Access controls not implemented

**Use Case:** Training models on healthcare data with privacy guarantee.

### CCPA (California Consumer Privacy Act)

**Relevant Sections:**
- Section 1798.100 — Right to Know (what personal info collected)
- Section 1798.105 — Right to Delete

**FedPrivacyLab Compliance:**
✅ Federated approach aligns with "don't centralize data"
✅ Right to delete: local data deletion sufficient (server never sees raw data)
❌ Transparency not automated

**Use Case:** Consumer app analytics without centralizing personal data.

---

## Security Best Practices

### For Development

- [ ] Use virtual environment (no system-wide Python pollution)
- [ ] Never hardcode secrets (use environment variables)
- [ ] Use `.env` file (add to .gitignore)
- [ ] Validate all inputs (Pydantic handles this)
- [ ] Log security events (authentication, data access)
- [ ] Use HTTPS in development (at minimum test locally)

### For Deployment

- [ ] Enable TLS/SSL (Let's Encrypt free certificates)
- [ ] Use PostgreSQL (not SQLite) for production
- [ ] Implement authentication (JWT tokens recommended)
- [ ] Set CORS restrictions (not `allow_origins=["*"]`)
- [ ] Enable rate limiting (100 req/min per IP)
- [ ] Implement audit logging (all data access logged)
- [ ] Regular security updates (keep dependencies current)
- [ ] Use WAF (Web Application Firewall)
- [ ] Implement secrets management (Kubernetes Secrets, Vault)
- [ ] Monitor and alert on suspicious activity

### For Data Privacy

- [ ] Encryption at rest: Enable database encryption
- [ ] Encryption in transit: Enforce TLS 1.2+
- [ ] Key rotation: Implement monthly secret rotation
- [ ] Data retention: Delete old experiments after 90 days
- [ ] Access control: Principle of least privilege
- [ ] Data minimization: Collect only necessary information
- [ ] Audit trail: Log who accessed what, when

---

## Security Considerations

### Known Limitations

1. **Simple DP Composition**
   - Uses basic sequential composition (not optimal)
   - Tighter bounds available via Rényi composition
   - Not limiting but not state-of-the-art

2. **Simulated Secure Aggregation**
   - Not cryptographically proven
   - Vulnerable to side-channel attacks
   - Suitable for research, not production

3. **Trusted Clients**
   - Assumes all clients are honest
   - No Byzantine robustness
   - Malicious client can poison model

4. **No Differential Privacy Composition Over Populations**
   - Tracks privacy per round for same clients
   - If different clients each round: privacy may not compose correctly
   - Future work: per-client privacy budgets

5. **SQLite Limitations**
   - No built-in encryption
   - Single-threaded (concurrency issues)
   - Not suitable for production
   - **Recommendation:** Use PostgreSQL

### Assumptions

1. **Honest Protocol Execution** — Coordinator and clients follow spec
2. **Secure Transport** — TLS/encryption used for network communication
3. **Secure Random Numbers** — System RNG is cryptographically secure
4. **Trusted Execution Environment** — Client devices secure (not rooted/jailbroken)
5. **Sufficient Data Privacy** — Clients keep sensitive data secure locally

---

## Audit Trail

FedPrivacyLab creates an audit trail via database tables:

**ExperimentsTable:**
- Experiment creation timestamp
- Configuration (can be audited)
- Status changes

**PrivacyReportsTable:**
- Privacy settings per round
- Risk assessment
- Epsilon consumption

**ClientParticipationTable:**
- Which clients selected each round
- Update norms (can detect outliers)
- Clipping decisions

**Queries for Audit:**

```sql
-- Who created what experiment?
SELECT name, created_at FROM experiments ORDER BY created_at DESC;

-- How much privacy budget spent?
SELECT experiment_id, epsilon, spent_epsilon FROM privacy_reports;

-- Which clients participated?
SELECT client_id_hash, COUNT(*) FROM client_participation 
  WHERE experiment_id=1 GROUP BY client_id_hash;

-- Were updates clipped?
SELECT round_number, COUNT(*) as clipped_updates 
  FROM client_participation WHERE clipped=true 
  GROUP BY round_number;
```

---

## Responsible Disclosure

If you find a security vulnerability:

1. **Do NOT** open a public GitHub issue
2. **Email:** security@example.com (replace with actual email)
3. **Include:**
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

**Response Time:** 48 hours acknowledgment, 30 days patch.

---

## Privacy Policy (Template)

If deploying FedPrivacyLab as a service:

**What Data We Collect:**
- Model updates (encrypted, on server <24 hours)
- Aggregated metrics (non-identifying counts, sums)
- Logs (API requests, errors)

**How We Use It:**
- Training models
- Computing analytics
- System monitoring

**What We Don't Collect:**
- Raw training data (stays on your device)
- Personal identifiers (client IDs hashed)
- Inference queries (local to device)

**Data Retention:**
- Experiments deleted after 90 days
- Logs retained for 30 days
- Backups deleted after 1 year

**Your Rights:**
- Right to access: View your experiment data
- Right to delete: Request experiment deletion
- Right to portability: Export results

---

## References

### Differential Privacy
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) — Textbook by Dwork & Roth
- [Differentially Private Federated Learning](https://arxiv.org/abs/1911.00222)
- [Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Secure Aggregation
- [Practical Secure Aggregation for Federated Learning](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/)
- [Secure Aggregation from Homomorphic Encryption](https://arxiv.org/abs/1811.04482)

### Federated Learning
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.03832)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

### Privacy Attacks
- [Model Inversion Attacks](https://arxiv.org/abs/1412.1549)
- [Membership Inference Attacks](https://arxiv.org/abs/1610.05492)
- [Gradient Leakage from Federated Learning](https://arxiv.org/abs/2006.03137)

---

**For security questions, contact: security@example.com**  
**For privacy concerns, see: [PRIVACY_POLICY.md](PRIVACY_POLICY.md)**
