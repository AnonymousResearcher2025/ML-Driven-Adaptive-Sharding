# ML-Driven Adaptive Sharding: Comprehensive Project Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Theoretical Foundation](#theoretical-foundation)
5. [System Architecture](#system-architecture)
6. [Implementation Details](#implementation-details)
7. [Module Interactions](#module-interactions)
8. [Data Flow](#data-flow)
9. [Algorithm Walkthrough](#algorithm-walkthrough)
10. [Code Organization](#code-organization)
11. [Key Design Decisions](#key-design-decisions)
12. [Usage Guide](#usage-guide)
13. [Testing Strategy](#testing-strategy)
14. [Performance Considerations](#performance-considerations)
15. [Extension Points](#extension-points)

---

## 1. Executive Summary

### What This Project Does

This project implements a **dynamic blockchain sharding system** that uses **machine learning clustering** to intelligently partition blockchain accounts across multiple shards. The goal is to **minimize cross-shard communication** (which is expensive) while **maintaining balanced load** across shards and **minimizing the cost of moving accounts** between shards.

### The Core Problem

In blockchain systems:
- **Sharding** splits the blockchain into multiple parallel chains (shards) to improve throughput
- **Cross-shard transactions** (transactions accessing accounts in different shards) are expensive because they require coordination
- **Account placement** determines which shard stores which accounts
- **Transaction patterns change over time**, so static partitioning becomes inefficient

### Our Solution

We implement **Algorithm 2** from the research paper, which:
1. **Monitors** transaction patterns continuously
2. **Builds a graph** where accounts are nodes and transaction frequency is edge weight
3. **Uses ML clustering** to find better account-to-shard assignments
4. **Decides intelligently** whether the benefit of repartitioning outweighs the migration cost
5. **Migrates accounts** only when cost-effective

### Key Innovation

The system balances three competing objectives:
- ✅ **Minimize cross-shard transactions** (clustering objective)
- ✅ **Minimize migration overhead** (stability objective)
- ✅ **Maintain balanced shards** (fairness objective)

---

## 2. Problem Statement

### Formal Problem Definition

**Given:**
- `n` accounts that need to be partitioned into `s` shards
- Time-varying transaction patterns between accounts
- Cost parameters: `α` (migration), `β` (cross-shard penalty), `γ` (threshold)

**Find:**
- An account-to-shard mapping `π_t: A(t) → [s]` that evolves over time
- Minimize total cost = processing cost + migration cost

**Subject to:**
- Balance constraint: Each shard should have approximately `n/s` accounts (within tolerance `τ`)
- Online constraint: Decisions made without future knowledge

### Why This is Hard

1. **Dynamic patterns**: Transaction patterns change unpredictably
2. **Migration cost**: Moving accounts is expensive (state transfer, consistency)
3. **Balance requirement**: Can't just put all accounts in one shard
4. **Online setting**: Must make decisions in real-time without knowing future

### Real-World Analogy

Imagine organizing people into meeting rooms:
- People who frequently talk should be in the same room (minimize cross-room communication)
- But moving people between rooms is disruptive (migration cost)
- Rooms should be roughly equal size (balance)
- Conversation patterns change throughout the day (dynamic)

---

## 3. Solution Overview

### Three-Phase Algorithm

Our implementation follows **Algorithm 2** from the paper, which runs periodically:

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Global Cost Aggregation                           │
│ - Each shard reports: intra-shard txns (Li) and           │
│   cross-shard txns (Xi)                                    │
│ - Compute current processing cost                          │
│   C_proc^current = ΣLi + β·ΣXi                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Clustering-Based Repartition Evaluation          │
│ - Build transaction graph GT(t)                            │
│ - Run ML clustering → new partition S*(t)                  │
│ - Enforce balance constraint (repair if needed)            │
│ - Compute new processing cost C_proc^cluster               │
│ - Identify migration set M(t) = accounts that change      │
│ - Compute migration cost C_mig = Σ α·d(Si, Sj)           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Migration Decision Rule                          │
│ - Compute savings: Δ = C_proc^current - C_proc^cluster    │
│ - Decision: Migrate if Δ > γ·C_mig                        │
│                                                             │
│ IF YES:                     IF NO:                         │
│ - Pause scheduler           - Continue with                │
│ - Migrate accounts            current mapping              │
│ - Update mapping πt+1 ← πt* - πt+1 ← πt                   │
│ - Resume scheduler                                          │
└─────────────────────────────────────────────────────────────┘
```

### Cost Model

**Transaction Processing Cost:**
```
For each transaction Ti:
  - If all accessed accounts in same shard: cost = 1
  - If cross-shard: cost = β · max_distance
  
Total: C_proc(t) = Σ cost(Ti)
```

**Migration Cost:**
```
For each migrating account a:
  - cost = α · distance(source_shard, dest_shard)
  
Total: C_mig(t) = Σ α · d(Si, Sj)
```

**Decision Rule:**
```
Savings = C_proc^current - C_proc^cluster

IF Savings > γ · Migration_Cost THEN
  Migrate (benefit exceeds cost)
ELSE
  Don't migrate (too expensive)
```

### Key Parameters

| Parameter | Meaning | Typical Value | Constraint |
|-----------|---------|---------------|------------|
| `α` | Migration cost | 10.0 | α > β |
| `β` | Cross-shard penalty | 5.0 | β > 1 |
| `γ` | Migration threshold | 1.5 | γ ≥ 1 |
| `τ` | Balance tolerance | 1.2 | τ ≥ 1 |
| `s` | Number of shards | 8 | s ≥ 2 |
| `k` | Max accounts/txn | 5 | k ≥ 1 |

---

## 4. Theoretical Foundation

### Transaction Graph Model

**Definition:** `G_T(t) = (V(t), E(t), W(t))`

- **V(t):** Set of active accounts at time t
- **E(t):** Edges between accounts that appear in transactions together
- **W(t):** Edge weights = transaction frequency/intensity

**Example:**
```
Transaction 1: {Account_1, Account_2, Account_3}
  → Creates edges: (1,2), (2,3), (1,3) with weight 1.0

Transaction 2: {Account_1, Account_2}
  → Increases weight of edge (1,2) to 2.0
  
Result: Accounts 1 and 2 have strong connection (should be in same shard)
```

### Clustering Objective

**Goal:** Partition graph to minimize cut weight

**Cut weight** = sum of edge weights crossing partition boundaries

**Spectral Clustering:** Uses eigenvectors of graph Laplacian to find partitions that minimize normalized cut:
```
Normalized_Cut = Cut(S, S̄) / Vol(S) + Cut(S, S̄) / Vol(S̄)
```

This naturally aligns with our objective of minimizing cross-shard communication.

### Balance Constraint (Section 2.5, Constraint 5)

**Formula:**
```
|V(t)|/(τ·s) ≤ |Si(t)| ≤ τ·|V(t)|/s    ∀i ∈ [s]
```

**Meaning:**
- Lower bound: Each shard must have at least `n/(τ·s)` accounts
- Upper bound: Each shard can have at most `τ·n/s` accounts
- τ=1.0 means perfect balance (n/s accounts per shard)
- τ=1.2 means 20% tolerance (allows some imbalance)

**Example with n=100, s=4, τ=1.2:**
```
Lower bound = 100/(1.2·4) = 20.83 accounts
Upper bound = 1.2·100/4 = 30 accounts

Valid: Shard sizes [25, 25, 25, 25] ✓
Valid: Shard sizes [21, 28, 26, 25] ✓
Invalid: Shard sizes [10, 30, 30, 30] ✗ (shard 0 too small)
Invalid: Shard sizes [35, 20, 25, 20] ✗ (shard 0 too large)
```

---

## 5. System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    run_simulation.py                        │
│                   (Main Entry Point)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
┌────────────────┐        ┌──────────────────┐
│ Configuration  │        │ Logging & Utils  │
│  - Parameters  │        │  - Logger setup  │
│  - Validation  │        │  - Metrics       │
└────────────────┘        └──────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│              AdaptiveShardingAlgorithm                      │
│              (Algorithm 2 Implementation)                   │
└───┬─────────────┬─────────────┬─────────────┬──────────────┘
    ↓             ↓             ↓             ↓
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│Transaction│ │Clustering│ │   Cost   │ │  Scheduler   │
│  Graph   │ │  Engine  │ │  Model   │ │  Controller  │
│          │ │          │ │          │ │              │
│ GT(t)    │ │ Spectral │ │ C_proc   │ │ Pause/Resume │
│ Builder  │ │ K-Means  │ │ C_mig    │ │              │
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
    ↑             ↑             ↑
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Account  │ │ Balance  │ │  Migration   │
│  Model   │ │ Repair   │ │  Engine      │
└──────────┘ └──────────┘ └──────────────┘
    ↑
┌──────────────────────────────────┐
│     TransactionGenerator         │
│   (Synthetic Workload)           │
└──────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| **AdaptiveShardingAlgorithm** | Orchestrates Algorithm 2 | `execute()`, `apply_migration()` |
| **TransactionGraph** | Maintains G_T(t) | `add_transaction()`, `get_adjacency_matrix()` |
| **ClusteringEngine** | Partitions accounts | `cluster()` (spectral/kmeans/minibatch) |
| **CostModel** | Computes all costs | `compute_processing_cost()`, `compute_migration_cost()` |
| **BalanceRepair** | Enforces constraints | `repair()` |
| **SchedulerController** | Manages pause/resume | `request_pause()`, `resume()` |
| **TransactionGenerator** | Creates workload | `generate_transactions()` |
| **SimulationDriver** | Runs end-to-end | `run()` |

---

## 6. Implementation Details

### 6.1 Core Data Structures

#### Account Model (`models/account.py`)

```python
@dataclass
class Account:
    account_id: int          # Unique identifier
    shard_id: int           # Current shard assignment πt(a)
    status: AccountStatus   # ACTIVE, MIGRATING, INACTIVE
    state_size: int        # For migration cost (optional)
```

**Purpose:** Represents a blockchain account/address

#### Transaction Model (`models/transaction.py`)

```python
@dataclass
class Transaction:
    transaction_id: int
    timestamp: int                    # Time step t
    accessed_accounts: Set[int]       # ATi(t) - accounts accessed
    home_shard: int                   # Originating shard
    weight: float = 1.0              # Frequency/intensity
    
    def is_cross_shard(self, mapping) -> bool:
        """Check if transaction spans multiple shards"""
        shards = {mapping[acc] for acc in accessed_accounts}
        return len(shards) > 1
```

**Purpose:** Represents a transaction (Ti) that accesses multiple accounts

**Key constraint:** `|accessed_accounts| ≤ k` (transaction width)

#### Transaction Graph (`models/graph.py`)

```python
class TransactionGraph:
    def __init__(self):
        self.graph = nx.Graph()           # NetworkX graph
        self.edge_weights: Dict = {}      # (u,v) → weight
        self.time_step: int = 0
    
    def add_transaction(self, txn):
        """Add transaction to graph, update edge weights"""
        for each pair (u,v) in txn.accessed_accounts:
            if edge exists:
                graph[u][v]['weight'] += txn.weight
            else:
                graph.add_edge(u, v, weight=txn.weight)
    
    def get_adjacency_matrix(self) -> (np.ndarray, List[int]):
        """Export weighted adjacency matrix for clustering"""
        # Returns: (matrix, node_ids)
```

**Purpose:** Maintains G_T(t) = (V(t), E(t), W(t))

**Key operations:**
- Building graph from transactions (incremental)
- Exporting adjacency matrix for clustering
- Computing cut weights and volumes

#### Shard Model (`models/shard.py`)

```python
@dataclass
class Shard:
    shard_id: int
    accounts: Set[int]                    # Si(t)
    num_nodes: int                        # |Si| nodes running consensus
    
    # Statistics (Phase 1)
    intra_shard_transactions: int         # Li(t)
    cross_shard_transactions: int         # Xi(t)
```

**Purpose:** Represents a blockchain shard with its accounts

### 6.2 Algorithm 2 Implementation

The core algorithm is in `algorithm/adaptive_sharding.py`:

```python
class AdaptiveShardingAlgorithm:
    def execute(self, transactions, current_time):
        """Execute Algorithm 2 - returns MigrationDecision"""
        
        # PHASE 1: Global Cost Aggregation
        current_cost, intra, cross = self._phase1_cost_aggregation(txns)
        
        # PHASE 2: Clustering-Based Repartition
        new_mapping, cluster_cost, mig_cost, mig_set = \
            self._phase2_repartition_evaluation(txns)
        
        # PHASE 3: Migration Decision
        should_migrate, delta = self._phase3_migration_decision(
            current_cost, cluster_cost, mig_cost
        )
        
        return MigrationDecision(...)
```

#### Phase 1: Global Cost Aggregation

```python
def _phase1_cost_aggregation(self, transactions):
    """
    Each shard computes:
      - Li(t): count of intra-shard transactions
      - Xi(t): count of cross-shard transactions
    
    Reference shard aggregates:
      C_proc^current = Σ Li(t) + β·Σ Xi(t)
    """
    
    # Compute per-shard statistics
    intra_shard, cross_shard = self.cost_model.compute_shard_local_stats(
        transactions, self.current_mapping, num_shards
    )
    
    # Aggregate globally
    total_intra = sum(intra_shard.values())
    total_cross = sum(cross_shard.values())
    
    current_cost = total_intra + self.config.beta * total_cross
    
    return current_cost, intra_shard, cross_shard
```

**How it works:**
1. Iterate through all transactions
2. For each transaction, check if all accessed accounts in same shard
3. If yes: increment Li(t) for home shard
4. If no: increment Xi(t) for home shard
5. Compute total cost using formula

#### Phase 2: Clustering-Based Repartition

```python
def _phase2_repartition_evaluation(self, transactions):
    """
    1. Extract GT(t) adjacency matrix
    2. Run ML clustering → S*(t)
    3. Enforce balance constraint
    4. Compute new costs under S*(t)
    5. Identify migration set M(t)
    6. Compute migration cost
    """
    
    # Step 1: Get graph data
    adjacency_matrix, node_ids = self.graph.get_adjacency_matrix()
    
    # Step 2: Run clustering
    new_mapping_raw = self.clustering_method.cluster(
        adjacency_matrix, node_ids
    )
    
    # Step 3: Enforce balance
    new_mapping = enforce_balance(
        new_mapping_raw,
        num_shards=self.config.num_shards,
        tau=self.config.tau,
        adjacency_matrix=adjacency_matrix,
        node_ids=node_ids
    )
    
    # Step 4: Compute clustered cost
    clustered_cost, intra_new, cross_new = \
        self.cost_model.compute_clustered_processing_cost(
            transactions, new_mapping, num_shards
        )
    
    # Step 5-6: Migration cost
    migration_cost, migration_set = \
        self.cost_model.compute_migration_cost(
            self.current_mapping, new_mapping
        )
    
    return new_mapping, clustered_cost, migration_cost, migration_set
```

**Key insight:** Clustering finds account groupings that minimize edge cut (cross-shard communication)

#### Phase 3: Migration Decision

```python
def _phase3_migration_decision(self, current_cost, cluster_cost, mig_cost):
    """
    Decision rule: Migrate if Δ(t) > γ·C_mig(t)
    
    Where: Δ(t) = C_proc^current(t) - C_proc^cluster(t)
    """
    
    delta = current_cost - cluster_cost  # Expected savings
    threshold = self.config.gamma * mig_cost
    
    should_migrate = (delta > threshold)
    
    return should_migrate, delta
```

**Interpretation:**
- `delta > 0`: Clustering would improve processing cost
- `threshold`: The "break-even" point (migration cost × safety factor)
- Only migrate if savings significantly exceed migration cost

### 6.3 Clustering Implementation

#### Spectral Clustering (`clustering/spectral.py`)

```python
class SpectralClusteringMethod:
    def cluster(self, adjacency_matrix, node_ids):
        """
        Spectral clustering algorithm:
        1. Compute normalized Laplacian: L = D^(-1/2)(D-A)D^(-1/2)
        2. Compute first k eigenvectors
        3. Run k-means on eigenvector embedding
        4. Return account → shard mapping
        """
        
        # sklearn handles Laplacian computation internally
        spectral = SpectralClustering(
            n_clusters=self.num_shards,
            affinity='precomputed',
            random_state=self.random_seed,
            assign_labels='kmeans'
        )
        
        labels = spectral.fit_predict(adjacency_matrix)
        
        # Create mapping: account_id → shard_id
        mapping = {node_ids[i]: int(labels[i]) for i in range(len(node_ids))}
        
        return mapping
```

**Why spectral clustering?**
- Minimizes normalized cut (proven to approximate optimal partition)
- Works well on graph-structured data
- Finds non-convex clusters (unlike k-means on raw features)

#### K-Means with Laplacian Embedding (`clustering/kmeans.py`)

```python
class KMeansClusteringMethod:
    def cluster(self, adjacency_matrix, node_ids):
        """
        Alternative to spectral:
        1. Compute Laplacian eigenmap embedding
        2. Run k-means on embedding
        """
        
        # Compute embedding (similar to spectral step 1-2)
        embedding = self._compute_laplacian_embedding(adjacency_matrix)
        
        # Normalize for k-means
        embedding_norm = normalize(embedding, norm='l2')
        
        # Run k-means
        kmeans = KMeans(n_clusters=self.num_shards, random_state=seed)
        labels = kmeans.fit_predict(embedding_norm)
        
        return mapping
```

**Difference from spectral:** More control over embedding and clustering separately

#### Mini-Batch K-Means (`clustering/kmeans.py`)

```python
class MiniBatchKMeansMethod:
    """Scalable version for large graphs (n > 10,000)"""
    
    def cluster(self, adjacency_matrix, node_ids):
        embedding = self._compute_laplacian_embedding(adjacency_matrix)
        
        # Process in batches
        mbkmeans = MiniBatchKMeans(
            n_clusters=self.num_shards,
            batch_size=self.batch_size,
            random_state=self.random_seed
        )
        labels = mbkmeans.fit_predict(embedding)
        
        return mapping
```

**When to use:** Large-scale simulations (>5000 accounts)

### 6.4 Balance Repair (`clustering/balance_repair.py`)

After clustering, shard sizes may violate balance constraint. The repair algorithm:

```python
class BalanceRepair:
    def repair(self, mapping):
        """
        Enforce: |V(t)|/(τ·s) ≤ |Si(t)| ≤ τ·|V(t)|/s
        
        Strategy:
        1. Identify over-full and under-full shards
        2. Select accounts to move (prefer "loose" connections)
        3. Move to under-full shards (prefer high affinity)
        4. Minimize disruption to clustering quality
        """
        
        # Compute bounds
        lower, upper = self._compute_bounds(num_accounts)
        
        # Find violations
        overfull = [(shard, excess) for shard in range(s) 
                    if size[shard] > upper]
        underfull = [(shard, deficit) for shard in range(s)
                     if size[shard] < lower]
        
        # Greedy repair
        for src_shard, excess in overfull:
            # Get "loosest" accounts (weak intra-shard connections)
            candidates = self._get_migration_candidates(src_shard, excess)
            
            for account in candidates:
                # Find best destination (high affinity + under-full)
                dst_shard = self._find_best_destination(account, underfull)
                
                # Move account
                mapping[account] = dst_shard
```

**Key insight:** Move accounts with weak intra-shard connections to minimize disruption

**Affinity computation:**
```python
def _find_best_destination(self, account, underfull_shards):
    """
    For each underfull shard, compute:
      affinity = Σ edge_weight(account, acc_in_shard)
    
    Choose shard with highest affinity
    """
    
    best_shard = None
    best_affinity = -1
    
    for shard_id in underfull_shards:
        affinity = 0
        for other_account in shard:
            affinity += adjacency[account, other_account]
        
        if affinity > best_affinity:
            best_affinity = affinity
            best_shard = shard_id
    
    return best_shard
```

### 6.5 Cost Computation (`cost/processor.py`)

#### Transaction Processing Cost

```python
class TransactionCostProcessor:
    def compute_transaction_cost(self, accessed_accounts, mapping):
        """
        Cost model from section 2.4.1:
        
        Cost(Ti) = {
            1,      if all accounts in same shard (intra-shard)
            β·di,   otherwise (cross-shard)
        }
        
        where di = max distance between accessed shards
        """
        
        # Get shards accessed by transaction
        accessed_shards = {mapping[acc] for acc in accessed_accounts}
        
        if len(accessed_shards) == 1:
            return 1.0  # Intra-shard
        
        # Cross-shard: compute max distance
        max_dist = 0
        for si in accessed_shards:
            for sj in accessed_shards:
                dist = self.distance_matrix[si, sj]
                max_dist = max(max_dist, dist)
        
        return self.beta * max_dist
```

**Why this cost model?**
- Intra-shard transactions: fast, local consensus
- Cross-shard transactions: expensive coordination (2PC, state verification)
- Distance matters: further shards = higher latency

#### Migration Cost

```python
class MigrationCostProcessor:
    def compute_migration_cost(self, migration_set, current, new):
        """
        Cost model from section 2.4.2:
        
        C_mig(t) = Σ(a ∈ M(t)) α · d(S_πt(a), S_πt*(a))
        
        where:
        - M(t) = {a | πt(a) ≠ πt*(a)} (accounts that change shards)
        - α = base migration cost
        - d(Si, Sj) = distance between shards
        """
        
        total_cost = 0
        
        for account in migration_set:
            src_shard = current[account]
            dst_shard = new[account]
            
            distance = self.distance_matrix[src_shard, dst_shard]
            cost = self.alpha * distance
            
            total_cost += cost
        
        return total_cost
```

**Why α > β?** Migration is more expensive than processing cross-shard transactions (state transfer, consistency protocols)

### 6.6 Scheduler Integration (`scheduler/controller.py`)

The scheduler ensures safe migration (no transactions in-flight):

```python
class SchedulerController:
    def request_pause_for_migration(self, migration_callback):
        """
        Algorithm 2 requirement:
        "Stop Phase 2 of Algorithm 1 after completion of 
         currently processed color"
        
        Implementation:
        1. Set pause flag
        2. Wait for current batch to complete
        3. Enter MIGRATING state
        4. Execute migration callback
        """
        
        with self.lock:
            self.pause_requested = True
            self.migration_callback = migration_callback
    
    def process_batch(self, batch):
        """Process transaction batch"""
        
        if self.pause_requested:
            # Complete current batch first
            self._complete_current_batch()
            
            # Now pause
            self.state = PAUSED
            
            # Execute migration
            self.state = MIGRATING
            self.migration_callback()
            
            return False  # Signal pause
        
        # Normal processing
        self._execute_batch(batch)
        return True
    
    def resume(self):
        """Resume after migration"""
        self.state = RUNNING
```

**Key property:** Transactions never span migration (ensures consistency)

---

## 7. Module Interactions

### 7.1 Execution Flow Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    SIMULATION START                           │
└───────────────────┬───────────────────────────────────────────┘
                    ↓
            ┌───────────────┐
            │ Configuration │ ← Read parameters (α,β,γ,τ,s,k)
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │  Initialize   │ ← Create components:
            │  Components   │   - TransactionGraph
            └───────┬───────┘   - Clustering method
                    │            - CostModel
                    ↓            - Algorithm instance
    ┌───────────────────────────────────────┐
    │ Initialize Account-to-Shard Mapping   │
    │ πt: distribute accounts evenly        │
    └───────────────┬───────────────────────┘
                    ↓
    ┌───────────────────────────────────────┐
    │ FOR t = 0 TO time_steps:              │
    │                                        │
    │   ┌─────────────────────────────┐    │
    │   │ 1. Generate Transactions    │◄───┼── TransactionGenerator
    │   │    T(t) = {T1, T2, ..., Tn} │    │
    │   └──────────┬──────────────────┘    │
    │              ↓                        │
    │   ┌─────────────────────────────┐    │
    │   │ 2. Update Graph GT(t)       │◄───┼── TransactionGraph
    │   │    Add edges, update weights│    │
    │   └──────────┬──────────────────┘    │
    │              ↓                        │
    │   ┌─────────────────────────────┐    │
    │   │ 3. Process Transactions     │◄───┼── Scheduler
    │   │    Execute via scheduler    │    │
    │   └──────────┬──────────────────┘    │
    │              ↓                        │
    │   ┌─────────────────────────────┐    │
    │   │ 4. IF t % interval == 0:    │    │
    │   │                              │    │
    │   │  ┌────────────────────────┐ │    │
    │   │  │ Run Algorithm 2        │◄┼────┼── AdaptiveShardingAlgorithm
    │   │  │                        │ │    │
    │   │  │ Phase 1: Cost Agg     │ │    │
    │   │  │ Phase 2: Clustering   │ │    │
    │   │  │ Phase 3: Decision     │ │    │
    │   │  └──────────┬─────────────┘ │    │
    │   │             ↓                │    │
    │   │  ┌────────────────────────┐ │    │
    │   │  │ IF should_migrate:     │ │    │
    │   │  │ - Pause scheduler      │ │    │
    │   │  │ - Migrate accounts     │ │    │
    │   │  │ - Update πt+1          │ │    │
    │   │  │ - Resume scheduler     │ │    │
    │   │  └────────────────────────┘ │    │
    │   └─────────────────────────────┘    │
    │              ↓                        │
    │   ┌─────────────────────────────┐    │
    │   │ 5. Collect Metrics          │◄───┼── MetricsCollector
    │   └─────────────────────────────┘    │
    │                                        │
    └───────────────┬────────────────────────┘
                    ↓
    ┌───────────────────────────────────────┐
    │ END: Analyze & Report Results         │
    └───────────────────────────────────────┘
```

### 7.2 File Dependency Graph

```
run_simulation.py
    ├─→ config.py
    ├─→ utils/logger.py
    ├─→ algorithm/adaptive_sharding.py
    │       ├─→ models/graph.py
    │       │       ├─→ models/transaction.py
    │       │       └─→ models/account.py
    │       ├─→ clustering/spectral.py
    │       │       ├─→ clustering/base.py
    │       │       └─→ clustering/balance_repair.py
    │       ├─→ cost/processor.py
    │       │       ├─→ cost/migration.py
    │       │       └─→ models/transaction.py
    │       └─→ scheduler/controller.py
    ├─→ simulation/driver.py
    │       ├─→ simulation/generator.py
    │       │       └─→ models/transaction.py
    │       └─→ utils/metrics.py
    └─→ utils/metrics.py
```

### 7.3 Data Flow Through System

#### Step-by-Step Example

**Initial State (t=0):**
```
Accounts: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Shards: 2
Initial Mapping: {0→S0, 1→S0, 2→S0, 3→S0, 4→S0,
                  5→S1, 6→S1, 7→S1, 8→S1, 9→S1}
```

**Transaction Generation (t=1):**
```python
# TransactionGenerator creates:
T1 = Transaction(id=1, accessed={0, 1, 5}, home=S0)  # Cross-shard!
T2 = Transaction(id=2, accessed={2, 3}, home=S0)     # Intra-shard
T3 = Transaction(id=3, accessed={5, 6, 7}, home=S1)  # Intra-shard
```

**Graph Update:**
```python
# TransactionGraph.add_transaction() for T1:
graph.add_edge(0, 1, weight=1.0)
graph.add_edge(0, 5, weight=1.0)  # Cross-shard edge!
graph.add_edge(1, 5, weight=1.0)  # Cross-shard edge!

# After all transactions:
Edges: (0,1):1.0, (0,5):1.0, (1,5):1.0, (2,3):1.0, (5,6):1.0, (6,7):1.0, (5,7):1.0
```

**Cost Computation (Phase 1):**
```python
# Intra-shard transactions: T2, T3 → cost = 2 × 1 = 2
# Cross-shard transactions: T1 → cost = 1 × β = 5
# Total: C_proc^current = 2 + 5 = 7
```

**Clustering (Phase 2):**
```python
# Adjacency matrix shows accounts 0,1,5 strongly connected
# Spectral clustering suggests new partition:
# S0*: {0, 1, 5, 2, 3}  ← Account 5 moved from S1 to S0!
# S1*: {6, 7, 8, 9, 4}  ← Account 4 moved from S0 to S1

# New cost under S*:
# All transactions now intra-shard!
# C_proc^cluster = 3 × 1 = 3
```

**Migration Cost:**
```python
# Migration set M(t) = {4, 5} (changed shards)
# C_mig = 2 accounts × α × d(S0,S1) = 2 × 10 × 1 = 20
```

**Decision (Phase 3):**
```python
# Savings: Δ = 7 - 3 = 4
# Threshold: γ·C_mig = 1.5 × 20 = 30
# Decision: Δ > threshold? → 4 > 30? → NO

# Don't migrate (migration too expensive for small benefit)
```

### 7.4 Module Communication Patterns

#### Pattern 1: Configuration Propagation
```
config.py
    ↓ (parameters)
    ├─→ AdaptiveShardingAlgorithm (α,β,γ,τ,s)
    ├─→ CostModel (α,β,γ)
    ├─→ ClusteringMethod (s, seed)
    └─→ TransactionGenerator (k, locality)
```

#### Pattern 2: Graph Building
```
TransactionGenerator
    ↓ (generates transactions)
TransactionGraph
    ↓ (accumulates edges)
AdaptiveShardingAlgorithm
    ↓ (requests adjacency matrix)
ClusteringMethod
    ↓ (returns new mapping)
BalanceRepair
    ↓ (enforces constraints)
AdaptiveShardingAlgorithm (updated πt*)
```

#### Pattern 3: Cost Computation
```
Transactions + CurrentMapping
    ↓
TransactionCostProcessor
    ├─→ compute_shard_local_stats() → (Li, Xi)
    └─→ compute_processing_cost() → C_proc

CurrentMapping + NewMapping
    ↓
MigrationCostProcessor
    ├─→ compute_migration_set() → M(t)
    └─→ compute_migration_cost() → C_mig
```

---

## 8. Data Flow

### 8.1 Complete Data Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ INPUT: System Parameters                                 │
│ α=10, β=5, γ=1.5, τ=1.2, s=8, k=5, n=1000              │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ INITIALIZATION                                           │
│ πt: Account → Shard mapping                             │
│ {0→S0, 1→S0, ..., 124→S0, 125→S1, ..., 999→S7}         │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ TIME STEP t                                              │
│                                                          │
│ [TransactionGenerator]                                   │
│   ↓ generates                                           │
│ T(t) = {T1, T2, ..., T50}                               │
│   Example: T1 = {accessed: {42, 156, 789},              │
│                  home: S3, weight: 1.0}                 │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ GRAPH CONSTRUCTION                                       │
│                                                          │
│ [TransactionGraph.add_transaction()]                     │
│   For each Ti ∈ T(t):                                   │
│     For each pair (u,v) in Ti.accessed:                 │
│       E[u,v] += Ti.weight                               │
│                                                          │
│ Result: GT(t) = (V, E, W)                               │
│   V = {0,1,...,999}                                     │
│   E = {(42,156):1.0, (42,789):1.0, (156,789):1.0, ...} │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ PERIODIC CHECK: t % 10 == 0?                            │
│   NO  → Continue to next time step                      │
│   YES → Execute Algorithm 2                             │
└────────────────────┬─────────────────────────────────────┘
                     ↓ YES
┌──────────────────────────────────────────────────────────┐
│ ALGORITHM 2 - PHASE 1                                    │
│                                                          │
│ [compute_shard_local_stats()]                            │
│   For each Ti ∈ T(t):                                   │
│     shards = {πt(a) for a in Ti.accessed}               │
│     if |shards| == 1:                                   │
│       Li[home_shard] += 1                               │
│     else:                                                │
│       Xi[home_shard] += 1                               │
│                                                          │
│ Result: {L0:35, L1:42, ..., L7:38}                      │
│         {X0:12, X1:8,  ..., X7:15}                      │
│                                                          │
│ [compute_current_processing_cost()]                      │
│   C_proc^current = ΣLi + β·ΣXi                          │
│   = (35+42+...+38) + 5×(12+8+...+15)                    │
│   = 280 + 5×95 = 755                                    │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ ALGORITHM 2 - PHASE 2                                    │
│                                                          │
│ [graph.get_adjacency_matrix()]                           │
│   Export: A[1000×1000] where A[i,j] = edge_weight      │
│           nodes = [0,1,2,...,999]                       │
│                                                          │
│ [clustering.cluster(A, nodes)]                           │
│   1. Compute Laplacian: L = D^(-1/2)(D-A)D^(-1/2)      │
│   2. Eigenvectors of L → embedding[1000×8]              │
│   3. K-means on embedding → labels[1000]                │
│   Result: πt_raw* = {0→2, 1→2, 2→5, 3→2, ...}          │
│                                                          │
│ [enforce_balance(πt_raw*, τ=1.2, s=8)]                  │
│   Check: n/(τ·s) ≤ |Si| ≤ τ·n/s                        │
│         1000/(1.2·8) ≤ |Si| ≤ 1.2·1000/8               │
│         104.2 ≤ |Si| ≤ 150                              │
│                                                          │
│   If violated: repair by moving accounts                 │
│   Result: πt* = {0→1, 1→1, 2→5, 3→1, ...} (balanced)   │
│                                                          │
│ [compute_clustered_processing_cost(πt*)]                 │
│   Re-compute Li*, Xi* under new mapping                 │
│   C_proc^cluster = ΣLi* + β·ΣXi*                        │
│   = 350 + 5×45 = 575                                    │
│                                                          │
│ [compute_migration_set(πt, πt*)]                         │
│   M(t) = {a | πt(a) ≠ πt*(a)}                           │
│   = {2, 7, 15, 23, ..., 987}  (42 accounts)            │
│                                                          │
│ [compute_migration_cost(M(t), πt, πt*)]                  │
│   For each a ∈ M(t):                                    │
│     cost += α × d(πt(a), πt*(a))                        │
│   = 42 × 10 × 1 = 420                                   │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ ALGORITHM 2 - PHASE 3                                    │
│                                                          │
│ [evaluate_migration_decision()]                          │
│   Δ(t) = C_proc^current - C_proc^cluster                │
│         = 755 - 575 = 180                               │
│                                                          │
│   Threshold = γ · C_mig                                  │
│             = 1.5 × 420 = 630                            │
│                                                          │
│   Decision: Δ > Threshold?                              │
│            180 > 630? → NO                              │
│                                                          │
│ Result: MigrationDecision(                              │
│   should_migrate=False,                                  │
│   savings_delta=180,                                     │
│   migration_cost=420,                                    │
│   ...                                                    │
│ )                                                        │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ APPLY DECISION                                           │
│                                                          │
│ IF should_migrate == True:                              │
│   1. scheduler.request_pause()                          │
│   2. Wait for safe point                                │
│   3. For each a ∈ M(t):                                 │
│        notify_migration(a, πt(a), πt*(a))               │
│   4. πt+1 ← πt*                                         │
│   5. scheduler.resume()                                 │
│                                                          │
│ ELSE:                                                    │
│   πt+1 ← πt  (no change)                                │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ METRICS COLLECTION                                       │
│                                                          │
│ SimulationMetrics(                                       │
│   time_step=10,                                         │
│   num_transactions=50,                                   │
│   num_intra_shard=35,                                   │
│   num_cross_shard=15,                                   │
│   processing_cost=755,                                   │
│   num_migrations=0,                                     │
│   shard_sizes={0:125, 1:125, ..., 7:125}               │
│ )                                                        │
└────────────────────┬─────────────────────────────────────┘
                     ↓
                 t = t + 1
                 (Loop continues)
```

### 8.2 Data Transformations

| Stage | Input | Transformation | Output |
|-------|-------|----------------|--------|
| **Generation** | Config params | Random sampling | T(t) transactions |
| **Graph Build** | Transactions | Edge accumulation | GT(t) adjacency |
| **Clustering** | Adjacency matrix | Spectral decomposition | πt* mapping |
| **Balance** | πt* mapping | Greedy repair | Balanced πt* |
| **Cost Calc** | Transactions + πt | Aggregation | C_proc, C_mig |
| **Decision** | Costs + γ | Comparison | Boolean |
| **Migration** | πt + πt* | Account moves | Updated πt+1 |

---

## 9. Algorithm Walkthrough

### 9.1 Detailed Example with Real Numbers

**Scenario:** 10 accounts, 2 shards, simple transaction pattern

**Configuration:**
```
s = 2 shards
n = 10 accounts
α = 10.0
β = 5.0
γ = 1.5
τ = 1.2
k = 3 (max 3 accounts per transaction)
```

**Initial State (t=0):**
```
Current Mapping πt:
  S0: {0, 1, 2, 3, 4}
  S1: {5, 6, 7, 8, 9}

Shard sizes: [5, 5] ✓ (balanced)
```

**Transactions Generated (t=1):**
```
T1: {0, 1, 2}     home=S0  weight=1.0  → INTRA (all in S0)
T2: {5, 6, 7}     home=S1  weight=1.0  → INTRA (all in S1)
T3: {0, 5}        home=S0  weight=1.0  → CROSS (S0 ↔ S1)
T4: {1, 2, 6}     home=S0  weight=1.0  → CROSS (S0 ↔ S1)
T5: {3, 4, 5}     home=S0  weight=1.0  → CROSS (S0 ↔ S1)
```

**Graph Construction:**
```
Edges added:
  (0,1): 1.0,  (0,2): 1.0,  (1,2): 1.0  ← from T1
  (5,6): 1.0,  (5,7): 1.0,  (6,7): 1.0  ← from T2
  (0,5): 1.0                             ← from T3
  (1,6): 1.0,  (2,6): 1.0,  (1,2): +1.0 ← from T4
  (3,5): 1.0,  (4,5): 1.0,  (3,4): 1.0  ← from T5

Adjacency Matrix:
     0   1   2   3   4   5   6   7   8   9
  ┌────────────────────────────────────────┐
0 │ 0   1   1   0   0   1   0   0   0   0 │
1 │ 1   0   2   0   0   0   1   0   0   0 │
2 │ 1   2   0   0   0   0   1   0   0   0 │
3 │ 0   0   0   0   1   1   0   0   0   0 │
4 │ 0   0   0   1   0   1   0   0   0   0 │
5 │ 1   0   0   1   1   0   1   1   0   0 │
6 │ 0   1   1   0   0   1   0   1   0   0 │
7 │ 0   0   0   0   0   1   1   0   0   0 │
8 │ 0   0   0   0   0   0   0   0   0   0 │
9 │ 0   0   0   0   0   0   0   0   0   0 │
  └────────────────────────────────────────┘
```

**Phase 1: Cost Aggregation**
```
Intra-shard transactions:
  S0: T1 → L0 = 1
  S1: T2 → L1 = 1
  Total: ΣLi = 2

Cross-shard transactions:
  S0: T3, T4, T5 → X0 = 3
  S1: none → X1 = 0
  Total: ΣXi = 3

C_proc^current = ΣLi + β·ΣXi
               = 2 + 5×3
               = 17
```

**Phase 2: Clustering**
```
Step 1: Spectral clustering on adjacency matrix
  Observation: Accounts {0,1,2,5,6} form dense cluster
               Accounts {3,4} weakly connected
               Accounts {7,8,9} isolated

Step 2: Clustering suggests:
  πt_raw* = {
    0→S0, 1→S0, 2→S0, 5→S0, 6→S0,  ← Move 5,6 to S0
    3→S1, 4→S1, 7→S1, 8→S1, 9→S1
  }

Step 3: Check balance
  |S0*| = 5, |S1*| = 5
  Bounds: 10/(1.2×2) ≤ size ≤ 1.2×10/2
         4.17 ≤ size ≤ 6.0
  Both shards: 4.17 ≤ 5 ≤ 6.0 ✓ (balanced)

Step 4: Compute new costs under πt*
  Transactions under new mapping:
    T1: {0,1,2} → all in S0 → INTRA
    T2: {5,6,7} → 5,6 in S0, 7 in S1 → CROSS!
    T3: {0,5} → both in S0 → INTRA
    T4: {1,2,6} → all in S0 → INTRA
    T5: {3,4,5} → 3,4 in S1, 5 in S0 → CROSS!

  New stats:
    L0* = 3, L1* = 0
    X0* = 1, X1* = 1

  C_proc^cluster = (3+0) + 5×(1+1)
                 = 3 + 10
                 = 13

Step 5: Migration set
  M(t) = {a | πt(a) ≠ πt*(a)}
       = {5, 6}  (moved S1 → S0)

Step 6: Migration cost
  C_mig = Σ α·d(πt(a), πt*(a))
        = α·d(S1,S0) + α·d(S1,S0)
        = 10×1 + 10×1
        = 20
```

**Phase 3: Decision**
```
Savings:
  Δ(t) = C_proc^current - C_proc^cluster
       = 17 - 13
       = 4

Threshold:
  γ·C_mig = 1.5 × 20 = 30

Decision:
  Δ > γ·C_mig?
  4 > 30? → NO

Result: Don't migrate (savings insufficient)
  πt+1 ← πt (keep current mapping)
```

**Why Migration Rejected:**
- Savings from clustering: 4 units
- Cost to migrate 2 accounts: 20 units
- Safety factor: ×1.5 → threshold 30
- Benefit doesn't justify cost

**Alternative Scenario (lower α):**
```
If α = 2.0 instead:
  C_mig = 2×1 + 2×1 = 4
  Threshold = 1.5 × 4 = 6
  Decision: 4 > 6? → NO (still rejected, but closer)

If α = 1.0:
  C_mig = 1×1 + 1×1 = 2
  Threshold = 1.5 × 2 = 3
  Decision: 4 > 3? → YES (migration accepted!)
```

### 9.2 Edge Cases Handled

**Case 1: Empty Migration Set**
```python
if len(migration_set) == 0:
    # Clustering produced same partition
    # C_mig = 0, so decision depends on Δ
    # If Δ > 0, migrate (free improvement)
    # If Δ ≤ 0, don't migrate (no benefit)
```

**Case 2: Perfect Clustering**
```python
if C_proc^cluster == num_transactions:
    # All transactions now intra-shard
    # Maximum possible improvement achieved
```

**Case 3: Imbalanced Clustering**
```python
# Balance repair ensures constraint satisfaction
if shard_size > upper_bound:
    # Move "loose" accounts to under-full shards
    # Prioritize accounts with weak intra-shard connections
```

**Case 4: Isolated Accounts**
```python
# Accounts with no edges in GT(t)
# Assigned to under-full shards during balance repair
# Doesn't affect clustering quality (no edges to cut)
```

---

## 10. Code Organization

### 10.1 Directory Tree with Descriptions

```
adaptive_sharding/
│
├── __init__.py                    # Package initialization, exports main classes
│
├── config.py                      # CRITICAL: All system parameters
│   │                              # - ShardingConfig dataclass
│   │                              # - Parameter validation (α>β, τ≥1)
│   │                              # - Distance matrix initialization
│   │                              # - Preset configurations
│   └─ Classes: ShardingConfig
│      Functions: get_default_config(), get_high_load_config()
│
├── models/                        # Core data structures
│   ├── __init__.py
│   ├── account.py                 # Account representation
│   │   └─ Classes: Account, AccountStatus
│   ├── transaction.py             # Transaction + Graph
│   │   └─ Classes: Transaction, TransactionGraph
│   ├── shard.py                   # Shard state
│   │   └─ Classes: Shard
│   └── graph.py → transaction.py  # (TransactionGraph main implementation)
│
├── clustering/                    # ML clustering algorithms
│   ├── __init__.py
│   ├── base.py                    # Abstract base class
│   │   └─ Classes: ClusteringBase (ABC)
│   ├── spectral.py                # Spectral clustering (primary method)
│   │   └─ Classes: SpectralClusteringMethod
│   │      Functions: create_clustering_method()
│   ├── kmeans.py                  # K-Means variants
│   │   └─ Classes: KMeansClusteringMethod,
│   │              MiniBatchKMeansMethod
│   └── balance_repair.py          # Balance constraint enforcement
│       └─ Classes: BalanceRepair
│          Functions: enforce_balance()
│
├── cost/                          # Cost model (section 2.4)
│   ├── __init__.py
│   ├── processor.py               # Transaction processing cost
│   │   └─ Classes: TransactionCostProcessor,
│   │              MigrationCostProcessor,
│   │              CostModel (unified interface)
│   └── migration.py → processor.py # (Included in processor.py)
│
├── algorithm/                     # Algorithm 2 implementation
│   ├── __init__.py
│   └── adaptive_sharding.py       # CORE: Main algorithm
│       └─ Classes: AdaptiveShardingAlgorithm,
│                   MigrationDecision (dataclass)
│          Methods: execute() [runs Algorithm 2],
│                   apply_migration(),
│                   _phase1_cost_aggregation(),
│                   _phase2_repartition_evaluation(),
│                   _phase3_migration_decision()
│
├── scheduler/                     # Transaction scheduling
│   ├── __init__.py
│   ├── stub.py                    # Simplified scheduler
│   │   └─ Classes: SimpleScheduler, TransactionBatch
│   └── controller.py              # Pause/resume control
│       └─ Classes: SchedulerController, SchedulerState
│          Methods: request_pause_for_migration(),
│                   perform_migration(),
│                   resume()
│
├── simulation/                    # End-to-end simulation
│   ├── __init__.py
│   ├── generator.py               # Transaction generation
│   │   └─ Classes: TransactionGenerator
│   │      Methods: generate_transactions(),
│   │               _generate_single_transaction()
│   └── driver.py                  # Simulation orchestration
│       └─ Classes: SimulationDriver, SimulationMetrics
│          Methods: run(), _collect_metrics()
│
└── utils/                         # Utilities
    ├── __init__.py
    ├── logger.py                  # Logging setup
    │   └─ Functions: setup_logger()
    └── metrics.py                 # Metrics analysis & visualization
        └─ Classes: MetricsAnalyzer
           Functions: plot_metrics(),
                     export_metrics_csv(),
                     export_config_json()
```

### 10.2 Key Files and Their Responsibilities

#### 1. `run_simulation.py` (Entry Point)
**Purpose:** Main executable script
**Responsibilities:**
- Parse command-line arguments
- Create configuration
- Initialize all components
- Run simulation loop
- Generate reports

**Key Functions:**
```python
def main():
    args = parse_arguments()
    config = create_config(args)
    
    # Initialize components
    graph = TransactionGraph()
    clustering = create_clustering_method(...)
    cost_model = CostModel(...)
    algorithm = AdaptiveShardingAlgorithm(...)
    
    # Run
    driver = SimulationDriver(...)
    metrics = driver.run()
    
    # Analyze
    analyzer = MetricsAnalyzer(metrics, config)
    analyzer
