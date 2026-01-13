# RL-AG: Methodology Document

## Executive Summary

RL-AG (Reinforcement Learning for Attack Graph Generation) is a deep reinforcement learning system that generates optimal Advanced Persistent Threat (APT) attack paths at provenance-level granularity. The system integrates TAGAPT's provenance framework with Proximal Policy Optimization (PPO) to synthesize attack paths tailored to specific network topologies and vulnerabilities.

**Key Innovation**: Transforms the APT generation problem from statistical graph generation into a sequential decision-making process, enabling goal-directed attack path optimization while maintaining provenance-level fidelity required for cybersecurity applications.

---

## 1. Project Architecture Overview

### 1.1 System Components

The RL-AG system consists of five main components:

1. **Environment (`APTAttackEnv`)**: Gym-compatible environment orchestrating attack path generation
2. **Policy Network (`ActorCriticPolicy`)**: Graph neural network-based actor-critic architecture
3. **Training Loop (`PPOTrainer`)**: PPO training with curriculum learning support
4. **Data Layer**: Policy repositories, technique mappings, and instance libraries
5. **Evaluation Tools**: Policy evaluation and attack path export utilities

### 1.2 Data Flow

```
Topology JSON → TopologyGraph
Policy JSONs → PolicyRepository
Technique JSONs → TechniqueRepository
Instance Library → TechniqueInstanceLibrary
Dependencies JSON → TacticDependencyMap
                    ↓
            APTAttackEnv (MDP)
                    ↓
        RelationalGraphEncoder (GNN)
                    ↓
        ActorCriticPolicy (PPO)
                    ↓
        ProvenanceState (IAG output)
```

---

## 2. MDP Formulation

### 2.1 Markov Decision Process Components

The attack generation process is formalized as an MDP $M = (\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P})$:

- **State Space ($\mathcal{S}$)**: Heterogeneous graph combining network topology and dynamic provenance
- **Action Space ($\mathcal{A}$)**: Hierarchical macro-actions mapping to MITRE ATT&CK techniques
- **Reward Function ($\mathcal{R}$)**: Multi-tiered reward structure guiding policy optimization
- **Transition Dynamics ($\mathcal{P}$)**: Environment state transitions based on action execution

### 2.2 State Space ($\mathcal{S}$)

**Definition**: The state $S_t$ at time step $t$ is an augmented heterogeneous graph $G_t$ that combines:

1. **Static Infrastructure Graph** ($G_{Net}$):
   - Nodes: Network devices (hosts, servers, PLCs, workstations)
   - Node Features: Device type, platform (OS), IP address, vulnerabilities (CVE IDs, CVSS scores, EPSS), mapped MITRE techniques
   - Edges: Network connectivity, firewall rules, communication protocols

2. **Dynamic Provenance Graph** ($IAG_{t-1}$):
   - Nodes: Provenance entities (MP, TP, MF, SF, TF, SO)
   - Edges: Provenance relations (FR, RD, WR, EX, IJ, ST, RF, UK, CD)
   - Metadata: Technique IDs, policy keys, target devices

**State Representation**: The graph $G_t$ is encoded using a Relational Graph Convolutional Network (R-GCN) to produce a fixed-size embedding $s_t = \text{GNN}(G_t)$.

**Implementation**: See `src/env/apt_env.py:AttackObservation` and `src/models/features.py:encode_observation()`.

### 2.3 Action Space ($\mathcal{A}$)

**Hierarchical Action Model**:

#### Macro-Actions ($\mathcal{A}_{Macro}$)

A macro-action $a_{Macro}$ is a tuple:
$$a_{Macro} = (\text{Stage}_k, \text{Technique}_T, \text{Target Device}_v)$$

Where:
- $\text{Stage}_k$: Attack stage (1-4: Initial Invasion, Privilege Exploration, Sensitive Action, Target Achievement)
- $\text{Technique}_T$: MITRE ATT&CK technique ID (e.g., T1204, T1555)
- $\text{Target Device}_v$: Network device identifier

**Action Space Construction**:
- Enumerates all valid combinations of policies, techniques, and target devices
- Filters actions based on device platform compatibility and technique support
- Respects temporal dependencies between tactics (via `TacticDependencyMap`)

**Implementation**: See `src/env/actions.py:MacroActionSpace` and `src/env/actions.py:AttackMacroAction`.

#### Micro-Actions (Implicit)

When a macro-action is executed, it triggers a sequence of micro-actions that instantiate provenance triples:
- Adding provenance nodes (MP, TP, MF, SF, TF, SO)
- Adding provenance edges (FR, RD, WR, EX, IJ, ST, RF, UK, CD)
- Selecting concrete instance names from the technique instance library

**Implementation**: See `src/env/apt_env.py:_apply_policy()`.

### 2.4 Reward Function ($\mathcal{R}$)

The reward function is structured in three tiers:

#### Tier 1: Structure and Rationality ($R_{Structure}$)

**Purpose**: Ensure valid provenance graph construction

- **Rationality Penalty** ($R_{Penalty}$): Large negative reward (-0.5) for domain constraint violations
  - Violations include: operations on deleted entities, invalid entity-relation combinations, lifecycle violations
- **Completeness Reward**: Small positive reward (+0.1) per successfully instantiated provenance triple

**Implementation**: See `src/env/domain.py:DomainConstraintEngine` and `src/env/apt_env.py:step()`.

#### Tier 2: Temporal Progression ($R_{Temporal}$)

**Purpose**: Guide attack stage progression

- **Stage Transition Bonus**: Medium reward (+1.0) when unlocking next attack stage
- **Stage Completion Bonus**: Large reward (+5.0) when completing a stage's mandatory tactics
- **Stagnation Penalty**: Small negative reward for lack of progress (scales with steps since last progress)

**Implementation**: See `src/env/apt_env.py:_complete_stage()` and `src/env/apt_env.py:_stage_stagnation_penalty()`.

#### Tier 3: Utility and Terminal ($R_{Utility}$)

**Purpose**: Reward goal achievement and efficiency

- **Action Cost**: Small negative reward (-0.01) per action to encourage efficiency
- **Final Stage Bonus**: Large reward (+11.0) when reaching Impact stage (Target Achievement)
- **Stage Multiplier**: Reward scaling based on stage index (later stages receive higher multipliers)

**Total Reward**:
$$R(s_t, a_t) = R_{Structure} + R_{Temporal} + R_{Utility}$$

**Implementation**: See `src/env/apt_env.py:step()` reward breakdown.

---

## 3. Neural Network Architecture

### 3.1 Graph Encoder

**Architecture**: Relational Graph Convolutional Network (R-GCN)

**Components**:
1. **Node Feature Encoding**: One-hot encoding of entity types (MP, TP, MF, SF, TF, SO)
2. **Edge Feature Encoding**: Relation type indices (FR, RD, WR, EX, IJ, ST, RF, UK, CD)
3. **Relational Convolution Layers**: Multiple R-GCN layers with relation-specific weight matrices
4. **Graph-Level Readout**: Mean pooling over node embeddings

**Configuration** (default):
- Input dimension: 6 (entity types)
- Hidden dimension: 256
- Number of layers: 3
- Activation: ReLU
- Dropout: 0.1

**Implementation**: See `src/models/encoders.py:RelationalGraphEncoder`.

### 3.2 Policy Network

**Architecture**: Actor-Critic with shared encoder

**Components**:
1. **Shared Encoder**: R-GCN encoder producing graph embedding
2. **Actor Head**: Linear layer mapping embedding to action logits
3. **Critic Head**: Linear layer mapping embedding to state value

**Action Selection**:
- Action masking: Invalid actions (locked stages, unsupported techniques) are masked
- Sampling: Categorical distribution over valid actions
- Deterministic mode: Argmax for evaluation

**Implementation**: See `src/models/policy.py:ActorCriticPolicy`.

---

## 4. Training Methodology

### 4.1 Proximal Policy Optimization (PPO)

**Algorithm**: PPO-Clip with Generalized Advantage Estimation (GAE)

**Hyperparameters** (default):
- Learning rate: 5e-4
- Rollout length: 512 steps
- PPO epochs: 4
- Minibatch size: 256
- Discount factor ($\gamma$): 0.99
- GAE lambda ($\lambda$): 0.95
- Clip coefficient: 0.3
- Value function coefficient: 0.5
- Entropy coefficient: 0.02
- Target KL divergence: 0.015

**Training Loop**:
1. Collect rollout trajectories (512 steps)
2. Compute advantages using GAE
3. Normalize advantages (optional reward normalization)
4. Update policy for multiple epochs on minibatches
5. Log metrics and save checkpoints

**Implementation**: See `src/training/ppo.py:PPOTrainer`.

### 4.2 Curriculum Learning

**Strategy**: Progressive stage unlocking

**Mechanism**:
- Initially train on early attack stages only
- Advance to next stage when success rate threshold is met (default: 70% over 5 rollouts)
- Track stage completion rates per curriculum phase
- Reset environment when curriculum advances

**Configuration**:
- Default curriculum: Stages 1-14, 20,000 rollouts per phase
- Custom curriculum: User-specified stage sequences and phase lengths

**Implementation**: See `src/training/ppo.py:_apply_curriculum()` and `src/training/config.py:CurriculumConfig`.

### 4.3 Reward Normalization

**Optional Feature**: Running statistics normalization

**Mechanism**:
- Maintains running mean and variance of rewards
- Normalizes rewards to zero mean, unit variance
- Improves training stability for reward scales

**Implementation**: See `src/training/ppo.py:RunningStats`.

---

## 5. Domain Constraints and Validation

### 5.1 Domain Rules

The system enforces TAGAPT's five fundamental domain constraints:

1. **Schema Validation**: Only valid entity-relation-object triples from policies are permitted
2. **Lifecycle Validation**: Deleted entities (via UK relation) cannot participate in operations
3. **Entity Type Matching**: Subject/object nodes must match triple entity types
4. **Temporal Constraints**: Processes must be forked before they can execute operations
5. **Parent-Child Constraints**: Child processes cannot manipulate parent processes

**Implementation**: See `src/env/domain.py:DomainConstraintEngine`.

### 5.2 Action Masking

**Purpose**: Prevent invalid actions from being selected

**Masking Rules**:
- Locked stages: Actions for stages not yet unlocked are masked
- Platform compatibility: Actions requiring unsupported techniques are masked
- Temporal dependencies: Actions violating tactic ordering are masked

**Implementation**: See `src/env/apt_env.py:_build_observation()` and `src/models/policy.py:_masked_logits()`.

---

## 6. Data Structures and Formats

### 6.1 Topology Format

**File**: JSON files in `data/sample_topologies/`

**Structure**:
```json
{
  "nodes": [
    {
      "device_id": "device_1",
      "name": "PLC Controller",
      "device_type": "PLC",
      "platform": "Linux",
      "ip_address": "192.168.1.10",
      "neighbors": ["device_2"],
      "connections": [...],
      "vulnerabilities": [
        {
          "cve_id": "CVE-2023-1234",
          "cvss_base": 7.5,
          "epss": 0.85,
          "mapped_techniques": ["T1204", "T1555"]
        }
      ]
    }
  ]
}
```

**Implementation**: See `src/env/topology.py:TopologyGraph`.

### 6.2 Policy Format

**File**: JSON files in `data/regulation_dic/stage*_regulation.json`

**Structure**:
```json
{
  "3.Initial Access-21": [
    ["MP", "FR", "TP"],
    ["MP", "EX", "MF"]
  ]
}
```

Each key is parsed as: `{stage}.{tactic}-{policy_number}`

**Implementation**: See `src/data/policy_loader.py:PolicyRepository`.

### 6.3 Dependency Map Format

**File**: `data/tactic_dependency_map/dependencies.json`

**Structure**:
```json
{
  "tactic_order": [1, 2, 3, 4],
  "tactics": {
    "1": {
      "mandatory": ["Reconnaissance", "Initial Access"],
      "optional": ["Resource Development"],
      "parent": "stage_1"
    }
  },
  "critical_tactics": ["Execution", "Command and Control", "Impact"],
  "movement_tactics": ["Lateral Movement"]
}
```

**Implementation**: See `src/data/dependency_loader.py:TacticDependencyMap`.

---

## 7. Evaluation and Usage

### 7.1 Training

**Command**:
```bash
PYTHONPATH=src python -m training.train_agent \
  --topology data/sample_topologies/chemical_plant.json \
  --total-steps 500000 \
  --use-default-curriculum
```

**Key Parameters**:
- `--total-steps`: Total training steps
- `--rollout-length`: Steps per rollout (default: 512)
- `--ppo-epochs`: PPO update epochs (default: 4)
- `--learning-rate`: Optimizer learning rate (default: 5e-4)
- `--curriculum-stages`: Custom curriculum stage sequence
- `--curriculum-lengths`: Rollout counts per curriculum phase

**Outputs**:
- Checkpoints: `checkpoints/ppo_step_{step}.pt`
- TensorBoard logs: `logs/` (metrics: reward, done_rate, losses, entropy)

**Implementation**: See `src/training/train_agent.py`.

### 7.2 Evaluation

**Command**:
```bash
PYTHONPATH=src python -m cli.evaluate \
  --checkpoint checkpoints/ppo_step_500000.pt \
  --episodes 10 \
  --export-dir outputs/
```

**Metrics**:
- Average reward
- Average episode length
- Success rate (terminal stage completion)
- Technique diversity

**Export Options**:
- `--top-k`: Export top-K paths by reward
- `--require-plc-impact`: Filter paths reaching PLC impact

**Implementation**: See `src/cli/evaluate.py`.

### 7.3 Generated Attack Path Format

**Output**: JSON files containing:
- Provenance graph (nodes and edges)
- Action sequence with techniques and targets
- Reward and episode metadata
- PLC impact detection (if applicable)

**Structure**:
```json
{
  "reward": 15.23,
  "length": 42,
  "techniques": ["T1204", "T1555", "T1498"],
  "actions": [...],
  "provenance": {
    "nodes": [...],
    "edges": [...]
  },
  "plc_impact": true,
  "impact_device": {...}
}
```

---

## 8. Key Design Decisions

### 8.1 Hierarchical Action Space

**Rationale**: Decouples high-level tactical decisions from low-level provenance instantiation, enabling efficient learning while maintaining provenance fidelity.

**Trade-off**: Larger action space but clearer semantic meaning per action.

### 8.2 Graph Neural Network Encoding

**Rationale**: Preserves topological structure and relational information that would be lost in flat vector representations.

**Alternative Considered**: Flat feature vectors (rejected due to information loss).

### 8.3 Multi-Tiered Reward Structure

**Rationale**: Provides dense local feedback (structure) and sparse strategic guidance (temporal, utility), balancing exploration and exploitation.

**Trade-off**: Requires careful reward scaling to prevent one tier from dominating.

### 8.4 Curriculum Learning

**Rationale**: Reduces initial exploration space, improving sample efficiency and training stability.

**Alternative**: End-to-end training (possible but less efficient).

### 8.5 Domain Constraint Enforcement

**Rationale**: Prevents learning invalid attack patterns, dramatically improving sample efficiency compared to learning constraints through trial-and-error.

**Trade-off**: Less flexible but ensures output validity.

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Static Topology**: Network topology is fixed during training (no dynamic network changes)
2. **Discrete Actions**: Action space is discrete (no continuous parameter tuning)
3. **Single Agent**: No multi-agent or adversarial training
4. **Manual Policy Curation**: Policies must be manually maintained as new TTPs emerge

### 9.2 Future Directions

1. **Adversarial RL**: Train attacker (Red) against defender (Blue) agent for stealth optimization
2. **LLM Integration**: Automatically extract policies and instances from unstructured CTI reports
3. **Dynamic Topology**: Support network topology changes during episodes
4. **Transfer Learning**: Pre-train on general attack patterns, fine-tune for specific topologies
5. **Multi-Objective Optimization**: Optimize for multiple objectives (efficiency, stealth, impact)

---

## 10. References and Related Work

### 10.1 Foundation

- **TAGAPT**: Toward Automatic Generation of APT Samples With Provenance-Level Granularity
- **MITRE ATT&CK**: Framework for understanding adversary tactics and techniques
- **PPO**: Proximal Policy Optimization algorithms (Schulman et al., 2017)

### 10.2 Technical Components

- **R-GCN**: Relational Graph Convolutional Networks (Schlichtkrull et al., 2018)
- **GAE**: Generalized Advantage Estimation (Schulman et al., 2016)
- **Curriculum Learning**: Progressive training strategies (Bengio et al., 2009)

---

## Appendix A: File Structure

```
RL-AG/
├── src/
│   ├── env/              # Environment implementation
│   │   ├── apt_env.py   # Main MDP environment
│   │   ├── actions.py   # Action space construction
│   │   ├── domain.py    # Domain constraint validation
│   │   ├── provenance.py # Provenance graph state
│   │   └── topology.py   # Network topology representation
│   ├── models/          # Neural network architectures
│   │   ├── policy.py    # Actor-critic policy
│   │   ├── encoders.py  # Graph neural network encoder
│   │   └── features.py  # Feature engineering utilities
│   ├── training/        # Training infrastructure
│   │   ├── train_agent.py # Training entry point
│   │   ├── ppo.py       # PPO trainer implementation
│   │   └── config.py    # Configuration dataclasses
│   ├── data/            # Data loading and models
│   │   ├── policy_loader.py      # Policy repository
│   │   ├── technique_loader.py   # Technique mappings
│   │   ├── dependency_loader.py  # Tactic dependencies
│   │   ├── instance_library.py   # Instance name library
│   │   └── models.py    # Data model definitions
│   └── cli/             # Command-line utilities
│       ├── evaluate.py  # Policy evaluation and export
│       └── sample_paths.py # Path sampling utilities
├── data/                # Data files
│   ├── sample_topologies/    # Network topology JSONs
│   ├── regulation_dic/       # Policy definitions
│   ├── tech_dic/             # Technique mappings
│   ├── tactic_dependency_map/ # Tactic dependencies
│   └── instance_lib/         # Technique instance names
├── checkpoints/         # Saved model checkpoints
├── logs/               # TensorBoard logs
└── README.md           # Quick start guide
```

---

## Appendix B: Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | GNN hidden dimension |
| `num_layers` | 3 | Number of GNN layers |
| `rollout_length` | 512 | Steps per rollout |
| `ppo_epochs` | 4 | PPO update epochs |
| `minibatch_size` | 256 | Minibatch size |
| `learning_rate` | 5e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_coef` | 0.3 | PPO clip coefficient |
| `entropy_coef` | 0.02 | Entropy regularization |
| `stage_patience` | 16 | Steps before stagnation penalty |
| `stage_completion_bonus` | 5.0 | Reward for stage completion |
| `stage_transition_bonus` | 1.0 | Reward for stage transition |
| `max_steps` | 200 | Maximum episode length |

---

## Document Version

**Version**: 1.0  
**Date**: 2025-01-27  
**Author**: Methodology Review  
**Status**: Complete


