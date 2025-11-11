

# **Technical Blueprint: Policy Optimization for Provenance-Level APT Generation via Graph Reinforcement Learning**

**技术蓝图：基于图强化学习的溯源级 APT 生成策略优化**

## **I. Executive Summary and Foundational Context**

**一、执行摘要和基础背景**

The primary challenge in developing robust, supervised Advanced Persistent Threat (APT) detection systems lies in the scarcity of diverse, high-fidelity APT samples documented at the system provenance level.1 Traditional approaches relying on data provenance—the record of causal dependencies between system entities and events—suffer when training data is limited, rendering powerful machine learning techniques, effective in other domains, impractical for APT defense.1

开发稳健的、受监督的高级持续性威胁 (APT) 检测系统的主要挑战在于缺乏在系统溯源级别记录的各种高保真 APT 样本。1 依赖数据来源（系统实体和事件之间因果依赖关系的记录）的传统方法在训练数据有限时会遇到困难，这使得在其他领域有效的强大机器学习技术对于 APT 防御而言不切实际。1

The TAGAPT system was initially proposed to mitigate this limitation by automatically generating numerous APT samples with provenance-level granularity.1 TAGAPT leverages a serialized deep graph generation model (GraphAF) to learn structural patterns from existing Cyber Threat Intelligence (CTI) reports, synthesizing new Abstract Attack Graphs (AAGs), which are then divided, interpreted, and instantiated into usable Instantiated Attack Graphs (IAGs).1

TAGAPT 系统最初是为了缓解这一限制而提出的，它能够自动生成大量具有溯源级别粒度的 APT 样本。1 TAGAPT 利用序列化深度图生成模型 (GraphAF) 从现有的网络威胁情报 (CTI) 报告中学习结构模式，合成新的抽象攻击图 (AAG)，然后将这些 AAG 分割、解释并实例化为可用的实例化攻击图 (IAG)。1

The current requirement necessitates moving beyond purely statistical generation towards policy optimization. While TAGAPT successfully generates diverse samples, integrating its methodology into a Deep Reinforcement Learning (DRL) framework allows for the synthesis of *optimal* and *goal-directed* attack paths specifically tailored to a given environment (the network topology and its inherent vulnerabilities, or the "terrain").2 This approach transforms the problem from generalized graph generation into a sequential decision-making process, seeking the most efficient or damaging path to compromise critical assets.

当前的需求要求我们从纯粹的统计生成转向策略优化。TAGAPT 虽然能够成功生成多样化的样本，但将其方法集成到深度强化学习 (DRL) 框架中，可以合成针对特定环境（网络拓扑及其固有漏洞，即“地形”）量身定制的*最优*且*目标导向的*攻击路径。2 这种方法将问题从一般的图生成转变为顺序决策过程，寻找破坏关键资产的最有效或最具破坏性的路径。

### **MDP Formulation Overview  MDP 配方概述**

To achieve this goal, the entire attack generation process must be formalized as a Markov Decision Process (MDP), $M \= (\\mathcal{S, A, R, P})$, where an intelligent agent learns an optimal policy $\\pi(a|s)$.3 This policy dictates which actions ($a$) to take in a given state ($s$), maximizing the expected cumulative reward ($\\mathcal{R}$). The DRL framework will iterate through the following components:

为了实现这一目标，整个攻击生成过程必须形式化为马尔可夫决策过程（MDP）， $M \= (\\mathcal{S, A, R, P})$ ，其中智能代理学习最优策略 $\\pi(a|s)$ 。3 该策略决定在给定状态 ( $s$ ) 下采取哪些行动 ( $a$ )，以最大化预期累积奖励 ( $\\mathcal{R}$ )。深度强化学习框架将迭代执行以下组件：

* **State ($\\mathcal{S}$):** The comprehensive, evolving representation of the network topology fused with the attack provenance accumulated thus far.  
  **状态（ $\\mathcal{S}$ ）：** 网络拓扑的全面、不断演进的表示，与迄今为止积累的攻击来源融合在一起。  
* **Action ($\\mathcal{A}$):** A granular, hierarchical selection of ATT\&CK techniques mapped directly from TAGAPT's technical policies.  
  **操作（ $\\mathcal{A}$ ）：** 从 TAGAPT 的技术策略直接映射的 ATT\&CK 技术的细粒度、分层选择。  
* **Reward ($\\mathcal{R}$):** A multi-tiered, rule-based function translating TAGAPT’s structural and strategic constraints into dynamic feedback signals.  
  **奖励（ $\\mathcal{R}$ ）：** 一个多层级的、基于规则的函数，将 TAGAPT 的结构和战略约束转化为动态反馈信号。  
* Transition Probability ($\\mathcal{P}$): The inherent dynamics of the environment, governing the shift from state $s\_t$ to $s\_{t+1}$ based on the action $a\_t$.转移概率（ $\\mathcal{P}$ ）： 环境固有的动态特性，决定了状态 s 到状态 s 的转变。  
  t  
  ​  
  到 s  
  t+1  
  ​  
  基于该行为  
  t  
  ​  
  .

The subsequent sections detail the technical blueprint for defining these components, ensuring the DRL output maintains the provenance-level granularity and tactical rationality required for practical cybersecurity applications.

接下来的章节详细介绍了定义这些组件的技术蓝图，确保 DRL 输出保持实际网络安全应用所需的溯源级粒度和战术合理性。

## **II. Deconstruction of TAGAPT: The Blueprint for Provenance Granularity**

**二、TAGAPT 的解构：溯源粒度蓝图**

To successfully integrate DRL, the intrinsic structure and constraints of the TAGAPT output must be precisely mapped to the MDP formulation. The core components of TAGAPT—provenance granularity, domain constraints, and attack staging—are essential for defining $\\mathcal{S}$ and constraining $\\mathcal{A}$.

为了成功集成深度强化学习（DRL），TAGAPT 输出的内在结构和约束必须精确映射到马尔可夫决策过程（MDP）模型。TAGAPT 的核心组件——溯源粒度、领域约束和攻击阶段——对于定义 $\\mathcal{S}$ 和约束 $\\mathcal{A}$ 至关重要。

### **A. Provenance Graph Granularity and Heterogeneity**

**A. 溯源图粒度和异构性**

A provenance graph, $G=(V, E)$, is fundamentally a heterogeneous graph that captures the causal dependencies between system entities and events.1 The TAGAPT system defines a specific, high-granularity structure that must be maintained in the DRL environment.

溯源图 $G=(V, E)$ 从本质上讲是一个异构图，它捕获系统实体和事件之间的因果依赖关系。1 TAGAPT 系统定义了一个特定的、高粒度的结构，该结构必须在 DRL 环境中保持。

The six types of entities (nodes) defined in TAGAPT serve as the atomic components of the provenance record: Malicious Process (MP), Benign Process (TP), Malicious File (MF), System File (SF), Temporary File (TF), and Socket (SO).1 These entities interact through nine types of relations (edges), which represent system events such as Fork (FR), Read (RD), Write (WR), Execute (EX), Inject (IJ), Send To (ST), and Receive From (RF).1

TAGAPT 中定义的六种实体（节点）类型作为溯源记录的原子组件：恶意进程 (MP)、良性进程 (TP)、恶意文件 (MF)、系统文件 (SF)、临时文件 (TF) 和套接字 (SO)。1 这些实体通过九种类型的关系（边）进行交互，这些关系代表系统事件，例如 Fork (FR)、Read (RD)、Write (WR)、Execute (EX)、Inject (IJ)、Send To (ST) 和 Receive From (RF)。1

The critical implication of this definition is that the DRL State $\\mathcal{S}$ cannot simply be an abstract network topology; it must incorporate these fine-grained provenance features. Every action taken by the DRL agent must generate a valid provenance triple, linking a subject (e.g., a process) and an object (e.g., a file or socket) through a time-ordered system event (relation). This ensures that the generated IAGs are usable for downstream tasks that rely on system-level logs, such as threat hunting or filtering false positives.1

该定义的关键含义在于，DRL 状态 $\\mathcal{S}$ 不能仅仅是一个抽象的网络拓扑结构；它必须包含这些细粒度的溯源特征。DRL 代理执行的每个操作都必须生成一个有效的溯源三元组，通过一个按时间顺序排列的系统事件（关系）将主体（例如，进程）和客体（例如，文件或套接字）关联起来。这确保了生成的 IAG 可用于依赖系统级日志的下游任务，例如威胁狩猎或过滤误报。1

### **B. Architectural Foundation: Generative Constraints and Staging**

**B. 建筑基础：生成约束和阶段性**

The original AAG generation in TAGAPT relied on the GraphAF model, augmented by five foundational domain rules to ensure the generated graph structures were logically sound.1 These rules govern the validity of system events (e.g., a Fork relation should not occur between a process and a file, as mandated by Rule I and verified against Table II) and the lifecycle of entities (e.g., a process cannot be involved in operations before it is forked, Rule IV; deleted entities should not accept operations, Rule II/III; child processes cannot reverse-manipulate the parent, Rule V).1

TAGAPT 中的原始 AAG 生成依赖于 GraphAF 模型，并辅以五条基础领域规则，以确保生成的图结构在逻辑上是合理的。1 这些规则控制着系统事件的有效性（例如，根据规则 I 的规定，进程和文件之间不应该发生 Fork 关系，并根据表 II 进行了验证）和实体的生命周期（例如，进程在被 fork 之前不能参与操作，规则 IV；已删除的实体不应该接受操作，规则 II/III；子进程不能反向操作父进程，规则 V）。1

In the context of DRL, these five domain constraints translate into essential safety and rationality mechanisms. They must be encoded as strict constraints or policy masking operations on the action space selection. This serves as a foundational layer of the reward structure, specifically the $R\_{Rationality}$ component, which prevents the RL agent from selecting nonsensical provenance triples. By enforcing these constraints from the start, the learning efficiency of the agent is dramatically improved, avoiding the lengthy process of discovering fundamental system logic through trial-and-error.4在深度强化学习（DRL）的背景下，这五个领域约束转化为必要的安全性和合理性机制。它们必须被编码为对动作空间选择的严格约束或策略掩蔽操作。这构成了奖励结构的基础层，特别是 R 值。  
理性  
​  
该组件可防止强化学习智能体选择无意义的来源三元组。通过从一开始就强制执行这些约束，智能体的学习效率显著提高，避免了通过反复试错来发现基本系统逻辑的漫长过程。4  
Furthermore, TAGAPT introduced a structural division, separating a complete attack (AAG) into four key attack stages, defining the sequential progression of an Advanced Persistent Threat: Initial Invasion, Privilege Exploration, Sensitive Action, and Target Achievement.1 This decomposition, achieved using the Attack Stage Division Algorithm, identifies Stage Characterization Nodes (SCNs)—central processes marking the beginning of a stage—based on metrics like Globality, Locality, and Specificity.1

此外，TAGAPT 引入了结构划分，将完整的攻击（AAG）分为四个关键攻击阶段，定义了高级持续性威胁的顺序进展：初始入侵、权限探索、敏感操作和目标达成。1 这种分解是利用攻击阶段划分算法实现的，它基于全局性、局部性和特异性等指标来识别阶段特征节点（SCN）——标志着阶段开始的中心进程。1

These four stages inherently define the macro-tactical phases of the MDP. The DRL agent's policy optimization must, therefore, be shaped to favor transitions that complete one stage and initiate the next. The metrics used to identify SCNs, such as the ranking score $R(SCN\_{i}^{can})$, which combines normalized flow counts and degree counts, can be leveraged as high-value, sparse progressive rewards in the DRL framework:

这四个阶段从本质上定义了 MDP 的宏观战术阶段。因此，DRL 智能体的策略优化必须有利于完成一个阶段并启动下一个阶段的转换。用于识别 SCN 的指标，例如结合了归一化流计数和度计数的排名分数 $R(SCN\_{i}^{can})$ ，可以作为 DRL 框架中的高价值、稀疏的渐进式奖励：

$$R(SCN\_{i}^{can})=\\omega\\times\\frac{Id(SCN\_{i}^{can},L^{flow})}{len(L^{flow})} \+ (1-\\omega)\\frac{Id(SCN\_{i}^{can}, L^{degree})}{len(L^{degree})} \\text{ \[1\]}$$  
where $L^{flow}$ and $L^{degree}$ are lists ranking candidate SCNs by flow count and degree, respectively, and $\\omega$ is the flow weight (set to $0.5$ in TAGAPT evaluations).1 Achieving a high $R(SCN\_{i}^{can})$ score provides measurable intermediate objectives, guiding the agent toward rational attack progression.

其中 $L^{flow}$ 和 $L^{degree}$ 分别是按流数和度对候选 SCN 进行排名的列表， $\\omega$ 是流权重（在 TAGAPT 评估中设置为 $0.5$ ）。1 获得较高的 $R(SCN\_{i}^{can})$ 分数可以提供可衡量的中间目标，引导智能体进行合理的攻击进程。

Finally, the integrity of the attack sequencing is ensured by adhering to explicit temporal dependencies between MITRE ATT\&CK tactics, which TAGAPT enforces during its Genetic Algorithm instantiation phase (referenced in Figure 3 of the original methodology).1 For instance, if a tactical policy requires "Initial Access" to occur before "Resource Development" in Stage I, the DRL agent's policy must respect this temporal sequence when selecting macro-actions. This structural guidance is crucial for shaping the DRL policy towards synthesizing attacks that are strategically coherent.

最后，通过遵守 MITRE ATT\&CK 策略之间的明确时间依赖性来保证攻击序列的完整性，TAGAPT 在其遗传算法实例化阶段强制执行此依赖性（在原始方法的图 3 中引用）。1 例如，如果战术策略要求在第一阶段“初始访问”之前进行“资源开发”，那么深度强化学习（DRL）智能体在选择宏观行动时必须遵循这一时间顺序。这种结构性指导对于构建战略连贯的攻击策略至关重要。

### **C. Core Provenance Entities and Relations**

**C. 核心溯源实体和关系**

The underlying data structure inherited by the DRL environment is summarized below, highlighting the high granularity inherited from TAGAPT.

下面总结了 DRL 环境继承的底层数据结构，重点介绍了从 TAGAPT 继承的高粒度。

Table II.1: Core Provenance Entities and Relations in TAGAPT

表 II.1：TAGAPT 中的核心溯源实体和关系

| Component  成分 | Entity Symbol  实体符号 | Description  描述 | Relevant Relations  相关关系 |
| :---- | :---- | :---- | :---- |
| Processes  流程 | MP, TP  MP，TP | Malicious Process, Benign Process 恶意过程，良性过程 | FR (Fork), UK (Unlink), IJ (Inject), EX (Exec), RD (Read), WR (Write), CD (Change Mode), ST (Send To), RF (Receive From) FR（分叉），UK（断开连接），IJ（注入），EX（执行），RD（读取），WR（写入），CD（更改模式），ST（发送到），RF（接收自） |
| Files  文件 | MF, SF, TF  MF、SF、TF | Malicious File, System File, Temporary File 恶意文件、系统文件、临时文件 | RD, WR, UK, EX  RD、WR、英国、EX |
| Socket  插座 | SO | Socket/Network Endpoint  套接字/网络端点 | ST, RF  ST，RF |

## **III. MDP Formulation: State Space $\\mathcal{S}$ and Environment Mapping**

**三、MDP 建模：状态空间 $\\mathcal{S}$ 和环境映射**

The State Space $\\mathcal{S}$ represents the complex and constantly evolving environment the DRL agent interacts with. It must integrate the static infrastructure ("network topology" and "device vulnerabilities") with the dynamic provenance generated by the agent.

状态空间 $\\mathcal{S}$ 代表了 DRL 智能体所交互的复杂且不断变化的环境。它必须将静态基础设施（“网络拓扑”和“设备漏洞”）与智能体生成的动态溯源信息整合起来。

### **A. Formalizing the Cyber-Physical Environment**

**A. 网络物理环境的形式化**

The foundational environment is the static network topology, $G\_{Net}=(V\_{N}, E\_{N})$, where nodes $V\_{N}$ represent hosts (servers, workstations) and edges $E\_{N}$ represent network connectivity.5基础环境是静态网络拓扑结构 $G\_{Net}=(V\_{N}, E\_{N})$ ，其中节点 V  
N  
​  
表示主机（服务器、工作站）和边 E  
N  
​  
表示网络连接。5  
The state at any time step $t$, $S\_t$, is defined by an augmented heterogeneous graph, $G\_t$. This graph $G\_t$ is a continuous amalgamation of the static $G\_{Net}$ infrastructure graph and the dynamically generated provenance subgraph $IAG\_{t-1}$.7 As the agent acts, it appends provenance edges and nodes (derived from TAGAPT entities) to $G\_{Net}$, thereby changing its state.任意时间步 $t$ 时的状态，S  
t  
​  
由增广异构图 G 定义。  
t  
​  
这张图 G  
t  
​  
是静态 G 的连续融合  
网  
​  
基础设施图和动态生成的溯源子图 IAG  
t−1  
​  
。7 当代理执行操作时，它会将来源边和节点（源自 TAGAPT 实体）添加到 G 中。  
网  
​  
从而改变其状态。

### **B. GNN-based State Representation $\\mathcal{S}$**

**B. 基于图神经网络的状态表示 $\\mathcal{S}$**

Since the graph $G\_t$ is non-Euclidean, large, and highly heterogeneous, traditional DRL approaches that compress the state into a simple vector would incur significant information loss, particularly regarding topological structure.8 Therefore, Graph Neural Networks (GNNs) are indispensable for effectively processing the relational data and deriving a rich state representation.5 The state embedding $s\_t$ fed to the DRL policy network is derived via a GNN encoder: $s\_t \= \\text{GNN}(G\_t)$.由于图 G  
t  
​  
如果系统是非欧几里得的、大的、高度异构的，那么将状态压缩成简单向量的传统 DRL 方法将会造成严重的信息损失，尤其是在拓扑结构方面。8 因此，图神经网络（GNN）对于有效处理关系数据和导出丰富的状态表示是必不可少的。5 状态嵌入  
t  
​  
输入到 DRL 策略网络的参数是通过 GNN 编码器得到的： $s\_t \= \\text{GNN}(G\_t)$ 。

#### **Node Feature Encoding ($F\_V$)**

**节点特征编码（F） V ​ )**

Each host node in $G\_{Net}$ must be initialized with features that accurately describe the "terrain" as required by the query (vulnerabilities and configurations):G 中的每个主机节点  
网  
​  
必须使用能够准确描述查询所需的“地形”（漏洞和配置）的特征进行初始化：

1. **Vulnerability and Configuration Features:** Static features include asset type, operating system (Linux/Windows, as distinguished by TAGAPT’s instantiation library for rationality 1), CVSS scores of known vulnerabilities, and running service configurations. Crucially, these features must include a dense vector describing the host's susceptibility to specific MITRE ATT\&CK Techniques (TTPs).9  
   **漏洞和配置特性：** 静态特性包括资产类型、操作系统（Linux/Windows，由 TAGAPT 的实例化库进行区分，以实现理性化）。1 ）、已知漏洞的 CVSS 评分以及正在运行的服务配置。至关重要的是，这些特征必须包含一个密集向量，用于描述主机对特定 MITRE ATT\&CK 技术 (TTP) 的易受攻击性。9  
2. **Attack Provenance Features (Dynamic):** Dynamic features reflecting the attack's presence, derived directly from TAGAPT's entity types, must be mapped onto the host. Examples include the number of active Malicious Processes (MP), the current access privilege level achieved (low, high, system), and markers indicating the state of attack progression (e.g., successful presence of a Stage Characterization Node).  
   **攻击溯源特征（动态）：** 反映攻击存在的动态特征，直接来源于 TAGAPT 的实体类型，必须映射到主机上。例如，活跃恶意进程（MP）的数量、当前获得的访问权限级别（低、高、系统）以及指示攻击进展状态的标记（例如，成功存在阶段特征节点）。

#### **Edge Feature Encoding ($F\_E$)**

**边缘特征编码（F） E ​ )**

Edges in $G\_{Net}$ define the initial connectivity and trust. Features include network metrics (latency, bandwidth), firewall rules, and latent representations of potential communication channels that could support provenance relations like ST (Send To), RF (Receive From), or remote execution events.G 中的边  
网  
​  
定义初始连接和信任。功能包括网络指标（延迟、带宽）、防火墙规则以及可能支持溯源关系（例如 ST（发送到）、RF（接收自）或远程执行事件）的潜在通信通道的潜在表示。

#### **GNN Architecture Selection**

**GNN 架构选择**

Given that the combined graph $G\_t$ is highly heterogeneous, integrating system entities (processes, files) with network hosts, an R-GCN (Relational Graph Convolutional Network) or a GAT (Graph Attention Network) is preferred.1 The R-GCN approach is particularly suitable as it was already utilized by TAGAPT's GraphAF model for learning embedding vectors $V\_{i}^{L}\\in\\mathbb{R}^{n\\times k}$ based on relational types \[1, Equation 1 (implicit)\]. By using a similar architecture for the state encoder, the resulting embedding $s\_t$ captures the complex structure of the combined network and provenance information, which is essential for accurate policy prediction.给定组合图 G  
t  
​  
由于系统高度异构，需要将系统实体（进程、文件）与网络主机集成，因此最好使用 R-GCN（关系图卷积网络）或 GAT（图注意力网络）。1 R-GCN 方法特别合适，因为它已被 TAGAPT 的 GraphAF 模型用于基于关系类型学习嵌入向量 $V\_{i}^{L}\\in\\mathbb{R}^{n\\times k}$ \[1 ，公式 1（隐式）\]。通过使用类似的架构作为状态编码器，得到的嵌入 s  
t  
​  
捕捉组合网络和来源信息的复杂结构，这对于准确的政策预测至关重要。

## **IV. Granular Action Space $\\mathcal{A}$: Decomposing AAGs into Atomic Steps**

**IV. 颗粒作用空间 $\\mathcal{A}$ ：将 AAG 分解为原子步骤**

The most significant technical synthesis required is the translation of TAGAPT’s abstract attack graph generation process into a series of discrete actions suitable for an RL agent \[User Query\]. An AAG or AASG fragment represents the *outcome* of multiple coordinated system activities and cannot be an atomic action itself.11 Therefore, a hierarchical action model is necessary.

最重要的技术整合是将 TAGAPT 的抽象攻击图生成过程转化为一系列适用于强化学习智能体的离散动作\[用户查询\]。AAG 或 AASG 片段代表多个协同系统活动的*结果* ，其本身不能是原子动作。11 因此，需要采用层级式行动模型。

### **A. Hierarchical Action Model**

**A. 层级行动模型**

The action space $\\mathcal{A}$ is structured to decouple tactical choice from provenance instantiation:

行动空间 $\\mathcal{A}$ 的结构旨在将战术选择与来源实例化解耦：

#### **1\. Macro-Action ($\\mathcal{A}\_{Macro}$): Tactical Selection**

**1\. 宏观行动（A 宏 ​ 战术选择**

The agent selects a high-level strategic move $a\_{Macro}$ corresponding to an ATT\&CK technique (TTP), which is directly mapped from one of the hundreds of specific TAGAPT Technical Policies defined for each attack stage.1 This action space is discrete, defined by the total enumeration of policies across the four stages (58 policies for Stage I, 119 for Stage II, 94 for Stage III, and 101 for Stage IV).1代理人选择了一个高层次的战略举措。  
宏  
​  
对应于 ATT\&CK 技术（TTP），该技术直接映射到为每个攻击阶段定义的数百个特定 TAGAPT 技术策略之一。1 该行动空间是离散的，由四个阶段的策略总数定义（第一阶段 58 个策略，第二阶段 119 个策略，第三阶段 94 个策略，第四阶段 101 个策略）。1  
A Macro-Action $a\_{Macro}$ is defined as a tuple: $a\_{Macro} \= (\\text{Stage}\_k, \\text{Technique}\_{T}, \\text{Target Device}\_{v})$.宏观行动  
宏  
​  
定义为元组： $a\_{Macro} \= (\\text{Stage}\_k, \\text{Technique}\_{T}, \\text{Target Device}\_{v})$ 。  
The selection must be constrained by the current state $s\_t$, ensuring the chosen technique $T$ is relevant to the current stage $k$ and technically feasible on the target device $v$, based on $v$'s vulnerability features ($F\_V$).选择必须受当前状态的约束  
t  
​  
确保所选技术 $T$ 与当前阶段 $k$ 相关，并且在目标设备 $v$ 上技术上可行，基于 $v$ 的漏洞特征(F)。  
V  
​  
).

#### **2\. Micro-Action ($\\mathcal{A}\_{Micro}$): Provenance Instantiation**

**2\. 微行动（A 微 ​ ）：溯源实例化**

Executing $a\_{Macro}$ triggers a sequence of Micro-Actions that instantiate the provenance events dictated by the corresponding TAGAPT technical policy (e.g., the policy for Execution Tactic User Execution (T1204) includes the actions and).1 These Micro-Actions are the low-level graph modification primitives: adding a node of a specific type (MP, TP, MF, etc.) or adding an edge of a specific relation type (RD, WR, FR, etc.) between existing nodes.11执行  
宏  
​  
触发一系列微操作，这些微操作实例化由相应的 TAGAPT 技术策略规定的溯源事件（例如， 执行策略用户执行 (T1204) 的策略包括 and 操作）。1 这些微操作是低级的图修改原语：在现有节点之间添加特定类型（MP、TP、MF 等）的节点或添加特定关系类型（RD、WR、FR 等）的边。11  
The transition $G\_t \\rightarrow G\_{t+1}$ resulting from the execution of $a\_{Macro}$ involves the creation of a new Abstract Attack Subgraph (AASG) segment that is appended to the current state graph $G\_t$.过渡 G  
t  
​

→G  
t+1  
​  
执行  
宏  
​  
这包括创建一个新的抽象攻击子图（AASG）片段，并将其附加到当前状态图 G 中。  
t  
​  
.

### **B. Incorporating TAGAPT Instantiation Logic**

**B. 整合 TAGAPT 实例化逻辑**

TAGAPT’s original Instantiation phase employed a Genetic Algorithm (GA) to find the optimal technique-level interpretation for each AASG, maximizing the entity matching ratio while respecting temporal dependencies.1 In the DRL framework, the trained Policy Network $\\pi(a|s)$ *replaces* the function of this GA. Instead of passively evaluating pre-generated graphs, the DRL agent actively *generates* the optimal technical explanation sequentially, step-by-step, maximizing the expected future reward.14

TAGAPT 的原始实例化阶段采用遗传算法 (GA) 为每个 AASG 找到最佳的技术级解释，在尊重时间依赖性的同时最大化实体匹配率。1 在 DRL 框架中，训练好的策略网络 $\\pi(a|s)$  *取代了*遗传算法的功能。DRL 智能体不再被动地评估预先生成的图，而是主动地按顺序、逐步地*生成*最优的技术解释，从而最大化预期的未来奖励。14

To ensure the final output is a usable IAG—not just an abstract graph—the agent must access TAGAPT's Instantiation Library.1 This library, compiled from CTI reports, CAPEC, ATT\&CK, and execution examples (like ANY.RUN data), provides the necessary semantic instance names. Upon executing a Micro-Action that creates an abstract entity (e.g., MP), the agent must select a concrete instance name (e.g., changing abstract MP to the string identifier powershell.exe) from the library that aligns with the chosen technique $T$.1 This semantic grounding ensures the utility of the generated sample for system security analysts.

为了确保最终输出是可用的 IAG（而不仅仅是抽象图），代理必须访问 TAGAPT 的实例化库。1 该库由 CTI 报告、CAPEC、ATT\&CK 和执行示例（例如 ANY.RUN 数据）编译而成，提供了必要的语义实例名称。执行创建抽象实体（例如，MP）的微操作时，代理必须从库中选择一个与所选技术相符的具体实例名称（例如，将抽象 MP 更改为字符串标识符 powershell.exe ） $T$ 。1 这种语义基础确保了生成的样本对系统安全分析人员的实用性。

## **V. Policy Optimization and Multi-Tiered Reward Engineering $\\mathcal{R}$**

**V. 策略优化和多层奖励工程 $\\mathcal{R}$**

The success of the DRL training process is critically dependent on designing an effective reward function $\\mathcal{R}$ that translates the complex, domain-specific requirements of a successful APT path—which are inherently static in the original TAGAPT GA fitness metric—into a dynamic, sequential signal.15 Rule-based rewards are necessary to guide the policy optimization process towards structured, coherent graph generation outputs.4

DRL 训练过程的成功与否，关键在于设计一个有效的奖励函数 $\\mathcal{R}$ ，该函数将成功的 APT 路径的复杂、特定领域的要求（在原始的 TAGAPT GA 适应度指标中本质上是静态的）转化为动态的、序列的信号。15 基于规则的奖励对于引导策略优化过程生成结构化、连贯的图输出至关重要。4

### **A. Translating GA Fitness to RL Reward**

**A. 将遗传算法适应度转化为强化学习奖励**

The Genetic Algorithm in TAGAPT maximized individual fitness based on two criteria: the ratio of matched entities (provenance completeness) and the rationality of the tactical combinations (temporal sequence adherence).1 These map directly to maximizing local provenance completeness ($R\_{Structure}$) and ensuring global tactical progression ($R\_{Temporal}$) in the DRL framework.TAGAPT 中的遗传算法根据两个标准最大化个体适应度：匹配实体的比例（溯源完整性）和战术组合的合理性（时间顺序遵守情况）。1 这些直接对应于最大化本地溯源完整性（R）。  
结构  
​  
）并确保全球战术进展（R  
颞  
​  
在 DRL 框架中。

### **B. Reward Function $\\mathcal{R}(s\_t, a\_t)$  B. 奖励函数 $\\mathcal{R}(s\_t, a\_t)$**

The reward function must be defined hierarchically to provide both dense, local feedback for atomic actions and sparse, high-value signals for strategic objectives.

奖励函数必须按层级定义，以便既能为原子动作提供密集、局部的反馈，又能为战略目标提供稀疏、高价值的信号。

#### **1\. Tier 1: Provenance Structure and Rationality Reward ($R\_{Structure}$)**

**1\. 第一层级：溯源结构和理性奖励（R） 结构 ​ )**

This tier provides immediate feedback on the integrity of the provenance subgraph generated by the action $a\_t$.此层级可立即反馈由操作 a 生成的溯源子图的完整性。  
t  
​  
.

* Rationality Penalty ($R\_{Penalty}$): A large, immediate negative reward is administered if the action $a\_t$ results in the violation of any of TAGAPT’s five fundamental Domain Constraints (I-V) (e.g., attempting a Write operation to a File entity that has already been Unlinked/deleted).1 This ensures basic system logic is respected.理性惩罚（ $R\_{Penalty}$ ）： 如果行为不理性，则会立即给予较大的负面奖励。  
  t  
  ​  
  导致违反 TAGAPT 的五个基本域约束中的任何一个（IV）（例如，尝试对已取消链接/删除的文件实体执行写入操作）。1 这样可以确保基本系统逻辑得到遵循。  
* **Completeness Reward:** A small, dense positive reward is proportional to the successful generation and instantiation of new entities and relations, effectively maximizing the local entity matching ratio. This promotes the creation of well-formed provenance segments.  
  **完整性奖励：** 少量但密集的正向奖励与新实体和关系的成功生成和实例化成正比，从而有效最大化本地实体匹配率。这有助于创建结构良好的溯源片段。

#### **2\. Tier 2: Progressive Stage Achievement Reward ($R\_{Temporal}$)**

**2\. 第二层级：渐进阶段成就奖励（R） 颞 ​ )**

This sparse reward guides the agent across the macro-tactical stages of the attack.

这种稀少的奖励引导着智能体完成攻击的宏观战术阶段。

* Tactical Progression: A medium positive reward is granted when the agent successfully executes a Macro-Action $a\_{Macro}$ that adheres to the established temporal dependencies between tactics (e.g., successfully moving from the Reconnaissance tactic to the Initial Access tactic within Stage 1, based on the prescribed sequence).1战术进展： 当智能体成功执行宏动作时，将获得中等的正向奖励。  
  宏  
  ​  
  遵循既定的战术间时间依赖关系（例如，根据规定的顺序，在第一阶段内成功地从侦察战术过渡到初始进入战术）。1  
* **SCN Achievement:** A significant, shaping reward is provided when the generated provenance confirms the emergence of a robust Stage Characterization Node (SCN), calculated using the metrics for Globality and Locality described by the SCN ranking score $R(SCN\_{i}^{can})$.1 This rewards complex, multi-operation achievements that centralize control within a stage.  
  **SCN 成就：** 当生成的来源证实了稳健的阶段特征节点 (SCN) 的出现时，将提供重要的塑造性奖励，该奖励是使用 SCN 排名分数 $R(SCN\_{i}^{can})$ 描述的全局性和局部性指标计算得出的。1 这种奖励机制旨在实现舞台内集中控制的复杂、多操作的成就。

#### **3\. Tier 3: Utility and Terminal Reward ($R\_{Utility}$)**

**3\. 第三层级：实用性和最终奖励（R） 公用事业 ​ )**

This reward defines the ultimate objective and ensures the practicality of the output IAG.

该奖励定义了最终目标，并确保了输出 IAG 的实用性。

* **Goal Achievement:** A large positive terminal reward is delivered upon successfully generating a complete IAG that reaches the final Target Achievement stage, confirming the execution of a critical tactic such as Exfiltration or Impact.1  
  **目标达成：** 成功生成完整的 IAG 并达到最终目标达成阶段后，将获得大量积极的最终奖励，确认执行了诸如撤离或冲击之类的关键策略。1  
* **Efficiency Metric:** A small, constant negative reward (cost) is applied for every action taken, encouraging the DRL agent to discover the minimal, most efficient attack path to the target, thus minimizing the time-to-compromise.  
  **效率指标：** 对采取的每个行动应用一个较小的、恒定的负奖励（成本），鼓励 DRL 智能体发现到达目标的最小、最有效的攻击路径，从而最大限度地减少妥协时间。  
* **Diversity Metric (Optional):** Rewards can be structured to favor action sequences that leverage diverse techniques (e.g., promoting Living Off The Land (LOTL) tactics over standard Malware execution), ensuring the generated IAGs possess the breadth of features necessary for robust defense training.1  
  **多样性指标（可选）：** 奖励可以设计成有利于利用各种技术的行动序列（例如，鼓励使用“自给自足”（LOTL）策略而不是标准恶意软件执行），以确保生成的 IAG 具备强大的防御训练所需的广泛功能。1

### **C. Multi-Tiered DRL Reward Mechanism**

**C. 多层级深度强化学习奖励机制**

The reward framework represents a structured translation of static constraints into dynamic incentives, optimizing both local graph validity and global strategic progression.

奖励框架代表了将静态约束结构化地转化为动态激励，从而优化了局部图的有效性和全局战略进展。

Table V.1: Translation of TAGAPT Success Metrics to DRL Reward Components

表 V.1：TAGAPT 成功指标到 DRL 奖励组成部分的转换

| Reward Tier  奖励等级 | Purpose  目的 | TAGAPT Metric/Constraint AnalogTAGAPT 度量/约束模拟 | Reward Type  奖励类型 |
| :---- | :---- | :---- | :---- |
| $R\_{Structure}$ | Local Provenance Validity 本地产地有效性 | Domain Constraints (I-V); Entity Matching Ratio 领域约束（IV）；实体匹配率 | Immediate, Dense Penalty/Sparse Positive 立即生效，密集惩罚/稀疏正向 |
| $R\_{Temporal}$ | Attack Progression Guidance 攻击进程指南 | Temporal Tactic Dependencies (Fig. 3); SCN Identification Metrics ($R(SCN\_{i}^{can})$) 时间策略依赖性（图 3）；SCN 识别指标（ $R(SCN\_{i}^{can})$ ） | Sparse Positive Shaping  稀疏正整形 |
| $R\_{Utility}$ | Global Attack Success  全球攻击成功 | Reaching Target Achievement; Overall Instantiation Completeness 达到目标达成；整体实例化完成 | Terminal, Large Positive  终端，大正极 |

## **VI. DRL Architecture and Training Methodology**

**六、深度强化学习架构和训练方法**

The implementation requires a stable DRL architecture capable of handling the high-dimensional, structured state space derived from the GNN encoder and the large, discrete action space defined by the TAGAPT policies.

该实现需要一个稳定的 DRL 架构，能够处理从 GNN 编码器导出的高维结构化状态空间和由 TAGAPT 策略定义的大型离散动作空间。

### **A. Model Selection: GNN-DRL Integration**

**A. 模型选择：GNN-DRL 集成**

An **Actor-Critic** framework is recommended, specifically Proximal Policy Optimization (PPO) or an evolution thereof (like Generalized Advantage Estimation, GAE), due to its stability and effectiveness in large, complex state spaces.3

建议采用 **Actor-Critic** 框架，特别是近端策略优化 (PPO) 或其改进版本（如广义优势估计，GAE），因为它在大型、复杂的状态空间中具有稳定性和有效性。3

The system architecture comprises two main components:

该系统架构包含两个主要组成部分：

1. GNN Encoder: An R-GCN or GAT module takes the heterogeneous state graph $G\_t$ as input and produces the fixed-size state embedding $s\_t$.GNN 编码器： R-GCN 或 GAT 模块接收异构状态图 G。  
   t  
   ​  
   作为输入，并生成固定大小的状态嵌入 s  
   t  
   ​  
   .  
2. Policy and Value Heads: The Actor head takes $s\_t$ and outputs a probability distribution over the discrete Macro-Action space $\\mathcal{A}\_{Macro}$ (the collection of TAGAPT technical policies). The Critic head estimates the expected return $V(s\_t)$ from the current state.政策和价值方向： 行动者方向承担  
   t  
   ​  
   并输出离散宏观动作空间 A 上的概率分布  
   宏  
   ​  
   （TAGAPT 技术策略的集合）。Critic 负责人估计当前状态下的预期收益 $V(s\_t)$ 。

### **B. Addressing the Grounding Problem**

**B. 解决接地问题**

A significant challenge in applying RL to automated penetration testing is the "grounding problem"—ensuring that abstract actions chosen by the agent are technically feasible within the real-world network context.7

将强化学习应用于自动化渗透测试的一个重大挑战是“接地问题”——确保代理选择的抽象动作在现实世界的网络环境中具有技术可行性。7

In this framework, the issue is mitigated through meticulous state engineering and reward calibration. The initial feature set $F\_V$ must rigorously encode the network's vulnerabilities and configuration dependencies (the "terrain").1 The $R\_{Structure}$ reward then acts as a crucial filter: if the agent selects an action that represents a technique (e.g., Credential Dumping T1555) that cannot logically operate on the target device (due to OS type or access rights derived from $F\_V$), the action is immediately masked or heavily penalized by $R\_{Penalty}$. This ensures the policy learns to select only grounded, technically sound actions, which aligns with the observed technical prerequisites derived from CTI data used by TAGAPT.在此框架下，通过精细的状态工程和奖励校准来缓解该问题。初始特征集 F  
V  
​  
必须严格编码网络的漏洞和配置依赖关系（“地形”）。1 R  
结构  
​  
奖励机制则起到关键的过滤作用：如果代理选择的动作代表了一种技术（例如，凭证转储 T1555），而该技术由于操作系统类型或从 F 派生的访问权限等原因，无法在目标设备上进行逻辑操作，则该动作将被判定为无效。  
V  
​  
），该行为会立即被 R 掩盖或受到严厉惩罚。  
惩罚  
​  
这样可以确保该策略学会只选择有理有据、技术上合理的行动，这与 TAGAPT 使用的 CTI 数据得出的观察到的技术先决条件相一致。

### **C. Computational Complexity and Scalability**

**C. 计算复杂性和可扩展性**

The original TAGAPT process demonstrated acceptable overhead, averaging approximately 18 seconds to generate one fully usable IAG, with the majority of time spent on the GraphAF generation phase.1 While generating 1,000 samples required significant effort, the goal was sample volume, not real-time decision-making.

原始的 TAGAPT 流程表现出可接受的开销，平均生成一个完全可用的 IAG 大约需要 18 秒，其中大部分时间都花在了 GraphAF 生成阶段。1 虽然生成 1,000 个样本需要付出巨大的努力，但目标是样本量，而不是实时决策。

In the DRL paradigm, the computational bottleneck shifts from *sample generation complexity* to *policy optimization complexity*. Training a GNN-DRL agent can be computationally intensive due to the GNN message passing required for state embedding.17 However, the use of structured, rule-based reward signals (Tiers 1 and 2\) acts as a powerful regularization and shaping mechanism. This guidance significantly improves the sample efficiency of the DRL algorithm compared to black-box trial-and-error, helping to mitigate the protracted training periods often associated with complex Q-learning or DQN methods on high-dimensional state/action spaces.3 Efficient architectures, like MulVAL, often used for attack graph analysis, are known for their relative scalability, suggesting that integrating the network representation efficiently is achievable.18

在深度强化学习（DRL）范式中，计算瓶颈从*样本生成复杂度*转移到*策略优化复杂度* 。由于状态嵌入需要进行 GNN 消息传递，因此训练 GNN-DRL 智能体可能需要大量的计算资源。17 然而，使用结构化的、基于规则的奖励信号（第一层和第二层）可以起到强大的正则化和塑造作用。与黑盒试错法相比，这种指导显著提高了深度强化学习算法的样本效率，有助于缓解在高维状态/动作空间上复杂 Q 学习或 DQN 方法通常伴随的漫长训练周期。3 像 MulVAL 这样的高效架构经常用于攻击图分析，它们以相对的可扩展性而闻名，这表明高效地集成网络表示是可以实现的。18

### **D. Training and Evaluation Strategy**

To maximize training efficiency given the sequential nature of the APT attack path (I-IV), a **Curriculum Learning** strategy is highly effective. The agent can be initially trained and rewarded solely for mastering early-stage tactics (Stage I: Initial Invasion), before progressively introducing the complexities and dependencies of later stages.

The final policy is evaluated using multiple metrics:

1. **Success Rate:** The frequency with which the agent generates complete, coherent IAGs that achieve the terminal $R\_{Utility}$ reward.  
2. **Path Efficiency:** The average number of actions required to achieve the goal (minimizing the negative cost per action).  
3. **Policy Diversity:** The breadth of techniques utilized, ensuring the generated samples span different attack patterns (e.g., verifying the successful synthesis of LOTL attacks, which TAGAPT showed an ability to generalize 1).

## **VII. Strategic Implications and Conclusions**

The integration of the TAGAPT provenance framework with DRL represents a significant advancement in synthetic cyber threat intelligence. The generated IAGs are not merely statistically plausible; they are topologically validated, technique-specific, and optimized attack policies, providing unparalleled fidelity for security training and defense testing.

### **A. Enhancing Downstream Utility**

The primary benefit lies in the utility of the generated IAGs for enhancing existing security operations systems:

* **False Positive Filtering:** The DRL-generated IAGs represent optimal, *worst-case* attack scenarios tailored to a specific network's vulnerabilities. This highly targeted training data is superior for systems like Kairos, the state-of-the-art anomaly detection system previously tested by TAGAPT.1 Training a one-class classification model (like one-class SVM) on these optimal IAGs allows for the identification of legitimate malicious patterns with higher precision.1 This policy-driven generation is expected to improve upon TAGAPT’s original achievement of filtering out $73\\%$ of observed false positives, delivering higher confidence alerts and significantly reducing human analyst fatigue.1  
* **Threat Hunting Precision:** In threat hunting scenarios, security analysts search for malicious activities starting from a Point of Interest (POI) by constructing query graphs.1 The DRL-synthesized IAGs, generated for the exact target topology and leveraging specific techniques (like those assessed in DARPA scenarios), serve as immensely precise hunting query graphs. Since these queries are policy-optimized, they reduce both false positives and false negatives compared to generalized, manually constructed queries, mirroring the success observed when comparing TAGAPT's generalized IAGs against expert-provided query graphs (Table IX in the original study).1

### **B. Future Directions and Policy Refinement**

The proposed DRL architecture inherently addresses several limitations noted in the original TAGAPT system. The emphasis on efficiency in the $R\_{Utility}$ reward naturally encourages the agent to select only the provenance steps essential for state transition. This policy optimization should mitigate the minor redundancy and noise introduced by the GraphAF model during the original AAG generation process, where edge identification accuracy for attack staging was measured at $91.91\\%$.1

Future work should focus on automating the system maintenance identified as a challenge in TAGAPT (the manual curation of policies and the instantiation library as new TTPs emerge).1 The integration of large language models (LLMs) with GNNs could be explored to automatically extract new policies, techniques, and provenance instantiation details from unstructured CTI reports, thereby dynamically updating the $\\mathcal{A}\_{Macro}$ action space and the Instantiation Library.1

Furthermore, moving from maximizing a static reward to **Adversarial Reinforcement Learning** would allow the DRL agent (Red Agent/Attacker) to be trained against a second DRL agent (Blue Agent/Defender), generating highly robust attack paths that actively seek to minimize the detectability or impact probability, thereby synthesizing samples that maximize stealth alongside destructive efficacy.20

#### **Works cited**

1. TAGAPT\_Toward\_Automatic\_Generation\_of\_APT\_Samples\_With\_Provenance-Level\_Granularity.pdf  
2. A reinforcement learning approach for attack graph analysis Yousefi, Mehdi \- Glasgow Caledonian University, accessed on November 10, 2025, [https://researchonline.gcu.ac.uk/files/26084628/H.Tianfield\_attack\_graph.pdf](https://researchonline.gcu.ac.uk/files/26084628/H.Tianfield_attack_graph.pdf)  
3. Employing Deep Reinforcement Learning to Cyber-Attack Simulation for Enhancing Cybersecurity \- MDPI, accessed on November 10, 2025, [https://www.mdpi.com/2079-9292/13/3/555](https://www.mdpi.com/2079-9292/13/3/555)  
4. Compile Scene Graphs with Reinforcement Learning \- arXiv, accessed on November 10, 2025, [https://arxiv.org/html/2504.13617v1](https://arxiv.org/html/2504.13617v1)  
5. (PDF) GNN-based Deep Reinforcement Learning with Adversarial Training for Robust Optimization of Modern Tactical Communication Systems \- ResearchGate, accessed on November 10, 2025, [https://www.researchgate.net/publication/371513204\_GNN-based\_Deep\_Reinforcement\_Learning\_with\_Adversarial\_Training\_for\_Robust\_Optimization\_of\_Modern\_Tactical\_Communication\_Systems](https://www.researchgate.net/publication/371513204_GNN-based_Deep_Reinforcement_Learning_with_Adversarial_Training_for_Robust_Optimization_of_Modern_Tactical_Communication_Systems)  
6. Achieving Network Resilience through Graph Neural Network-enabled Deep Reinforcement Learning \- arXiv, accessed on November 10, 2025, [https://arxiv.org/html/2501.11074v1](https://arxiv.org/html/2501.11074v1)  
7. A Layered Reference Model for Penetration Testing with Reinforcement Learning and Attack Graphs \- arXiv, accessed on November 10, 2025, [https://arxiv.org/pdf/2206.06934](https://arxiv.org/pdf/2206.06934)  
8. Automated Cyber Defense with Generalizable Graph-based Reinforcement Learning Agents, accessed on November 10, 2025, [https://arxiv.org/html/2509.16151v1](https://arxiv.org/html/2509.16151v1)  
9. MITRE ATT\&CK®, accessed on November 10, 2025, [https://attack.mitre.org/](https://attack.mitre.org/)  
10. KillChainGraph: ML Framework for Predicting and Mapping ATT\&CK Techniques \- arXiv, accessed on November 10, 2025, [https://arxiv.org/pdf/2508.18230](https://arxiv.org/pdf/2508.18230)  
11. Graph Reinforcement Learning Overview \- Emergent Mind, accessed on November 10, 2025, [https://www.emergentmind.com/topics/graph-reinforcement-learning-grl](https://www.emergentmind.com/topics/graph-reinforcement-learning-grl)  
12. Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation, accessed on November 10, 2025, [http://papers.neurips.cc/paper/7877-graph-convolutional-policy-network-for-goal-directed-molecular-graph-generation.pdf](http://papers.neurips.cc/paper/7877-graph-convolutional-policy-network-for-goal-directed-molecular-graph-generation.pdf)  
13. Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints \- Frontiers, accessed on November 10, 2025, [https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.00059/full](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.00059/full)  
14. Knowledge Graphs and Reinforcement Learning: A Hybrid Approach for Cybersecurity Problems \- UMBC ebiquity, accessed on November 10, 2025, [https://ebiquity.umbc.edu/paper/html/id/1132/Knowledge-Graphs-and-Reinforcement-Learning-A-Hybrid-Approach-for-Cybersecurity-Problems](https://ebiquity.umbc.edu/paper/html/id/1132/Knowledge-Graphs-and-Reinforcement-Learning-A-Hybrid-Approach-for-Cybersecurity-Problems)  
15. Niekum, S., A. Barto, and L. Spector. 2010\. Genetic Programming for Reward Function Search. In IEEE Transactions \- Faculty, accessed on November 10, 2025, [https://faculty.hampshire.edu/lspector/pubs/TAMD-webpost.pdf](https://faculty.hampshire.edu/lspector/pubs/TAMD-webpost.pdf)  
16. Exploration-Driven Genetic Algorithms for Hyperparameter Optimisation in Deep Reinforcement Learning \- MDPI, accessed on November 10, 2025, [https://www.mdpi.com/2076-3417/15/4/2067](https://www.mdpi.com/2076-3417/15/4/2067)  
17. A Hybrid Graph Neural Network-Based Reinforcement Learning Approach for Adaptive Cybersecurity Risk Management in FinTech \- ResearchGate, accessed on November 10, 2025, [https://www.researchgate.net/publication/395750290\_A\_Hybrid\_Graph\_Neural\_Network-Based\_Reinforcement\_Learning\_Approach\_for\_Adaptive\_Cybersecurity\_Risk\_Management\_in\_FinTech](https://www.researchgate.net/publication/395750290_A_Hybrid_Graph_Neural_Network-Based_Reinforcement_Learning_Approach_for_Adaptive_Cybersecurity_Risk_Management_in_FinTech)  
18. GNN-enhanced Traffic Anomaly Detection for Next-Generation SDN-Enabled Consumer Electronics \- arXiv, accessed on November 10, 2025, [https://arxiv.org/html/2510.07109v1](https://arxiv.org/html/2510.07109v1)  
19. TAGAPT: Toward Automatic Generation of APT Samples With Provenance-Level Granularity, accessed on November 10, 2025, [https://www.scholars.northwestern.edu/en/publications/tagapt-toward-automatic-generation-of-apt-samples-with-provenance](https://www.scholars.northwestern.edu/en/publications/tagapt-toward-automatic-generation-of-apt-samples-with-provenance)  
20. Top K Enhanced Reinforcement Learning Attacks on Heterogeneous Graph Node Classification \- arXiv, accessed on November 10, 2025, [https://arxiv.org/html/2408.01964v1](https://arxiv.org/html/2408.01964v1)