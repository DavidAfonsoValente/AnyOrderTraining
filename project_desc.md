# Any-Order Masked Training for Trajectory-Level Supervised Learning in LLM-Based Agents

**PhD Research Proposal**

---

## Abstract

We propose **Any-Order Masked Training (AOMT)**, a supervised fine-tuning (SFT) paradigm for LLM-based agents that reframes agent learning as a trajectory-level reconstruction problem. Rather than fixing a single directional prediction order — as in standard next-action SFT or single-step state-transition prediction — AOMT repeatedly samples arbitrary masks at the observation/action unit level across training epochs, training a masked diffusion language model (masked DLM) to reconstruct masked units from arbitrary bidirectional context in a single forward pass. We show formally that standard SFT and the offline variant of ALEE's Implicit World Modeling objective are degenerate special cases of AOMT under constrained mask samplers. AOMT is strictly reward-free, requires no environment interaction, and operates on the same offline expert trajectory datasets as standard SFT. Its contribution is entirely on the **training-objective axis** — orthogonal to, and composable with, data-collection paradigms such as ALEE and ETO. Experiments are conducted on ALFWorld, ScienceWorld, and WebShop using the publicly available ETO expert trajectory dataset. We evaluate task success on seen and unseen test splits, and introduce **observation-masked NLL** as a novel proxy metric for world-model quality that is uniquely natural for masked DLMs and unavailable to autoregressive baselines without modification.

---

## 1. Introduction

Supervised fine-tuning (SFT) on expert trajectories remains the dominant approach for bootstrapping capable LLM-based agents (Yao et al., 2022; Deng et al., 2023; Zeng et al., 2023). The standard SFT objective is autoregressive next-action prediction: given the full causal prefix, predict the next action. This imposes a fixed **training order** — a predetermined assignment of trajectory elements to context versus target roles. Every training step asks the same question of the same trajectory: *what comes next?*

This rigidity is a structural limitation, not a necessary one. A trajectory of length $T$ contains $O(2^T)$ possible context/target assignments, of which standard SFT exploits exactly one per action. The supervision signal contained in the relationship between, say, a future observation and a preceding action — the kind of bidirectional reasoning that distinguishes a good plan from a brittle one — is inaccessible to any single-order training method.

Recent methods have explored alternative fixed orders. ALEE (Zhang et al., 2025) trains agents on self-generated rollout data using a single-step next-state prediction objective: given a current state and one action, predict the immediately resulting observation. Learning from Verbal Feedback (Scheurer et al., 2023) appends hindsight feedback to the context and predicts a fixed target action. Both methods are valuable but structurally limited: they each fix one training order and commit in advance to which trajectory elements are context and which are targets.

**Any-Order Masked Training (AOMT)** removes this constraint by treating training-order assignment as a variable. For each training step, an arbitrary subset of observation and action units in an expert trajectory is masked; the unmasked units serve as bidirectional context; and the model reconstructs the masked units in a single forward pass. By independently resampling the mask across training epochs, the same expert trajectory provides supervision for a large family of conditional distributions simultaneously — without additional data collection, environment interaction, or reward signal.

This framing is made practical by masked diffusion language models (masked DLMs). LLaDA (Nie et al., 2025) demonstrates that a masked DLM trained with a random masking ratio $t \sim U[0,1]$ achieves performance competitive with autoregressive LLMs of the same scale, and by construction supports single-pass reconstruction from arbitrary partial context. AOMT extends this capability to the semantic unit level of agent trajectories, masking whole observations and whole actions rather than arbitrary subtoken positions, thereby aligning the training objective with the granularity at which agent decisions are made.

The proposal makes three claims:

1. **AOMT extracts strictly more supervisory signal from fixed expert trajectory data** than any single-order method by varying the context/target assignment across epochs.
2. **Observation-masking within AOMT functions as an implicit world-model objective**, and the quality of the induced world model — measurable via observation-masked NLL — correlates with task success on causally structured environments such as ScienceWorld.
3. **Unit-level masking is the correct granularity** for trajectory-level reconstruction, and produces qualitatively different and more semantically grounded learning signal than token-level masking.

---

## 2. Background and Related Work

### 2.1 Standard SFT: Next-Action Prediction

The standard SFT objective for LLM agents is behavioral cloning (BC): maximum likelihood prediction of expert action tokens conditioned on the full preceding context:

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(s,a)\sim\mathcal{D}_{\text{expert}}} \log \pi_\theta(a \mid s)$$

where $s$ encodes the task instruction and all preceding observations and actions, and $a$ is the expert action. Using interleaved trajectory notation $(O_0, A_0, O_1, A_1, \ldots)$, a standard SFT training example is:

```
Context : O0, A0, O1, A1, O2, A2, O3
Target  : A3
```

The training order is fixed: observations are always context, actions are always targets, information flows strictly left-to-right. The model never sees future observations as context when learning an earlier action.

Standard SFT generalizes poorly to out-of-distribution states because the agent is only ever trained to complete the next step from a valid expert prefix — it has no mechanism to reason about the consequences of its actions or to reason backward from future states (Chu et al., 2025).

### 2.2 ALEE: A Data-Collection Paradigm with a Fixed Objective

Agent Learning via Early Experience (ALEE; Zhang et al., 2025) addresses SFT's generalization limitations by augmenting the training data through active environment interaction. Within its paradigm, ALEE's **Implicit World Modeling (IWM)** strategy uses the following training objective:

$$\mathcal{L}_{\text{IWM}} = -\sum_{(s_i,\, a_i^j,\, s_i^j)\,\in\,\mathcal{D}_{\text{rollout}}} \log p_\theta\!\left(s_i^j \mid s_i,\, a_i^j\right)$$

Each training example is a **single-step state transition**: a current state $s_i$ and one action $a_i^j$ form the context; the immediately resulting next state $s_i^j$ is the target:

```
Context : O_t, A_t
Target  : O_{t+1}
```

Two properties of this objective are critical for positioning AOMT correctly:

**First:** ALEE IWM **never predicts actions**. The objective exclusively targets observations (future states). Action prediction requires a separate standard SFT stage. Policy learning and world-model learning are two separate training phases.

**Second:** ALEE IWM **requires live environment interaction**. The data triples $(s_i, a_i^j, s_i^j)$ are collected by rolling out the partially-trained agent's own policy, not by sampling from a fixed expert dataset. This self-generated data contains states outside the expert distribution — that diversity is ALEE's primary contribution. Without environment access, ALEE IWM cannot be run as intended.

**Positioning relative to AOMT.** ALEE is a data paradigm: its contribution is generating richer training data through active rollouts. AOMT is a training-objective paradigm: its contribution is extracting richer supervision from fixed offline expert data by varying the context/target assignment. These axes are independent and composable. AOMT does not compete with ALEE's data-collection contribution; it addresses a different bottleneck.

For experimental purposes, we construct a **Prefix SFT** baseline by applying the IWM objective to offline expert trajectory data: the same state-action-nextstate structure, but sourced entirely from the expert dataset rather than agent rollouts. This isolates ALEE's objective contribution from its data contribution, enabling a clean comparison with AOMT on equal data footing.

### 2.3 ETO and IPR: Contrastive Methods Requiring Rollouts and Rewards

Exploration-Based Trajectory Optimization (ETO; Song et al., 2024) trains an SFT base agent, collects failure trajectories through environment interaction, and applies DPO-style contrastive learning. ETO achieves +22% over SFT on the out-of-distribution ScienceWorld test set. Iterative Preference Refinement (IPR; Zhang et al., 2024) extends this with step-level reward estimation.

Both methods require environment rollouts and binary reward signals. AOMT requires neither. ETO and IPR are included in results tables as **cross-paradigm reference points** — their numbers establish the performance ceiling achievable with additional resources, contextualizing how much of the SFT-to-ETO gap AOMT can close using only offline expert data. They are not primary experimental opponents.

### 2.4 Masked Diffusion Language Models

LLaDA (Nie et al., 2025) introduces a masked diffusion language model that, during pretraining, independently masks each token with probability $t \sim U[0,1]$ and trains a bidirectional Transformer to reconstruct all masked tokens simultaneously. The training objective is an upper bound on the negative log-likelihood of the model distribution (a variational ELBO), making LLaDA a principled generative model with Fisher consistency — unlike BERT, whose fixed masking ratio prevents it from being used as a generative model. During SFT, only response tokens are subject to masking; the prompt is left as clean context (Nie et al., 2025). This prompt-clean / response-masked structure is the direct architectural precedent for AOMT's context/target distinction.

LLaDA 8B is competitive with LLaMA3 8B on general benchmarks and surpasses GPT-4o on reversal poem completion, demonstrating the practical capability of bidirectional masked DLMs at scale. LLaDA 2.0 (InclusionAI, 2025) scales this paradigm to 100B/16B total parameters using a Mixture-of-Experts architecture, trained with the dFactory post-training framework (built on VeOmni with FSDP2). These serve as our base infrastructure.

**Distinction from token-level masking.** Standard masked DLM training masks at the individual token level with no respect for semantic boundaries. Applying this directly to agent trajectories would allow partial token-level information leakage within a masked unit — e.g., seeing the first word of an action command while being asked to reconstruct the action. AOMT masks at the **observation/action unit level**: entire units are either fully visible as context or fully replaced with mask tokens. This enforces that reconstruction must exploit inter-unit trajectory dependencies rather than intra-unit token completion, and aligns the learning signal with the semantic granularity of agent decisions.

### 2.5 XLNet and Permutation-Based Any-Order Training

XLNet (Yang et al., 2019) trains an autoregressive model over all possible token-level permutation orderings, capturing bidirectional context. AOMT shares the any-order motivation but operates via masked DLM single-pass reconstruction rather than permutation-based sequential factorization. This avoids XLNet's two-stream attention overhead and training-inference discrepancy. More importantly, AOMT applies any-order training at the semantic trajectory-unit level — a qualitatively different granularity that respects the compositional structure of agent trajectories rather than treating them as flat token sequences.

---

## 3. Method

### 3.1 Notation

An agent trajectory of length $T$:

$$\tau = (O_0, A_0, O_1, A_1, \ldots, O_{T-1}, A_{T-1}, O_T)$$

Each $O_t$ is a multi-token text observation; each $A_t$ is a multi-token text action. The set of **trajectory units** is $\mathcal{U}(\tau) = \{O_0, A_0, O_1, \ldots, O_T\}$, with $|\mathcal{U}| = 2T+1$.

A **training order** is a partition $\sigma: \mathcal{U} \to \{\texttt{context}, \texttt{target}\}$. Standard SFT uses a fixed order: $A_t$ is target, all preceding units are context, for each $t$. ALEE IWM applied offline uses a fixed order: $O_{t+1}$ is target, $(O_t, A_t)$ is context, for each $t$. AOMT uses a random order resampled per training step.

### 3.2 Training Objective

For a binary unit-level mask $m \in \{0,1\}^{|\mathcal{U}|}$ where $m_i = 1$ indicates unit $u_i$ is masked, the AOMT loss on trajectory $\tau$ is:

$$\mathcal{L}_{\text{AOMT}}(\tau, m) = -\sum_{\{i\,:\,m_i=1\}} \log p_\theta\!\left(u_i \;\Big|\; \{u_j : m_j = 0\}\right)$$

where $p_\theta$ is the masked DLM with full bidirectional attention over unmasked (context) positions. The loss is summed over all token positions within each masked unit. The full objective over dataset $\mathcal{D}$ is:

$$\mathcal{L}_{\text{AOMT}} = \mathbb{E}_{\tau \sim \mathcal{D}}\;\mathbb{E}_{m \sim \mathrm{Mask}(p)}\;\mathcal{L}_{\text{AOMT}}(\tau, m)$$

where $m \sim \mathrm{Mask}(p)$ draws each $m_i \sim \mathrm{Bernoulli}(p)$ independently. Masks are resampled per training step; across epochs, the same trajectory is presented under many distinct orders.

### 3.3 AOMT Subsumes Prior Methods

**Proposition 1.** AOMT with a mask sampler constrained to always mask exactly the current action $A_t$ — and providing all preceding units as clean context — reduces to standard SFT.

**Proposition 2.** AOMT with a mask sampler constrained to always mask exactly the next observation $O_{t+1}$ — and providing only $(O_t, A_t)$ as clean context — reduces to the offline Prefix SFT objective (ALEE IWM applied to offline data).

Both are strict special cases. AOMT generalizes both by allowing any unit(s) to be masked and providing all remaining units as bidirectional context, including future units.

### 3.4 Why Unit-Level Masking

Token-level masking on an action like `"heat potato with oven"` might leave three of six tokens visible, allowing the model to exploit intra-unit partial text to reconstruct the action rather than exploiting inter-unit trajectory context. Unit-level masking removes this shortcut: the model cannot see any part of a masked unit and must reconstruct it entirely from surrounding trajectory units. This is semantically correct — the capability we wish to induce is inference of complete states and complete actions from trajectory context, not token completion.

Unit-level masking also respects the structured nature of agent observations, which are multi-sentence environment descriptions whose semantic content is not recoverable from partial token access.

### 3.5 Training vs. Inference

**At inference time**, AOMT is operationally identical to standard SFT. The model receives the current observation history and generates the next action via LLaDA's standard iterative unmasking procedure, starting from a fully masked response and progressively unmasking over multiple decoding steps. No observation prediction is performed; no bidirectional context from future steps is available. The world-model learning induced during training shapes the model's representations but does not alter the inference interface.

**At training time**, the mask sampler may designate observations, actions, or both as targets simultaneously. This allows the model to be trained on policy and world-model objectives jointly within one loss, without requiring separate training stages.

---

## 4. Baselines

All baselines use **identical training data** (`agent-eto/eto-sft-trajectory`), **identical base model** (LLaDA 2.0), and **identical compute** (matched gradient steps). No baseline except ETO and IPR (reference only) uses environment interaction or reward signals.

---

### Baseline 1 — Standard SFT

**Objective.** For every action $A_t$ in a trajectory, predict $A_t$ from the full causal prefix $(O_0, A_0, \ldots, O_t)$.

**Concrete example** — ALFWorld task: *"Put a heated potato on the countertop."*

```
Training example 1:
  Context : O0 = "You are in a kitchen. A potato is on the floor."
  Target  : A0 = "pick up potato"

Training example 2:
  Context : O0, A0, O1 = "You pick up the potato. You are holding it."
  Target  : A1 = "go to oven"

Training example 3:
  Context : O0, A0, O1, A1, O2 = "You are in front of the oven."
  Target  : A2 = "heat potato with oven"

Training example 4:
  Context : O0, A0, O1, A1, O2, A2, O3 = "You heated the potato."
  Target  : A3 = "go to countertop"

Training example 5:
  Context : O0...O4 = "You are at the countertop, holding a hot potato."
  Target  : A4 = "put potato on countertop"
```

**What it learns.** Next-action prediction from strictly causal context. Observations are always context; actions are always targets. The model has no access to future observations when learning any past action. This is the degenerate $p=0$ (action-only, causal) limit of AOMT.

**Role.** Primary ablation baseline. Any gain from AOMT-Action-Only over Standard SFT is attributable purely to providing bidirectional context (future observations) when predicting masked actions.

---

### Baseline 2 — Prefix SFT (ALEE IWM Objective, Applied Offline)

**Objective.** For every consecutive step $(O_t, A_t, O_{t+1})$ in a trajectory, predict $O_{t+1}$ from $(O_t, A_t)$. This is ALEE's IWM loss applied to the offline expert corpus rather than agent rollout data.

**Why this construction.** ALEE IWM cannot be run without environment access (it requires self-generated rollout data). By applying the IWM objective to expert data, we strip ALEE down to its training-objective component alone, enabling a clean equal-data comparison with AOMT. This is not a claim that this baseline is equivalent to ALEE — ALEE's data-collection contribution is explicitly absent — but it isolates whether the IWM objective alone (fixed single-step next-state prediction) offers any benefit, and whether AOMT's varied masking outperforms it.

**Implementation.** Stage 1 trains on $\mathcal{L}_{\text{IWM}}$. Stage 2 fine-tunes the Stage 1 checkpoint with $\mathcal{L}_{\text{SFT}}$ to restore action prediction capability. Task success is evaluated after Stage 2.

**Concrete example** (same trajectory):

```
Training example 1:
  Context : O0 = "You are in a kitchen.", A0 = "pick up potato"
  Target  : O1 = "You pick up the potato. You are holding it."

Training example 2:
  Context : O1 = "You pick up the potato.", A1 = "go to oven"
  Target  : O2 = "You are in front of the oven."

Training example 3:
  Context : O2 = "You are in front of the oven.", A2 = "heat potato with oven"
  Target  : O3 = "You heated the potato. You are holding a hot potato."
```

**What it learns.** Single-step state transition prediction (Stage 1 only). Actions are always context; observations are always targets; context is always exactly one preceding state-action pair — future observations never inform predictions; actions are never predicted in Stage 1. **Prefix SFT alone cannot predict actions and is not deployable after Stage 1.** Stage 2 SFT is required.

**Role.** Isolates whether AOMT's gains over Standard SFT come from the any-order variety of masking versus simply adding a world-model pre-training stage. If AOMT-Mixed outperforms Prefix SFT + SFT, the gain is attributable to the unified any-order objective rather than just having two training stages.

---

### Baseline 3 — AOMT-Action-Only

**Objective.** AOMT with the mask sampler restricted to masking only action units. Each action $A_t$ is independently masked with probability $p$; observations are never masked and always serve as context, including future observations.

**Concrete examples** (same trajectory, different epochs show different masks):

```
Epoch 3:
  Context : O0, [MASK:A0], O1, A1, O2, A2, O3, A3, O4
  Target  : A0 = "pick up potato"
  ← A0 is predicted with O1...O4 as context — the model knows what happened next

Epoch 7:
  Context : O0, A0, O1, [MASK:A1], O2, A2, O3, A3, O4
  Target  : A1 = "go to oven"
  ← A1 is predicted knowing O2 = "You are in front of the oven" already happened

Epoch 12:
  Context : O0, [MASK:A0], O1, A1, O2, [MASK:A2], O3, A3, O4
  Target  : A0, A2 (two actions masked simultaneously)
  ← Multiple actions predicted jointly from full surrounding observation context
```

**What it learns.** Action prediction with bidirectional context — the model can exploit future observations when predicting masked actions. Observations are never targets; no world-model training occurs. This is the "bidirectional policy" ablation.

**Role.** Isolates the contribution of bidirectional context for policy learning, independent of world-model training. The comparison Standard SFT → AOMT-Action-Only answers: *does providing future observations as context improve action prediction?* The comparison AOMT-Action-Only → AOMT-Mixed answers: *does additionally training on observation reconstruction improve policy performance?*

---

### Baseline 4 — AOMT-Mixed (Full Proposed Method)

**Objective.** AOMT with an unconstrained mask sampler. Each unit — whether observation or action — is independently masked with probability $p$ per training step.

**Concrete examples** (same trajectory, three epochs):

```
Epoch 1:
  Context : O0, [MASK:A0], O1, A1, O2, A2, O3, A3, O4
  Target  : A0 = "pick up potato"          [policy learning, bidirectional context]

Epoch 5:
  Context : O0, A0, [MASK:O1], A1, O2, A2, O3, A3, O4
  Target  : O1 = "You pick up the potato."  [world-model learning]

Epoch 9:
  Context : O0, [MASK:A0], [MASK:O1], A1, O2, A2, O3, A3, O4
  Target  : A0, O1                          [joint policy + world-model]

Epoch 14:
  Context : O0, A0, O1, A1, [MASK:O2], A2, [MASK:O3], A3, O4
  Target  : O2, O3                          [multi-step world-model reconstruction]
```

**What it learns.** A unified model trained as both a policy (action prediction from any context) and an implicit world model (observation reconstruction from any context), within a single training objective. The model never needs to predict observations at inference, but the world-model training shapes its representations to better understand the causal structure of the environment.

**Role.** The full proposed method. The hypothesis is that joint any-order training of policy and world model outperforms all single-order methods on task success (especially on out-of-distribution test sets) and produces significantly lower observation-masked NLL, particularly on ScienceWorld where causal understanding is required.

---

### Reference Points: ETO and IPR

ETO (Song et al., 2024) and IPR (Zhang et al., 2024) are included in the results table as published reference numbers. Both require environment rollouts and reward signals unavailable to AOMT. This asymmetry is explicitly stated in every table and results discussion. Their numbers establish the performance level achievable with additional resources; they quantify the gap that offline-only methods cannot be expected to close, and they provide a ceiling for interpreting AOMT's improvements over Standard SFT.

---

## 5. Evaluation

### 5.1 Benchmark Suite

**ALFWorld** (Shridhar et al., 2021). Six household task types (pick-and-place, heat, cool, clean, examine, pick-two) requiring multi-step navigation and object manipulation in text-based environments. Seen and unseen task splits; binary success/failure. Expert trajectories are from the ETO dataset (`agent-eto/eto-sft-trajectory`), augmented with GPT-4-generated rationales. Direct comparison with published SFT, ETO, IPR, and WKM numbers.

**ScienceWorld** (Wang et al., 2022). Thirty science experiment task types requiring agents to execute multi-step lab procedures (measuring temperature, growing plants, identifying conductors, etc.) across sequences of 10–120 steps. Average normalized score on [0,1]. The key benchmark for world-model quality: tasks require understanding causal rules (conductors carry electricity; heating changes state of matter) that are exactly the knowledge observation-masking training is designed to induce. ETO achieves +22% over SFT on the unseen split — the most significant generalization gap in the standard suite, and AOMT's primary target.

**WebShop** (Yao et al., 2022). Sequential product search and selection against natural language instructions across 12,087 products. Continuous reward on [0,1] measuring instruction-product alignment. Tests whether AOMT's gains generalize across interaction modalities (web navigation vs. embodied manipulation).

These three benchmarks are the standard evaluation suite for LLM agent SFT methods (ETO, IPR, WKM, ALEE). They have established expert trajectory datasets, published competitive baselines, and together span the major modalities of text-based agent interaction.

### 5.2 Primary Metric: Task Success and Average Reward

For all benchmarks, we report the metric used in prior work: binary success rate for ALFWorld (seen/unseen), average normalized score for ScienceWorld (seen/unseen), and average reward for WebShop. Seen-vs-unseen split performance is the primary signal for generalization quality.

### 5.3 World-Model Quality: Observation-Masked NLL

We introduce **observation-masked NLL** as a novel auxiliary metric measuring the quality of the implicit world model induced by AOMT training:

$$\text{NLL}_{\text{obs}}(\theta) = -\mathbb{E}_{\tau \sim \mathcal{D}_{\text{test}}}\;\mathbb{E}_t\!\left[\log p_\theta(O_t \mid \tau \setminus O_t)\right]$$

This metric asks: given all other observations and actions in a trajectory, how well can the model predict a held-out observation? It directly quantifies the model's understanding of environment dynamics. Standard SFT models have no natural mechanism for conditioning on future context, making this metric structurally unavailable to them without modification. Only masked DLM models with bidirectional attention — specifically AOMT-Action-Only and AOMT-Mixed — can be evaluated on this metric without architectural change.

We compute NLL-obs for AOMT-Action-Only and AOMT-Mixed at every training checkpoint and report it alongside task success. The analysis question is: does NLL-obs on the test set correlate with task success, and is the correlation stronger on ScienceWorld (where causal understanding matters most) than on WebShop (where it matters less)?

**Why this metric matters for the paper.** It provides a result that purely task-success comparisons cannot: a direct measure of what AOMT's observation-masking objective is actually learning. If NLL-obs is significantly lower for AOMT-Mixed than AOMT-Action-Only while task success on ScienceWorld also improves, this constitutes strong evidence for the world-model hypothesis. If NLL-obs is lower but task success does not improve, this reveals a generalization gap worth analyzing. Either result is scientifically interesting and publishable.

### 5.4 Robustness Under Observation Noise

As a targeted evaluation of AOMT's partial-observability hypothesis, we run all models on ALFWorld and ScienceWorld with a fixed fraction $\rho \in \{0.1, 0.2, 0.3\}$ of observation tokens randomly replaced with noise tokens at test time. Standard SFT has no mechanism to handle corrupted observations; AOMT-Mixed has been explicitly trained to reconstruct observations from surrounding context. The hypothesis is that AOMT-Mixed degrades more gracefully under increasing $\rho$ than Standard SFT or AOMT-Action-Only.

### 5.5 Ablation Study

The ablation study is organized around four clean scientific questions:

| Question | Comparison |
|---|---|
| Does bidirectional context improve action prediction? | Standard SFT → AOMT-Action-Only |
| Does adding world-model training improve policy? | AOMT-Action-Only → AOMT-Mixed |
| Does AOMT's varied masking outperform fixed single-step IWM? | Prefix SFT → AOMT-Mixed |
| What mask probability $p$ optimizes the tradeoff? | $p \in \{0.15, 0.25, 0.40, 0.50\}$ on ALFWorld |

Each question maps to a clean pair of experimental conditions, enabling crisp attribution of performance differences to specific design choices.

---

## 6. Implementation

### 6.1 Base Model and Framework

We use **LLaDA 2.0-mini** (`inclusionAI/LLaDA2.0-mini`, 16B total / 1.4B active parameters, MoE architecture) as the base model for all experiments. LLaDA 2.0-mini is practical for research-scale experiments while sharing the architectural properties of the full 100B flash model. The training framework is **dFactory**, fine-tuned with FSDP2 parallelism.

The key property that makes LLaDA 2.0-mini the correct base model for AOMT is its native SFT structure: during fine-tuning, prompt tokens are kept as clean unmasked context and only response tokens are masked (Nie et al., 2025). AOMT extends this by making the prompt/response boundary dynamic — the "prompt" at any training step consists of the unmasked trajectory units, and the "response" consists of the masked units. This maps directly onto dFactory's existing SFT masking code path with a modified mask constructor.

### 6.2 Data

All experiments use the publicly available `agent-eto/eto-sft-trajectory` dataset from Hugging Face, which contains expert trajectories in ReAct format for ALFWorld, ScienceWorld, and WebShop with GPT-4-generated rationales. Trajectory parsing, unit boundary extraction, and dataset preprocessing are described in the engineering specification document.

### 6.3 Timeline

| Phase | Weeks | Deliverables |
|---|---|---|
| **Infrastructure** | 1–2 | Unit-level mask sampler; dFactory integration; sanity-check on toy trajectories; Prefix SFT two-stage pipeline confirmed working |
| **Ablations** | 3–6 | Full $p$-sweep on ALFWorld; action-only vs. mixed; Standard SFT vs. AOMT-Action-Only; NLL-obs metric implemented and tracked |
| **Full Benchmarks** | 7–12 | All three benchmarks, seen + unseen splits, noise-robustness experiments; Prefix SFT results; ETO/IPR reference numbers from published papers |
| **Analysis and Writing** | 13–16 | NLL-obs vs. task success correlation; attention pattern analysis; failure mode analysis; paper draft |

---

## 7. Expected Contributions

1. **A novel training-objective paradigm for offline SFT on expert agent trajectories** that extracts more supervision from fixed data without environment interaction, reward signals, or data augmentation.

2. **Formal unification of Standard SFT and ALEE's offline IWM objective as constrained special cases of AOMT**, with a clear decomposition of the data axis vs. objective axis across the agent training literature.

3. **Unit-level masking as a principled semantic design choice** for trajectory-level masked DLM training, with formal justification differentiating it from token-level masking.

4. **Observation-masked NLL as a novel evaluation metric** for implicit world-model quality in text-based interactive agents — the first metric that directly measures this capability from within the training paradigm rather than through downstream task proxies.

5. **Empirical validation on ALFWorld, ScienceWorld, and WebShop** with complete ablation study, noise-robustness analysis, and cross-paradigm contextualization against ETO and IPR.

---

## 8. Limitations and Future Work

AOMT's contribution is on the objective axis only. The natural extension is composing it with ALEE's data-collection paradigm: applying AOMT's any-order objective to ALEE-generated rollout data rather than fixed expert data. This is likely strictly better than either method alone and is the most immediate future direction.

AOMT is evaluated on text-based trajectories. Extending to visual observations requires adapting unit-level masking to image patch or frame-level units, building on multimodal masked DLMs such as LLaDA-V (2025).

The theoretical relationship between AOMT's objective and LLaDA's likelihood lower bound at the unit level requires formal derivation. We conjecture that AOMT inherits Fisher consistency from LLaDA but this must be proven carefully given the non-standard masking granularity.

---

## References

- Bie, T. et al. (2025). LLaDA 2.0: Scaling up diffusion language models to 100B. *arXiv:2512.15745*. InclusionAI, Ant Group.
- Chu, T. et al. (2025). SFT memorizes, RL generalizes. *arXiv:2501.17161*.
- Deng, X. et al. (2023). Mind2Web: Towards a generalist agent for the web. *NeurIPS 2023*.
- Nie, S. et al. (2025). Large language diffusion models (LLaDA). *arXiv:2502.09992*.
- Scheurer, J. et al. (2023). Language models can learn from verbal feedback without scalar rewards. *arXiv:2305.15766*.
- Shridhar, M. et al. (2021). ALFWorld: Aligning text and embodied environments for interactive learning. *ICLR 2021*.
- Song, Y. et al. (2024). Trial and error: Exploration-based trajectory optimization for LLM agents. *ACL 2024*. Dataset: `agent-eto/eto-sft-trajectory`.
- Wang, R. et al. (2022). ScienceWorld: Is your agent smarter than a 5th grader? *EMNLP 2022*.
- Yang, Z. et al. (2019). XLNet: Generalized autoregressive pretraining for language understanding. *NeurIPS 2019*.
- Yao, S. et al. (2022). WebShop: Towards scalable real-world web interaction with grounded language agents. *NeurIPS 2022*.
- Yao, S. et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*.
- Zeng, A. et al. (2023). AgentTuning: Enabling generalized agent abilities for LLMs. *arXiv:2310.12823*.
- Zhang, J. et al. (2024). Watch every step! LLM agent learning via iterative step-level process refinement (IPR). *EMNLP 2024*.
- Zhang, K. et al. (2025). Agent learning via early experience (ALEE). *arXiv:2510.08558*.