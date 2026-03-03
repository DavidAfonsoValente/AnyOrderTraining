# PhD Research Project Specification

## Any-Order Masked Training for Trajectory-Level Learning in LLM-Based Agents

| **Model** | **Framework** | **Training Setup** |
|---|---|---|
| LLaDA 2.0 8B · Qwen-2.5-7B | dFactory | Expert-only SFT · No rewards |

**Version 13.0** — Primary comparison: Qwen-2.5-7B+IL vs Qwen-2.5-7B+IWM vs LLaDA-2.0-8B+AOMT

---

## 1. Research Question and Motivation

Zhang et al. (2025) demonstrated that the supervision structure applied to expert trajectory data during fine-tuning is a consequential design choice for autoregressive LLMs: Implicit World Modelling (IWM) and Self-Reflection consistently outperform standard behaviour cloning. Their work is framed around what data to include and how to structure it. This project asks a different question: what happens when the model itself is a masked diffusion LM, where the notion of a training objective is fundamentally different?

> **Central Research Question**
>
> Does Any-Order Masked Training (AOMT) — fine-tuning a masked DLM by independently resampling which trajectory units are masked on every training step — produce a better agent than:
>
> (a) Qwen-2.5-7B fine-tuned with standard causal IL / behaviour cloning, and
>
> (b) Qwen-2.5-7B fine-tuned with IWM (the best method from Zhang et al.)?
>
> Measured on task success rate in ALFWorld — a fair, model-agnostic downstream metric.

This is a meaningful question because masked DLMs have a structural property autoregressive models lack: during inference they can condition on any subset of context tokens bidirectionally. If AOMT training teaches the model to exploit this property for trajectory understanding, it may outperform autoregressive methods even on their home turf of sequential agent tasks.

**Why not just use LLaDA for all baselines?** A previous version of this proposal defined B-SingleUnit (mask one action unit per step) and B-Prefix (mask all future units from a split point) as masked-DLM analogs of IL and IWM, and compared all three on LLaDA. The problem: those are masked-DLM approximations, not the real methods. Comparing AOMT to B-Prefix-on-LLaDA does not answer whether AOMT beats IWM — it only tells you which masking strategy works best within LLaDA. That is a useful ablation, but it is not the primary question. The primary comparison must use the actual IL and IWM methods on a causal model. Because LLaDA 2.0 is initialised from Qwen-2.5-7B before continued masked-DLM pre-training, Qwen-2.5-7B is the most principled causal baseline available — same architecture family, same pre-training corpus.

> **The Architecture Confound and Why It Is Manageable**
>
> Qwen-2.5-7B+IWM vs LLaDA-8B+AOMT differs in two ways: (1) LLaDA has undergone continued masked-DLM pre-training after Qwen initialisation, and (2) the fine-tuning objective differs. Factor (1) cannot be controlled away without retraining LLaDA from scratch with a causal objective, which is out of scope.
>
> This is acknowledged explicitly as a confound. It is mitigated by three properties: (a) LLaDA's base architecture and pre-training data are identical to Qwen-2.5's; (b) the within-LLaDA ablation (Phase 2) cleanly isolates factor (2) by holding factor (1) constant; (c) the primary comparison is on task success rate — a model-agnostic metric — not on NLL, which is incomparable across objectives.

---

## 2. Method: Any-Order Masked Training

### 2.1 Trajectory Representation

Trajectories are represented as alternating unit sequences:

```
τ = (O₀, A₀, O₁, A₁, ..., Oₙ, Aₙ)
```

where each observation `Oₜ` and action `Aₜ` is a complete token span. Unit boundaries are stored alongside `input_ids`; all tokens in a unit are masked simultaneously, ensuring each prediction task requires genuine inter-unit reasoning rather than intra-unit pattern completion.

### 2.2 Training Objective

All LLaDA conditions share one loss:

```
L(θ, τ, M) = −(1/|M|) · Σᵢ∈M Σₜ∈tokens(Xⁱ) log pθ(xₜ | {Xʲ : j∉M})
```

This is identical to LLaDA 2.0's pre-training objective. The only experimental variable is how `M` is sampled. Loss is normalised by number of masked units (treating each unit as equally important); evaluation scripts normalise by token count for comparability. When the Bernoulli draw produces an empty mask, the sampler redraws until `|M| ≥ 1`.

### 2.3 The AnyOrderSampler

All three LLaDA conditions are modes of one sampler class; experiments differ only by YAML config:

```python
class AnyOrderSampler:
    # mode: 'b_singleunit' | 'b_prefix' | 'any_order'

    def sample_mask(self, unit_indices, action_indices):
        if mode == 'b_singleunit':
            return [random.choice(action_indices)]
        elif mode == 'b_prefix':
            k = random.randint(1, len(unit_indices)-1)
            return unit_indices[k:]
        elif mode == 'any_order':
            return [i for i in unit_indices if random.random() < mask_prob]
```

The mask is resampled inside `__getitem__` — every training step sees a fresh mask. No new data is generated between epochs.

---

## 3. Experimental Conditions

The study has two tiers. Phase 2 is a within-LLaDA ablation that isolates mask sampling strategy. Phase 3 is the primary cross-paradigm comparison that answers the main research question.

### 3.1 Primary Comparison Conditions (Phase 3)

| **ID** | **Model** | **Training Method** | **Compares Against** | **Metrics** |
|---|---|---|---|---|
| Qwen+IL | Qwen-2.5-7B | Causal IL / behaviour cloning on expert trajectories. Standard left-to-right fine-tuning. | Primary baseline (C1) | Task SR, ROUGE-L |
| Qwen+IWM | Qwen-2.5-7B | Implicit World Modelling (Zhang et al., 2025): expert data + rollout data structured as future-from-past prediction. | Primary baseline (C2) | Task SR, ROUGE-L |
| LLaDA+AOMT | LLaDA 2.0 8B | Any-Order Masked Training: Bernoulli(0.5) independent unit masking, resampled every step. Expert data only. | Proposed method | Task SR, ROUGE-L, NLL (within-family only) |
| LLaDA+B-SU | LLaDA 2.0 8B | B-SingleUnit: mask one action unit per step, all others visible. Masked-DLM analog of IL. | Within-LLaDA ablation | Task SR, NLL |
| LLaDA+B-Pfx | LLaDA 2.0 8B | B-Prefix: mask all units from split k onward; bidirectional over prefix. Masked-DLM analog of IWM structure. | Within-LLaDA ablation | Task SR, NLL |

**Note on data:** Qwen+IWM requires rollout data in addition to expert trajectories, as per Zhang et al. LLaDA+AOMT, LLaDA+B-SU, and LLaDA+B-Pfx use expert trajectories only. Qwen+IL also uses expert trajectories only. This means Qwen+IWM has a training data advantage that is acknowledged as a limitation — it replicates the original Zhang et al. setup faithfully rather than artificially handicapping it.

### 3.2 Qwen+IWM Replication Protocol

Qwen+IWM follows Zhang et al. (2025) exactly: (1) fine-tune Qwen-2.5-7B on the public ALFWorld expert corpus with causal IL to produce an initial agent; (2) run this agent in the ALFWorld simulator to collect rollout trajectories (including failed and partial episodes); (3) reformat rollouts as future-from-past prediction examples combined with the original expert data; (4) fine-tune Qwen-2.5-7B on the combined corpus with the IWM structured objective. We use the hyperparameters, rollout budget, and train/test split from their paper. Any deviation required by environment constraints will be documented.

Infrastructure note: Qwen and LLaDA use separate training stacks — Qwen uses standard causal fine-tuning; LLaDA uses dFactory with the AnyOrderSampler. Rollout collection for Qwen+IWM requires live ALFWorld simulator access during training. These pipelines run sequentially, not concurrently, to manage GPU resources.

### 3.3 Mask Sampling Ablation Conditions (Phase 2 only)

| **ID** | **mask_prob** | **mask_strategy** | **Tests** |
|---|---|---|---|
| B-SingleUnit | — | actions only | Fixed-pattern baseline; masked-DLM IL analog |
| B-Prefix | — | prefix (future) | Fixed-pattern baseline; masked-DLM IWM analog |
| AO-p15 | 0.15 | mixed | Sparse any-order; lowest p in sweep |
| AO-p30 | 0.30 | mixed | Moderate any-order |
| AO-p50 | 0.50 | mixed | Core AOMT condition (≈ LLaDA pre-training rate) |
| AO-p75 | 0.75 | mixed | Dense any-order; context sparsity risk |
| AO-ObsOnly | 0.50 | observations only | Attribution: world model component |
| AO-ActOnly | 0.50 | actions only | Attribution: policy component |

---

## 4. Evaluation

### 4.1 Metrics and Their Validity

| **Metric** | **Definition** | **Valid Across** | **Used For** |
|---|---|---|---|
| Task Success Rate (SR) | Episode pass/fail via ALFWorld simulator. Standard metric in agent literature. | All five conditions — model-agnostic | Primary: C1 (vs Qwen+IL), C2 (vs Qwen+IWM), C3 (vs LLaDA+B-SU) |
| ROUGE-L / Exact Match | String-level Oₜ₊₁ reconstruction accuracy, past-only context, 50 held-out trajectories. | All conditions — string match does not involve log-probabilities | Secondary: world model quality demonstration |
| NLL_total (LOO) | −log pθ(Xⁱ \| all other units). Leave-one-unit-out, bidirectional. | Within LLaDA ONLY — different objectives make cross-model NLL invalid | Within-LLaDA ablation (Phase 2 + LLaDA conditions in Phase 3) |
| NLL_obs_forward | −log pθ(Oₜ₊₁ \| O₀,A₀,...,Oₜ,Aₜ). Past-only context. Causal world model quality. | Within LLaDA ONLY | Within-LLaDA ablation; AOMT-Best selection criterion |
| NLL_act_forward | −log pθ(Aₜ \| O₀,A₀,...,Oₜ). Past-only context. Causal policy quality. | Within LLaDA ONLY | Within-LLaDA ablation; continuous complement to task SR |

> **Why NLL Cannot Cross the Architecture Boundary**
>
> Autoregressive NLL: `−Σ log p(xₜ | x₀...xₜ₋₁)`. Causal prefix, teacher-forced, computed in one pass.
>
> Masked DLM marginal NLL: `−log pθ(Xⁱ | all other units)`. Bidirectional conditioning; not the same quantity.
>
> These measure different mathematical objects. Subtracting one from the other is meaningless. Qwen NLL values are never reported in this project. Task success rate and string-match metrics (ROUGE-L / EM) are the only valid cross-model comparisons.

### 4.2 Causal Evaluation Suite (LLaDA only)

`NLL_obs_forward` and `NLL_act_forward` together form a complete causal evaluation under past-only context — identical to rollout inference conditions. Together they test whether the model knows what will happen next (world model) and what to do next (policy), on the same numerical scale.

**Formulae:**

```
NLL_obs_fwd = −(1/|T|) · Σ_τ (1/n_fwd(τ)) · Σₜ log pθ(Oₜ₊₁ | O₀,A₀,...,Oₜ,Aₜ)
              [n_fwd = number of consecutive triples]

NLL_act_fwd = −(1/|T|) · Σ_τ (1/n_act(τ)) · Σₜ log pθ(Aₜ | O₀,A₀,...,Oₜ)
              [n_act = number of action units]
```

### 4.3 Rollout Protocol

At each step t, all models receive `(O₀, A₀, ..., Oₜ)` and must predict `Aₜ`. Future observations are absent for all models — a level playing field. LLaDA models mask only the current action unit. **100 episodes per seed × 3 seeds per condition = 300 episodes per condition.**

### 4.4 Trajectory Completion Demonstration (LLaDA only)

50 held-out ALFWorld trajectories. For each, `Oₜ₊₁` is masked and the three LLaDA conditions (LLaDA+B-SU, LLaDA+B-Pfx, LLaDA+AOMT) reconstruct it via masked inference under past-only context. ROUGE-L and exact match measure which masking strategy produces the best causal world model predictions within the masked DLM family.

Qwen conditions are excluded. Autoregressive models generate `Oₜ₊₁` by continuing the token sequence; masked DLMs infill a masked span of fixed length. ROUGE-L scores are confounded by generation verbosity and stopping criteria, which differ systematically between AR and masked generation. The valid cross-model comparison is task success rate; trajectory completion is an explanatory tool for the within-LLaDA comparison only.

---

## 5. Phase 2 — Mask Sampling Ablation in MiniGrid

### 5.1 Purpose

Phase 2 cleanly isolates the effect of mask sampling strategy by holding the model (LLaDA 2.0 8B) and data constant. It answers: within the masked DLM paradigm, which masking strategy best develops world model quality and policy quality? It also selects AOMT-Best (the `p` and `mask_strategy` configuration) to carry forward to Phase 3.

### 5.2 Environments

| **Environment** | **Description** | **Inter-Unit Dependencies** | **Expected AOMT Advantage** |
|---|---|---|---|
| GoToDoor-v0 | Navigation only, ~15 steps | Minimal | Small or zero — short dependencies offer little gain from any-order coverage |
| PickupDist-v0 | Navigate + one object interaction | Moderate | Moderate — object pickup creates observation–action dependencies |
| Unlock-v0 | Find key → pick up → navigate → unlock | Strong, long-range | Most pronounced — sequential dependencies reward training on diverse conditional subsets |

### 5.3 Runs and Falsifiable Predictions

**8 conditions × 3 environments × 3 seeds = 72 runs.** Fixed 80/20 train/test split across all. 1,000 expert bot trajectories per environment.

> **Falsifiable Predictions (Phase 2)**
>
> **P1 — Dependency scaling:** The NLL_total gap (B-SingleUnit − AO-p50) increases monotonically GoToDoor → PickupDist → Unlock. Failure indicates AOMT's advantage is not driven by inter-unit dependency coverage.
>
> **P2 — Causal world model:** NLL_obs_forward gap (B-Prefix − AO-p50) is positive across all three environments, largest on Unlock. B-Prefix is trained specifically on forward prediction; AOMT beating it on NLL_obs_forward despite this is the non-trivial result.
>
> **P3 — B-SingleUnit double mismatch:** B-SingleUnit has the worst NLL_obs_forward across all environments. At evaluation it faces wrong unit type (trained on actions only) AND wrong context (full bidirectional at train, past-only at eval). B-Prefix faces only the context mismatch.
>
> **P4 — Attribution dissociation:** AO-ObsOnly outperforms AO-ActOnly on NLL_obs_forward; AO-ActOnly outperforms AO-ObsOnly on NLL_act_forward. A clean double dissociation confirms that observation masking drives world model gains and action masking drives policy gains.

### 5.4 AOMT-Best Selection

AOMT-Best is the any-order condition with the lowest `NLL_obs_forward` on the Unlock-v0 validation set, as this is the causal world model quality criterion central to the project. Near-ties (best condition mean minus second-best mean < one standard deviation of the best condition across seeds) are broken by `NLL_total`. The selected configuration is carried forward to Phase 3 unchanged.

### 5.5 Compute Fairness

Training is equalised by epochs. AO-p50 produces approximately `p × |Uτ|` masked-unit gradient contributions per step (where `|Uτ|` is the number of units in trajectory `τ`) versus exactly one for B-SingleUnit. Robustness check: B-SingleUnit is additionally trained for `p × |Uτ|` additional epochs to match AO-p50's total gradient contribution count. If B-SingleUnit still underperforms, the AOMT advantage is robust to this asymmetry.

---

## 6. Phase 3 — Primary Evaluation in ALFWorld

### 6.1 Setup

All five conditions (Qwen+IL, Qwen+IWM, LLaDA+B-SU, LLaDA+B-Pfx, LLaDA+AOMT-Best) are evaluated on the ALFWorld benchmark. Expert corpus: **3,553 trajectories / 21,031 state-action pairs** (public dataset). Qwen+IWM additionally uses rollout data per the original Zhang et al. protocol. **3 seeds per condition = 15 training runs.**

### 6.2 The Three Primary Claims

> **C1 — AOMT vs Causal IL**
>
> If LLaDA+AOMT achieves higher task success than Qwen+IL: masked DLM training with any-order masking matches or exceeds the standard supervised learning baseline despite: (a) operating on a different architecture/objective family, and (b) using no rollout data.
>
> This establishes that AOMT is at minimum competitive with the simplest causal alternative.

> **C2 — AOMT vs IWM (the central claim)**
>
> If LLaDA+AOMT achieves competitive or higher task success than Qwen+IWM: any-order masking on expert-only data matches or beats the best method from Zhang et al., which requires rollout data collection and a structured world-model training objective.
>
> This is the strongest result: AOMT achieves IWM-quality world modelling from simpler training. If AOMT loses to IWM, the gap is quantified and the result is still informative — masked DLM training with AOMT is a meaningful new baseline for the field.
>
> Trajectory completion ROUGE-L provides mechanistic evidence: does AOMT produce qualitatively better future-state predictions than both Qwen models under the same past-only context?

> **C3 — Within-LLaDA masking strategy**
>
> Does LLaDA+AOMT outperform LLaDA+B-SU and LLaDA+B-Pfx on both task success and NLL metrics?
>
> C3 is the ablation that supports C1 and C2. If AOMT wins in the cross-paradigm comparison, C3 confirms the gain comes from any-order masking specifically — not from simply using a masked DLM. LLaDA+B-SU represents using LLaDA with the most naive fine-tuning strategy; if it were already competitive with Qwen+IWM, the credit would belong to LLaDA as an architecture, not to AOMT. C3 rules that out. It also validates that Phase 2 NLL findings (on short symbolic MiniGrid sequences) scale to long natural language ALFWorld trajectories.

### 6.3 Expected Results Table

| **Condition** | **Task SR** | **NLL_obs_fwd** | **Key Interpretation** |
|---|---|---|---|
| Qwen+IL | ~78% (Zhang) | N/A (invalid cross-model) | Causal BC floor |
| Qwen+IWM | ~83% (Zhang) | N/A | Best published causal baseline |
| LLaDA+B-SU | TBD | Weakest (double mismatch) | Masked-DLM IL analog; should lag Qwen+IL |
| LLaDA+B-Pfx | TBD | Intermediate | Masked-DLM IWM analog; should approach Qwen+IWM |
| LLaDA+AOMT | TBD — target ≥ Qwen+IWM | Best within LLaDA | Main result: any-order training outperforms focused objectives |

Qwen SR figures are from Zhang et al. (2025) and represent the landscape against which AOMT is evaluated. They are not controlled comparisons; the architecture confound is explicitly acknowledged.

---

## 7. Statistical Testing

All metrics reported as **mean ± SD** over 3 seeds. Effect sizes reported as Cohen's d for NLL and absolute percentage points for task success rate.

**Phase 2** (within-LLaDA ablation): Bonferroni correction over 12 primary comparisons (AO-p50 vs B-SingleUnit + AO-p50 vs B-Prefix, on NLL_total and NLL_obs_forward, across 3 environments; α/12 ≈ 0.004). NLL_act_forward and LOO decompositions are descriptive secondary metrics not included in the correction.

**Phase 3** (primary cross-paradigm comparison): C2 (LLaDA+AOMT vs Qwen+IWM on task success rate) is pre-specified as the single primary hypothesis test. This pre-specification prevents inflation from testing C1, C2, and C3 simultaneously. C1 (vs Qwen+IL) and C3 (within-LLaDA) are secondary and interpreted descriptively. Phase 3 uses bootstrap CIs (1,000 resamples) alongside paired t-tests for all comparisons. Three seeds is a pragmatic constraint; bootstrap CIs mitigate this.

**Pre-registration:** The primary hypothesis (C2), AOMT-Best selection criterion (best NLL_obs_forward on Unlock-v0), and statistical thresholds are fixed prior to running Phase 3. Any post-hoc analysis is labelled as exploratory.

---

## 8. Data

LLaDA+AOMT, LLaDA+B-SU, LLaDA+B-Pfx, and Qwen+IL all train on the same expert-only ALFWorld corpus (Phase 3) and expert MiniGrid bot trajectories (Phase 2). No rollout data, hindsight relabelling, or augmentation is used for these four conditions. Qwen+IWM additionally uses rollout data as per Zhang et al. — this replicates the original setup faithfully. The data difference is explicit and acknowledged as a limitation; it is not hidden or minimised.

AOMT's contribution is in how existing trajectories are consumed — not what data was collected. If AOMT matches IWM on expert-only data while IWM requires rollout collection, that is itself a significant practical finding.

---

## 9. Codebase

| **Component** | **File** | **Responsibility** |
|---|---|---|
| AnyOrderSampler | `sampler/any_order_sampler.py` | All three LLaDA modes. Mask resampled every `__getitem__` call. |
| DataTransform | `transform/trajectory_transform.py` | Tokenise trajectory JSON; track unit boundaries; apply `[MASK]` at unit level. |
| Train task | `tasks/train_any_order.py` | dFactory training loop. All LLaDA conditions via YAML config only. |
| Marginal NLL evaluator | `scripts/evaluate_marginal_nll.py` | Leave-one-unit-out NLL. Produces NLL_obs, NLL_act, NLL_total. Token-normalised. |
| Forward NLL evaluator | `scripts/evaluate_forward_nll.py` | NLL_obs_forward and NLL_act_forward. Past-only context. Correct denominators n_fwd(τ), n_act(τ). |
| Rollout runner | `scripts/run_alfworld_rollout.py` | 100 episodes × 3 seeds. All five conditions. Past-only context. |
| Trajectory completion demo | `scripts/trajectory_completion_demo.py` | ROUGE-L and EM on 50 held-out trajectories. All five conditions. Oₜ₊₁ masked, past-only. |

---

## 10. Timeline (14 Weeks)

| **Phase** | **Weeks** | **Tasks** | **Success Gate** |
|---|---|---|---|
| 1 — Setup | 1–2 | Implement AnyOrderSampler (3 modes), DataTransform, both NLL evaluators, rollout runner. Sanity check on GoToDoor-v0: all three LLaDA modes train; NLL_obs_forward < uninformed baseline; B-SingleUnit NLL_obs_forward > B-Prefix (P3 sanity). | All modes train. Sanity checks pass. |
| 2A — MiniGrid runs | 3–5 | 72 LLaDA ablation runs (8 conditions × 3 envs × 3 seeds). Collect NLL_obs, NLL_act, NLL_total, NLL_obs_forward, NLL_act_forward. | All 72 runs complete. AO-p50 < B-SingleUnit on NLL_total for Unlock across all seeds. |
| 2B — Ablation analysis | 6 | Compute-matched robustness check (B-SU trained for p×\|Uτ\| additional epochs). Select AOMT-Best by NLL_obs_forward on Unlock-v0 validation set. Write Phase 2 results section. | AOMT-Best selected. P1–P4 documented (confirmed or analysed). |
| 3A — ALFWorld training | 7–11 | Probe stage (3 runs): run LLaDA+B-SU, LLaDA+B-Pfx, LLaDA+AOMT-Best at 1 seed each on ALFWorld to validate that AOMT-Best selection from MiniGrid transfers before committing full compute. If AOMT-Best does not beat B-Pfx on NLL_obs_forward in probe, run 3-condition p-sweep (p30/p50/p75, 1 seed each) and reselect before proceeding. Full stage (15 runs): 3 seeds × 5 conditions. Qwen+IWM requires live ALFWorld rollout collection per Zhang et al. Pipelines run sequentially. | Probe validates AOMT-Best. All 15 full runs complete. Qwen+IL reproduces ~78% (sanity check). |
| 3B — ALFWorld evaluation | 12 | 100-episode rollouts for all 5 conditions. LOO NLL and causal NLL for 3 LLaDA conditions. Trajectory completion demo (all 5). | C1/C2/C3 results in hand. |
| 3C — Write-up | 13–14 | Full manuscript: intro, related work, method, Phase 2 ablations, Phase 3 primary results, limitations, discussion. | Submission-ready manuscript. |

---

## 11. Expected Contributions

**1.** First direct comparison of masked DLM agent fine-tuning (AOMT) against the best published causal agent training methods (IL and IWM from Zhang et al.) using model-agnostic downstream metrics. Establishes whether the masked DLM paradigm is competitive with causal LLMs on sequential agent tasks.

**2.** Any-Order Masked Training as a principled fine-tuning method for masked DLMs on trajectory data. Demonstrated to outperform fixed-pattern masked DLM alternatives (B-SingleUnit, B-Prefix) on both bidirectional and causal evaluation metrics.

**3.** A complete causal evaluation suite for masked DLM trajectory models: NLL_obs_forward (world model quality) + NLL_act_forward (policy quality), both evaluated under past-only context matching rollout inference conditions. Applicable to any masked DLM trajectory evaluation.

**4.** Empirical characterisation of mask sampling strategy effects across trajectory complexity levels (Phase 2), including optimal mask probability, data efficiency, and unit-type attribution (AO-ObsOnly / AO-ActOnly dissociation).

**5.** Reusable codebase: AnyOrderSampler, five evaluation scripts, dFactory YAML configs. Immediately applicable to new environments and masked DLM checkpoints.

---

## 12. Known Limitations

| **Limitation** | **Description** | **Mitigation** |
|---|---|---|
| Architecture confound | Qwen+IWM vs LLaDA+AOMT conflates masked-DLM pre-training with fine-tuning method. Cannot be controlled without retraining. | Acknowledged explicitly. Within-LLaDA ablation (Phase 2 + LLaDA conditions in Phase 3) cleanly isolates fine-tuning method by holding model constant. |
| Qwen+IWM data advantage | Qwen+IWM uses rollout data; LLaDA+AOMT uses expert data only. | Stated explicitly as a design choice replicating Zhang et al. faithfully. If AOMT matches IWM on less data, this is a positive finding, not a flaw. |
| Three seeds | Marginal for strong statistical claims, especially Phase 3. | Bootstrap CIs (1,000 resamples) alongside t-tests. Effect sizes reported. |
| Mask probability transfer | AOMT-Best selected on short MiniGrid trajectories; applied to longer ALFWorld text. | Acknowledged as assumption. Contingency: targeted 3-run p-sweep on ALFWorld if Phase 3 results deviate substantially. |
| Variable gradient quality | AO-p50 produces steps with near-empty to near-full masks; gradient signal varies in a way not captured by epoch count. | Compute-matched robustness check (§5.5). |

---

## 13. Key References

| **Reference** | **Relevance** |
|---|---|
| Zhang et al. (2025). *Agent Learning via Early Experience.* arXiv:2510.08558. | Motivation and primary baseline. Qwen+IL and Qwen+IWM conditions replicate their setup. AOMT is evaluated against their published task success rates. |
| Nie et al. (2025). *LLaDA: Large Language Diffusion with Masking.* arXiv:2502.09992. | Base model. LLaDA 2.0 8B is the masked DLM used in all LLaDA conditions. Its Qwen-2.5 initialisation makes Qwen the most principled causal baseline. |
| Wu et al. (2023). *Masked Trajectory Models.* ICML 2023. | Closest prior work on masking trajectories for offline RL. Motivates unit-level masking; does not study mask sampling strategy as a variable and operates in continuous state spaces. |
| Yang et al. (2019). *XLNet.* NeurIPS 2019. | Permutation language modelling: theoretical motivation for any-order coverage producing richer representations than fixed-order training. |
| Devlin et al. (2019). *BERT.* NAACL 2019. | Bidirectional masked training. Establishes context advantage; AOMT's novel claim is which masking strategy best exploits it for trajectory data. |
| Shridhar et al. (2021). *ALFWorld.* ICLR 2021. | Phase 3 benchmark. Natural-language agent environment; public expert dataset and simulator; basis for Zhang et al. comparison. |
| Chevalier-Boisvert et al. (2023). *MiniGrid.* | Phase 2 benchmark. Controlled complexity gradient for mask sampling ablation. |
| Sahoo et al. (2024). *Masked Diffusion LMs.* NeurIPS 2024. | Masked DLM training objective underlying LLaDA's pre-training and this project's fine-tuning loss. |

---

*End of Specification — v13.0 — Primary: Qwen-2.5-7B (IL/IWM) vs LLaDA-2.0-8B (AOMT)*