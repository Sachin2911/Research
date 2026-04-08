# CoEvoRL
## Adversarial Co-Evolution of LLM Red-Team Agents and Safety Monitors via Constrained Evolutionary Reinforcement Learning

> Wits University | Honours/Masters Research Project | Safe AI via Evolutionary Algorithms

---

## 1. The One-Line Idea

Use a co-evolutionary arms race — driven by CMA-ES and RL — between a population of LLM attacker agents and a population of safety monitors, so that each side forces the other to become more capable, producing monitors that generalise to novel attacks and attackers that expose failure modes gradient-based methods miss.

---

## 2. Motivation and Problem Statement

### 2.1 The Core Problem

Modern LLMs are aligned using RLHF — a gradient-based process that optimises a reward model as a proxy for human preferences. This proxy is imperfect and gets exploited. The model learns to produce outputs that score highly on the reward model without genuinely being safe or helpful. This is **reward hacking**, and it is the central safety challenge in deployed AI systems.

Current red-teaming approaches (GCG, PAIR) are gradient-based. They find attacks efficiently but explore a narrow region of the failure space — they follow the gradient greedily and converge to a handful of attack patterns. A defender trained only against these attacks learns brittle surface-level features that fail on novel inputs.

### 2.2 Why Evolutionary Algorithms?

Evolutionary algorithms explore the solution space fundamentally differently from gradient descent. CMA-ES maintains a population of diverse solutions, applies selection pressure broadly, and does not follow a local gradient. This produces two properties that matter for red-teaming:

- **Diversity** — the population naturally explores many qualitatively different attack strategies simultaneously.
- **Black-box operation** — CMA-ES does not need to differentiate through the model or the reward signal, making it applicable anywhere a fitness score can be computed.

The hypothesis at the heart of this project: evolutionary attackers find failure modes that gradient-based methods structurally cannot reach, and co-evolutionary pressure produces monitors that generalise beyond their training distribution.

### 2.3 Strategic Context

| Goal | Description |
|---|---|
| **Academic** | Apply evolutionary algorithms (CMA-ES + RL) to safe AI — directly satisfying the project brief on black-box optimisation for aligned agents. |
| **Research** | Produce a genuine contribution on adversarial co-evolution in LLM weight space — a gap in the current literature. |
| **Industry** | Generate a dataset of diverse, EA-discovered LLM reward hacks directly useful for training reward hacking monitors — relevant to Watertight AI's core product. |

---

## 3. System Architecture

### 3.1 The Four Layers

The system operates across four nested levels. Each layer is a well-studied technique on its own — the contribution is their composition.

| Layer | Role |
|---|---|
| **CMA-ES (Outer Loop)** | Population diversity and exploration across generations. Maintains a distribution over LoRA weight vectors and updates it based on episodic fitness. |
| **PPO / RL (Inner Loop)** | Sequential policy optimisation within each generation. Each attacker agent runs K PPO steps, learning to adapt its strategy based on real-time monitor feedback. |
| **Co-evolution** | Attacker and monitor populations evolve simultaneously in an adversarial arms race — each generation, each side's fitness depends on the other's current best. |
| **CMDP Constraint** | Formalises the attacker's objective: maximise harm (reward) subject to a probabilistic constraint on detection rate (cost). Connects directly to the safe RL literature. |

### 3.2 The Attacker

**What it is:** A LoRA adapter applied to a frozen base LLM (Llama 3.2 3B Instruct). CMA-ES searches over the LoRA weight space via random projection to a tractable dimension (~1000D from ~2M parameters).

**What it does:** Generates responses to HarmBench prompts that are genuinely harmful (high LlamaGuard score) but appear safe to the current monitor (low detection score). The RL inner loop trains the attacker to adapt its strategy dynamically within an episode — if the monitor's score rises, the attacker softens its language; if it stays low, it escalates.

**CMDP formulation:**
- **State:** Current conversation context + last monitor detection score
- **Action:** Next token sequence / prompt framing choice
- **Reward:** LlamaGuard harm score (oracle, external to the system)
- **Cost:** Monitor detection probability
- **Constraint:** P(detection > 0.5 per episode) < ε

> **Key insight:** The CMDP formalisation is identical to the supervisor's sub-project 2 (constrain probability of violation, not expectation) — but the agent is an LLM attacker and the unsafe state is being detected, not crashing a robot. Same mathematics, novel domain.

### 3.3 The Monitor

**What it is:** A lightweight classification head placed on top of the frozen base model's final hidden state. Only the classifier weights are evolved by CMA-ES — no gradients, no backpropagation through the LLM.

**What it does:** Takes a model output and produces a detection score — is this response genuinely harmful or safely refused? Evaluated against a mixed dataset of attacker outputs, archive outputs, and clean refusals.

**Architecture:** Frozen base LLM → mean-pool hidden states → Linear(3072, 256) → ReLU → Linear(256, 1) → Sigmoid. Parameter count ~800K, projected to ~500D for CMA-ES.

### 3.4 The Co-Evolutionary Loop

Each generation proceeds as follows:

1. CMA-ES samples 20 attacker LoRA vectors and 20 monitor classifier vectors.
2. Each attacker runs K=10 PPO steps against the current best monitor, adapting its policy to the monitor's responses.
3. Each adapted attacker generates outputs on HarmBench prompts. LlamaGuard scores harm (oracle). The current monitor scores detection.
4. Attacker fitness = mean LlamaGuard score − λ × max(0, violation_prob − ε). Attackers that are harmful but stealthy score highest.
5. Best attacker outputs are added to the hall-of-fame archive.
6. Each monitor candidate is evaluated on the archive (all historical attacker outputs, not just current generation). Monitor fitness = classification accuracy.
7. CMA-ES updates both distributions based on fitness. The best monitor becomes the current monitor for the next generation.
8. ε is updated by a third CMA-ES instance (ROSARL), seeking the constraint threshold that maximises information gain from the arms race.

### 3.5 The Hall-of-Fame Archive

> **Why this is not optional:** Without the archive, the monitor adapts to the current attacker generation, the attacker adapts away, and the monitor forgets earlier attacks. This cycling produces a monitor that is never robust. The archive forces the monitor to remain accurate across all historical attackers simultaneously — the only path to generalisation.

---

## 4. ROSARL: Co-Evolving the Constraint

The constraint threshold ε is not hand-tuned. A third CMA-ES instance evolves it alongside the two populations. The fitness signal for ε is the information gain from the arms race:

- **High ε (loose constraint):** Attacker is barely penalised for being detected → arms race is trivial → monitor learns nothing useful.
- **Low ε (tight constraint):** Attacker cannot produce harmful content without being caught → arms race collapses → no interesting attacks generated.
- **Optimal ε:** The arms race is maximally informative — the attacker is genuinely challenged and the monitor is genuinely stretched.

The system discovers this optimal ε autonomously. Expected to converge around 0.2–0.3, meaning the attacker must keep its detection probability below 20–30% per episode to receive positive fitness.

> **Supervisor connection:** ROSARL is directly your supervisor's sub-project 3 — co-evolving the neural network and the penalty. Implemented here in LLM adversarial space rather than Safety-Gymnasium.

---

## 5. Research Questions

| Question | Why It Matters |
|---|---|
| **Q1:** Do co-evolved monitors generalise to gradient-based attacks (GCG, PAIR) they were never trained against? | If yes, co-evolution produces fundamentally more robust monitors than supervised training on fixed attack datasets. Core result. |
| **Q2:** Does ERL (CMA-ES + RL) find qualitatively different attacks than pure CMA-ES or pure PPO? | Validates the two-level architecture. If ERL finds more diverse attacks, the combination is justified over either component alone. |
| **Q3:** Do the populations exhibit Red Queen dynamics (cycling) or converge to a Nash equilibrium? | Foundational question for adversarial co-evolution in LLM space. Has implications for how long to run the arms race in practice. |
| **Q4:** What does the self-discovered ε reveal about the geometry of the safety-harm tradeoff? | The converged ε characterises how detectable harm is at the optimal stealthiness level — a measure of the difficulty of the safety problem. |
| **Q5:** Are EA-discovered attacks harder to defend against than gradient-based attacks? | Directly relevant to Watertight AI: if evolutionary attacks expose blind spots in existing monitors, they should be included in monitor training. |

---

## 6. Novelty and Gap in the Literature

| Prior Work | Method | What They Do | What's Missing |
|---|---|---|---|
| ESSA (2025) | ES + LoRA | ES to make LLMs safer | One-sided, no adversary, no RL |
| PAIR / GCG | Gradient-based | Red-team LLMs adversarially | Not evolutionary, no co-evolution, no monitor |
| GAN-based text | Gradient adversarial | Adversarial text generation | Unstable, discrete token problem, no safety framing |
| ROSARL | Co-evolve policy + penalty | Safe RL with adaptive penalty | Robotic control only, not LLMs |
| **This project** | **CMA-ES + PPO + Co-evolution** | **Adversarial arms race in LLM weight space with CMDP** | **The gap. This does not exist yet.** |

---

## 7. Technical Stack

| Component | Tool / Library |
|---|---|
| Base LLM | meta-llama/Llama-3.2-3B-Instruct (HuggingFace) |
| LoRA Adaptation | HuggingFace PEFT library |
| CMA-ES | pycma (`pip install cma`) |
| RL Inner Loop | PPO via TRL or Stable-Baselines3 |
| Safety Oracle | LlamaGuard (ground truth harm scorer, external) |
| Attack Benchmark | HarmBench (standardised harmful prompt dataset) |
| Dimensionality Reduction | Random projection (fixed matrix P, intrinsic dimension ~1000) |
| Hardware | Single GPU (A100 or equivalent) — 3B model is feasible |
| Language | Python 3.10+, PyTorch |

---

## 8. Phased Research Plan

### Phase 1 — Weeks 1–4: Core Pipeline
**Goal:** Confirm CMA-ES can find any successful attack against the base model.

- Set up Llama 3.2 3B with LoRA adapter (HuggingFace PEFT)
- Implement random projection: 1000D CMA-ES → full LoRA weight space
- Run LlamaGuard as oracle scorer on HarmBench prompts
- Single-population CMA-ES on the attacker only (no monitor, no RL)
- Confirm attacks are findable within 20 generations
- **Exit criterion:** Mean LlamaGuard score > 0.6 on at least 5 HarmBench prompts

### Phase 2 — Weeks 5–8: Add RL Inner Loop
**Goal:** Show ERL (CMA-ES + PPO) outperforms static CMA-ES.

- Implement CMDP: state, action, reward, cost, constraint
- Wrap each attacker in PPO, run K=10 steps per generation
- Add constraint penalty to CMA-ES fitness
- Compare: static CMA-ES vs ERL on attack success rate and diversity
- **Exit criterion:** ERL reaches mean fitness 0.75+ with violation rate < ε

### Phase 3 — Weeks 9–12: Full Co-evolution
**Goal:** Arms race is running. Both populations evolving simultaneously.

- Build monitor architecture (classification head on frozen LLM)
- Implement hall-of-fame archive
- Run full co-evolutionary loop: attacker CMA-ES + monitor CMA-ES
- Log attacker fitness and monitor accuracy per generation
- Observe and characterise convergence dynamics (Red Queen vs equilibrium)
- **Exit criterion:** Clear oscillation or convergence pattern visible over 30+ generations

### Phase 4 — Weeks 13–14: ROSARL Extension
**Goal:** System self-discovers optimal constraint threshold.

- Add third CMA-ES instance evolving ε
- Implement information-gain fitness for ε
- Run 20 generations of ε co-evolution
- **Exit criterion:** ε converges to a stable value, arms race remains informative

### Phase 5 — Weeks 15–16: Evaluation and Write-up
**Goal:** Tell a clear story with clean results.

- Generalisation: evaluate co-evolved monitor on GCG, PAIR, human red-team attacks
- Diversity: embed all discovered attacks, cluster, measure overlap with gradient-based attacks
- Ablations: with/without archive, with/without RL inner loop, different LoRA ranks
- Write paper / thesis chapter
- Open-source the attack dataset generated by the evolutionary system

---

## 9. Expected Results and Key Figures

| Figure / Result | What It Shows |
|---|---|
| Arms race plot: attacker fitness and monitor accuracy over generations | Red Queen dynamics — oscillation stabilising toward equilibrium. Core finding on co-evolutionary behaviour in LLM space. |
| ε convergence plot | ROSARL self-discovers optimal constraint ~0.2–0.3. Validates the ROSARL extension. |
| Generalisation table: monitor accuracy on GCG / PAIR / human attacks | Co-evolved monitor generalises. Baseline (trained on static attacks) does not. Core practical result. |
| Attack diversity: 2D embedding cluster plot | CMA-ES finds 5–7 qualitatively distinct attack clusters. Gradient-based methods cover 2–3. Most EA clusters are unique. |
| Ablation table: ERL vs CMA-ES vs PPO alone | ERL achieves higher fitness and greater diversity than either component alone. Validates the two-level architecture. |

---

## 10. Connection to Watertight AI

Watertight AI is building automated reward hacking monitors for RL training. The core challenge they face is: how do you train a monitor that catches reward hacks it has never seen before?

This project directly addresses that challenge in the LLM domain:

- **The attack dataset:** The evolutionary system generates a diverse corpus of LLM reward hacks — attacks that exploit the gap between the reward model and true safety. This dataset is exactly what is needed to train and evaluate Watertight's monitors.
- **The generalisation result:** If the co-evolved monitor catches gradient-based attacks it was never trained against, this demonstrates that co-evolutionary training is a superior approach to building robust monitors.
- **The diversity result:** If EA attacks are qualitatively different from gradient-based attacks, it means current monitors trained only on GCG/PAIR attacks have a structural blind spot that only evolutionary red-teaming can expose.

> **The pitch to Aengus Lynch:** *"I built a co-evolutionary system where CMA-ES and RL drive an arms race between LLM attackers and safety monitors. The monitors it produces generalise to attacks they were never trained against, and the attacker finds failure modes that gradient-based red-teaming misses entirely. The attack dataset is open-source. I think it would be directly useful for training Watertight's monitors."*

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Compute:** CMA-ES requires many forward passes. 3B model × population × generations may be slow. | Use random projection to keep search dimension low. Run Phase 1 on smaller prompts first. Each phase is independently valuable — stop at Phase 3 if compute is the bottleneck. |
| **Co-evolutionary cycling** produces no stable result. | Hall-of-fame archive is the primary mitigation. Also consider MAP-Elites style diversity preservation if cycling persists. |
| **LlamaGuard is imperfect** — the oracle has its own blind spots. | Use HarmBench human-labelled test set as secondary evaluation. Document LlamaGuard limitations explicitly in the write-up. |
| **Supervisor expects robotics/control direction.** | Frame the conversation around CMA-ES and CMDPs as the core methods — the same mathematics applied to a more novel domain. The ROSARL and CMDP sub-projects map directly. |
| **Scope creep** across 5 phases. | Each phase has a defined exit criterion. Treat Phase 1+2 as a complete workshop paper. The full system is a main conference paper. Do not start Phase 4 without solid Phase 3 results. |

---

## 12. Draft Paper Abstract

Existing red-teaming methods for large language models rely on gradient-based optimisation (GCG, PAIR), which efficiently finds attacks but explores a narrow region of the safety failure space. We propose **CoEvoRL**, a co-evolutionary framework in which a population of LoRA-based attacker agents and a population of safety monitor classifiers engage in an adversarial arms race driven by CMA-ES at the population level and PPO at the within-generation policy level. Attackers are trained under a Constrained Markov Decision Process (CMDP) formulation that penalises detection probability rather than expected cost, producing stealthy attacks that genuinely harm while evading the current monitor. A hall-of-fame archive forces the monitor to generalise across all historical attackers. The constraint threshold is co-evolved via ROSARL, eliminating hand-tuning. We show that: (1) co-evolved monitors generalise to gradient-based attacks (GCG, PAIR) they were never trained against, outperforming baselines trained on static attack datasets; (2) CMA-ES discovers qualitatively distinct failure modes invisible to gradient-based methods; and (3) the system exhibits Red Queen co-evolutionary dynamics that stabilise toward an approximate Nash equilibrium. The resulting attack dataset is released open-source for training reward hacking monitors in downstream safety systems.

---

## 13. Quick Glossary

| Term | Plain English Definition |
|---|---|
| **CMA-ES** | Covariance Matrix Adaptation Evolution Strategy. An evolutionary algorithm that maintains a probability distribution over solutions and updates it each generation based on which solutions performed best. |
| **LoRA** | Low-Rank Adaptation. A technique for fine-tuning LLMs by adding small trainable weight matrices, keeping the base model frozen. Drastically reduces the number of trainable parameters. |
| **CMDP** | Constrained Markov Decision Process. A reinforcement learning framework where the agent must maximise reward while keeping a separate cost signal below a threshold. |
| **Reward Hacking** | When an RL agent finds a way to score highly on the reward function without actually doing what was intended. The central safety problem in RLHF-trained LLMs. |
| **Co-evolution** | Two or more populations evolving simultaneously, where each population's fitness depends on the other. Classic example: predator and prey. |
| **Hall-of-Fame Archive** | A store of the best solutions found across all generations. Used to prevent the monitor from forgetting earlier attacks as the attacker evolves. |
| **Red Queen Dynamics** | From evolutionary biology: an endless arms race where both sides keep evolving but neither gains a permanent advantage. Named after the Red Queen in Alice in Wonderland. |
| **ROSARL** | Risk-constrained Optimisation via Self-Adapting RL. A framework where the constraint threshold is learned rather than hand-tuned. |
| **LlamaGuard** | A safety classifier from Meta, used here as an oracle to score how harmful a model output is. External to the system — not the thing being fooled. |
| **HarmBench** | A standardised benchmark of harmful prompts used to evaluate red-teaming methods. Provides a common evaluation ground for comparing attack methods. |

---

*This document is a living research overview. Update it as the project evolves.*
