# Summary: Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning

**Authors:** Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen
**Venue:** arXiv Preprint, February 2026
**Affiliation:** Cognizant AI Lab; MIT; UCLA; UT Austin

## Motivation / Problem Statement

RL-based fine-tuning (PPO, GRPO) is the dominant post-training paradigm for LLMs but suffers from four key limitations: (1) low sample efficiency with long-horizon/sparse rewards due to credit assignment difficulties, (2) sensitivity to the choice of base LLM, (3) susceptibility to reward hacking, and (4) instability across runs. Evolution Strategies (ES) could address these issues but was assumed to be infeasible for billion-parameter models without dimensionality reduction. This paper overturns that assumption.

## Key Contributions

1. **First full-parameter ES fine-tuning of billion-parameter LLMs:** Demonstrated that ES can directly optimise the full parameter space of modern LLMs (up to 7B+ parameters) without any dimensionality reduction (no LoRA, no SVD compression), using a population of only 30 -- orders of magnitude smaller than prior ES work (which used 10,000+).
2. **Outperforms RL on Countdown task:** ES substantially outperformed PPO, GRPO, and Dr.GRPO across all 8 tested models (Qwen2.5 0.5B--7B, Llama3 1B--8B), using a single fixed set of hyperparameters, while each RL method required per-model hyperparameter sweeps.
3. **Reduced reward hacking:** In conciseness fine-tuning, GRPO produced nonsensical short outputs (reward hacking) without a carefully tuned KL-divergence penalty. ES achieved comparable reward without any KL penalty and without reward hacking, producing a dominant Pareto front (higher reward, lower KL divergence).
4. **Greater robustness:** ES provided consistent improvements across different base LLMs and across independent runs, whereas RL methods failed entirely on some models and showed high variance between runs.
5. **Competitive on math reasoning:** On standard benchmarks (MATH500, AIME2024, Minerva, OlympiadBench, AMC) using Qwen2.5-Math-7B, ES achieved competitive performance with state-of-the-art RL methods (SimpleRL-Zero, OpenReasoner-Zero, Oat-Zero), despite using a vanilla implementation without common enhancements.
6. **Inference-only, no backprop:** ES fine-tuning only requires forward passes, saving significant GPU memory and eliminating the need for gradient computation infrastructure.

## Methodology

- Based on simplified NES (following Salimans et al., 2017) with fixed-covariance Gaussian perturbation noise.
- **Key implementation details:**
  - *Random seed storage:* Only seeds stored (not full noise tensors), drastically reducing memory.
  - *Layer-level in-place perturbation/restoration:* Parameters perturbed and restored layer-by-layer, requiring only one extra layer-sized tensor.
  - *Greedy decoding:* Perturbed models evaluated deterministically, ensuring all performance differences come from parameter-space exploration.
  - *Z-score reward normalisation:* Rewards normalised within each iteration for cross-iteration consistency.
  - *Decomposed parameter update:* Aggregated update applied layer-by-layer and seed-by-seed, minimising peak memory.
- Intentionally omits common ES enhancements (rank transformation, mirrored sampling, weight decay, Adam, virtual batch normalisation) to isolate the core algorithm's effectiveness.
- Fixed hyperparameters: $N = 30$, $\sigma = 0.001$, $\alpha = 5 \times 10^{-4}$ for all Countdown experiments.

## Main Results

- **Countdown task:** ES outperformed all RL baselines across all 8 models. Notable margins: Qwen2.5-3B (ES: 60.5% vs. best RL: 43.8%), Qwen2.5-7B (ES: 66.8% vs. best RL: 57.5%), Llama3.1-8B (ES: 61.2% vs. best RL: 51.3%).
- **Conciseness fine-tuning:** ES Pareto front dominated GRPO's (higher reward at lower KL divergence). No reward hacking observed with ES, even without KL penalty.
- **Math reasoning (Qwen2.5-Math-7B):** Competitive with SOTA RL -- e.g. Minerva Math: ES 33.1% vs. best RL 30.5%; AMC: ES 62.7% matching Dr.GRPO 62.7%; MATH500: ES 78.6% vs. GRPO 78.0%.
- **Consistency:** ES showed lower variance across independent runs compared to RL methods.

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

- **Direct evidence for ES as a safe fine-tuning paradigm:** The reduced reward hacking is a key safety property. ES optimises a solution *distribution* rather than a single solution, making it structurally harder to exploit reward function loopholes -- directly relevant to ensuring aligned, safe behaviour.
- **Robustness = reliability for safety-critical deployment:** ES's consistency across base models and runs reduces the risk of deploying a poorly fine-tuned model due to bad hyperparameter choices or training instability.
- **Full-parameter optimisation without gradients:** Opens the possibility of optimising arbitrary (non-differentiable) safety objectives -- e.g. constraint violation counts, human preference scores, or toxicity classifiers -- directly as the fitness function.
- **Complements ESSA:** While ESSA (Korotyshova et al.) compresses the search space via SVD-LoRA for maximum efficiency, this work shows ES can also work in the full parameter space, suggesting a spectrum of ES approaches for LLM alignment depending on compute budget.
- **Democratisation of safe fine-tuning:** The paper's Impact Statement notes that ES lowers the barrier to fine-tuning (no expert gradient-training knowledge needed) and reduces the risk of losing ethical guardrails, making safe fine-tuning more accessible to non-experts.
