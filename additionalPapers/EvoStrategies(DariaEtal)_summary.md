# Summary: ESSA: Evolutionary Strategies for Scalable Alignment

**Authors:** Daria Korotyshova, Boris Shaposhnikov, Alexey Malakhov, Alexey Khokhulin, Nikita Surnachev, Kirill Ovcharenko, George Bredis, Alexey Gorbatovski, Viacheslav Sinii, Daniil Gavrilov
**Venue:** Preprint (undated)
**Affiliation:** T-Tech

## Motivation / Problem Statement

Aligning LLMs via RLHF (PPO, GRPO) requires complex distributed training pipelines: actor/critic networks, backpropagation through long sequences, gradient synchronisation across GPUs, and sensitive hyperparameter tuning. These costs grow severely at billion-parameter scale. The question is whether a gradient-free, inference-only approach can match gradient-based alignment quality while being simpler, faster, and more scalable.

## Key Contributions

1. **ESSA framework:** A gradient-free alignment method that replaces the online RL stage of RLHF with CMA-ES optimisation over a drastically compressed parameter space. The method operates entirely through forward inference -- no backpropagation is needed.
2. **SVD-compressed LoRA search space:** LoRA adapters (on Q/K/V/O attention projections) are initialised via SFT, then decomposed via SVD. Only the singular values are optimised by CMA-ES, reducing the search space to a compact, task-aligned subspace while preserving LoRA expressiveness.
3. **Quantisation compatibility:** Since ESSA is inference-only, it runs natively in INT4 and INT8 precision, enabling alignment of models up to ~72B parameters on a single high-memory GPU.
4. **Superior scaling:** ESSA shows near-linear scaling with GPU count. On Qwen2.5-32B (PRM800K), it reaches target accuracy 2x faster on 16 GPUs and 6x faster on 128 GPUs compared to GRPO, due to embarrassingly parallel evaluation with minimal communication (only random seeds and scalar rewards).
5. **Competitive or superior quality:** Across multiple benchmarks, ESSA matches or outperforms GRPO -- improving Qwen2.5-Math-7B accuracy by +12.6% on GSM8K and +14.8% on PRM800K, and LLaMA3.1-8B accuracy by +22.5% on IFEval.

## Methodology

- **Initialisation:** Train standard LoRA adapters via a short SFT stage (same for both ESSA and GRPO baselines). SFT quality is important -- reducing SFT data from 100% to 5% drops final accuracy by >15 percentage points.
- **SVD decomposition:** Each LoRA factor (A, B) is decomposed as $U\Sigma V^\top$. Orthogonal matrices $U, V$ are frozen; only the top singular values in $\Sigma$ are trainable. Solution length = num_layers x num_matrices_per_layer x LoRA_rank x 2.
- **CMA-ES optimisation:** Each iteration samples $\lambda$ candidate singular-value vectors from a multivariate normal. Each candidate reconstructs updated LoRA factors, evaluates the model on alignment data to get a scalar reward, and CMA-ES updates the mean, step-size, and covariance matrix.
- **Communication:** Only random seeds and scalar rewards are exchanged between GPU workers, enabling near-linear parallel scaling.

## Main Results

- **GSM8K (Qwen2.5-Math-7B):** ESSA achieves 87.2% accuracy vs. GRPO baseline (~77.5%), a +12.6% improvement.
- **PRM800K / MATH500 (Qwen2.5-Math-7B):** +14.8% over GRPO.
- **IFEval (LLaMA3.1-8B):** +22.5% over GRPO on instruction following.
- **Scaling:** On 128 GPUs, ESSA reaches target accuracy in ~20 minutes vs. ~150 minutes for GRPO (~6x faster).
- **Quantisation:** INT4 and INT8 precision incur minimal accuracy loss (BFLOAT16: 0.847, INT8: 0.844, INT4: 0.838 on PRM800K).
- **Hyperparameter robustness:** Accuracy is stable across moderate LoRA ranks (2--16), population sizes (24--96), and singular value fractions ($\alpha \geq 0.4$).

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

- **Gradient-free alignment of LLMs:** ESSA demonstrates that evolutionary methods can directly align LLM behaviour -- the core goal of safe AI. Replacing the RL alignment stage with ES removes a major source of training instability and hyperparameter fragility.
- **Inference-only safety optimisation:** Since ESSA never computes gradients through the model, it could optimise safety-related objectives (e.g. constraint satisfaction, toxicity reduction) without requiring differentiable reward/cost functions.
- **Scalability for safe alignment:** The near-linear scaling with hardware makes ESSA practical for aligning very large models under safety constraints, where multiple evaluation criteria may need to be assessed per candidate.
- **Connection to CMA-ES literature:** ESSA builds on CMA-ES, the same optimizer used in Quality-Diversity methods (MAP-Elites variants), potentially enabling future work that maintains diverse archives of aligned policies with different safety-performance tradeoffs.
