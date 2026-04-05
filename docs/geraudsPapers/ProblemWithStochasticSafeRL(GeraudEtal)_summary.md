# Summary: ROSARL: Reward-Only Safe Reinforcement Learning

**Authors:** Geraud Nangue Tasse, Tamlin Love, Mark Nemecek, Steven James, Benjamin Rosman
**Venue:** Reinforcement Learning Conference (RLC), August 2024
**Affiliation:** University of the Witwatersrand; UPC Barcelona; Duke University

## Motivation / Problem Statement

Designing safe RL agents typically requires either hand-crafted penalties for unsafe states or separate cost functions to be minimised (as in Constrained MDPs). Both approaches are non-trivial: penalties that are too small lead to unsafe behaviour, while penalties that are too large slow down learning. Existing constraint-based methods (CPO, TRPO-Lagrangian) still frequently violate safety constraints in practice, especially under stochastic dynamics.

## Key Contributions

1. **Minmax Penalty concept:** Formalised the smallest penalty for unsafe states that guarantees safe optimal policies regardless of task rewards.
2. **Theoretical bounds:** Derived upper and lower bounds on the Minmax penalty using environment *diameter* (longest expected path to a goal) and *controllability* (minimum non-zero probability of reaching safe goals). Proved these bounds guarantee safe optimal policies (Theorem 2).
3. **Practical model-free algorithm (Algorithm 2):** Since exact estimation is NP-hard (Theorem 3), proposed a simple online method that estimates the Minmax penalty from the agent's learned value function. The penalty is updated as `V_MIN - V_MAX` using observed rewards and value estimates, and replaces the reward whenever an unsafe state is encountered.
4. **Integrates into any value-based RL pipeline** by only modifying the reward for unsafe transitions.

## Methodology

- Formulates safety in undiscounted stochastic shortest path MDPs with absorbing unsafe states.
- Bounds the Minmax penalty as R_MIN_bar = (R_MIN - R_MAX) * D / C, where D is the diameter and C is the controllability.
- The practical algorithm tracks running estimates of reward bounds and value bounds, producing an increasingly negative penalty each time the agent visits an unsafe state (self-correcting loop).
- Applied to TRPO (denoted TRPO-Minmax) by replacing rewards at unsafe transitions with the learned penalty.

## Main Results

- **Lava Gridworld (tabular):** Algorithm learns near-optimal penalties and safe policies across varying slip probabilities (0, 0.25, 0.5). Higher stochasticity leads to larger penalties and safer (longer) paths.
- **Safety Gym Pillar (continuous control):** TRPO-Minmax learns safe policies even in high-dimensional continuous domains with function approximation, adapting penalty magnitude to noise level.
- **Comparison with baselines:** Under stochastic dynamics (noise=2.5), TRPO, TRPO-Lagrangian, CPO, and Saute-TRPO all learn risky short-trajectory policies with high cumulative cost. TRPO-Minmax consistently maintains low failure rates while still maximising returns. When noise is too high (>=5), TRPO-Minmax prioritises safety over reward.

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

This paper is directly from the research supervisor's group and represents the **reward-only paradigm** for Safe RL -- an alternative to CMDP-based approaches. Key connections:

- Demonstrates that safety can be encoded purely through reward shaping rather than separate cost constraints, aligning with the reward hypothesis.
- The Minmax penalty framework could be combined with evolutionary search methods (e.g., MAP-Elites) to evolve diverse safe policies.
- The failure of standard Safe RL baselines under stochastic dynamics motivates exploring evolutionary approaches that may be more robust to environmental stochasticity.
- The simplicity of the approach (modifying only unsafe-state rewards) makes it easy to integrate with population-based / evolutionary RL methods.
