# Summary: Approximating Gradients for Differentiable Quality Diversity in Reinforcement Learning

**Authors:** Bryon Tjanaka, Matthew C. Fontaine, Julian Togelius, Stefanos Nikolaidis
**Venue:** GECCO 2022 (Genetic and Evolutionary Computation Conference)
**Affiliation:** University of Southern California; New York University

## Motivation / Problem Statement

Quality Diversity (QD) algorithms like MAP-Elites produce diverse archives of high-performing solutions, which is valuable for training robustly capable RL agents (e.g., agents that can adapt to damage by switching policies). Differentiable QD (DQD) algorithms such as CMA-MEGA vastly outperform standard QD when exact gradients of the objective and measure functions are available. However, in RL settings the environment is typically non-differentiable, so exact gradients do not exist. The challenge is: how can we approximate these gradients to bring DQD's efficiency to RL?

## Key Contributions

1. **Formalised QD-RL as DQD:** Defined the Quality Diversity for Reinforcement Learning (QD-RL) problem and reduced it to an instance of DQD, where objective = expected discounted return and measures = behavioural descriptors of policies.
2. **Two CMA-MEGA variants for RL:**
   - **CMA-MEGA (ES):** Approximates both objective and measure gradients using OpenAI-ES (evolution strategies).
   - **CMA-MEGA (TD3, ES):** Approximates the objective gradient using TD3 (actor-critic) and measure gradients using OpenAI-ES.
3. **Benchmarking on QDGym locomotion tasks:** Evaluated on QD Ant, QD Half-Cheetah, QD Hopper, and QD Walker (4 simulated locomotion environments). Compared against PGA-MAP-Elites (state-of-the-art), ME-ES, and MAP-Elites.
4. **Key insight on DQD limitations:** Revealed that CMA-MEGA's advantage diminishes when gradients must be approximated rather than computed exactly, especially when the main challenge is objective optimisation rather than measure-space exploration.

## Methodology

- **CMA-MEGA framework:** Maintains a solution point phi*, a CMA-ES distribution over gradient coefficients, and a MAP-Elites archive. Each iteration: compute gradients at phi*, sample coefficient vectors to create new solutions by linearly combining objective and measure gradients, insert into archive, update phi* and CMA-ES towards maximum archive improvement.
- **ES gradient approximation:** Uses OpenAI-ES formula with mirror sampling and rank normalisation to estimate gradients of both objective f and measures m_i as black boxes.
- **TD3 gradient approximation:** Trains critic networks on replay buffer experience; provides objective gradient without additional environment interaction.
- **Evaluation:** 1 million solution evaluations per algorithm, 5 random seeds, metrics: QD score, archive coverage, best performance.

## Main Results

- **CMA-MEGA (TD3, ES):** Achieved comparable QD score to PGA-MAP-Elites across all 4 tasks, though less sample-efficient in QD Ant and QD Half-Cheetah.
- **CMA-MEGA (ES):** Comparable to PGA-MAP-Elites in QD Hopper and QD Walker, but weaker in QD Ant and QD Half-Cheetah.
- **PGA-MAP-Elites** remained the most consistent performer, benefiting from its simpler combination of MAP-Elites mutation operators with TD3 gradient steps.
- **MAP-Elites** (no gradients) and **ME-ES** generally underperformed the gradient-informed methods.
- **Key finding:** The theoretical advantage of CMA-MEGA (which outperforms by orders of magnitude with exact gradients) does not fully transfer to RL domains where gradient approximation introduces noise. The bottleneck is objective optimisation, not measure-space exploration.

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

- **Bridges evolutionary and gradient-based RL:** Demonstrates how to combine Evolution Strategies, actor-critic methods, and Quality Diversity in a unified framework -- directly relevant to "Safe AI via Evolutionary Algorithms."
- **Diverse policy archives for safety:** QD-RL produces archives of behaviourally diverse policies. These archives could be extended with safety measures (e.g., constraint violations as a measure dimension) to produce diverse *safe* policies.
- **Practical algorithm design:** Shows that simpler hybrid approaches (PGA-MAP-Elites) can match more theoretically elegant ones (CMA-MEGA) when gradients are noisy, informing design choices for a safe evolutionary RL system.
- **Foundation for safety-aware QD-RL:** The QD-RL formulation could be extended to include safety constraints, creating a "Safe QD-RL" problem where the archive explicitly tracks both performance and safety.
