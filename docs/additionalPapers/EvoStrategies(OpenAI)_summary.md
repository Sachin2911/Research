# Summary: Evolution Strategies as a Scalable Alternative to Reinforcement Learning

**Authors:** Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever
**Venue:** arXiv Preprint, 2017
**Affiliation:** OpenAI

## Motivation / Problem Statement

Standard deep RL algorithms (Q-learning, policy gradients) have achieved impressive results but are inherently difficult to parallelise: they require value function approximation, backpropagation, and frequent communication of large gradient vectors between workers. Black-box optimisation methods like Evolution Strategies (ES) avoid these bottlenecks, yet were perceived as unable to compete on hard RL benchmarks. This paper challenges that perception.

## Key Contributions

1. **Massively parallel ES for RL:** Demonstrated that Natural Evolution Strategies (NES) with a novel shared-random-seed communication strategy can scale to over 1,440 parallel CPU workers. Workers only communicate scalar returns (not gradients), enabling near-linear speedups.
2. **Competitive with state-of-the-art RL:** ES matched the final performance of A3C on most Atari games (using 3--10x more data but ~3x less computation due to no backprop/value function). On MuJoCo, ES matched TRPO performance. The 3D humanoid walking task was solved in just 10 minutes.
3. **Virtual batch normalisation trick:** Showed that reparameterising the policy network with virtual batch normalisation is critical for ES on Atari -- without it, perturbed policies collapse to a single action. This made ES reliable across a wide variety of environments.
4. **Structural advantages over policy gradients:** ES is invariant to action frequency, tolerant of arbitrarily long time horizons (gradient variance does not grow with episode length), handles sparse/delayed rewards naturally, and does not require temporal discounting or value function approximation.
5. **Qualitatively different exploration:** ES discovered diverse gaits on MuJoCo humanoid (e.g. walking sideways, backwards) that were never observed with TRPO, suggesting parameter-space perturbation yields broader exploration than action-space noise.

## Methodology

- Uses NES with an isotropic Gaussian perturbation distribution over policy parameters: each worker samples noise $\epsilon_i$, evaluates the perturbed policy $\theta + \sigma\epsilon_i$ over a full episode, and reports the scalar return.
- **Shared random seeds** allow all workers to reconstruct each other's perturbations, so only scalar returns need to be communicated -- drastically reducing bandwidth.
- Variance reduction via antithetic (mirrored) sampling ($\epsilon, -\epsilon$ pairs) and rank-based fitness shaping.
- Virtual batch normalisation ensures the policy responds meaningfully to different inputs even with random initial weights.
- Weight decay prevents parameters from growing large relative to perturbation scale.

## Main Results

- **MuJoCo:** Solved 3D humanoid walking in 10 minutes on 1,440 CPUs. Matched TRPO final performance on standard tasks using at most 10x more data.
- **Atari:** Competitive with A3C on most games after 1 hour of training. Better on 23/51 games, worse on 28, using comparable total computation to A3C's published 1-day results.
- **Parallelisation:** Achieved linear speedup from 1 to 1,440 workers due to minimal communication overhead.
- **Robustness:** Used fixed hyperparameters across all Atari environments and a separate fixed set across all MuJoCo environments.

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

- **Foundational ES-for-RL paper:** This is the seminal work demonstrating that ES can solve modern deep RL benchmarks at scale. It directly motivates both ESSA (Korotyshova et al.) and ES-at-Scale (Qiu et al.) for LLM alignment.
- **Safety-relevant properties:** ES's tolerance of long horizons and sparse rewards, combined with its resistance to reward shaping artifacts, aligns with SafeRL goals where reward signals may be infrequent or require careful design.
- **Exploration diversity:** The qualitatively different exploration behaviour (diverse gaits) parallels Quality-Diversity ideas and could be leveraged to discover robust, diverse safe policies.
- **Scalability without backprop:** The inference-only, gradient-free nature of ES makes it a natural fit for optimising safety-constrained objectives where differentiating through constraint functions may be difficult or undesirable.
