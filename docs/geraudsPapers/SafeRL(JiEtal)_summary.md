# Summary: Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark

**Authors:** Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Juntao Dai, Yaodong Yang
**Venue:** NeurIPS 2023 (Datasets and Benchmarks Track)
**Affiliation:** Institute for AI, Peking University; BIGAI

## Motivation / Problem Statement

Safe RL requires dedicated simulation environments for evaluating algorithms, yet the field lacks comprehensive, well-maintained benchmarks. The original OpenAI Safety Gym relies on the deprecated mujoco-py library, supports only 3 agents and limited tasks, and has poor visual rendering. There is a need for a unified, extensible platform covering single-agent, multi-agent, and vision-based SafeRL scenarios, along with a reliable algorithm library for fair comparison.

## Key Contributions

1. **Safety-Gymnasium environment suite:** A modernised and expanded benchmark built on Gymnasium and MuJoCo (replacing mujoco-py). Provides 54+ distinct environments spanning:
   - **Single-agent tasks:** Velocity, Run, Circle, Goal, Push, Button
   - **Multi-agent tasks:** Cooperative control with deconstructed robot bodies
   - **Vision-only tasks:** RGB and RGB-D inputs with realistic rendering
   - **Isaac Gym integration:** GPU-accelerated dexterous manipulation with safety constraints (Safety-DexterousHands)
   - **Constraint types:** Hazards, Pillars, Sigwalls, Vases, Gremlins, Velocity constraints
   - **Robots:** Point, Car, Doggo, Racecar, Ant (single-agent); multi-agent MuJoCo variants

2. **SafePO algorithm library:** A single-file-style library housing 16 state-of-the-art SafeRL algorithms, including:
   - Lagrangian-based: PPO-Lag, TRPO-Lag, RCPO, CPPO-PID
   - Trust-region / projection: CPO, PCPO, CUP, FOCOPS
   - Multi-agent: MACPO, MAPPO-Lag, HAPPO
   - Pure policy: PG, Natural PG, TRPO, PPO, MAPPO
   - Each algorithm verified line-by-line against original implementations.

3. **Comprehensive empirical analysis:** Evaluated all 16 algorithms across 54 environments with detailed metrics (normalised reward, normalised cost). Provides insights into each algorithm's strengths and weaknesses across different constraint complexities.

## Methodology

- Formulates SafeRL as a Constrained MDP (CMDP) for single-agent and Constrained Markov Game for multi-agent settings.
- Environments return a cost signal alongside reward at each step, with agents required to keep cumulative cost below a threshold.
- SafePO supports TensorBoard and WandB logging with 40+ visualisable parameters.
- Easy installation (`pip install safety-gymnasium`) and customisation (~100 lines to create a new environment).

## Main Results

- SafePO implementations match or outperform existing open-source SafeRL codebases (Safety-Starter-Agents, RL-Safety-Algorithms) across navigation tasks.
- Lagrangian-based methods generally achieve good reward-cost tradeoffs but can violate constraints in harder tasks.
- CPO provides theoretical near-constraint satisfaction but suffers from expensive Fisher information matrix inversions.
- Multi-agent SafeRL remains significantly more challenging than single-agent, with all algorithms showing higher cost violations.

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

- **Primary benchmark:** Safety-Gymnasium is the standard evaluation platform for SafeRL research. Any proposed evolutionary approach to safe RL should be evaluated here.
- **Algorithm baselines:** The 16 algorithms in SafePO provide the baselines that an evolutionary SafeRL method would need to outperform or complement.
- **CMDP formulation context:** Understanding the CMDP framework is essential since most SafeRL literature (and this benchmark) uses it. The Geraud et al. ROSARL paper proposes an alternative reward-only approach.
- **Multi-agent and vision tasks** present additional challenges where evolutionary methods (population-based search) may offer advantages through parallelism and diversity.
