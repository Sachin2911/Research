# Summary: Illuminating Search Spaces by Mapping Elites

**Authors:** Jean-Baptiste Mouret, Jeff Clune
**Venue:** arXiv Preprint, April 2015
**Affiliation:** Universite Pierre et Marie Curie-Paris 6 (CNRS); University of Wyoming

## Motivation / Problem Statement

Traditional search/optimisation algorithms aim to find a single highest-performing solution. This is limiting for two reasons: (1) users often want to understand *how* performance varies across different solution characteristics, and (2) local optima trap algorithms that only follow performance gradients. A fundamentally different approach is needed -- one that provides a holistic map of high-performing solutions across user-defined dimensions of variation.

## Key Contributions

1. **MAP-Elites algorithm:** Introduced a new class of "illumination algorithms" (now called Quality-Diversity algorithms) that return the highest-performing solution for each cell in a discretised feature space chosen by the user.
2. **Simple and effective:** The algorithm is conceptually simple -- maintain an archive grid, randomly select an elite, mutate it, and place the offspring in the appropriate cell if it outperforms the current occupant.
3. **Three key benefits:**
   - Illuminates the relationship between performance and user-chosen features
   - Returns a large set of diverse, high-performing solutions
   - Often finds better global optima than pure optimisation algorithms (by avoiding deception through diverse search)
4. **Comparison with related algorithms:** Outperforms Novelty Search with Local Competition (NS+LC) and Multi-Objective Landscape Exploration (MOLE) on all metrics while being simpler to implement and understand.

## Methodology

- User chooses: (1) a performance/fitness measure f(x), and (2) N dimensions of variation defining a low-dimensional feature space.
- Each dimension is discretised into cells.
- **Initialisation:** Generate G random solutions, evaluate performance and features, place in appropriate cells.
- **Main loop:** Randomly select an elite from the archive, mutate (and/or crossover) to create offspring, evaluate, place in cell if empty or if offspring outperforms current occupant.
- O(1) cell lookup per evaluation (vs. O(n log n) for NS+LC).
- Hierarchical variant starts with coarse cells and refines; parallelised variant farms evaluations to cluster nodes.

## Main Results

- **Neural networks (retina problem):** MAP-Elites significantly outperformed traditional EA, NS+LC, and random sampling on all four metrics: global performance, global reliability, precision, and coverage. Lineage analysis showed that elites often descend from distant parents, demonstrating the value of simultaneous diverse search.
- **Simulated soft robots:** Produced diverse high-performing locomoting morphologies within a single run (unlike traditional EAs that converge to one type per run).
- **Real soft robotic arm:** Effective even in low-dimensional, real-world settings.
- Qualitative finding: most elites descend from nearby cells in feature space, but lineages traverse long paths through many regions -- "stepping stones" from one area enable discoveries in distant areas.

## Relevance to Research Project (Safe AI via Evolutionary Algorithms)

- **Foundational QD algorithm:** MAP-Elites is the basis for all Quality-Diversity methods used in evolutionary RL (e.g., PGA-MAP-Elites, CMA-MEGA, ME-ES). Understanding it is essential.
- **Diversity for robustness:** The archive of diverse, high-performing solutions directly enables robust agent behaviour -- if one policy fails (e.g., due to damage or safety violation), the agent can switch to a qualitatively different policy from the archive.
- **Safety as a feature dimension:** Safety-related metrics (e.g., constraint violations, proximity to hazards) could be used as feature dimensions in MAP-Elites, creating archives that explicitly map the safety-performance tradeoff.
- **Stepping-stone effect:** The finding that diverse search helps escape deception is relevant to safe RL, where overly greedy policies may exploit unsafe shortcuts.
