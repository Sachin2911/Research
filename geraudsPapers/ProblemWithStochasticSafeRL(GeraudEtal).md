Reinforcement Learning Conference (August 2024)

# ROSARL: Reward-Only Safe Reinforcement Learning

**Geraud Nangue Tasse**<sup>1</sup>, **Tamlin Love**<sup>1,2</sup>, **Mark Nemecek**<sup>3</sup>, **Steven James**<sup>1</sup> & **Benjamin Rosman**<sup>1</sup>
<sup>1</sup> School of Computer Science and Applied Mathematics, University of the Witwatersrand
<sup>2</sup> Institut de Robòtica i Informàtica Industrial, Universidad Politécnica de Cataluna
<sup>3</sup> Department of Computer Science, Duke University
tlove@iri.upc.edu, mark.nemecek@duke.edu
{geraud.nanguetasse1, steven.james, benjamin.rosman1}@wits.ac.za

## Abstract

An important problem in reinforcement learning is designing agents that learn to solve tasks safely in an environment. A common solution is to define either a penalty in the reward function or a cost to be minimised when reaching unsafe states. However, designing reward or cost functions is non-trivial and can increase with the complexity of the problem. To address this, we investigate the concept of a *Minmax penalty*, the smallest penalty for unsafe states that leads to safe optimal policies, regardless of task rewards. We derive an upper and lower bound on this penalty by considering both environment *diameter* and *controllability*. Additionally, we propose a simple algorithm for agents to estimate this penalty while learning task policies. Our experiments demonstrate the effectiveness of this approach in enabling agents to learn safe policies in high-dimensional continuous control environments.

## 1 Introduction

Reinforcement learning (RL) has recently achieved success across a variety of domains, such as video games (Shao et al., 2019), robotics (Kalashnikov et al., 2018; Kahn et al., 2018) and autonomous driving (Kiran et al., 2021). However, if we hope to deploy RL in the real world, agents must be capable of completing tasks while avoiding unsafe or costly behaviour. For example, a navigating robot must avoid colliding with objects and actors around it, while simultaneously learning to solve the required task. Figure 1 shows an example.

Many approaches in RL deal with this problem by allocating arbitrary penalties to unsafe states when hand-crafting the reward function. However, the problem of specifying a reward function for desirable, safe behaviour is notoriously difficult (Amodei et al., 2016). *Importantly, penalties that are too small may result in unsafe behaviour, while penalties that are too large may result in increased learning times.* Furthermore,



Figure 1: Example trajectories of prior work—TRPO (Schulman et al., 2015) (left-most), TRPO-Lagrangian (Ray et al., 2019) (middle-left), CPO (Achiam et al., 2017) (middle-right)—compared to ours (right-most) in the Safety Gym domain (Ray et al., 2019). For each, a point mass agent learns to reach a goal location (green cylinder) while avoiding unsafe regions (blue circles). The cyan block is a randomly placed movable obstacle. Our approach learns safer policies than the baselines, and works by simply changing the rewards received for entering unsafe regions to a learned penalty (keeping the rewards received for all other transitions unchanged).

1

Reinforcement Learning Conference (August 2024)

these rewards must be specified by an expert for each new task an agent faces. If our aim is to design truly autonomous, general agents, it is then simply impractical to require that a human designer specify penalties to guarantee optimal but safe behaviours for every task.

When safety is an explicit goal, a common approach is to constrain policy learning according to some threshold on cumulative cost (Schulman et al., 2015; Ray et al., 2019; Achiam et al., 2017). While effective, these approaches require the design of a cost function whose specification can be as challenging as designing a reward function. Additionally, these methods may still result in unacceptably frequent constraint violations in practice, due to the large cost threshold typically used. See Appendix C for further discussion of related works.

Rather than attempting to both maximise a reward function and minimise a cost function, which requires specifying both rewards and costs and a new learning objective, we should simply aim to have a better reward function—since we then do not have to specify yet another scalar signal nor change the learning objective. This approach is consistent with the reward hypothesis (Sutton & Barto, 2018) which states: “All of what we mean by goals and purposes can be well thought of as maximisation of the expected value of the cumulative sum of a received scalar signal (reward).” Therefore, the question we examine in this work is how to determine the Minmax penalty—the smallest penalty assigned to unsafe states such that the probability of reaching safe goals is maximised by an optimal policy. Rather than requiring an expert’s input, we show that this penalty can be bounded by taking into account the diameter and controllability of an environment, and a practical estimate of it can be learned by an agent using its current value estimates. We make the following main contributions:

(i) Bounding the Minmax penalty (Section 3.3): We obtain the analytical form of an upper and lower bound on the Minmax penalty and prove that using the upper bound results in learned behaviours that minimise the probability of visiting unsafe states (Theorem 2); We also show that these bounds can be accurately estimated using policy evaluation (Sutton & Barto, 2018) (Theorem 1).

(ii) Learning safe policies (Section 4): We show that accurately estimating the Minmax penalty or bounds is NP-hard (Theorem 3). Hence, we propose a simple model-free algorithm for learning a practical estimate of the Minmax penalty while learning the task policy. Since the approach only modifies the rewards for unsafe transitions with the estimated penalty (keeping the rewards for other transitions unchanged), it can be integrated into any RL pipeline that learns value functions.

(iii) Experiments (Section 5): Finally, we investigate the behaviour of agents that only rely on their learned Minmax penalty to solve tasks safely. Our results demonstrate that these reward-only agents are capable of learning to solve tasks while avoiding unsafe states. Additionally, while prior methods often violate safety constraints, we observe that reward-only agents consistently learn safer policies.

## 2 Background

We consider the typical RL setting where the task faced by an agent is modelled by a Markov Decision Process (MDP). An MDP is defined as a tuple $\langle S, A, P, R \rangle$, where $S$ is a finite set of states, $A$ is a finite set of actions, $P : S \times A \times S \to [0, 1]$ is the transition probability function, and $R : S \times A \times S \to [R_{MIN}, R_{MAX}]$ is the reward function. Our focus is on undiscounted MDPs that model stochastic shortest path problems (Bertsekas & Tsitsiklis, 1991) in which an agent must reach some goals in the non-empty set of absorbing states $G \subset S$ while avoiding unsafe absorbing states $G^! \subset G$. The set of non-absorbing states $S \setminus G$ are referred to as internal states. We will also refer to the tuple $\langle S, A, P \rangle$ as the environment, and the MDP $\langle S, A, P, R \rangle$ as a task to be solved.

A policy $\pi : S \to A$ is a mapping from states to actions. The value function $V^\pi(s) = E[\sum_{t=0}^\infty R(s_t, a_t, s_{t+1})]$ associated with a policy specifies the expected return under that policy starting from state $s$. The goal of an agent is to learn an optimal policy $\pi^*$ that maximises the value function $V^{\pi^*}(s) = V^*(s) = \max_\pi V^\pi(s)$ for all $s \in S$. Since tasks are undiscounted, $\pi^*$ is guaranteed to exist by assuming that the value function of improper policies is unbounded from below—where proper policies are those that are guaranteed to reach an absorbing state (Van Niekerk et al., 2019). Since there always exists a deterministic $\pi^*$ (Sutton & Barto, 1998), and $\pi^*$ is proper, we will focus our attention on the set of all deterministic proper policies $\Pi$.

2

Reinforcement Learning Conference (August 2024)

## 3 Avoiding Unsafe Absorbing States

Given an environment, we aim to bound the smallest penalty (hence the largest reward) to use as feedback for unsafe transitions to guarantee safe optimal policies. We formally define a safe policy as a proper policy that minimises the probability of reaching any unsafe terminal states:

**Definition 1** *Consider an environment $\langle S, A, P \rangle$. Where $s_T$ is the final state of a trajectory and $G^! \subset G$ is the non-empty set of unsafe absorbing states, let $P_s^\pi(s_T \in G^!)$ be the probability of reaching $G^!$ from $s$ under a proper policy $\pi \in \Pi$. Then $\pi$ is called safe if $\pi \in \underset{\pi' \in \Pi}{\arg \min} P_s^{\pi'}(s_T \in G^!)$ for all $s \in S$.*

**Remark 1** *Since proper policies reach $G$, Definition 1 equivalently says that safe policies are those that maximise the probability of reaching safe goal states $G \setminus G^!$. Since optimal policies are also proper, this means that safe optimal policies also maximise the probability of reaching $G \setminus G^!$. For example, looping forever in a non-absorbing region of the state space is neither proper, nor safe, nor optimal.*

We now define the Minmax penalty $R_{Minmax}$ as the largest reward for unsafe transitions that lead to safe optimal policies:

**Definition 2** *Consider an environment $\langle S, A, P \rangle$ where task rewards $R(s, a, s')$ are bounded by $[R_{MIN} \ R_{MAX}]$ for all $s' \notin G^!$. Let $\pi^*$ be an optimal policy for one such task $\langle S, A, P, R \rangle$. We define the Minmax penalty of this environment as the scalar $R_{Minmax} \in R$ that satisfies the following:*
*   (i) If $R(s, a, s') < R_{Minmax}$ for all $s' \in G^!$, then $\pi^*$ is safe for all $R$;
*   (ii) If $R(s, a, s') > R_{Minmax}$ for some $s' \in G^!$ reachable from $S \setminus G$, then there exists an $R$ s.t. $\pi^*$ is unsafe.

### 3.1 A Motivating Example: The Chain-Walk Environment

To illustrate the difficulty in designing reward functions for safe behaviour, consider the simple *chain-walk* environment in Figure 2a. It consists of four states $s_0, \textcolor{orange}{s_1}, s_2, \textcolor{blue}{s_3}$ where $G = \{ \textcolor{orange}{s_1}, \textcolor{blue}{s_3} \}$ and $G^! = \{ \textcolor{orange}{s_1} \}$. The agent has two actions $a_1, a_2$, the initial state is $s_0$, and the diagram denotes the transition probabilities. Task rewards for safe transitions are bounded by $[R_{MIN} \ R_{MAX}] = [-1 \ 0]$. The absorbing transitions have a reward of 0 while all other transitions have a reward of $R_{step} = -1$, and the agent must reach the goal state $\textcolor{blue}{s_3}$, but not the unsafe state $\textcolor{orange}{s_1}$. Hence, the question here is what penalty to give for transitions from $s_0$ into $\textcolor{orange}{s_1}$ such that the optimal policies are safe. Figures 2b-2d exemplify how too large penalties result in longer convergence times, while too small ones result in unsafe policies, demonstrating the need to find the Minmax penalty.



Figure 2: The effect of different penalties for unsafe transitions ($s_0$ to $\textcolor{orange}{s_1}$) on optimal policies in the chain-walk environment. (a) The transition probabilities of the chain-walk environment (where $p_1, p_2 \in [0 \ 1]$); (b) The failure rate for each penalty in $[-10 \ 0]$ and each transition probabilities ($p_1 = p_2 \in [0 \ 1]$), with a task reward of $R_{step} = -1$; (c) The failure rate for each penalty in $[-10 \ 0]$ and each task reward in $[-1 \ 0]$, with transition probabilities given by $p_1 = p_2 = 0.4$; (d) The total timesteps needed to learn optimal policies to convergence (using value iteration (Sutton & Barto, 1998)) for each penalty in $[-10 \ 0]$ and each task reward in $[-1 \ 0]$, with transition probabilities given by $p_1 = p_2 = 0.4$. The black dashed lines in (b) and (c) show the Minmax penalty.

3

Reinforcement Learning Conference (August 2024)

Since the transitions per action are stochastic, controlled by $p_1, p_2 \in [0, 1]$, and <mark>s<sub>3</sub></mark> is further from the start state $s_0$ than <mark>s<sub>1</sub></mark>, the agent may not always be able to avoid <mark>s<sub>1</sub></mark>. In fact, for $p_1 = p_2 = 0$ and $-1$ penalty for transitions into <mark>s<sub>1</sub></mark>, the optimal policy is to always pick $a_2$ which always reaches <mark>s<sub>1</sub></mark>. For a sufficiently high penalty for reaching <mark>s<sub>1</sub></mark> (any penalty higher than $-2$), the optimal policy is to always pick action $a_1$, which always reaches <mark>s<sub>3</sub></mark>. However, for $p_1 = p_2 = 0.4$ (Figure 2c), a higher penalty is required for $a_1$ to stay optimal. To capture this relationship between the stochasticity of an environment and the required penalty to obtain safe policies, we introduce a notion of *controllability*, which measures the ability of an agent to reach safe goals. Additionally, observe that as $p_2$ increases, the probability that the agent can transition from $s_2$ to <mark>s<sub>3</sub></mark> decreases—thereby increasing the number of timesteps spent to reach the goal. Therefore, the penalty for <mark>s<sub>1</sub></mark> must also consider the environment's *diameter* to ensure an optimal policy will not simply reach <mark>s<sub>1</sub></mark> to avoid self-transitions in $s_2$.

### 3.2 On the Diameter and Controllability of Environments

Clearly, the size of the penalty that needs to be given for unsafe states depends on the *size* of the environment. We define this size as the *diameter* of the environment, which is the highest expected timesteps to reach an absorbing state from an internal state when following a proper policy:

**Definition 3** *Define the diameter of an environment as $D := \max_{s \in S \setminus G} \max_{\pi \in \Pi} E \left[ T(s_T \in G | \pi) \right]$, where $T(s_T \in G | \pi)$ is the timesteps taken to reach $G$ from $s$ when following a proper policy $\pi$.*

Given the diameter of an environment, a possible natural choice for the reward for unsafe states is to give a penalty that is as large as receiving the smallest task reward for the longest path to safe goal states: $\bar{R}_{MAX} := R_{MIN} D'$, where $D'$ is the diameter for safe policies $D' := \max_{s \in S \setminus G} \max_{\pi \in \Pi} E \left[ T(s_T \in G \setminus G^! | \pi) \right]$.

However, while $\bar{R}_{MAX}$ aims to make reaching unsafe states worse than reaching safe goals, it does not consider the controllability of an environment, nor the possibility that an unsafe policy receives $R_{MAX}$ everywhere in its trajectory. We can formally define the controllability of an environment as follows:

**Definition 4** *Define the degree of controllability as $C := \min_{s \in S \setminus G} \min_{\substack{\pi \in \Pi \\ P_s^\pi(s_T \notin G^!) \neq 0}} P_s^\pi(s_T \notin G^!)$.*

$C$ measures the degree of controllability of the environment by simply taking the smallest non-zero probability of reaching safe goal states by following a proper policy. For example, if the dynamics are deterministic, then any deterministic policy $\pi$ will either reach a safe goal or not. That is, $P_s^\pi(s_T \notin G^!)$ will either be 0 or 1. Since we require $P_s^\pi(s_T \notin G^!) \neq 0$, it must be that $C = 1$. Consider, for example, the chain-walk environment with different choices for $p$. Since actions in $s_2$ do not affect the transition probability, there are only 2 relevant deterministic policies $\pi_1(s) = a_1$ and $\pi_2(s) = a_2$. This gives $P_{s_1}^{\pi_1}(s_T \notin G^!) = (1 - p_1)1(p_2 = 1)$ and $P_{s_1}^{\pi_2}(s_T \notin G^!) = p_11(p_2 = 1)$. Here, $C = 1$ when $p_1 = p_2 = 0$ because the task is deterministic and <mark>s<sub>3</sub></mark> is reachable. $C$ then tends to 0.5 as $p_1$ and $p_2$ gets closer to 0.5, making the environment uniformly random. Finally, the environment is not controllable when $p = 1$ since <mark>s<sub>3</sub></mark> is unreachable from $s_2$.

**Remark 2** *We can think of $C = 0$ as the limit of $C$ when safe goals are unreachable.*

Given the diameter and controllability of an environment, we can now define a choice for the Minmax penalty that takes into account both $D$, $C$, and $R_{MAX}$: $\bar{R}_{MIN} := (R_{MIN} - R_{MAX}) \frac{D}{C}$. This choice of penalty says that since stochastic shortest path tasks require an agent to learn to achieve desired terminal states, if the agent enters an unsafe terminal state, it should receive the largest penalty possible by a proper policy. We now investigate the effect of these penalties on the failure rate of optimal policies.

### 3.3 On the Failure Rate of Optimal Policies

We begin by proposing a simple model-based algorithm for estimating the diameter and controllability, from which the penalties are then obtained. We describe the method here and present the pseudo-code in **Algorithm**

4

Reinforcement Learning Conference (August 2024)


<table>
  <tbody>
    <tr>
      <td><b>(a) $R_{step} = -1$</b></td>
      <td><b>(b) $R_{step} = -1$</b></td>
      <td><b>(c) $R_{step} = -1$</b></td>
      <td><b>(d) $p_1=p_2=0.4$</b></td>
      <td><b>(e) $p_1=0, p_2=0.4$</b></td>
      <td><b>(f) $p_1=0.4, p_2=0$</b></td>
    </tr>
    <tr>
      <td>\begin{tabular}{@{}c@{}} \includegraphics[width=0.15\textwidth]{placeholder_a.png}</td>
    <td></td><td></td><td></td><td></td><td></td></tr>
    <tr>
      <td>$p_1 = p_2 \in [0 \ 1]$ \end{tabular}</td>
      <td>\begin{tabular}{@{}c@{}} \includegraphics[width=0.15\textwidth]{placeholder_b.png}</td>
    <td></td><td></td><td></td><td></td></tr>
    <tr>
      <td>$p_1 = 0, p_2 \in [0 \ 1]$ \end{tabular}</td>
      <td>\begin{tabular}{@{}c@{}} \includegraphics[width=0.15\textwidth]{placeholder_c.png}</td>
    <td></td><td></td><td></td><td></td></tr>
    <tr>
      <td>$p_2 = 0, p_1 \in [0 \ 1]$ \end{tabular}</td>
      <td>\begin{tabular}{@{}c@{}} \includegraphics[width=0.15\textwidth]{placeholder_d.png}</td>
    <td></td><td></td><td></td><td></td></tr>
    <tr>
      <td>$R_{step} \in [-1 \ 0]$ \end{tabular}</td>
      <td>\begin{tabular}{@{}c@{}} \includegraphics[width=0.15\textwidth]{placeholder_e.png}</td>
    <td></td><td></td><td></td><td></td></tr>
    <tr>
      <td>$R_{step} \in [-1 \ 0]$ \end{tabular}</td>
      <td>\begin{tabular}{@{}c@{}} \includegraphics[width=0.15\textwidth]{placeholder_f.png}</td>
    <td></td><td></td><td></td><td></td></tr>
    <tr>
      <td>$R_{step} \in [-1 \ 0]$ \end{tabular}</td>
    <td></td><td></td><td></td><td></td><td></td></tr>
  </tbody>
</table>

Figure 3: Failure rates of optimal policies in the chain-walk environment. We show the effect of stochasticity ($p_1$ and $p_2$) and task rewards ($R_{step}$) on the bounds ($\bar{R}_{MIN}$ and $\bar{R}_{MAX}$) of the Minmax penalty ($R_{Minmax}$). The controllability and diameter for the bounds are estimated using Algorithm 1.

1 in Appendix B. Here, the diameter is estimated as follows: (i) For each deterministic policy $\pi$, estimate its expected timesteps $T(s_T \in G)$ (or $T(s_T \in G \setminus G^!)$ for $D'$) by using policy evaluation (Sutton & Barto, 2018) with rewards of 1 at all internal states; (ii) Then, calculate $D$ using the equation in Definition 3. Similarly, the controllability is estimated by estimating the reach probability $P_s^\pi(s_T \notin G^!)$ of each deterministic policy $\pi$ using rewards of 1 for transitions into safe goal states and zero rewards otherwise. This approach converges via the convergence of policy evaluation (**Theorem 1**).

**Theorem 1 (Estimation)** *Algorithm 1 converges to $D$ and $C$ for any given controllable environment.*

Figure 3 shows the result of applying this algorithm in the chain-walk MDP. Here, $R_{Minmax}$ is compared to accounting for $D$ only ($\bar{R}_{MAX}$) and accounting for both $C$ and $D$ ($\bar{R}_{MIN}$). Interestingly, we can observe $\bar{R}_{MIN} \le R_{Minmax}$ and $\bar{R}_{MAX} \ge R_{Minmax}$ consistently, highlighting how considering the diameter only is insufficient to guarantee safe optimal policies. It also indicates that these penalties may bound $R_{Minmax}$ in general. We show in **Theorem 2** that this is indeed the case.

**Theorem 2 (Safety Bounds)** *Consider a controllable environment where task rewards are bounded by $[R_{MIN} \ R_{MAX}]$ for all $s' \notin G^!$. Then $\bar{R}_{MIN} \le R_{Minmax} \le \bar{R}_{MAX}$.*

Theorem 2 says that for any MDP whose rewards for unsafe transitions are bounded above by $\bar{R}_{MIN}$, the optimal policy both minimises the probability of reaching unsafe states and maximises the probability of reaching safe goal states. Hence, any penalty $\bar{R}_{MIN} - \epsilon$, where $\epsilon > 0$ can be arbitrarily small, will guarantee safe optimal policies. Similarly, the theorem shows that any reward higher than $\bar{R}_{MAX}$ may have optimal policies that do not minimise the probability of reaching unsafe states. These can be observed in Figure 3. The figure demonstrates why considering both the diameter and controllability of an MDP is necessary to guarantee safe policies, because the diameter alone does not always minimise the failure rate.

## 4 Practical Algorithm for Learning Safe Policies

While the Minmax penalty of an MDP can be accurately estimated using policy evaluation (Algorithm 1), it requires knowledge of the environment dynamics (or an estimate of it). These are difficult quantities to estimate from an agent's experience, which is further complicated by the need to also learn the true optimal policy for the estimated Minmax penalty. Hence, obtaining an accurate estimate of the Minmax penalty is impractical in model-free and function approximation settings where the state and action spaces are large. In fact, it is NP-hard since it depends on the diameter, which requires solving a longest-path problem.

**Theorem 3 (Complexity)** *Estimating the Minmax penalty $R_{Minmax}$ accurately is NP-hard.*

Given the above challenges, we require a practical method for learning the Minmax penalty. Ideally, this method should require no knowledge of the environment dynamics and should easily integrate with existing RL approaches. To achieve this, we first note that $(R_{MIN} - R_{MAX}) \frac{D}{C} = (DR_{MIN} - DR_{MAX}) \frac{1}{C} = (V_{MIN} - V_{MAX}) \frac{1}{C}$, where $V_{MIN}$ and $V_{MAX}$ are the value function bounds. Hence, a practical estimate of the Minmax penalty can be efficiently learned by estimating the value gap $V_{MIN} - V_{MAX}$ using observations of the reward and the agent's

5

Reinforcement Learning Conference (August 2024)

estimate of the value function. We describe the method here and present the pseudo-code in **Algorithm 2** in Appendix **B**. This algorithm requires initial estimates of $R_{MIN}$ and $R_{MAX}$, which in this work are initialised to 0. The agent receives a reward $r_t$ after each environment interaction and updates its estimate of the reward bounds $R_{MIN} \leftarrow \min(R_{MIN}, r_t)$ and $R_{MAX} \leftarrow \max(R_{MAX}, r_t)$, the value bounds $V_{MIN} \leftarrow \min(V_{MIN}, R_{MIN}, V(s_t))$ and $V_{MAX} \leftarrow \max(V_{MAX}, R_{MAX}, V(s_t))$, and the Minmax penalty $\bar{R}_{MIN} \leftarrow V_{MIN} - V_{MAX}$, where $V(s_t)$ is the learned value function at time step $t$. Since the controllability $C$ is also expensive to estimate, it is not explicitly considered in this estimate of $\bar{R}_{MIN}$. Instead, given that the main purpose of $C$ is to make $\bar{R}_{MIN}$ more negative the more stochastic the environment is, we notice that this is already achieved in practice by the reward and value estimates. Since $R_{MIN}$ is estimated using $R_{MIN} \leftarrow \min(R_{MIN}, r_t)$, then every time the agent enters an unsafe state, we have that: $r_t \leftarrow \bar{R}_{MIN}$, $R_{MIN} \leftarrow \bar{R}_{MIN}$, and then $\bar{R}_{MIN} \leftarrow \bar{R}_{MIN} - V_{MAX}$. This means that when the estimated $V_{MAX}$ is greater than zero, the penalty estimate $\bar{R}_{MIN}$ become more negative every time the agent enters an unsafe state. **Finally, whenever an agent encounters an unsafe state, the reward $r_t$ is replaced by $\bar{R}_{MIN}$ to disincentivise unsafe behaviour.** Since $V_{MAX}$ is estimated using $V_{MAX} \leftarrow \max(V_{MAX}, R_{MAX}, V(s_t))$, it leads to an optimistic estimation of $\bar{R}_{MIN}$. Hence, we observe no need to add an $\epsilon > 0$ to $\bar{R}_{MIN}$.

## 5 Experiments

While the theoretical Minmax penalty is guaranteed to lead to optimal safe policies, it is unclear whether this also holds for the practical estimate proposed in Section 4. Hence, this section aims to investigate three main natural questions regarding the proposed practical algorithm (see Appendix **D** for additional experiments): How does Algorithm 2 (i) behave when the theoretical assumptions are satisfied? (ii) behave when the theoretical assumptions are *not* satisfied? (iii) compare to prior approaches towards Safe RL? For each result, we report the mean (solid line) and one standard deviation around it (shaded region).

### 5.1 How does Algorithm 2 behave when the theoretical assumptions are satisfied?

**Domain (LAVA GRIDWORLD)** This is a simple gridworld environment with 11 positions ($|S| = 11$) and 4 cardinal actions ($|A| = 4$). The agent here must reach a goal location $G$ while avoiding a lava location $L$ (hence $G = \{L, G\}$ and $G^! = \{L\}$). A wall is also present in the environment and, while not unsafe, must be navigated around. The environment has a *slip probability* ($sp$), so that with probability $sp$ the agent’s action is overridden with a random action. The agent receives $R_{MAX} = +1$ reward for reaching the goal, as well as $R_{step} = -0.1$ reward at each timestep to incentivise taking the shortest path to the goal. To test our approach, we modify Q-learning (Watkins, 1989) with $\epsilon$-greedy exploration such that the agent updates its estimate of the Minmax penalty as learning progresses and uses it as the reward whenever the lava state is reached, following the procedure outlined in Section 4. The action-value function is initialised to 0 for all states and actions, $\epsilon = 0.1$ and the learning rate $\alpha = 0.1$. The experiments are run over 10,000 episodes and averaged over 70 runs.

**Setup and Results** We examine the performance of our modified Q-learning approach across three values of the slip probability of the LAVA GRIDWORLD. A slip probability of 0 represents a fully deterministic environment, while a slip probability of 0.5 represents a more stochastic environment. Results are plotted in Figure 4. In the case of the fully deterministic environment, the Minmax penalty bound obtained via Algorithm 1 is $\bar{R}_{MIN} = -9.9$, since $C = 1$ and $D = 9$. However, the agent is able to learn a relatively smaller penalty ($-1.1$ in Figure 4b) to consistently minimise failure rate and maximise returns (Figures 4c and 4d). The resulting optimal policy then chooses the shorter path that passes near the lava location ($sp = 0$ in Figure 4a). As the stochasticity of the environment increases, a larger penalty is learned to incentivise longer, safer policies. Given the starting position of the agent next to the lava, the failure rate inevitably increases with increased stochasticity. The resulting optimal policy then chooses the longer path that passes to the left of the centre wall ($sp = 0.25$ and $sp = 0.5$ in Figure 4a). We can, therefore, conclude that while there is a gap between the true Minmax penalty and the one learned via Algorithm 2, this algorithm can still learn optimal safe policies when the theoretical setting holds.

### 5.2 How does Algorithm 2 behave when the theoretical assumptions are not satisfied?

**Domain (Safety Gym PILLAR)** This is a custom Safety Gym environment (Ray et al., 2019), in which the simple point robot must navigate to a goal location <mark>🟢</mark> around a large pillar <mark>🔵</mark> (hence $G = \{🔵, 🟢\}$ and

6

Reinforcement Learning Conference (August 2024)


**Figure 4:** Effect of increase in the slip probability of the LAVA GRIDWORLD on the learned Minmax penalty and corresponding failure rate and returns. The black circle in (a) represents the agent.

<table>
    <tr>
        <th>(a) Trajectories</th>
        <th>(b) Learned penalty</th>
        <th>(c) Failure rate</th>
        <th>(d) Average returns</th>
    </tr>
    <tr>
        <td>**sp=0.5**</td>
        <td>**penalties**</td>
        <td>**failures**</td>
        <td>**returns**</td>
    </tr>
    <tr>
        <td>**sp=0.25**</td>
        <td>**sp=0**</td>
        <td>**sp=0**</td>
        <td>**sp=0**</td>
    </tr>
    <tr>
        <td>**sp=0**</td>
        <td>**sp=0.25**</td>
        <td>**sp=0.25**</td>
        <td>**sp=0.25**</td>
    </tr>
    <tr>
        <td></td>
        <td>**sp=0.5**</td>
        <td>**sp=0.5**</td>
        <td>**sp=0.5**</td>
    </tr>
    <tr>
        <td></td>
        <td>0.0 0.2 0.4 0.6 0.8 1.0 1e5</td>
        <td>0.0 0.2 0.4 0.6 0.8 1.0 1e5</td>
        <td>0.0 0.2 0.4 0.6 0.8 1.0 1e5</td>
    </tr>
    <tr>
        <td></td>
        <td>episode</td>
        <td>episode</td>
        <td>episode</td>
    </tr>
</table>$$ G^! = \{ O \} $$
Just as in Ray et al. (2019), the agent uses *pseudo-lidar* to observe the distance to objects around it ($|S| = R^{60}$), and the action space is continuous over two actuators controlling the direction and forward velocity ($|A| = R^2$). The goal, pillar, and agent locations remain unchanged for all episodes. The discount factor is $\gamma = 0.99$, and the agent is rewarded for reaching the goal (with a reward of 1) as well as for moving towards it (the default dense distance-based reward). Each episode terminates once the agent reaches the goal or collides with the pillar (with a reward of $-1$). Otherwise, episodes terminate after 1000 timesteps. This domain does not satisfy the shortest path setting we assume since: it is discounted, optimal policies are not guaranteed to reach $G$ and policies that do not reach $G$ are not guaranteed to have value functions that are unbounded from below (due to the dense rewards). To test our approach in this setting, we modify TRPO (Schulman et al., 2015) (denoted TRPO-Minmax) to use the estimate of the Minmax penalty as described in Algorithm 2. The experiments are run over 10 million steps and averaged over 10 runs.

**Setup and Results** We examine the performance of TRPO-Minmax for five levels of noise in the PILLAR environment, similarly to the experiments in Section 5.1. Here, the value of the noise denotes the number by which a random action vector is scaled before vector addition with the agent’s action. Results are plotted in Figure 5. We observe similar results to Section 5.1, where the agent uses its learned Minmax penalty (Figure 5b) to successfully learn safe policies (Figure 5c) while solving the task (Figure 5d), using safer paths for more noisy dynamics (Figure 5a). Interestingly, it also correctly prioritises low failure rates when the dynamics are too noisy to safely reach the goal ($noise \ge 5$). We can, therefore, conclude that Algorithm 2 can learn safe policies even in discounted high-dimensional continuous-control domains requiring function approximation.


**Figure 5:** Performance of TRPO-Minmax in the PILLAR environment with varying noise levels.

<table>
    <tr>
        <th>(a) Trajectories</th>
        <th>(b) Learned penalty</th>
        <th>(c) Failure rate</th>
        <th>(d) Average returns</th>
    </tr>
    <tr>
        <td>**noise=5**</td>
        <td>**AveragePenalty**</td>
        <td>**AverageEpCost**</td>
        <td>**AverageEpRet**</td>
    </tr>
    <tr>
        <td>**noise=0**</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>**noise=2.5**</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>2 4 6 8 1e6</td>
        <td>2 4 6 8 1e6</td>
        <td>2 4 6 8 1e6</td>
    </tr>
    <tr>
        <td></td>
        <td>TotalEnvInteracts</td>
        <td>TotalEnvInteracts</td>
        <td>TotalEnvInteracts</td>
    </tr>
</table>### 5.3 How does Algorithm 2 compare to prior approaches towards Safe RL?

**Baselines** As a baseline representative of typical RL approaches, we use Trust Region Policy Optimisation (TRPO) (Schulman et al., 2015). To represent constraint-based approaches, we compare against Constrained Policy Optimisation (CPO) (Achiam et al., 2017), TRPO with Lagrangian constraints (TRPO-Lagrangian) (Ray et al., 2019), and Sauté RL with TRPO (Sauté-TRPO) (Sootla et al., 2022). All baselines except Sauté-TRPO use the implementations provided by Ray et al. (2019), and form a set of widely used baselines in safety domains (Zhang et al., 2020; Sootla et al., 2022; Yang et al., 2023). Sauté-TRPO uses the implementation provided by Sootla et al. (2022). As in Ray et al. (2019), all approaches use feed-forward MLPs, value

7

Reinforcement Learning Conference (August 2024)

networks of size (256,256), and *tanh* activation functions. The cost threshold for the constrained algorithms is set to 0, the best we found. The experiments are run over 10 million episodes and averaged over 10 runs.

**Setup and Results** We compare the performance of TRPO-Minmax to that of the baselines for different levels of noise in the PILLAR domain. Figure 6 shows the results. We observe that in the deterministic case $noise = 0$, all the algorithms achieve similar performance (except Sauté-TRPO), successfully maximising returns (Figure 6d **top**) while minimising the failure rates (Figure 6c **top**). However, in the stochastic case $noise = 2.5$, we can observe that all the baselines except Sauté-TRPO achieve significantly high returns (Figure 6d **bottom**) at the expense of a rapidly increasing cumulative cost (Figure 6b **bottom**). These results are also consistent with the benchmarks of Ray et al. (2019) where the cumulative cost of TRPO is greater than that of TRPO-Lagrangian, which is greater than that of CPO. Interestingly, Sauté-TRPO is the worst-performing of all the baselines. It successfully maximises returns while minimising cost only for the deterministic environment ($noise = 0$), but completely fails for the stochastic one ($noise = 2.5$). Finally, by examining the episode length (Figure 6a) and failure rates (Figure 6c) for all the baselines in the stochastic case, we can conclude that they have all learned risky policies that maximise rewards over short trajectories that are highly likely to result in collisions. We also provide additional results in the appendix for $noise \ge 5$ (Figures 9-11) to further demonstrate this point. In contrast, the results obtained show that TRPO-Minmax successfully maximises returns while minimising cost for both deterministic and stochastic environments. In addition, when the noise level is too high $noise \ge 5$, TRPO-Minmax consistently prioritises maintaining low failure rates over maximising returns.

<center>
<table>
  <tr>
    <td style="text-align:center"><span style="color:purple">■</span> TRPO</td>
    <td style="text-align:center"><span style="color:blue">■</span> TRPO Lagrangian</td>
    <td style="text-align:center"><span style="color:green">■</span> CPO</td>
    <td style="text-align:center"><span style="color:brown">■</span> SauteTRPO</td>
    <td style="text-align:center"><span style="color:red">■</span> TRPO Minmax (Ours)</td>
  </tr>
</table>
</center>

<center>
<table>
  <tr>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">1000</td></tr>
<tr><td style="text-align:right">800</td></tr>
<tr><td style="text-align:right">600</td></tr>
<tr><td style="text-align:right">400</td></tr>
<tr><td style="text-align:right">200</td></tr>
<tr><td style="text-align:right">0</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">EpLen</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">7000</td></tr>
<tr><td style="text-align:right">6000</td></tr>
<tr><td style="text-align:right">5000</td></tr>
<tr><td style="text-align:right">4000</td></tr>
<tr><td style="text-align:right">3000</td></tr>
<tr><td style="text-align:right">2000</td></tr>
<tr><td style="text-align:right">1000</td></tr>
<tr><td style="text-align:right">0</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">CumulativeCost</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">1.0</td></tr>
<tr><td style="text-align:right">0.8</td></tr>
<tr><td style="text-align:right">0.6</td></tr>
<tr><td style="text-align:right">0.4</td></tr>
<tr><td style="text-align:right">0.2</td></tr>
<tr><td style="text-align:right">0.0</td></tr>
<tr><td style="text-align:right">-0.2</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">AverageEpCost</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">4</td></tr>
<tr><td style="text-align:right">3</td></tr>
<tr><td style="text-align:right">2</td></tr>
<tr><td style="text-align:right">1</td></tr>
<tr><td style="text-align:right">0</td></tr>
<tr><td style="text-align:right">-1</td></tr>
<tr><td style="text-align:right">-2</td></tr>
<tr><td style="text-align:right">-3</td></tr>
<tr><td style="text-align:right">-4</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">AverageEpRet</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
  </tr>
<tr>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">1100</td></tr>
<tr><td style="text-align:right">1000</td></tr>
<tr><td style="text-align:right">900</td></tr>
<tr><td style="text-align:right">800</td></tr>
<tr><td style="text-align:right">700</td></tr>
<tr><td style="text-align:right">600</td></tr>
<tr><td style="text-align:right">500</td></tr>
<tr><td style="text-align:right">400</td></tr>
<tr><td style="text-align:right">300</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">EpLen</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">12000</td></tr>
<tr><td style="text-align:right">10000</td></tr>
<tr><td style="text-align:right">8000</td></tr>
<tr><td style="text-align:right">6000</td></tr>
<tr><td style="text-align:right">4000</td></tr>
<tr><td style="text-align:right">2000</td></tr>
<tr><td style="text-align:right">0</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">CumulativeCost</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">0.8</td></tr>
<tr><td style="text-align:right">0.6</td></tr>
<tr><td style="text-align:right">0.4</td></tr>
<tr><td style="text-align:right">0.2</td></tr>
<tr><td style="text-align:right">0.0</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">AverageEpCost</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
    <td style="text-align:center">
      <table>
        <tr><td style="text-align:right">3</td></tr>
<tr><td style="text-align:right">2</td></tr>
<tr><td style="text-align:right">1</td></tr>
<tr><td style="text-align:right">0</td></tr>
<tr><td style="text-align:right">-1</td></tr>
<tr><td style="text-align:right">-2</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">AverageEpRet</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">2</td></tr>
<tr><td style="text-align:center">4</td></tr>
<tr><td style="text-align:center">6</td></tr>
<tr><td style="text-align:center">8</td></tr>
<tr><td style="text-align:center">1e6</td></tr>
      </table>
      <br>
      <table>
        <tr><td style="text-align:center">TotalEnvInteracts</td></tr>
      </table>
    </td>
  </tr>
</table>
</center>

(a) Episode length (b) Cumulative cost (c) Failure rate (d) Average returns

Figure 6: Comparison with baselines in the PILLAR environment. (**top**) $noise = 0$, (**bottom**) $noise = 2.5$.

## 6 Discussion and Future Work

This paper investigates a new approach towards safe RL by asking the question: *Is a scalar reward enough to solve tasks safely?* To answer this question, we bound the Minmax penalty, which takes into account the diameter and controllability of an environment in order to minimise the probability of encountering unsafe states. We prove that the penalty does indeed minimise this probability, and present a method that uses an agent’s value estimates to learn an estimate of the penalty. Our results in tabular and high-dimensional continuous settings have demonstrated that, by encoding the safe behaviour directly in the reward function via the Minmax penalty, agents are able to solve tasks while prioritising safety, learning safer policies than popular constraint-based approaches. Finally, while we show that scalar rewards are indeed enough for safe RL, the current analysis is only applicable to unsafe terminal states—which only covers tasks that can be naturally represented by stochastic-shortest path MDPs. Given that other popular RL settings like discounted MDPs can be converted to stochastic shortest path MDPs (Bertsekas, 1987; Sutton & Barto, 1998), a promising future direction could be to find the dual of our results for other theoretically equivalent settings. In conclusion, we see this reward-only approach as a promising direction towards truly autonomous agents capable of independently learning to solve tasks safely.

8

Reinforcement Learning Conference (August 2024)

# References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In *International Conference on Machine Learning*, pp. 22–31. PMLR, 2017.

Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum, and Ufuk Topcu. Safe reinforcement learning via shielding. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 32, 2018.

Eitan Altman. *Constrained Markov decision processes: stochastic modeling*. Routledge, 1999.

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. *arXiv preprint arXiv:1606.06565*, 2016.

Dimitri P Bertsekas. *Dynamic Programming: Determinist. and Stochast. Models*. Prentice-Hall, 1987.

Dimitri P Bertsekas and John N Tsitsiklis. An analysis of stochastic shortest path problems. *Mathematics of Operations Research*, 16(3):580–595, 1991.

Yinlam Chow, Ofir Nachum, Edgar Duenez-Guzman, and Mohammad Ghavamzadeh. A Lyapunov-based approach to safe reinforcement learning. *Advances in Neural Information Processing Systems*, 31, 2018.

Gal Dalal, Krishnamurthy Dvijotham, Matej Vecerik, Todd Hester, Cosmin Paduraru, and Yuval Tassa. Safe exploration in continuous action spaces. *arXiv preprint arXiv:1801.08757*, 2018.

Rati Devidze, Goran Radanovic, Parameswaran Kamalaruban, and Adish Singla. Explicable reward design for reinforcement learning agents. *Advances in Neural Information Processing Systems*, 34:20118–20131, 2021.

Aria HasanzadeZonuzy, Archana Bura, Dileep Kalathil, and Srinivas Shakkottai. Learning with safety constraints: Sample complexity of reinforcement learning for constrained MDPs. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pp. 7667–7674, 2021.

Gregory Kahn, Adam Villaflor, Bosen Ding, Pieter Abbeel, and Sergey Levine. Self-supervised deep reinforcement learning with generalized computation graphs for robot navigation. In *2018 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 5129–5136. IEEE, 2018.

Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In *Conference on Robot Learning*, pp. 651–673. PMLR, 2018.

B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A Al Sallab, Senthil Yogamani, and Patrick Pérez. Deep reinforcement learning for autonomous driving: A survey. *IEEE Transactions on Intelligent Transportation Systems*, 2021.

Zachary C Lipton, Kamyar Azizzadenesheli, Abhishek Kumar, Lihong Li, Jianfeng Gao, and Li Deng. Combating reinforcement learning’s Sisyphean curse with intrinsic fear. *arXiv preprint arXiv:1611.01211*, 2016.

Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In *International Conference on Machine Learning*, volume 99, pp. 278–287, 1999.

Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking Safe Exploration in Deep Reinforcement Learning. 2019.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In *International Conference on Machine Learning*, pp. 1889–1897. PMLR, 2015.

Kun Shao, Zhentao Tang, Yuanheng Zhu, Nannan Li, and Dongbin Zhao. A survey of deep reinforcement learning in video games. *arXiv preprint arXiv:1912.10944*, 2019.

9

Reinforcement Learning Conference (August 2024)

Satinder Singh, Richard L Lewis, and Andrew G Barto. Where do rewards come from? In *Proceedings of the Annual Conference of the Cognitive Science Society*, pp. 2601–2606. Cognitive Science Society, 2009.

Aivar Sootla, Alexander I Cowen-Rivers, Taher Jafferjee, Ziyan Wang, David H Mguni, Jun Wang, and Haitham Ammar. Sauté RL: Almost surely safe reinforcement learning using state augmentation. In *International Conference on Machine Learning*, pp. 20423–20443. PMLR, 2022.

Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive safety in reinforcement learning by PID Lagrangian methods. In *International Conference on Machine Learning*, pp. 9133–9143. PMLR, 2020.

Richard Sutton and Andrew Barto. *Introduction to reinforcement learning*, volume 135. MIT press Cambridge, 1998.

Richard Sutton and Andrew Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

Guy Tennenholtz, Nadav Merlis, Lior Shani, Shie Mannor, Uri Shalit, Gal Chechik, Assaf Hallak, and Gal Dalal. Reinforcement learning with a terminator. *Advances in Neural Information Processing Systems*, 35: 35696–35709, 2022.

Benjamin Van Niekerk, Steven James, Adam Earle, and Benjamin Rosman. Composing value functions in reinforcement learning. In *International Conference on Machine Learning*, pp. 6401–6409. PMLR, 2019.

Nolan C Wagener, Byron Boots, and Ching-An Cheng. Safe reinforcement learning using advantage-based intervention. In *International Conference on Machine Learning*, pp. 10630–10640. PMLR, 2021.

C. Watkins. *Learning from delayed rewards*. PhD thesis, King’s College, Cambridge, 1989.

Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, and Peter J Ramadge. Projection-based constrained policy optimization. *arXiv preprint arXiv:2010.03152*, 2020.

Yujie Yang, Yuxuan Jiang, Yichen Liu, Jianyu Chen, and Shengbo Eben Li. Model-free safe reinforcement learning through neural barrier certificate. *IEEE Robotics and Automation Letters*, 8(3):1295–1302, 2023.

Yiming Zhang, Quan Vuong, and Keith Ross. First order constrained optimization in policy space. *Advances in Neural Information Processing Systems*, 33:15338–15349, 2020.

10

Reinforcement Learning Conference (August 2024)

# A Proofs of Theoretical Results

**Theorem 1 (Estimation)** *Algorithm 1 converges to $D$ and $C$ for any given controllable environment.*

**Proof** This follows from the convergence guarantee of policy evaluation (Sutton & Barto, 1998). ■

**Theorem 2 (Safety Bounds)** *Consider a controllable environment where task rewards are bounded by $[R_{MIN} \ R_{MAX}]$ for all $s' \notin G^!$. Then $\bar{R}_{MIN} \le R_{Minmax} \le \bar{R}_{MAX}$.*

**Proof** Let $\pi^*$ be an optimal policy for an arbitrary task $\langle S, A, P, R \rangle$ in the environment. Given the definition of the Minmax penalty (Definition 2), we need to show the following:

(i) If $R(s, a, s') < \bar{R}_{MIN}$ for all $s' \in G^!$, then $\pi^*$ is safe for all $R$; and
(ii) If $R(s, a, s') > \bar{R}_{MAX}$ for some $s' \in G^!$ reachable from $S \setminus G$, then there exists an $R$ s.t. $\pi^*$ is unsafe.

(i) Since $\pi^*$ is optimal, it is also proper and hence must reach $G$.

Assume $\pi^*$ is unsafe. Then there exists another proper policy $\pi$ that is safe, such that
$$P_s^\pi(s_T \in G^!) < P_s^{\pi^*}(s_T \in G^!) \quad for some  s \in S.$$

Then,
$$V^{\pi^*}(s) \ge V^\pi(s)$$
$$\implies E_s^{\pi^*} \left[ \sum_{t=0}^\infty R(s_t, a_t, s_{t+1}) \right] \ge E_s^\pi \left[ \sum_{t=0}^\infty R(s_t, a_t, s_{t+1}) \right]$$
$$\implies E_s^{\pi^*} \left[ G^{T-1} + R(s_T, a_T, s_{T+1}) \right] \ge E_s^\pi \left[ G^{T-1} + R(s_T, a_T, s_{T+1}) \right],$$
where $G^{T-1} = \sum_{t=0}^{T-1} R(s_t, a_t, s_{t+1})$ and $T$ is a random variable denoting when $s_{T+1} \in G$.
$$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \notin G^!) R(s_T, a_T, s_{T+1}) + P_s^{\pi^*}(s_T \in G^!) \bar{R}_{unsafe}(s_T, a_T, s_{T+1}) \right)$$
$$\ge E_s^\pi \left[ G^{T-1} \right] + \left( P_s^\pi(s_T \notin G^!) R(s_T, a_T, s_{T+1}) + P_s^\pi(s_T \in G^!) \bar{R}_{unsafe}(s_T, a_T, s_{T+1}) \right),$$
where $\bar{R}_{unsafe}$ denotes the rewards for transitions into $G^!$ and $a_T = \pi^*(s_T)$.
$$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \notin G^!) R(s_T, a_T, s_{T+1}) + \bar{R}_{unsafe}(s_T, a_T, s_{T+1}) \right)$$
$$\ge E_s^\pi \left[ G^{T-1} \right] + \left( P_s^\pi(s_T \notin G^!) R(s_T, a_T, s_{T+1}) + P_s^\pi(s_T \in G^!) \bar{R}_{unsafe}(s_T, a_T, s_{T+1}) \right),$$
$$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + \left( 1 - P_s^\pi(s_T \in G^!) \right) \bar{R}_{unsafe}(s_T, a_T, s_{T+1})$$
$$\ge E_s^\pi \left[ G^{T-1} \right] + \left( P_s^\pi(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1})$$
$$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + \left( 1 - P_s^\pi(s_T \in G^!) \right) \bar{R}_{MIN}$$
$$> E_s^\pi \left[ G^{T-1} \right] + \left( P_s^\pi(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1}),$$
since $\bar{R}_{unsafe}(s_T, a_T, s_{T+1}) < \bar{R}_{MIN}$.
$$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + \left( 1 - P_s^\pi(s_T \in G^!) \right) (R_{MIN} - R_{MAX}) \frac{D}{C}$$
$$> E_s^\pi \left[ G^{T-1} \right] + \left( P_s^\pi(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1})$$
$$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + (R_{MIN} - R_{MAX})D$$
$$> E_s^\pi \left[ G^{T-1} \right] + \left( P_s^\pi(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1}),  using definition of  C.$$

11

Reinforcement Learning Conference (August 2024)

$\implies E_s^{\pi^*} \left[ G^{T-1} \right] - R_{MAX} D$
$> E_s^{\pi} \left[ G^{T-1} \right] + \left( P_s^{\pi}(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1}) - R_{MIN} D$
$\implies E_s^{\pi^*} \left[ G^{T-1} \right] - R_{MAX} D > 0,$
since $E_s^{\pi} \left[ G^{T-1} \right] + \left( P_s^{\pi}(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1}) \ge R_{MIN} D$
$\implies E_s^{\pi^*} \left[ G^{T-1} \right] > R_{MAX} D.$

But this is a contradiction since the expected return of following an optimal policy up to a terminal state without the reward for entering the terminal state must be less than receiving $R_{MAX}$ for every step of the longest possible trajectory to $G$. Hence we must have $\pi^* \in \arg \min_{\pi} P_s^{\pi}(s_T \in G^!)$.

(ii) Assume $\pi^*$ is safe. Then, $P_s^{\pi^*}(s_T \notin G^!) \ge P_s^{\pi'}(s_T \notin G^!)$ for all $s \in S$, $\pi' \in \Pi$.
Let $\pi$ be the policy that maximises the probability of reaching $s' \in G^!$ from some state $s \in G$. Then, similarly to (i), we have

$V^{\pi^*}(s) \ge V^{\pi}(s)$
$\implies E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \in G^!) - P_s^{\pi}(s_T \in G^!) \right) \bar{R}_{unsafe}(s_T, a_T, s_{T+1})$
$\ge E_s^{\pi} \left[ G^{T-1} \right] + \left( P_s^{\pi}(s_T \notin G^!) - P_s^{\pi^*}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1})$
$\implies E_s^{\pi} \left[ G^{T-1} \right] + \left( P_s^{\pi}(s_T \in G^!) - P_s^{\pi^*}(s_T \in G^!) \right) \bar{R}_{unsafe}(s_T, a_T, s_{T+1})$
$\le E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \notin G^!) - P_s^{\pi}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1})$
$\implies E_s^{\pi} \left[ G^{T-1} \right] + \left( P_s^{\pi}(s_T \in G^!) - P_s^{\pi^*}(s_T \in G^!) \right) \bar{R}_{MAX}$
$< E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \notin G^!) - P_s^{\pi}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1}),  since  \bar{R}_{unsafe} > \bar{R}_{MAX}.$
$\implies E_s^{\pi} \left[ G^{T-1} \right] + \left( P_s^{\pi}(s_T \in G^!) - P_s^{\pi^*}(s_T \in G^!) \right) R_{MIN} D'$
$< E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \notin G^!) - P_s^{\pi}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1}),  by definition of  \bar{R}_{MAX}.$
$\implies E_s^{\pi} \left[ G^{T-1} \right] + R_{MIN} D'$
$< E_s^{\pi^*} \left[ G^{T-1} \right] + \left( P_s^{\pi^*}(s_T \notin G^!) - P_s^{\pi}(s_T \notin G^!) \right) R(s_T, a_T, s_{T+1})$
$\implies E_s^{\pi} \left[ G^{T-1} \right] + R_{MIN} D' < 0$

But this is a contradiction when $R$ is such that the agent receives a reward of $R_{MAX} \ge |R_{MIN}| D'$ at least once in its trajectory when following $\pi$ and zero everywhere else.

**Theorem 3 (Complexity)** *Estimating the Minmax penalty $R_{Minmax}$ accurately is NP-hard.*

**Proof** This follows from the NP-hardness of longest-path problems. Since the Minmax penalty is bounded by $\bar{R}_{MIN}$ and $\bar{R}_{MAX}$, both are defined by the diameter, which is in turn defined as the expected total timesteps of the longest path.

12

Reinforcement Learning Conference (August 2024)

# B Algorithms

**Algorithm 1:** Estimating the Diameter and Controllability
**Input** : $\langle S, A, P \rangle$, $R_D(s') := 1(s' \notin G)$, $R_C(s, a, s') := 1(s \notin G  and  s' \in G \setminus G^!)$
**Initialise** : Diameter $D = 0$, Controllability $C = 1$, Value functions $V_D^\pi(s) = 0$, $V_C^\pi(s) = 0$, Error $\Delta = 1$

<table>
    <tr>
        <td>**for** $\pi \in \Pi$ **do**&lt;br&gt;/* Policy evaluation for D */&lt;br&gt;**while** $\Delta &gt; 0$ **do**&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$\Delta \leftarrow 0$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;**for** $s \in S$ **do**&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$v' \leftarrow \sum_{s'} P(s' \mid s, \pi(s)) (R_D(s') + V_D^\pi(s'))$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$\Delta = \max \{ \Delta,</td>
        <td>V_D^\pi(s) - v'</td>
        <td>\}$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$V_D^\pi(s) \leftarrow v'$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;**end for**&lt;br&gt;**end while**&lt;br&gt;**for** $s \in S$ **do**&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$D = \max \{ D, V_D^\pi(s) \}$&lt;br&gt;**end for**&lt;br&gt;**end for**</td>
        <td>**for** $\pi \in \Pi$ **do**&lt;br&gt;/* Policy evaluation for C */&lt;br&gt;**while** $\Delta &gt; 0$ **do**&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$\Delta \leftarrow 0$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;**for** $s \in S$ **do**&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$v' \leftarrow \sum_{s'} P(s' \mid s, \pi(s)) (R_C(s, \pi(s), s') + V_C^\pi(s'))$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$\Delta = \max \{ \Delta,</td>
        <td>V_C^\pi(s) - v'</td>
        <td>\}$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$V_C^\pi(s) \leftarrow v'$&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;**end for**&lt;br&gt;**end while**&lt;br&gt;**for** $s \in S$ **do**&lt;br&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;$C = \min \{ C, V_C^\pi(s) \}$ **if** $V_C^\pi(s) \neq 0$ **else** $C$&lt;br&gt;**end for**&lt;br&gt;**end for**</td>
    </tr>
</table>**Algorithm 2:** RL while learning Minmax penalty
**Input** : RL algorithm $A$, max timesteps $T$
**Initialise** : $R_{MIN} = 0, R_{MAX} = 0, V_{MIN} = R_{MIN}, V_{MAX} = R_{MAX}, \pi$ and $V$ as per $A$
**for** t in T **do**
    **observe** a state $s_t$, **take** an action $a_t$ using $\pi$ as per $A$, and **observe** $s_{t+1}, r_t$
    $R_{MIN}, R_{MAX} \leftarrow \min(R_{MIN}, r_t), \max(R_{MAX}, r_t)$
    $V_{MIN}, V_{MAX} \leftarrow \min(V_{MIN}, R_{MIN}, V(s_t)), \max(V_{MAX}, R_{MAX}, V(s_t))$
    $\bar{R}_{MIN} \leftarrow V_{MIN} - V_{MAX}$
    $r_t \leftarrow \bar{R}_{MIN}$ **if** $s_{t+1} \in G^!$ **else** $r_t$
    **update** $\pi$ and $V$ with $(s_t, a_t, s_{t+1}, r_t)$ as per $A$
**end for**

13

Reinforcement Learning Conference (August 2024)

# C Related Work

**Reward shaping**: The problem of designing reward functions to produce desired policies in RL settings is well-studied (**Singh et al., 2009**). Particular focus has been placed on the practice of *reward shaping*, in which an initial reward function provided by an MDP is augmented in order to improve the rate at which an agent learns the same optimal policy (**Ng et al., 1999; Devidze et al., 2021**). While sacrificing some optimality, other approaches like **Lipton et al. (2016)** propose shaping rewards using an idea of intrinsic fear. Here, the agent trains a supervised fear model representing the probability of reaching unsafe states in a fixed horizon, scales said probabilities by a fear factor, and then subtracts the scaled probabilities from Q-learning targets. These approaches differ from ours in that they seek to find reward functions that improve convergence while preserving the optimality from an initial reward function. In contrast, we seek to determine the optimal rewards for terminal states in order to minimise undesirable behaviours irrespective of the original reward function and optimal policy.

**Constrained RL**: Disincentivising or preventing undesirable behaviours is core to the field of safe RL. A popular approach is to define constraints on the behaviour of an agent, tasking the agent with limiting the accumulation of costs associated with violating safety constraints while simultaneously maximising reward (**Altman, 1999; Achiam et al., 2017; Chow et al., 2018; Ray et al., 2019; HasanzadeZonuzy et al., 2021**). Widely used examples of these approaches include constrained policy optimisation (CPO) (**Achiam et al., 2017**), which augments TRPO (**Schulman et al., 2015**) with constraints to satisfy a constrained MDP, and TRPO-Lagrangian (**Ray et al., 2019**), which combines Lagrangian methods with TRPO. Another example is Sauté RL (**Sootla et al., 2022**), which incorporates the cost function into the rewards and augments the state with the remaining "cost budget" spent by violating safety constraints. Other constraint-based approaches include Projection-based CPO (**Yang et al., 2020**), which projects a TRPO policy onto a space defined by constraints, and PID Lagrangian methods (**Stooke et al., 2020**), which augment Lagrangian methods with PID control.

**Shielding**: Another important line of work involves relying on interventions from a model (**Dalal et al., 2018; Wagener et al., 2021**) or human (**Tennenholtz et al., 2022**) to prevent unsafe actions from being considered by the agent (shielding the agent) or prevent the environment from executing those unsafe actions by correcting them (shielding the environment). Other approaches here also look at using temporal logics to define or enforce safety constraints on the actions considered or selected by the agent (**Alshiekh et al., 2018**). These approaches fit seamlessly into our proposed reward-only framework since they are primarily about modifications on the transition dynamics and not the reward function—for example, unsafe actions here can simply lead to unsafe goal states.

14

Reinforcement Learning Conference (August 2024)

# D Additional Experiments and Figures

<table>
    <tr>
        <th>TRPO</th>
        <th>TRPO Lagrangian</th>
        <th>CPO</th>
        <th>SauteTRPO</th>
        <th>TRPO Minmax (Ours)</th>
    </tr>
    <tr>
        <td>&lt;mark style="background-color: purple"&gt;TRPO&lt;/mark&gt;</td>
        <td>&lt;mark style="background-color: blue"&gt;TRPO Lagrangian&lt;/mark&gt;</td>
        <td>&lt;mark style="background-color: green"&gt;CPO&lt;/mark&gt;</td>
        <td>&lt;mark style="background-color: brown"&gt;SauteTRPO&lt;/mark&gt;</td>
        <td>&lt;mark style="background-color: red"&gt;TRPO Minmax (Ours)&lt;/mark&gt;</td>
    </tr>
</table>## Figure 7: Performance comparison with baselines in the PILLAR environment with noise = 0.

### (a) Cumulative cost
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~1500</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~2000</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~2500</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~3000</td>
    </tr>
</table>### (b) Episode length
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>EpLen</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~200</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~200</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~200</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~200</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~200</td>
    </tr>
</table>### (c) Failure rate
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~1.0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.1</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.05</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.05</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.05</td>
    </tr>
</table>### (d) Average returns
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpRet</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~-4</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~3</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~3</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~3</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~3</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~3</td>
    </tr>
</table>## Figure 8: Performance comparison with baselines in the PILLAR environment with noise = 2.5.

### (a) Cumulative cost
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2000</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~4000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~6000</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~8000</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~10000</td>
    </tr>
</table>### (b) Episode length
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>EpLen</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~800</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~500</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~400</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~400</td>
    </tr>
</table>### (c) Failure rate
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~0.8</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.4</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.1</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.1</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.1</td>
    </tr>
</table>### (d) Average returns
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpRet</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~-2</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~2.5</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~2.5</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~2.5</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~2.5</td>
    </tr>
</table>## Figure 9: Performance comparison with baselines in the PILLAR environment with noise = 5.

### (a) Cumulative cost
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2000</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~4000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~6000</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~8000</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~10000</td>
    </tr>
</table>### (b) Episode length
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>EpLen</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~800</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~700</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~600</td>
    </tr>
</table>### (c) Failure rate
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~0.7</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.4</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.3</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.2</td>
    </tr>
</table>### (d) Average returns
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpRet</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~-1.0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~1.0</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~1.5</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~1.5</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~1.5</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~1.5</td>
    </tr>
</table>## Figure 10: Performance comparison with baselines in the PILLAR environment with noise = 7.5.

### (a) Cumulative cost
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2000</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~4000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~6000</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~7000</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~8000</td>
    </tr>
</table>### (b) Episode length
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>EpLen</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~800</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~700</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~600</td>
    </tr>
</table>### (c) Failure rate
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~0.7</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.4</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.3</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.2</td>
    </tr>
</table>### (d) Average returns
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpRet</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~-0.8</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.0</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.1</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.1</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.1</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.1</td>
    </tr>
</table>## Figure 11: Performance comparison with baselines in the PILLAR environment with noise = 10.

### (a) Cumulative cost
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2000</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~4000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~6000</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~7000</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~8000</td>
    </tr>
</table>### (b) Episode length
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>EpLen</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~800</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~700</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~600</td>
    </tr>
</table>### (c) Failure rate
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~0.7</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.4</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.3</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.2</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.2</td>
    </tr>
</table>### (d) Average returns
<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>AverageEpRet</th>
    </tr>
    <tr>
        <td>0</td>
        <td>~-0.75</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.0</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.25</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.25</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.25</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.25</td>
    </tr>
</table>15

Reinforcement Learning Conference (August 2024)

noise = 0.0      noise = 2.5      noise = 5.0      noise = 7.5      noise = 10.0



(a) **TRPO**. Failures per noise left to right: $0, 0, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}$

(b) **TRPO-Lagrangian**. Failures per noise left to right: $0, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}$

(c) **CPO**. Failures per noise left to right: $0, 0, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}$

(d) **Sauté-RL**. Failures per noise left to right: $0, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}$

(e) **TRPO-Minmax**. Failures per noise left to right: $0, 0, 0, \frac{1}{3}, 0$

Figure 12: Sample trajectories of policies learned by each baseline and our **TRPO-Minmax** approach in the Safety Gym PILLAR environment with varying noise levels. To sample the trajectories for each noise level, we use the same three environment random seeds across all the algorithms.

16

Reinforcement Learning Conference (August 2024)



Figure 13: Additional Safety-Gym domains. (a) is a modified version of the POINTGOAL1 task from OpenAI's Safety Gym environments (Ray et al., 2019), which represents complex, high-dimensional, continuous control tasks. In all of the original domains, $G = \emptyset$ by default. We only modify POINTGOAL1 to make unsafe transitions terminal $G = G^! = \{states with cost > 0\}$, leaving the safe goal states non-terminal ($G \setminus G^! = \emptyset$). Here, a simple robot must navigate to a goal location $\textcolor{green}{🟢}$ across a 2D plane while avoiding several hazards $\textcolor{blue}{🔵}$ (where $G = G^! = \{\textcolor{blue}{🔵}\}$). The agent's sensors, actions, and rewards are identical to the PILLAR domain. Unlike the PILLAR domain, the goal's location is randomly reset when the agent reaches it, but does not terminate the episode. (b-d) are modified similarly to the POINTGOAL1-HARD environment. POINTPUSH1-HARD is similar to POINTGOAL1-HARD, but with the addition of a pillar obstacle $\textcolor{blue}{🔵}$ and a large box $\textcolor{yellow}{🟡}$ the agent must push to the goal location $\textcolor{green}{🟢}$ to receive the goal reward (where $G = G^! = \{\textcolor{blue}{🔵}, \textcolor{blue}{🔵}\}$). Finally, POINTBUTTON1-HARD and CARBUTTON1-HARD are also similar to POINTGOAL1-HARD, but with the more complex car robot for CARBUTTON1-HARD and the addition of these to both: (i) Gremlins $\textcolor{red}{🔴}$, which are dynamic obstacles that move around the environment and must be avoided; and (ii) Buttons $\textcolor{green}{🟢}$, where the agent must reach the goal button with a cylinder $\textcolor{green}{🟢}$ to receive the goal reward (where $G = G^! = \{\textcolor{blue}{🔵}, \textcolor{red}{🔴}, \textcolor{green}{🟢}\}$).

17

# Reinforcement Learning Conference (August 2024)

<figure>
  <img src="image1.png" alt="Figure 14: Performance in POINTGOAL1-HARD">
  <figcaption>Figure 14: Performance in POINTGOAL1-HARD (where $G = G^1 = \{○\}$). Here, higher episode lengths are better (in addition to higher returns) since episodes only terminate when the agent reaches a hazard or after 1000 timesteps. Similar to Figure 6, all the baselines except Sauté-RL achieve significantly high returns at the expense of a rapidly increasing cumulative cost. By comparison, TRPO-Minmax dramatically reduces the failure rate while still being able to solve the task, as observed by average returns achieved as well as the trajectories observed. However, returns are lower since TRPO-Minmax learns safer paths to the goals but the dense reward function incentivises moving towards the goal despite the large number of hazards in-between.</figcaption>
</figure>

<figure>
  <img src="image2.png" alt="Figure 15: Performance in POINTPUSH1-HARD">
  <figcaption>Figure 15: Performance in POINTPUSH1-HARD (where $G = G^1 = \{○, ●\}$). Here, higher episode lengths are better (in addition to higher returns) since episodes only terminate when the agent reaches a hazard or after 1000 timesteps. Similar to Figure 6, the baselines achieve significantly high returns at the expense of a rapidly increasing cumulative cost while TRPO-Minmax consistently prioritises maintaining low failure rates.</figcaption>
</figure>

<figure>
  <img src="image3.png" alt="Figure 16: Performance in POINTBUTTON1-HARD">
  <figcaption>Figure 16: Performance in POINTBUTTON1-HARD (where $G = G^1 = \{○, ■, ●\}$). Here, higher episode lengths are better (in addition to higher returns) since episodes only terminate when the agent reaches a hazard or after 1000 timesteps. Similar to Figure 6, the baselines achieve significantly high returns at the expense of a rapidly increasing cumulative cost while TRPO-Minmax consistently prioritises maintaining low failure rates.</figcaption>
</figure>

<figure>
  <img src="image4.png" alt="Figure 17: Performance in CARBUTTON1-HARD">
  <figcaption>Figure 17: Performance in CARBUTTON1-HARD (where $G = G^1 = \{○, ■, ●\}$). Here, higher episode lengths are better (in addition to higher returns) since episodes only terminate when the agent reaches a hazard or after 1000 timesteps. Similar to Figure 6, the baselines achieve significantly high returns at the expense of a rapidly increasing cumulative cost while TRPO-Minmax consistently prioritises maintaining low failure rates.</figcaption>
</figure>

18

Reinforcement Learning Conference (August 2024)

(a) **TRPO** successes (top) and failures (bottom)

(b) **TRPO-Lagrangian** successes (top) and failures (bottom)

(c) **CPO** successes (top) and failures (bottom)

(d) **Sauté-RL** successes (top) and failures (bottom)

(e) **TRPO-Minmax** successes (top) and failures (bottom)

Figure 18: Sample trajectories of policies learned by each baseline and our Minmax approach in the Safety Gym POINTGOAL1-HARD domain, in the experiments of Figure 14. Trajectories that hit hazards or take more than 1000 timesteps to reach the goal location are considered failures.

19

Reinforcement Learning Conference (August 2024)

TRPO | TRPO Lagrangian | CPO | SauteTRPO | TRPO Minmax (Ours)
---|---|---|---|---
Chart 1 | Chart 2 | | |
(a) The cumulative cost. | (b) Failure rate | | |

<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2500</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~5000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~7500</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~10000</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~12500</td>
    </tr>
    <tr>
        <td></td>
    <td></td></tr>
    <tr>
        <td>TotalEnvInteracts</td>
        <td>AverageEpCost</td>
    </tr>
    <tr>
        <td>0</td>
        <td>~0.8</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~0.7</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~0.6</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~0.5</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~0.4</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~0.3</td>
    </tr>
    <tr>
        <td></td>
    <td></td></tr>
    <tr>
        <td>TotalEnvInteracts</td>
        <td>EpLen</td>
    </tr>
    <tr>
        <td>0</td>
        <td>~500</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~600</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~700</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~800</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~900</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~1000</td>
    </tr>
    <tr>
        <td></td>
    <td></td></tr>
    <tr>
        <td>TotalEnvInteracts</td>
        <td>AverageEpRet</td>
    </tr>
    <tr>
        <td>0</td>
        <td>~0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>~2.5</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>~5.0</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>~7.5</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>~10.0</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>~12.5</td>
    </tr>
</table>(c) Average episode length | (d) Average returns

Figure 19: Comparison with baselines in the POINTGOAL1-HARD environment (where $G = G^! = \{circle, square\}$). Here, higher episode lengths are better (in addition to higher returns) since episodes only terminate when the agent reaches a hazard or after 1000 timesteps. This experiment is similar to Figure 14, but uses a cost threshold of 25 for the baselines (as in Ray et al. (2019)) to check its effect on the performance of the baselines when episodes immediately terminate at unsafe states. We can observe drastically worse failure rates and cumulative costs for the baselines compared to their performance in Figure 14 (where the cost threshold was 0). Similar results were obtained when using a cost threshold of 1. These show how sensitive such approaches are to the cost threshold, while a reward-only approach like TRPO-Minmax does not depend on such hyperparameters.

20

Reinforcement Learning Conference (August 2024)


(a) **TRPO** successes (top) and failures (bottom)


(b) **TRPO-Lagrangian** successes (top) and failures (bottom)


(c) **CPO** successes (top) and failures (bottom)


(d) **Sauté-RL** successes (top) and failures (bottom)


(e) **TRPO-Minmax** successes (top) and failures (bottom)

Figure 20: Sample trajectories of policies learned by each baseline and our Minmax approach in the Safety Gym **POINTGOAL1-HARD** domain, in the experiments of Figure 19. Trajectories that hit hazards or take more than 1000 timesteps to reach the goal location are considered failures.

21

Reinforcement Learning Conference (August 2024)

TRPO | TRPO Lagrangian | CPO | SauteTRPO | TRPO Minmax (Ours)
---|---|---|---|---
Chart 1 | Chart 2 | | |
(a) The cumulative cost. | (b) Failure rate | | |

<table>
    <tr>
        <th>TotalEnvInteracts</th>
        <th>CumulativeCost</th>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>100000</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>200000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>300000</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>400000</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>500000</td>
    </tr>
    <tr>
        <td>1.2e7</td>
        <td>600000</td>
    </tr>
    <tr>
        <td></td>
    <td></td></tr>
    <tr>
        <td>TotalEnvInteracts</td>
        <td>AverageEpCost</td>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>20</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>40</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>60</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>80</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>100</td>
    </tr>
    <tr>
        <td>1.2e7</td>
        <td>120</td>
    </tr>
    <tr>
        <td></td>
    <td></td></tr>
    <tr>
        <td>TotalEnvInteracts</td>
        <td>EpLen</td>
    </tr>
    <tr>
        <td>0</td>
        <td>960</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>980</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>1000</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>1020</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>1040</td>
    </tr>
    <tr>
        <td></td>
    <td></td></tr>
    <tr>
        <td>TotalEnvInteracts</td>
        <td>AverageEpRet</td>
    </tr>
    <tr>
        <td>0</td>
        <td>-10</td>
    </tr>
    <tr>
        <td>2e6</td>
        <td>-5</td>
    </tr>
    <tr>
        <td>4e6</td>
        <td>0</td>
    </tr>
    <tr>
        <td>6e6</td>
        <td>5</td>
    </tr>
    <tr>
        <td>8e6</td>
        <td>10</td>
    </tr>
    <tr>
        <td>1e7</td>
        <td>15</td>
    </tr>
    <tr>
        <td>1.2e7</td>
        <td>20</td>
    </tr>
    <tr>
        <td>1.4e7</td>
        <td>25</td>
    </tr>
</table>(c) Average episode length | (d) Average returns

Figure 21: Comparison with baselines in the original Safety Gym POINTGOAL1 environment. This domain is the same as POINTGOAL1-HARD, except that episodes do not terminate when a hazard is hit (hence every episode only terminates after 1000 steps). We set the cost threshold for the baselines to 25 as in Ray et al. (2019) For TRPO Minmax, we replace the reward with the Minmax penalty every time the agent is in an unsafe state (that is every time the cost is greater than zero), as in previous experiments and as per Algorithm 2. While TRPO Minmax still beats the baselines in safe exploration (a-b), it struggles to maximise rewards while avoiding unsafe states (d).

22

Reinforcement Learning Conference (August 2024)


(a) **TRPO** successes (top) and failures (bottom)


(b) **TRPO-Lagrangian** successes (top) and failures (bottom)


(c) **CPO** successes (top) and failures (bottom)


(d) **Sauté-RL** successes (top) and failures (bottom)


(e) **TRPO-Minmax** successes (top) and failures (bottom)

Figure 22: Sample trajectories of policies learned by each baseline and our Minmax approach in the Safety Gym POINTGOAL1-HARD domain, in the experiments of Figure 21. Trajectories that hit hazards (the hits are highlighted by the red spheres) or take more than 1000 timesteps to reach the goal location are considered failures.

23