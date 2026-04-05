# Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark

**Jiaming Ji**<sup>1,∗</sup>, **Borong Zhang**<sup>1,∗</sup>, **Jiayi Zhou**<sup>1,∗</sup>, **Xuehai Pan**<sup>1</sup>, **Weidong Huang**<sup>1</sup>
**Ruiyang Sun**<sup>1</sup>, **Yiran Geng**<sup>1</sup>, **Yifan Zhong**<sup>1,2</sup>, **Juntao Dai**<sup>1</sup>, **Yaodong Yang**<sup>1,†</sup>

<sup>1</sup> Institute for AI, Peking University
<sup>2</sup> Beijing Institute for General Artificial Intelligence (BIGAI)

{jiamg.ji, borongzh}@gmail.com, gaiejj@outlook.com
yaodong.yang@pku.edu.cn

## Abstract

Artificial intelligence (AI) systems possess significant potential to drive societal progress. However, their deployment often faces obstacles due to substantial safety concerns. Safe reinforcement learning (SafeRL) emerges as a solution to optimize policies while simultaneously adhering to multiple constraints, thereby addressing the challenge of integrating reinforcement learning in safety-critical scenarios. In this paper, we present an environment suite called **Safety-Gymnasium**, which encompasses safety-critical tasks in both single and multi-agent scenarios, accepting vector and vision-only input. Additionally, we offer a library of algorithms named **Safe Policy Optimization (SafePO)**, comprising 16 state-of-the-art SafeRL algorithms. This comprehensive library can serve as a validation tool for the research community. By introducing this benchmark, we aim to facilitate the evaluation and comparison of safety performance, thus fostering the development of reinforcement learning for safer, more reliable, and responsible real-world applications. The website of this project can be accessed at https://sites.google.com/view/safety-gymnasium.

## 1 Introduction

AI systems possess enormous potential to spur societal progress. However, their deployment is frequently hindered by substantial safety considerations [1; 2; 3; 4]. Distinct from pure reinforcement learning (RL), Safe reinforcement learning (SafeRL) seeks to optimize policies while concurrently adhering to multiple constraints, addressing the challenge of employing RL in scenarios with critical safety implications [5; 6; 7; 8; 9]. This strategy proves particularly pertinent in real-world applications such as autonomous vehicles [10] and healthcare [11], where system failures or unsafe actions can result in grave consequences, such as accidents or harm to individuals. In large language models (LLMs), some studies have also shown that the toxicity of the models can be reduced through SafeRL [12; 13]. Incorporating safety constraints ensures adherence to predefined boundaries and regulatory standards, fostering trust and enabling exploration in environments with high-risk potential. Overall, SafeRL is instrumental in guaranteeing the dependable operation of intelligent systems in intricate and high-stake domains.

---
*Equal Contribution. †Corresponding author.
Work done when Jiayi Zhou visited Peking University.

37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks.

Simulation environments have become instrumental in fostering the advancement of RL. Eminent examples such as Gym [14], Atari [15], and dm-control [16] underline their importance. These versatile platforms permit researchers to swiftly design and execute varied tasks, thus enabling efficient evaluation of algorithmic effectiveness and intrinsic limitations. However, within the sphere of SafeRL, there is a notable dearth of dedicated simulation environments, which impedes comprehensive exploration of SafeRL. In recent years, there have been strides to address this gap. DeepMind presented AI-Safety-Gridworlds, a suite of RL environments showcasing various safety properties of intelligent agents [17]. Afterward, OpenAI introduced the Safety Gym benchmark suite, a collection of high-dimensional continuous control environments incorporating safety-robot tasks [18]. Over the past two years, several additional environments have been developed by researchers, including safe-control-gym [19], MetaDrive [20], etc.

**Compared to Safety Gym**<sup>1</sup> **Safety-Gymnasium** inherits and expands the settings of some tasks of Safety Gym, aiming to bolster the community’s growth further. Compared with Safety Gym, we have made the following major improvements:

*   **Refactoring of the physics engine.** Safety Gym utilizes *mujoco-py* to enable Python-based customization of MuJoCo components. However, *mujoco-py* stopped updates and support after 2021. In contrast, **Safety-Gymnasium** supports MuJoCo directly, eliminating the reliance on *mujoco-py*. This facilitates access to the latest MuJoCo features (e.g., rendering speed and accuracy improved, etc.) and lowers the entry barrier, particularly due to *mujoco-py*’s dependency on specific GCC versions and more.
*   **Extension of Agent and Task Components.** Safety Gym initially supports only three agents and tasks. On this basis, **Safety-Gymnasium** has been further expanded, introducing more diverse agents and task components and expanding safety tasks to cover multi-agent domains. Finally, **Safety-Gymnasium** launched a high-dimensional test component based on Issac-Gym [21], further enriching the benchmark.
*   **Enhanced Visual Task Support.** The visual components of Safety Gym are simplistic (consisting of basic geometric shapes), and *mujoco-py* relies on OpenGL for visual rendering, which results in significant virtualization performance loss on headless servers. In contrast, **Safety-Gymnasium**, built on MuJoCo, achieves rendering speeds on CPU that are twice as fast as the former. Additionally, it offers more comprehensive visual component support.
*   **Easy Installation and High Customization.** Safety Gym is cumbersome to install and relies heavily on the underlying software. One of the design motivations of **Safety-Gymnasium** is the ease of use so that everyone can focus on algorithm design. **Safety-Gymnasium** can be easily installed with one simple command `pip install safety-gymnasium`. While benefiting from the highly integrated framework, **Safety-Gymnasium** only needs 100 lines of code to customize the required environment.

In this work, we introduce **Safety-Gymnasium**, a collection of environments specifically for SafeRL, built upon the Gymnasium [14; 22] and MuJoCo [23]. Enhancing the extant Safety Gym framework [18], we address various concerns and expand the task scope to include vision-only and multi-agent scenarios. Additionally, we released **SafePO**, a single-file style algorithm library containing over 16 state-of-the-art algorithms. Collectively, our contributions are enumerated as follows:

*   **Environmental Components.** We provide various safety-oriented tasks under the umbrella of **Safety-Gymnasium**. These tasks encompass single-agent, multi-agent, and vision-based challenges, each with varying constraints. Our environments are categorized into two primary types: Gymnasium-based, featuring agents of escalating complexity for algorithm verification and comparison, and Issac-Gym-based, incorporating sophisticated agents that harness the parallel processing power of Issac-gym’s GPU. This empowers researchers to explore SafeRL algorithms in complex scenarios. Further details can be found in Section 4.
*   **Algorithm Components.** We offer the **SafePO** algorithm library, which comprises a single-file style housing 16 diverse algorithms. These algorithms encompass both single-agent and multi-agent approaches, along with first-order and second-order variants, as well as Lagrangian-based

---
<sup>1</sup>Again, we have no intention of attacking Safety Gym; the contribution of Safety Gym to the SafeRL community cannot be ignored, and Safety Gym also inspired this work. We hope that through our efforts, **Safety-Gymnasium** can further promote the development of SafeRL and give back to the entire RL community.

2

and Projection-based methods. Through meticulous decoupling, each algorithm’s code resides in an individual file. A more in-depth exploration of SafePO is presented in Section 5.

*   **Insights and Analysis.** Combining Safety-Gymnasium and SafePO, we conduct a detailed analysis of existing algorithms. Our analysis encompasses 16 algorithms across 54 distinct environments, covering various scenarios such as single-agent and multi-agent setups with varying constraint complexities. This analysis delves into each algorithm’s strengths, constraints, and avenues for enhancement. We provide access to all metadata, fostering community verification and encouraging further research. Further details can be found in Section 6.

## 2 Related Work

**Safety Environments** In RL, agents need to explore environments to learn optimal policies by trial and error. It is currently typical to train RL agents mostly or entirely in simulation, where safety concerns are minimal. However, we anticipate that challenges in simulating the complexities of the real world (e.g., human-AI collaborative control [1; 2]) will cause a shift towards training RL agents directly in the real world, where safety concerns are paramount [20; 24; 25]. OpenAI includes safety requirements in the Safety Gym [18], which is a suite of high-dimensional continuous control environments for measuring research progress on SafeRL. Safe-control-gym [19] allows for constraint specification and disturbance injection onto a robot’s inputs, states, and inertial properties through a portable configuration system. DeepMind also presents a suite of RL environments, AI-Safety-Gridworlds [17], illustrating various safety properties of intelligent agents.

**SafeRL Algorithms** CMDPs have been extensively studied for different constraint criteria [26; 27; 28; 29]. With the rise of deep learning, CMDPs are also moving to more high-dimensional continuous control problems. CPO [30] proposes the first general-purpose policy search algorithm for SafeRL with guarantees for near-constraint satisfaction at each iteration. However, CPO’s policy updates hinge on Taylor approximations and the inversion of high-dimensional Fisher information matrices. These approximations can occasionally lead to inappropriate policy updates. FOCOPS [31] applies a primal-dual approach to solve the constrained trust region problem directly and subsequently projects the solution back into the parametric policy space. Similarly, CUP [32] offers non-convex implementations through a first-order optimizer, thereby not requiring a strong approximation of the convexity of the objective.

## 3 Preliminaries

### 3.1 Constrained Markov decision process

SafeRL [6; 33] is often formulated as a Constrained Markov decision process (CMDP) [6], which is a tuple $M = (S, A, P, R, C, \mu, \gamma)$. Here $S$ and $A$ are the state space and action space correspondingly. $P(s'|s, a)$ is the probability of state transition from $s$ to $s'$ after taking action $a$. $R(s'|s, a)$ denotes the reward obtained by the agent performing action $a$ in state $s$ and transitioning to state $s'$. The set $C = \{(c_i, b_i)\}_{i=1}^m$, where $c_i$ are cost functions: $c_i : S \times A \to R$ and the cost thresholds are $b_i, i = 1, \cdots, m$. $\mu(\cdot) : S \to [0, 1]$ is the initial state distribution and the discount factor $\gamma \in [0, 1)$.

A stationary parameterized policy $\pi_\theta$ is a probability distribution defined on $S \times A$, $\pi_\theta(a|s)$ denotes the probability of taking action $a$ in state $s$. We use $\Pi_\theta = \{\pi_\theta : \theta \in R^p\}$ to denote the set of all stationary policies and $\theta$ is the network parameter needed to be learned. Let $P_{\pi_\theta} \in R^{|S| \times |S|}$ denotes a state transition probability matrix and the components are: $P_{\pi_\theta}[s, s'] = P_{\pi_\theta}(s'|s) = \sum_{a \in A} \pi_\theta(a|s)P(s'|s, a)$, which denotes one-step state transition probability from $s$ to $s'$ by executing $\pi_\theta$. Finally, we let $d_{\pi_\theta}^{s_0}(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P_{\pi_\theta}(s_t = s|s_0)$ to be the stationary state distribution of the Markov chain starting at $s_0$ induced by policy $\pi_\theta$ and $d_{\pi_\theta}^\mu(s) = E_{s_0 \sim \mu(\cdot)}[d_{\pi_\theta}^\mu(s)]$ to be the discounted state visitation distribution on initial distribution $\mu$.

The objective function is defined via the infinite horizon discounted reward function where for a given $\pi_\theta$, we have $J^R(\pi_\theta) = E[\sum_{t=0}^\infty \gamma^t R(s_{t+1}|s_t, a_t)|s_0 \sim \mu, a_t \sim \pi_\theta]$. The cost function is similarly specified via the following infinite horizon discount cost function: $J_i^C(\pi_\theta) = E[\sum_{t=0}^\infty \gamma^t C_i(s_{t+1}|s_t, a_t)|s_0 \sim \mu, a_t \sim \pi_\theta]$.

3

Then, we define the feasible policy set $\Pi_{C}$ as : $\Pi_{C} = \cap_{i=1}^{m} \{ \pi_{\theta} \in \Pi_{\theta}  and  J_{i}^{C}(\pi_{\theta}) \leq b_{i} \}$. The goal of CMDP is to search the optimal policy $\pi_{\star}$: $\pi_{\star} = \arg \max_{\pi_{\theta} \in \Pi_{C}} J^{R}(\pi_{\theta})$.

## 3.2 Constrained Markov Game

Safe multi-agent reinforcement learning is often formulated as a Constrained Markov Game $(N, S, A, P, \mu, \gamma, R, C, b)$. Here, $N = \{1, \dots, n\}$ is the set of agents, $S$ and $A = \prod_{i=1}^{n} A^{i}$ are the state space and the joint action space (i.e., the product of the agents’ action spaces), $P: S \times A \times S \rightarrow R$ is the probabilistic transition function, $\mu$ is the initial state distribution, $\gamma \in [0, 1)$ is the discount factor, $R: S \times A \rightarrow R$ is the joint reward function, $C = \{C_{j}^{i}\}_{1 \leq j \leq m^{i}}^{i \in N}$ is the set of sets of cost functions (every agent $i$ has $m^{i}$ cost functions) of the form $C_{j}^{i}: S \times A^{i} \rightarrow R$, and finally the set of corresponding cost threshold is given by $b = \{b_{j}^{i}\}_{1 \leq j \leq m^{i}}^{i \in N}$. At time step $t$, the agents are in a state $s_{t}$, and every agent $i$ takes an action $a_{t}^{i}$ according to its policy $\pi^{i}(a^{i} \mid s_{t})$. Together with other agents’ actions, it gives a joint action $a_{t} = (a_{t}^{1}, \dots, a_{t}^{n})$ and the joint policy $\pi(a \mid s) = \prod_{i=1}^{n} \pi^{i}(a^{i} \mid s)$. The agents receive the reward $R(s_{t}, a_{t})$, meanwhile each agent $i$ pays the costs $C_{j}^{i}(s_{t}, a_{t}^{i}), \forall j = 1, \dots, m^{i}$. The environment then transits to a new state $s_{t+1} \sim P(\cdot \mid s_{t}, a_{t})$.

The objective of reward function are $J(\pi) \triangleq E_{s_{0} \sim \rho^{0}, a_{0:\infty} \sim \pi, s_{1:\infty} \sim P} [\sum_{t=0}^{\infty} \gamma^{t} R(s_{t}, a_{t})]$, and costs function are $J_{j}^{i}(\pi) \triangleq E_{s_{0} \sim \rho^{0}, a_{0:\infty} \sim \pi, s_{1:\infty} \sim P} [\sum_{t=0}^{\infty} \gamma^{t} C_{j}^{i}(s_{t}, a_{t}^{i})] \leq c_{j}^{i}, \quad \forall j = 1, \dots, m^{i}$.

We are examining a fully cooperative setting where all agents share a common reward function. Consequently, the goal of safe multi-agent RL is to identify the optimal policy that maximizes the expected total reward while simultaneously ensuring that the safety constraints of each agent are satisfied. Then we define the feasible joint policy set $\pi_{C} = \cap_{i=1}^{n} \{ \pi_{\theta} \in \Pi_{\theta}  and  J_{j}^{i}(\pi) \leq c_{j}^{i}, \forall j = 1, \dots, m^{i} \}$. The goal of CMG is to search the optimal policy $\pi_{\star} = \arg \max_{\pi_{\theta} \in \Pi_{C}} J(\pi_{\theta})$.

## 4 Safety Environments: Safety-Gymnasium

Safety-Gymnasium provides a seamless installation process and minimalistic code snippets to basic examples, as shown in Figure 1. Due to the limited space of the paper, we provide a more detailed description (e.g., detailed instructions, the composition of the robot’s observation space and action space, dynamic structure, physical parameters, etc.) in Appendix B and Online Documentation$^{2}$.

```python
"""
Install from PyPI:
  pip install safety-gymnasium
"""
import safety_gymnasium
# Create the safety-task environment
env = safety_gymnasium.make("SafetyPointGoal1-v0", render_mode="human")
# Reset the environment
obs, info = env.reset()
while True:
    # Sample a random action
    act = env.action_space.sample()
    # Step the environment: costs are returned
    obs, reward, cost, terminated, truncated, info = env.step(act)
    if terminated or truncated:
        break
```

Figure 1: Using Safety-Gymnasium to create, step, render a specific safety-task environment.

### 4.1 Gymnasium-based Learning Environments

In this section, we introduce Gymnasium-based environment components from three aspects: (1) the robots (both single-agent and multi-agent); (2) the tasks that are supported within the environment; (3) the safety constraints that are upheld.

---
$^{2}$Online Documentation: www.safety-gymnasium.com

4

**Supported Robots** As shown in Figure 2, Safety-Gymnasium inherits three pre-existing agents from Safety Gym [18], namely Point, Car, and Doggo. By meticulously adjusting the model parameters, we have successfully mitigated the issue of excessive oscillations during the runtime of Point and Car agents. Building upon this foundation, we have introduced two additional robots: racecar [34; 35], and ant [23], to enrich the single-agent scenarios. As for multi-agent robots, we have leveraged certain configurations from multi-agent MuJoCo [36], deconstructing the original single-agent structure and enabling multiple agents to control distinct body segments. This design choice has been widely adopted in various research works [37; 38; 39].


Figure 2: **Upper:** The Single-Agent Robots of Gymnasium-based Environments. **Lower:** The Multi-Agent Robots of Gymnasium-based Environments.

**Supported Tasks** As shown in Figure 3, the Gymnasium-based learning environments support the following tasks. For a more detailed task specification, please refer to our online documentation$^3$.

*   *Velocity.* The robot aims to facilitate coordinated leg movement of the robot in the forward (right) direction by exerting torques on the hinges.
*   *Run.* The robot starts with a random initial direction and a specific initial speed as it embarks on a journey to reach the opposite side of the map.
*   *Circle.* The reward is maximized by moving along the green circle and not allowed to enter the outside of the red region, so its optimal path follows the line segments $AD$ and $BC$.
*   *Goal.* The robot navigates to multiple goal positions. After successfully reaching a goal, its location is randomly reset while maintaining the overall layout.
*   *Push.* The objective is to move a box to a series of goal positions. Like the goal task, a new random goal location is generated after each achievement.
*   *Button.* The objective is to activate a series of goal buttons distributed throughout the environment. The agent's goal is to navigate towards and contact the currently highlighted button, known as the goal button.

**Supported Constraints** As shown in Figure 3, the Gymnasium-based environments support the following constraints. For a more detailed task specification, please refer to our online documentation.

*   *Velocity-Constraint* involves safety tasks using MuJoCo agents [23]. In these tasks, agents aim for higher reward by moving faster, but they must also adhere to velocity constraints for safety. Specifically, in a two-dimensional plane, the cost is computed as the Euclidean norm of the agent's velocities ($v_x$ and $v_y$).

$^3$Task Specification Documentation: https://www.safety-gymnasium.com/en/latest/components_of_environments/tasks.html

5

(a). Velocity
(b). Run
(c). Circle
(d). Goal
(e). Button
(f). Push

(a). Velocity Constraints
(b). Pillars
(c). Hazards
(d). Sigwalls
(e). Vases
(f). Gremlins

**Figure 3:** **Upper:** Tasks of Gymnasium-based Environments; **Lower:** Constraints of Gymnasium-based Environments.

*   *Pillars* are employed to represent large cylindrical obstacles within the environment. In the general setting, contact with a pillar incurs costs.
*   *Hazards* are utilized to model areas within the environment that pose a risk, resulting in costs when an agent enters such areas.
*   *Sigwalls* are designed specifically for Circle tasks. Crossing the wall from inside the safe area to the outside incurs costs.
*   *Vases* represent static and fragile objects within the environment. Touching or displacing these objects incurs costs for the agent.
*   *Gremlins* represent moving objects within the environment that can interact with the agent.

### 4.1.1 Vision-only tasks

Vision-only SafeRL has gained significant attention as a focal point of research, primarily due to its applicability in real-world contexts [40; 41]. While the initial iteration of Safety Gym offered rudimentary visual input support, there is room for enhancing the realism of its environment. To effectively evaluate vision-based SafeRL algorithms, we have devised a more realistic visual environment utilizing MuJoCo. This enhanced environment facilitates the incorporation of both RGB and RGB-D inputs (as shown in Figure 5). An exemplar of this environment is depicted in Figure 4, while comprehensive descriptions are available in Appendix B.5.


(a) Race
(b) The Vision Input of Race
(c) FormulaOne
(d) The Vision Input of FormulaOne

**Figure 4:** Vision-only Tasks of Gymnasium-based Environments.

### 4.2 Issac-Gym-based Learning Environments

In this section, we introduce Safety-DexterousHands, a collection of environments built upon DexterousHands [42] and the Isaac Gym engine [21]. Leveraging GPU capabilities, Safety-DexterousHands enables large-scale parallel sample collection, significantly accelerating the training process. The environments support both single-agent and multi-agent settings. These environments involve two robotic hands (refer to Figure 6 (a) and (b)). In each episode, a ball randomly descends near the right hand. The right hand needs to grasp and launch the ball toward the left hand, which subsequently catches and deposits it at the target location.

6

Figure 5: The RGB and RGB-D input of Gymnasium-based Environments.


Figure 6: Tasks of Safety-DexterousHands.

For timestep $t$, let $x_{b,t}$, $x_{g,t}$ to be the position of the ball and the goal, $d_{p,t}$ to denote the positional distance between the ball and the goal $d_{p,t} = \|x_{b,t} - x_{g,t}\|_2$. Let $d_{a,t}$ denote the angular distance between the object and the goal, and the rotational difference is $d_{r,t} = 2 \arcsin \min\{|d_{a,t}|, 1.0\}$. The reward is defined as follows, $r_t = \exp\{-0.2(\alpha d_{p,t} + d_{r,t})\}$, where $\alpha$ is a constant balance of positional and rotational reward.

**Safety Joint** constrains the freedom of joint ④ of the forefinger (refer to Figure 6 (c) and (d)). Without the constraint, joint ④ has freedom of $[-20^\circ, 20^\circ]$. The safety tasks restrict joint ④ within $[-10^\circ, 10^\circ]$. Let ang\_4 be the angle of joint ④, and the cost is defined as: $c_t = I(ang\_4 \notin [-10^\circ, 10^\circ])$.

**Safety Finger** constrains the freedom of joints ②, ③ and ④ of forefinger (refer to Figure 6 (c) and (e)). Without the constraint, joints ② and ③ have freedom of $[0^\circ, 90^\circ]$ and joint ④ of $[-20^\circ, 20^\circ]$. The safety tasks restrict joints ②, ③, and ④ within $[22.5^\circ, 67.5^\circ]$, $[22.5^\circ, 67.5^\circ]$, and $[-10^\circ, 10^\circ]$ respectively. Let ang\_2, ang\_3, ang\_4 be the angles of joints ②, ③, ④, and the cost is defined as:
$$c_t = I(ang\_2 \notin [22.5^\circ, 67.5^\circ],  or  ang\_3 \notin [22.5^\circ, 67.5^\circ],  or  ang\_4 \notin [-10^\circ, 10^\circ]). \quad (1)$$

## 5 Safe Policy Optimization Algorithms: SafePO

This section provides a detailed discussion of the design of SafePO. Features such as strong performance, extensibility, customization, visualization, and documentation are all presented to demonstrate the advantages and contributions of SafePO.

**Correctness** For a benchmark, it is critical to ensure its correctness and reliability. Firstly, each algorithm is implemented strictly according to the original paper (e.g., ensuring consistency with the gradient flow of the original paper, etc.). Secondly, we compare our implementation with those line by line for algorithms with a commonly acknowledged open-source code base to double-check the correctness. Finally, we compare SafePO with existing benchmarks (e.g., Safety-Starter-Agents$^4$ and RL-Safety-Algorithms$^5$) and SafePO outperforms or achieves comparable performance with other existing implementations, as shown in Table 1.

$^4$Safety-Starter-Agents: https://github.com/openai/safety-starter-agents
$^5$RL-Safety-Algorithms: https://github.com/SvenGronauer/RL-Safety-Algorithms

7

<table>
    <tr><th></th>
        <th>**Safe Navigation**: Button, Push, Multi-Goal, etc.</th>
        <th>**Safe Policy**</th>
        <th>**Constrained Optimization**</th>
        <th>**SafePO**</th>
        <th>**Pure Policy**</th>
    </tr>
    <tr>
        <td>**Safe Manipulation**: ShadowHandOver, FreightFrankaCloseDrawer, etc.</td>
        <td>MAPPO-Lag</td>
        <td>Lagrangian</td>
        <td></td>
        <td>HAPPO</td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>PPO-Lag</td>
        <td></td>
        <td></td>
        <td>MAPPO</td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>RCPO</td>
        <td>PID Control</td>
        <td></td>
        <td>PPO</td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>TRPO-Lag</td>
        <td></td>
        <td></td>
        <td>PG</td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>CPPO-PID</td>
        <td></td>
        <td></td>
        <td>Natural PG</td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>CUP</td>
        <td>Two Stage</td>
        <td>Projection</td>
        <td>TRPO</td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>FOCOPS</td>
        <td></td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td>**Safe Velocity**: Multi/Single-agent, Ant, Swimmer, Humanoid, etc.</td>
        <td>PCPO</td>
        <td>One Stage</td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>CPO</td>
        <td></td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>MACPO</td>
        <td></td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td>**Safety Vision**: FormulaOne, Race, Building, Fading, etc.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>**Single / Multi-Agent Pipeline**</td>
        <td></td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td></td>
        <td>**Environment Wrapper**</td>
        <td>**Gymnasium / Isaac-Gym API**</td>
        <td></td>
        <td></td>
    <td></td></tr>
    <tr>
        <td>**User Interface**</td>
        <td>Keyboard Controller</td>
        <td>Efficient Command</td>
        <td>Policy Evaluator</td>
        <td>Customized Configuration</td>
        <td>Benchmarking Tools</td>
    </tr>
</table>Figure 7: The Architecture of SafePO

**Extensibility** SafePO enjoys high extensibility thanks to its architecture (as shown in Figure 7). New algorithms can be integrated into SafePO by inheriting from base algorithms and only implementing their unique features. For example, we integrate PPO by inheriting from policy gradient and only adding the clip ratio variable and rewriting the function that computes the loss of policy $\pi$. Similarly, algorithms can be easily added to SafePO.

**Logging and Visualization** Another necessary functionality of SafePO is logging and visualization. Supporting both TensorBoard and WandB, we offer code for visualizing more than 40 parameters and intermediate computation results to inspect the training process. Standard parameters and metrics such as KL-divergence, SPS (step per second), and cost variance are visualized universally. Special features of algorithms are also reported, such as the Lagrangian multiplier of Lagrangian-based methods, $g^T H^{-1} g, g^T H^{-1} b, \nu^*$, and $\lambda^*$ of CPO, proportional, integral, and derivative of PID-Lagrangian algorithms, etc. During training, users can inspect the changes of every parameter, collect the log file, and obtain saved checkpoint models. The complete and comprehensive visualization allows easier observation, model selection, and comparison.

**Documentation** In addition to its code implementation, SafePO comes with an extensive documentation$^6$. We include detailed guidance on installation and propose solutions to common issues. Moreover, we provide instructions on simple usage and advanced customization of SafePO. Official information concerning maintenance, ethical, and responsible use are stated clearly for reference.

Table 1: A comparison between SafePO and other implementations. Results are based on 10 evaluation iterations using over 3 seeds under `cost_limit`=25.00. $\bar{J}^R$ stands for normalized reward from PPO's performance, $\bar{J}^C$ signifies normalized cost relative to `cost_limit`, and AvgR/AvgC represents the ratio of the means of both across 10 environments. The $\uparrow$ indicates higher rewards are better, while the $\downarrow$ indicates lower costs (when beyond the threshold of 1.00) are better. *Gray* and *Black* depicts violation and compliance with the `cost_limit`.

<table>
    <tr>
        <th></th>
        <th>**CPO**</th>
        <th></th>
        <th></th>
        <th>**TRPO-Lag**</th>
        <th></th>
        <th></th>
        <th>**PPO-Lag**</th>
        <th></th>
        <th></th>
        <th>**FOCOPS**</th>
        <th></th>
        <th></th>
    <th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th></tr>
    <tr>
        <td></td>
        <td>**SafePO (Ours)**</td>
        <td>**Safety Starter Agents**</td>
        <td>**RL-Safety-Algorithms**</td>
        <td>**SafePO (Ours)**</td>
        <td>**Safety Starter Agents**</td>
        <td>**RL-Safety-Algorithms**</td>
        <td>**SafePO (Ours)**</td>
        <td>**Safety Starter Agents**</td>
        <td>**SafePO (Ours)**</td>
        <td>**Original Implementation**</td>
        <td></td>
        <td></td>
    <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr>
        <td>**Safety Navigation**</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
    <td></td><td></td><td></td><td></td></tr>
    <tr>
        <td>CARBUTTON1</td>
        <td>0.08</td>
        <td>1.75</td>
        <td>0.34</td>
        <td>3.65</td>
        <td>-0.06</td>
        <td>3.30</td>
        <td>-0.04</td>
        <td>1.08</td>
        <td>0.02</td>
        <td>0.78</td>
        <td>-0.05</td>
        <td>0.63</td>
        <td>0.01</td>
        <td>0.47</td>
        <td>0.02</td>
        <td>0.67</td>
        <td>0.04</td>
        <td>1.21</td>
        <td>0.53</td>
        <td>6.02</td>
    </tr>
    <tr>
        <td>CARGOAL1</td>
        <td>0.78</td>
        <td>1.63</td>
        <td>0.94</td>
        <td>2.49</td>
        <td>0.46</td>
        <td>1.25</td>
        <td>0.82</td>
        <td>1.09</td>
        <td>0.72</td>
        <td>1.04</td>
        <td>0.72</td>
        <td>0.91</td>
        <td>0.43</td>
        <td>0.39</td>
        <td>0.52</td>
        <td>0.52</td>
        <td>0.52</td>
        <td>0.93</td>
        <td>0.79</td>
        <td>2.45</td>
    </tr>
    <tr>
        <td>POINTBUTTON1</td>
        <td>0.12</td>
        <td>1.61</td>
        <td>0.70</td>
        <td>3.01</td>
        <td>0.03</td>
        <td>3.25</td>
        <td>0.27</td>
        <td>1.29</td>
        <td>0.21</td>
        <td>0.92</td>
        <td>0.04</td>
        <td>0.87</td>
        <td>0.22</td>
        <td>1.32</td>
        <td>0.17</td>
        <td>0.96</td>
        <td>0.25</td>
        <td>1.53</td>
        <td>0.70</td>
        <td>3.74</td>
    </tr>
    <tr>
        <td>POINTGOAL1</td>
        <td>0.78</td>
        <td>1.10</td>
        <td>0.81</td>
        <td>1.99</td>
        <td>0.28</td>
        <td>2.05</td>
        <td>0.72</td>
        <td>0.91</td>
        <td>0.65</td>
        <td>0.94</td>
        <td>0.33</td>
        <td>0.72</td>
        <td>0.47</td>
        <td>1.50</td>
        <td>0.66</td>
        <td>0.77</td>
        <td>0.56</td>
        <td>1.32</td>
        <td>0.81</td>
        <td>1.53</td>
    </tr>
    <tr>
        <td>**Safety Velocity**</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
        <td>$\bar{J}^R \uparrow$</td>
        <td>$\bar{J}^C \downarrow$</td>
    </tr>
    <tr>
        <td>ANTVEL</td>
        <td>0.52</td>
        <td>0.56</td>
        <td>0.31</td>
        <td>0.93</td>
        <td>0.40</td>
        <td>1.09</td>
        <td>0.53</td>
        <td>0.15</td>
        <td>0.32</td>
        <td>0.76</td>
        <td>0.44</td>
        <td>0.70</td>
        <td>0.54</td>
        <td>0.22</td>
        <td>0.31</td>
        <td>0.61</td>
        <td>0.55</td>
        <td>0.60</td>
        <td>0.52</td>
        <td>0.39</td>
    </tr>
    <tr>
        <td>HALFCHEETAHVEL</td>
        <td>0.40</td>
        <td>0.23</td>
        <td>0.30</td>
        <td>1.13</td>
        <td>0.31</td>
        <td>0.97</td>
        <td>0.43</td>
        <td>1.01</td>
        <td>0.25</td>
        <td>0.79</td>
        <td>0.43</td>
        <td>0.67</td>
        <td>0.44</td>
        <td>0.04</td>
        <td>0.30</td>
        <td>0.93</td>
        <td>0.42</td>
        <td>0.12</td>
        <td>0.44</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>HOPPERVEL</td>
        <td>0.73</td>
        <td>0.48</td>
        <td>0.35</td>
        <td>0.93</td>
        <td>0.26</td>
        <td>0.68</td>
        <td>0.59</td>
        <td>0.71</td>
        <td>0.41</td>
        <td>1.11</td>
        <td>0.24</td>
        <td>0.57</td>
        <td>0.58</td>
        <td>0.89</td>
        <td>0.29</td>
        <td>1.20</td>
        <td>0.66</td>
        <td>0.30</td>
        <td>0.74</td>
        <td>0.53</td>
    </tr>
    <tr>
        <td>HUMANOIDVEL</td>
        <td>0.71</td>
        <td>0.01</td>
        <td>0.05</td>
        <td>0.19</td>
        <td>0.36</td>
        <td>0.83</td>
        <td>0.72</td>
        <td>2.38</td>
        <td>0.05</td>
        <td>0.01</td>
        <td>0.71</td>
        <td>0.79</td>
        <td>0.72</td>
        <td>0.76</td>
        <td>0.07</td>
        <td>0.09</td>
        <td>0.71</td>
        <td>0.93</td>
        <td>0.73</td>
        <td>0.43</td>
    </tr>
    <tr>
        <td>SWIMMERVEL</td>
        <td>0.51</td>
        <td>0.82</td>
        <td>0.38</td>
        <td>1.11</td>
        <td>0.41</td>
        <td>0.82</td>
        <td>0.66</td>
        <td>0.84</td>
        <td>0.43</td>
        <td>1.67</td>
        <td>0.41</td>
        <td>1.02</td>
        <td>0.57</td>
        <td>1.11</td>
        <td>0.38</td>
        <td>1.18</td>
        <td>0.47</td>
        <td>1.30</td>
        <td>0.68</td>
        <td>0.71</td>
    </tr>
    <tr>
        <td>WALKER2DVEL</td>
        <td>0.39</td>
        <td>0.81</td>
        <td>0.44</td>
        <td>1.85</td>
        <td>0.05</td>
        <td>0.67</td>
        <td>0.51</td>
        <td>0.77</td>
        <td>0.46</td>
        <td>0.67</td>
        <td>0.51</td>
        <td>1.34</td>
        <td>0.44</td>
        <td>0.20</td>
        <td>0.47</td>
        <td>0.81</td>
        <td>0.50</td>
        <td>0.68</td>
        <td>0.48</td>
        <td>0.74</td>
    </tr>
    <tr>
        <td>**AvgR/AvgC**</td>
        <td></td>
        <td>**0.56**</td>
        <td></td>
        <td>0.27</td>
        <td></td>
        <td>0.17</td>
        <td></td>
        <td>**0.51**</td>
        <td></td>
        <td>0.40</td>
        <td></td>
        <td>0.46</td>
        <td></td>
        <td>**0.64**</td>
        <td></td>
        <td>0.41</td>
        <td></td>
        <td>**0.52**</td>
        <td></td>
        <td>0.39</td>
    </tr>
</table>$^6$SafePO's Documentation: https://safe-policy-optimization.readthedocs.io

8

# 6 Experiments and Analysis


<table>
<caption>(a) Average Episodic Reward of Algorithms</caption>
<thead>
<tr>
<th>Algorithm</th>
<th>Normalized Value based on (MA)PPO-Lag</th>
</tr>
</thead>
<tbody>
<tr>
<td>PPO-Lag</td>
<td>1.00</td>
</tr>
<tr>
<td>TRPO-Lag</td>
<td>1.14</td>
</tr>
<tr>
<td>CPPO-PID</td>
<td>0.99</td>
</tr>
<tr>
<td>RCPO</td>
<td>1.13</td>
</tr>
<tr>
<td>CPO</td>
<td>1.10</td>
</tr>
<tr>
<td>PCPO</td>
<td>0.54</td>
</tr>
<tr>
<td>CUP</td>
<td>0.90</td>
</tr>
<tr>
<td>FOCOPS</td>
<td>1.02</td>
</tr>
<tr>
<td>MAPPO-Lag</td>
<td>1.00</td>
</tr>
<tr>
<td>MACPO</td>
<td>0.76</td>
</tr>
</tbody>
</table>
<table>
<caption>(b) Bar Chart Categorizing Algorithms into Four Classes Based on Average Episodic Cost</caption>
<thead>
<tr>
<th>Algorithm</th>
<th>Strongly Unsafe: [35, ∞)</th>
<th>Unsafe: [25, 35)</th>
<th>Safe: [15, 25)</th>
<th>Strongly Safe: [0, 15)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PPO-Lag</td>
<td>~1.0</td>
<td>~1.0</td>
<td>~1.0</td>
<td>~0.5</td>
</tr>
<tr>
<td>TRPO-Lag</td>
<td>~0.5</td>
<td>~1.5</td>
<td>~1.0</td>
<td>~0.5</td>
</tr>
<tr>
<td>CPPO-PID</td>
<td>~0.2</td>
<td>~1.3</td>
<td>~1.0</td>
<td>~0.5</td>
</tr>
<tr>
<td>RCPO</td>
<td>~0.3</td>
<td>~1.2</td>
<td>~1.0</td>
<td>~0.5</td>
</tr>
<tr>
<td>CPO</td>
<td>~0.6</td>
<td>~1.4</td>
<td>~0.5</td>
<td>~0.5</td>
</tr>
<tr>
<td>PCPO</td>
<td>~0.5</td>
<td>~1.0</td>
<td>~0.5</td>
<td>~1.0</td>
</tr>
<tr>
<td>CUP</td>
<td>~0.5</td>
<td>~0.5</td>
<td>~0.5</td>
<td>~1.5</td>
</tr>
<tr>
<td>FOCOPS</td>
<td>~0.7</td>
<td>~0.8</td>
<td>~0.5</td>
<td>~1.0</td>
</tr>
<tr>
<td>MAPPO-Lag</td>
<td>~0.2</td>
<td>~0.1</td>
<td>~0.2</td>
<td>~1.5</td>
</tr>
<tr>
<td>MACPO</td>
<td>~0.1</td>
<td>~0.4</td>
<td>~0.5</td>
<td>~1.0</td>
</tr>
</tbody>
</table>

Figure 8: A bar chart analyzing the performance of different algorithms. The left graph compares episodic reward with PPO-Lag [18] (or MAPPO-Lag [39] for multi-agent). The right graph shows episodic costs proportionally under varying constraints. Single-agent data is from 40 navigation and 6 velocity tasks, and multi-agent data is from all 8 velocity tasks in Safety-Gymnasium.

Table 2: The performance of single-agent algorithms. $\bar{J}^R$ stands for normalized reward from PPO's performance, and $\bar{J}^C$ signifies normalized cost relative to `cost_limit`. The $\uparrow$ indicates higher rewards are better, while the $\downarrow$ indicates lower costs (when beyond the threshold of 1.00) are better. *Gray* and *Black* depicts breach and compliance with the `cost_limit`, while *Green* represents the optimal policy, maximizing reward within safety constraints.

<table>
<thead>
<tr>
<th rowspan="2">Safety Navigation</th>
<th colspan="2">PPO</th>
<th colspan="2">PPO-Lag</th>
<th colspan="2">TRPO-Lag</th>
<th colspan="2">CPPO-PID</th>
<th colspan="2">RCPO</th>
<th colspan="2">CPO</th>
<th colspan="2">PCPO</th>
<th colspan="2">CUP</th>
<th colspan="2">FOCOPS</th>
</tr>
<tr>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th></th></tr>
</thead>
<tbody>
<tr>
<td>ANTBUTTON1</td>
<td>1.00</td>
<td>4.42</td>
<td>0.09</td>
<td>0.86</td>
<td>0.23</td>
<td>1.95</td>
<td>0.10</td>
<td>0.70</td>
<td>0.16</td>
<td>2.07</td>
<td>0.12</td>
<td>4.01</td>
<td>0.03</td>
<td>1.01</td>
<td>0.03</td>
<td>0.17</td>
<td>0.01</td>
<td>0.46</td>
</tr>
<tr>
<td>ANTCIRCLE1</td>
<td>1.00</td>
<td>16.81</td>
<td>0.79</td>
<td>2.56</td>
<td>0.65</td>
<td>1.05</td>
<td>0.69</td>
<td>1.90</td>
<td>0.63</td>
<td>1.04</td>
<td>0.47</td>
<td>1.07</td>
<td>0.28</td>
<td>1.87</td>
<td>0.60</td>
<td>0.82</td>
<td>0.02</td>
<td>1.22</td>
</tr>
<tr>
<td>ANTGOAL1</td>
<td>1.00</td>
<td>1.81</td>
<td>0.26</td>
<td>0.94</td>
<td>0.25</td>
<td>0.74</td>
<td>0.47</td>
<td>1.94</td>
<td>0.29</td>
<td>0.78</td>
<td>0.19</td>
<td>0.55</td>
<td>0.09</td>
<td>0.42</td>
<td>0.34</td>
<td>1.33</td>
<td>0.09</td>
<td>0.67</td>
</tr><tr>
<td>ANTPUSH1</td>
<td>1.00</td>
<td>1.90</td>
<td>0.13</td>
<td>0.00</td>
<td>0.30</td>
<td>0.00</td>
<td>0.13</td>
<td>0.00</td>
<td>0.33</td>
<td>0.00</td>
<td>0.17</td>
<td>0.00</td>
<td>0.07</td>
<td>0.00</td>
<td>0.20</td>
<td>0.00</td>
<td>-0.30</td>
<td>0.03</td>
</tr><tr>
<td>CARBUTTON1</td>
<td>1.00</td>
<td>16.09</td>
<td>0.01</td>
<td>0.47</td>
<td>-0.04</td>
<td>1.08</td>
<td>-0.10</td>
<td>0.40</td>
<td>-0.19</td>
<td>1.73</td>
<td>0.08</td>
<td>1.75</td>
<td>0.02</td>
<td>1.90</td>
<td>0.04</td>
<td>5.50</td>
<td>0.04</td>
<td>1.21</td>
</tr><tr>
<td>CARCIRCLE1</td>
<td>1.00</td>
<td>8.42</td>
<td>0.81</td>
<td>0.82</td>
<td>1.69</td>
<td>2.77</td>
<td>1.61</td>
<td>1.79</td>
<td>1.70</td>
<td>3.11</td>
<td>1.67</td>
<td>3.13</td>
<td>1.41</td>
<td>1.99</td>
<td>0.76</td>
<td>1.04</td>
<td>0.84</td>
<td>1.12</td>
</tr><tr>
<td>CARGOAL1</td>
<td>1.00</td>
<td>2.38</td>
<td>0.43</td>
<td>0.39</td>
<td>0.82</td>
<td>1.09</td>
<td>0.03</td>
<td>2.47</td>
<td>0.55</td>
<td>0.86</td>
<td>0.78</td>
<td>1.63</td>
<td>0.61</td>
<td>1.42</td>
<td>0.19</td>
<td>0.63</td>
<td>0.52</td>
<td>0.93</td>
</tr><tr>
<td>CARPUSH1</td>
<td>1.00</td>
<td>7.16</td>
<td>0.46</td>
<td>0.78</td>
<td>1.38</td>
<td>0.70</td>
<td>0.03</td>
<td>0.47</td>
<td>1.11</td>
<td>1.42</td>
<td>0.83</td>
<td>1.14</td>
<td>0.64</td>
<td>2.36</td>
<td>0.32</td>
<td>0.95</td>
<td>0.29</td>
<td>0.36</td>
</tr><tr>
<td>DOGGOBUTTON1</td>
<td>1.00</td>
<td>7.57</td>
<td>0.01</td>
<td>0.03</td>
<td>0.00</td>
<td>1.27</td>
<td>0.01</td>
<td>0.07</td>
<td>0.01</td>
<td>0.09</td>
<td>0.00</td>
<td>0.15</td>
<td>0.00</td>
<td>0.25</td>
<td>0.02</td>
<td>0.45</td>
<td>0.06</td>
<td>3.68</td>
</tr><tr>
<td>DOGGOCIRCLE1</td>
<td>1.00</td>
<td>33.14</td>
<td>0.77</td>
<td>0.46</td>
<td>0.67</td>
<td>1.37</td>
<td>0.82</td>
<td>2.16</td>
<td>0.55</td>
<td>1.32</td>
<td>0.66</td>
<td>1.22</td>
<td>0.31</td>
<td>0.55</td>
<td>0.80</td>
<td>2.04</td>
<td>0.73</td>
<td>4.49</td>
</tr><tr>
<td>DOGGOGOAL1</td>
<td>1.00</td>
<td>2.28</td>
<td>0.05</td>
<td>0.00</td>
<td>0.18</td>
<td>0.69</td>
<td>0.00</td>
<td>0.00</td>
<td>0.16</td>
<td>2.08</td>
<td>0.30</td>
<td>0.50</td>
<td>0.00</td>
<td>0.00</td>
<td>0.00</td>
<td>0.90</td>
<td>0.04</td>
<td>1.27</td>
</tr><tr>
<td>DOGGOPUSH1</td>
<td>1.00</td>
<td>1.31</td>
<td>0.09</td>
<td>0.00</td>
<td>0.53</td>
<td>0.78</td>
<td>0.32</td>
<td>0.44</td>
<td>0.54</td>
<td>1.55</td>
<td>0.46</td>
<td>0.00</td>
<td>0.36</td>
<td>0.00</td>
<td>0.30</td>
<td>0.68</td>
<td>0.64</td>
<td>3.40</td>
</tr><tr>
<td>POINTBUTTON1</td>
<td>1.00</td>
<td>6.06</td>
<td>0.22</td>
<td>1.32</td>
<td>0.27</td>
<td>1.29</td>
<td>0.00</td>
<td>0.84</td>
<td>0.12</td>
<td>1.13</td>
<td>0.12</td>
<td>1.61</td>
<td>0.08</td>
<td>2.19</td>
<td>0.18</td>
<td>1.26</td>
<td>0.25</td>
<td>1.53</td>
</tr><tr>
<td>POINTCIRCLE1</td>
<td>1.00</td>
<td>8.10</td>
<td>0.86</td>
<td>0.93</td>
<td>1.67</td>
<td>1.35</td>
<td>1.72</td>
<td>2.09</td>
<td>1.66</td>
<td>1.42</td>
<td>1.69</td>
<td>1.74</td>
<td>1.33</td>
<td>2.26</td>
<td>0.82</td>
<td>0.62</td>
<td>0.84</td>
<td>0.89</td>
</tr><tr>
<td>POINTGOAL1</td>
<td>1.00</td>
<td>1.93</td>
<td>0.47</td>
<td>1.50</td>
<td>0.72</td>
<td>0.91</td>
<td>0.31</td>
<td>1.05</td>
<td>0.53</td>
<td>0.99</td>
<td>0.78</td>
<td>1.10</td>
<td>0.71</td>
<td>0.82</td>
<td>0.46</td>
<td>0.73</td>
<td>0.56</td>
<td>1.32</td>
</tr><tr>
<td>POINTPUSH1</td>
<td>1.00</td>
<td>2.31</td>
<td>0.98</td>
<td>1.33</td>
<td>0.85</td>
<td>1.00</td>
<td>0.35</td>
<td>0.35</td>
<td>5.30</td>
<td>0.94</td>
<td>2.22</td>
<td>0.80</td>
<td>1.72</td>
<td>1.25</td>
<td>2.32</td>
<td>0.80</td>
<td>1.13</td>
<td>2.51</td>
</tr><tr>
<td>RACECARBUTTON1</td>
<td>1.00</td>
<td>13.73</td>
<td>-0.01</td>
<td>1.94</td>
<td>-0.02</td>
<td>1.77</td>
<td>-0.16</td>
<td>2.06</td>
<td>-0.07</td>
<td>1.19</td>
<td>0.00</td>
<td>2.44</td>
<td>0.02</td>
<td>1.82</td>
<td>0.00</td>
<td>5.23</td>
<td>-0.10</td>
<td>3.37</td>
</tr><tr>
<td>RACECARCIRCLE1</td>
<td>1.00</td>
<td>15.87</td>
<td>0.83</td>
<td>1.90</td>
<td>0.80</td>
<td>2.18</td>
<td>0.58</td>
<td>1.33</td>
<td>0.83</td>
<td>2.07</td>
<td>0.79</td>
<td>0.81</td>
<td>0.22</td>
<td>2.87</td>
<td>0.74</td>
<td>3.53</td>
<td>0.77</td>
<td>2.11</td>
</tr>
<tr>
<td>RACECARGOAL1</td>
<td>1.00</td>
<td>4.26</td>
<td>0.26</td>
<td>0.51</td>
<td>1.19</td>
<td>0.77</td>
<td>-0.04</td>
<td>1.07</td>
<td>0.88</td>
<td>0.83</td>
<td>1.18</td>
<td>2.58</td>
<td>0.33</td>
<td>0.24</td>
<td>0.13</td>
<td>1.22</td>
<td>0.31</td>
<td>0.62</td>
</tr>
<tr>
<td>RACECARPUSH1</td>
<td>1.00</td>
<td>2.34</td>
<td>-0.40</td>
<td>0.00</td>
<td>0.74</td>
<td>1.79</td>
<td>-0.84</td>
<td>2.87</td>
<td>0.58</td>
<td>1.92</td>
<td>0.94</td>
<td>0.13</td>
<td>-0.16</td>
<td>0.18</td>
<td>-0.06</td>
<td>3.79</td>
<td>0.30</td>
<td>2.04</td>
</tr>
<tr>
<td colspan="18"><b>Safety Velocity</b></td>
<td></td></tr>
<tr>
<td></td>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
<th>$\bar{J}^R \uparrow$</th>
<th>$\bar{J}^C \downarrow$</th>
</tr>
<tr>
<td>ANTVEL</td>
<td>1.00</td>
<td>38.33</td>
<td>0.54</td>
<td>0.22</td>
<td>0.53</td>
<td>0.15</td>
<td>0.51</td>
<td>0.41</td>
<td>0.52</td>
<td>0.56</td>
<td>0.52</td>
<td>0.56</td>
<td>0.38</td>
<td>0.41</td>
<td>0.55</td>
<td>0.94</td>
<td>0.55</td>
<td>0.60</td>
</tr>
<tr>
<td>HALFCHEETAHVEL</td>
<td>1.00</td>
<td>36.77</td>
<td>0.44</td>
<td>0.00</td>
<td>0.43</td>
<td>1.01</td>
<td>0.48</td>
<td>0.04</td>
<td>0.36</td>
<td>0.56</td>
<td>0.40</td>
<td>0.23</td>
<td>0.25</td>
<td>0.63</td>
<td>0.40</td>
<td>0.17</td>
<td>0.42</td>
<td>0.12</td>
</tr>
<tr>
<td>HOPPERVEL</td>
<td>1.00</td>
<td>22.00</td>
<td>0.58</td>
<td>0.89</td>
<td>0.59</td>
<td>0.71</td>
<td>0.73</td>
<td>0.44</td>
<td>0.58</td>
<td>0.59</td>
<td>0.73</td>
<td>0.48</td>
<td>0.65</td>
<td>0.51</td>
<td>0.73</td>
<td>0.21</td>
<td>0.66</td>
<td>0.30</td>
</tr>
<tr>
<td>HUMANOIDVEL</td>
<td>1.00</td>
<td>38.42</td>
<td>0.72</td>
<td>0.76</td>
<td>0.72</td>
<td>2.38</td>
<td>0.73</td>
<td>0.00</td>
<td>0.68</td>
<td>0.82</td>
<td>0.71</td>
<td>0.01</td>
<td>0.64</td>
<td>0.01</td>
<td>0.68</td>
<td>0.80</td>
<td>0.71</td>
<td>0.93</td>
</tr>
<tr>
<td>SWIMMERVEL</td>
<td>1.00</td>
<td>6.61</td>
<td>0.57</td>
<td>1.11</td>
<td>0.66</td>
<td>0.84</td>
<td>0.91</td>
<td>0.92</td>
<td>0.54</td>
<td>0.90</td>
<td>0.51</td>
<td>0.82</td>
<td>0.50</td>
<td>0.69</td>
<td>0.59</td>
<td>0.96</td>
<td>0.47</td>
<td>1.30</td>
</tr>
<tr>
<td>WALKER2DVEL</td>
<td>1.00</td>
<td>36.11</td>
<td>0.44</td>
<td>0.20</td>
<td>0.51</td>
<td>0.77</td>
<td>0.27</td>
<td>0.36</td>
<td>0.49</td>
<td>0.15</td>
<td>0.39</td>
<td>0.81</td>
<td>0.27</td>
<td>0.71</td>
<td>0.44</td>
<td>0.18</td>
<td>0.50</td>
<td>0.16</td>
</tr>
</tbody>
</table>

**Reward and Cost.** Episodic reward and cost exhibit a trade-off relationship. Unconstrained algorithms aim to maximize reward through risky behaviors. HAPPO [37] achieves higher rewards compared to MAPPO [38] across 8 velocity-based tasks, accompanied by a simultaneous increase in average costs. SafeRL algorithms tend to maximize reward while adhering to constraints. As depicted in Table 2, in the velocity task, compared to PPO [43], PPO-Lag [18] achieves a reduction of 98% in cost while only experiencing a decrease of 45% in reward.

**Randomness and Oscillation.** The randomness of tasks is correlated with the oscillation of algorithms' performance. All SafeRL algorithms achieve average episodic costs within the `cost_limit` for velocity tasks. The divergence in episodic rewards between algorithms is negligible, and the distribution of optimal policies is tightly clustered. However, pronounced oscillations are present in navigation tasks characterized by high stochasticity. Out of the 20 navigation tasks examined,

9

optimal policies are spread out more, leading to observable differences in algorithm performance across various tasks.

**Lagrangian vs. Projection.** In contrast to projection-based methods, the Lagrangian-based methods tend to display more oscillation. A notable disparity becomes apparent upon examining the oscillatory patterns in the episodic cost around the designated safety constraints during training, as presented in Figure 8(b). Both CPO [30] and PPO-Lag [18] demonstrate oscillations; however, those exhibited by PPO-Lag are more conspicuous. This discrepancy is manifested in a higher proportion of instances classified as *Strongly Unsafe* and *Strongly Safe* for PPO-Lag, while CPO maintains a more centered distribution. Nevertheless, an excessively cautious policy has the potential to undermine performance. In contrast, the projection-based method PCPO [3] exhibits lower average costs and rewards in navigation and velocity tasks than CPO. This distinction is further accentuated when examining the contrast between MACPO and MAPPO-Lag.

**Lagrangian vs. PID-Lagrangian.** Incorporating a PID controller within the Lagrangian-based framework proves to be effective in mitigating inherent oscillations. As shown in Figure 8, CPPO-PID [44] displays episodic rewards during training that closely resemble those of PPO-Lag. However, CPPO-PID demonstrates a reduced frequency of instances entering the *Strongly Unsafe* region, resulting in a more significant proportion of *Safe* states and improved safety performance.

Table 3: The normalized performance of SafePO's multi-agent algorithms on Safety-Gymnasium.

<table>
    <tr><th></th>
        <th>Safety Velocity</th>
        <th>MAPPO $\bar{J}^R \uparrow$</th>
        <th>MAPPO $\bar{J}^C \downarrow$</th>
        <th>HAPPO $\bar{J}^R \uparrow$</th>
        <th>HAPPO $\bar{J}^C \downarrow$</th>
        <th>MAPPO-Lag $\bar{J}^R \uparrow$</th>
        <th>MAPPO-Lag $\bar{J}^C \downarrow$</th>
        <th>MACPO $\bar{J}^R \uparrow$</th>
        <th>MACPO $\bar{J}^C \downarrow$</th>
    </tr>
    <tr>
        <td>2x4AntVel</td>
        <td>1.00</td>
        <td>35.76</td>
        <td>1.26</td>
        <td>39.12</td>
        <td>0.57</td>
        <td>0.00</td>
        <td>0.51</td>
        <td>0.14</td>
    <td></td></tr>
    <tr>
        <td>4x2AntVel</td>
        <td>1.00</td>
        <td>38.01</td>
        <td>1.07</td>
        <td>34.34</td>
        <td>0.50</td>
        <td>0.00</td>
        <td>0.50</td>
        <td>0.01</td>
    <td></td></tr>
    <tr>
        <td>2x3HalfCheetahVel</td>
        <td>1.00</td>
        <td>39.02</td>
        <td>1.11</td>
        <td>37.70</td>
        <td>0.35</td>
        <td>0.01</td>
        <td>0.49</td>
        <td>1.28</td>
    <td></td></tr>
    <tr>
        <td>6x1HalfCheetahVel</td>
        <td>1.00</td>
        <td>39.23</td>
        <td>1.09</td>
        <td>37.74</td>
        <td>0.28</td>
        <td>0.02</td>
        <td>0.36</td>
        <td>0.37</td>
    <td></td></tr>
    <tr>
        <td>3x1HopperVel</td>
        <td>1.00</td>
        <td>22.58</td>
        <td>1.04</td>
        <td>22.05</td>
        <td>0.47</td>
        <td>0.00</td>
        <td>0.22</td>
        <td>1.03</td>
    <td></td></tr>
    <tr>
        <td>9</td>
        <td>8HumanoidVel</td>
        <td>1.00</td>
        <td>6.34</td>
        <td>2.79</td>
        <td>17.18</td>
        <td>0.54</td>
        <td>0.84</td>
        <td>0.53</td>
        <td>1.30</td>
    </tr>
    <tr>
        <td>2x3Walker2dVel</td>
        <td>1.00</td>
        <td>22.99</td>
        <td>1.55</td>
        <td>33.67</td>
        <td>0.60</td>
        <td>0.01</td>
        <td>0.27</td>
        <td>1.21</td>
    <td></td></tr>
</table>## 7 Limitations and Future Works

Ensuring safety remains a paramount concern. Across various tasks, safety concerns can be transformed into corresponding constraints. However, a limitation of this study is its inability to encompass all forms of constraints. For instance, safety constraints related to human-centric considerations are paramount in human-AI collaboration, yet these considerations have not been fully integrated within the scope of this study. This work focuses on safety tasks within a simulated environment and introduces an extensive testing component. However, the transferability of the results to complex real-world safety-critical applications may be limited. A promising work for the future involves transferring policy refined within the Safety-Gymnasium to physical robotic platforms, which holds profound implications.

10

# References

[1] Tom Carlson and Yiannis Demiris. Increasing robotic wheelchair safety with collaborative control: Evidence from secondary task experiments. In *2010 IEEE International Conference on Robotics and Automation*, pages 5582–5587. IEEE, 2010.

[2] Zhu Ming Bi, Chaomin Luo, Zhonghua Miao, Bing Zhang, WJ Zhang, and Lihui Wang. Safety assurance mechanisms of collaborative robotic systems in manufacturing. *Robotics and Computer-Integrated Manufacturing*, 67:102022, 2021.

[3] Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, and Peter J Ramadge. Projection-based constrained policy optimization. *arXiv preprint arXiv:2010.03152*, 2020.

[4] Fengshuo Bai, Hongming Zhang, Tianyang Tao, Zhiheng Wu, Yanna Wang, and Bo Xu. Picor: Multi-task deep reinforcement learning with policy correction. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(6):6728–6736, Jun. 2023.

[5] Keith W Ross and Ravi Varadarajan. Markov decision processes with sample path constraints: the communicating case. *Operations Research*, 37(5):780–790, 1989.

[6] Eitan Altman. *Constrained Markov decision processes*. Routledge, 2021.

[7] Linrui Zhang, Qin Zhang, Li Shen, Bo Yuan, Xueqian Wang, and Dacheng Tao. Evaluating model-free reinforcement learning toward safety-critical tasks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37(12), pages 15313–15321, 2023.

[8] Jiaming Ji, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, and Yaodong Yang. Omnisafe: An infrastructure for accelerating safe reinforcement learning research, 2023.

[9] Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan, Aidan O’Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. Ai alignment: A comprehensive survey, 2024.

[10] Shuo Feng, Haowei Sun, Xintao Yan, Haojie Zhu, Zhengxia Zou, Shengyin Shen, and Henry X Liu. Dense reinforcement learning for safety validation of autonomous vehicles. *Nature*, 615(7953):620–627, 2023.

[11] Yafei Ou and Mahdi Tavakoli. Towards safe and efficient reinforcement learning for surgical robots using real-time human supervision and demonstration. In *2023 International Symposium on Medical Robotics (ISMR)*, pages 1–7. IEEE, 2023.

[12] Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Chi Zhang, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset, 2023.

[13] Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback, 2023.

[14] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. *arXiv preprint arXiv:1606.01540*, 2016.

[15] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*, 2013.

[16] Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel, Tom Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa. dm_control: Software and tasks for continuous control. *Software Impacts*, 6:100022, 2020.

[17] Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, and Shane Legg. Ai safety gridworlds. *arXiv preprint arXiv:1711.09883*, 2017.

11

[18] Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking safe exploration in deep reinforcement learning. *arXiv preprint arXiv:1910.01708*, 7(1):2, 2019.

[19] Zhaocong Yuan, Adam W Hall, Siqi Zhou, Lukas Brunke, Melissa Greeff, Jacopo Panerati, and Angela P Schoellig. Safe-control-gym: A unified benchmark suite for safe learning-based control and reinforcement learning in robotics. *IEEE Robotics and Automation Letters*, 7(4):11142–11149, 2022.

[20] Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, and Bolei Zhou. Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning. *IEEE transactions on pattern analysis and machine intelligence*, 45(3):3461–3475, 2022.

[21] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. *arXiv preprint arXiv:2108.10470*, 2021.

[22] Farama Foundation. A standard api for single-agent reinforcement learning environments, with popular reference environments and related utilities (formerly gym). https://github.com/Farama-Foundation/Gymnasium, 2022.

[23] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In *2012 IEEE/RSJ international conference on intelligent robots and systems*, pages 5026–5033. IEEE, 2012.

[24] Mengdi Xu, Zuxin Liu, Peide Huang, Wenhao Ding, Zhepeng Cen, Bo Li, and Ding Zhao. Trustworthy reinforcement learning against intrinsic vulnerabilities: Robustness, safety, and generalizability. *arXiv preprint arXiv:2209.08025*, 2022.

[25] Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, Yaodong Yang, and Alois Knoll. A review of safe reinforcement learning: Methods, theory and applications. *arXiv preprint arXiv:2205.10330*, 2022.

[26] Lodewijk CM Kallenberg. Linear programming and finite markovian control problems. *MC Tracts*, 1983.

[27] Javier García and Fernando Fernández. A comprehensive survey on safe reinforcement learning. *Journal of Machine Learning Research*, 16(1):1437–1480, 2015.

[28] Juntao Dai, Jiaming Ji, Long Yang, Qian Zheng, and Gang Pan. Augmented proximal policy optimization for safe reinforcement learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(6):7288–7295, Jun. 2023.

[29] Weidong Huang, Jiaming Ji, Borong Zhang, Chunhe Xia, and Yaodong Yang. Safedreamer: Safe reinforcement learning with world models, 2023.

[30] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In *International conference on machine learning*, pages 22–31. PMLR, 2017.

[31] Yiming Zhang, Quan Vuong, and Keith Ross. First order constrained optimization in policy space. *Advances in Neural Information Processing Systems*, 33:15338–15349, 2020.

[32] Long Yang, Jiaming Ji, Juntao Dai, Yu Zhang, Pengfei Li, and Gang Pan. Cup: A conservative update policy algorithm for safe reinforcement learning, 2022.

[33] Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

[34] Patrick L Jacobs, Stephen E Olvey, Brad M Johnson, and Kelly Cohn. Physiological responses to high-speed, open-wheel racecar driving. *Medicine and science in sports and exercise*, 34(12):2085–2090, 2002.

[35] Johannes Betz, Alexander Wischnewski, Alexander Heilmeier, Felix Nobis, Tim Stahl, Leonhard Hermansdorfer, and Markus Lienkamp. A software architecture for an autonomous racecar. In *2019 IEEE 89th Vehicular Technology Conference (VTC2019-Spring)*, pages 1–6. IEEE, 2019.

12

[36] Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Böhmer, and Shimon Whiteson. Deep multi-agent reinforcement learning for decentralized continuous cooperative control. *arXiv preprint arXiv:2003.06709*, 19, 2020.

[37] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fanglei Sun, Jun Wang, and Yaodong Yang. Trust region policy optimisation in multi-agent reinforcement learning. *arXiv preprint arXiv:2109.11251*, 2021.

[38] Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. *Advances in Neural Information Processing Systems*, 35:24611–24624, 2022.

[39] Shangding Gu, Jakub Grudzien Kuba, Munning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois Knoll, and Yaodong Yang. Multi-agent constrained policy optimisation. *arXiv preprint arXiv:2110.02793*, 2021.

[40] Yecheng Jason Ma, Andrew Shen, Osbert Bastani, and Jayaraman Dinesh. Conservative and adaptive penalty for model-based safe reinforcement learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 36(5), pages 5404–5412, 2022.

[41] Yarden As, Ilnura Usmanova, Sebastian Curi, and Andreas Krause. Constrained policy optimization via bayesian world models. *arXiv preprint arXiv:2201.09802*, 2022.

[42] Yuanpei Chen, Tianhao Wu, Shengjie Wang, Xidong Feng, Jiechuan Jiang, Zongqing Lu, Stephen McAleer, Hao Dong, Song-Chun Zhu, and Yaodong Yang. Towards human-level bimanual dexterous manipulation with reinforcement learning. *Advances in Neural Information Processing Systems*, 35:5150–5163, 2022.

[43] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

[44] Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive safety in reinforcement learning by pid lagrangian methods. In *International Conference on Machine Learning*, pages 9133–9143. PMLR, 2020.

[45] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*, 2015.

[46] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

13

# A Details of Experimental Results

## A.1 Hyperparameters Analysis

This section presents the disclosure of SafePO hyperparameters settings and their rationales. We employed the Generalized Advantage Estimation (GAE)[45] method to estimate the values of rewards and cost advantages and used Adam[46] for learning the neural network parameters.

**Single-agent Algorithm Settings.** The models employed in the single-agent algorithms were 3-layer MLPs with Tanh activation functions and hidden layer sizes of [64, 64], for more intricate navigation agents Ant and Doggo, hidden layers of [256, 256] were employed.

**Multi-agent Algorithms Settings.** The models employed in the multi-agent algorithms were 3-layer MLPs with ReLU activation functions and hidden layer sizes of [128, 128].

Table 4: Hyperparameters of SafePO algorithms in Safety-Gymnasium tasks. Second-order algorithms set the parameters to the actor model directly, instead of iterative gradient descent, so the *Actor Learning Rate* of them are marked Gray.

<table>
  <thead>
    <tr>
      <th colspan="2">PG/PPO/PPO-Lag</th>
      <th colspan="2">TRPO/TRPO-Lag</th>
      <th colspan="2">CPPO-PID</th>
      <th colspan="2">NPG/RCPO</th>
      <th colspan="2">HAPPO/MAPPO/MAPPO-Lag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
    </tr>
<tr>
      <td>Target KL</td>
      <td>0.02</td>
      <td>Target KL</td>
      <td>0.01</td>
      <td>Target KL</td>
      <td>0.02</td>
      <td>Target KL</td>
      <td>0.01</td>
      <td>Target KL</td>
      <td>0.016</td>
    </tr>
<tr>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
    </tr>
<tr>
      <td>Number of SGD Iterations</td>
      <td>40</td>
      <td>Number of SGD Iterations</td>
      <td>10</td>
      <td>Number of SGD Iterations</td>
      <td>40</td>
      <td>Number of SGD Iterations</td>
      <td>10</td>
      <td>Number of SGD Iterations</td>
      <td>5</td>
    </tr>
<tr>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>10000</td>
    </tr>
<tr>
      <td>Actor Learning Rate</td>
      <td>0.0003</td>
      <td>Actor Learning Rate</td>
      <td>None</td>
      <td>Actor Learning Rate</td>
      <td>0.0003</td>
      <td>Actor Learning Rate</td>
      <td>None</td>
      <td>Actor Learning Rate</td>
      <td>0.0005</td>
    </tr>
<tr>
      <td>Critic Learning Rate</td>
      <td>0.0003</td>
      <td>Critic Learning Rate</td>
      <td>0.001</td>
      <td>Critic Learning Rate</td>
      <td>0.0003</td>
      <td>Critic Learning Rate</td>
      <td>0.001</td>
      <td>Critic Learning Rate</td>
      <td>0.0005</td>
    </tr>
<tr>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
    </tr>
<tr>
      <td>Clip Coefficient</td>
      <td>0.20</td>
      <td>Conjugate Gradient Iterations</td>
      <td>15</td>
      <td>Clip Coefficient</td>
      <td>0.20</td>
      <td>Conjugate Gradient Iterations</td>
      <td>15</td>
      <td>Clip Coefficient</td>
      <td>0.20</td>
    </tr>
<tr>
      <td>Lagrangian Initial Value</td>
      <td>0.001</td>
      <td>Lagrangian Initial Value</td>
      <td>0.001</td>
      <td>PID Controller Kp</td>
      <td>0.10</td>
      <td>Lagrangian Initial Value</td>
      <td>0.001</td>
      <td>Lagrangian Initial Value</td>
      <td>0.00001</td>
    </tr>
<tr>
      <td>Lagrangian Learning Rate</td>
      <td>0.035</td>
      <td>Lagrangian Learning Rate</td>
      <td>0.035</td>
      <td>PID Controller Ki</td>
      <td>0.01</td>
      <td>Lagrangian Learning Rate</td>
      <td>0.035</td>
      <td>Lagrangian Learning Rate</td>
      <td>0.78</td>
    </tr>
<tr>
      <td>Lagrangian Optimizer</td>
      <td>Adam</td>
      <td>Lagrangian Optimizer</td>
      <td>Adam</td>
      <td>PID Controller Kd</td>
      <td>0.01</td>
      <td>Lagrangian Optimizer</td>
      <td>Adam</td>
      <td>Lagrangian Optimizer</td>
      <td>SGD</td>
    </tr>
  </tbody>
</table>
<table>
  <thead>
    <tr>
      <th colspan="2">CPO</th>
      <th colspan="2">PCPO</th>
      <th colspan="2">CUP</th>
      <th colspan="2">FOCOPS</th>
      <th colspan="2">MACPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
      <td>Discount Factor $\gamma$</td>
      <td>0.99</td>
    </tr>
<tr>
      <td>Target KL</td>
      <td>0.01</td>
      <td>Target KL</td>
      <td>0.01</td>
      <td>Target KL</td>
      <td>0.02</td>
      <td>Target KL</td>
      <td>0.02</td>
      <td>Target KL</td>
      <td>0.01</td>
    </tr>
<tr>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
      <td>GAE $\lambda$</td>
      <td>0.95</td>
    </tr>
<tr>
      <td>Number of SGD Iterations</td>
      <td>10</td>
      <td>Number of SGD Iterations</td>
      <td>10</td>
      <td>Number of SGD Iteration</td>
      <td>40</td>
      <td>Number of SGD Iteration</td>
      <td>40</td>
      <td>Number of SGD Iteration</td>
      <td>15</td>
    </tr>
<tr>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>20000</td>
      <td>Training Batch Size</td>
      <td>10000</td>
    </tr>
<tr>
      <td>Actor Learning Rate</td>
      <td>None</td>
      <td>Actor Learning Rate</td>
      <td>None</td>
      <td>Actor Learning Rate</td>
      <td>0.0003</td>
      <td>Actor Learning Rate</td>
      <td>0.0003</td>
      <td>Actor Learning Rate</td>
      <td>None</td>
    </tr>
<tr>
      <td>Critic Learning Rate</td>
      <td>0.001</td>
      <td>Critic Learning Rate</td>
      <td>0.001</td>
      <td>Critic Learning Rate</td>
      <td>0.0003</td>
      <td>Critic Learning Rate</td>
      <td>0.0003</td>
      <td>Critic Learning Rate</td>
      <td>0.0005</td>
    </tr>
<tr>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
      <td>Cost Limit</td>
      <td>25.00</td>
    </tr>
<tr>
      <td>Conjugate Gradient Iterations</td>
      <td>15</td>
      <td>Conjugate Gradient Iterations</td>
      <td>15</td>
      <td>Clip Coefficient</td>
      <td>0.20</td>
      <td>Clip Coefficient</td>
      <td>0.20</td>
      <td>Conjugate Gradient Iterations</td>
      <td>10</td>
    </tr>
<tr>
      <td>CPO Searching Steps</td>
      <td>15</td>
      <td>PCPO Searching Steps</td>
      <td>200</td>
      <td>CUP $\lambda$</td>
      <td>0.95</td>
      <td>FOCOPS $\lambda$</td>
      <td>1.50</td>
      <td>MACPO Searching Steps</td>
      <td>10</td>
    </tr>
<tr>
      <td>Step Fraction</td>
      <td>0.80</td>
      <td>Step Fraction</td>
      <td>0.80</td>
      <td>CUP $\nu$</td>
      <td>2.00</td>
      <td>FOCOPS $\nu$</td>
      <td>2.00</td>
      <td>Step Fraction</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>

**Lagrangian Multiplier Settings.** Lagrangian-based methods are sensitive to hyperparameters. We present the following detailed description of the settings for both the naive and the PID-controlled Lagrangian multiplier.

*   **Lagrangian Initial Value:** The initial value of the Lagrangian multiplier. It impacts the early-stage performance of the Lagrangian-based methods. A higher initial value promotes safer exploration but may impede task completion. Conversely, a lower initial value delays the agent's exploration of safe policies.
*   **Lagrangian Learning Rate:** The learning rate of the Lagrangian multiplier. A high learning rate induces excessive oscillations, impedes convergence speed, and hinders the algorithm's ability to attain the desired solution. Conversely, a low learning rate slows down convergence and adversely affects training.
*   **PID Controller Kp:** The PID controller's proportional gain determines the output's response to changes in the episodic costs. If `pid_kp` is too large, the Lagrangian multiplier oscillates, and performance deteriorates. If `pid_kp` is too small, the Lagrangian multiplier updates slowly, also impacting performance negatively.
*   **PID Controller Kd:** The PID controller's derivative gain governs the output's response to changes in the episodic costs. If `pid_kd` is too large, the Lagrangian multiplier becomes excessively sensitive to noise or changes in the episodic costs, leading to instability or oscillations. If `pid_kd` is too small, the Lagrangian multiplier may not respond quickly or accurately enough to changes in the episodic costs.
*   **PID Controller Ki:** The PID controller's integral gain determines the controller's ability to eliminate the steady-state error by integrating the episodic costs over time. If `pid_ki` is too large, the Lagrangian multiplier may become overly responsive to previous errors, adversely affecting performance.

14

# A.2 Performance Table of Safety-Gymnasium

Table 5: The performance of SafePO algorithms on Safety-Gymnasium. All experimental outcomes were derived from 10 assessment iterations encompassing multiple random seeds and under the experimental setting of `cost_limit=25.00`. The $\uparrow$ indicates higher rewards are better, while the $\downarrow$ indicates lower costs (when beyond the threshold of 25.00) are better. *Gray* and *Black* depicts breach and compliance with the `cost_limit`, while *Green* represents the optimal policy, maximizing reward within safety constraints.

## (a) The performance of SafePO single-agent algorithms on Safety-Gymnasium.

<table>
    <tr>
        <th>Safety Navigation</th>
        <th>PPO $J^R$</th>
        <th>PPO $J^C$</th>
        <th>PPO-Lag $J^R$</th>
        <th>PPO-Lag $J^C$</th>
        <th>CPPO-PID $J^R$</th>
        <th>CPPO-PID $J^C$</th>
        <th>TRPO-Lag $J^R$</th>
        <th>TRPO-Lag $J^C$</th>
        <th>RCPO $J^R$</th>
        <th>RCPO $J^C$</th>
        <th>CPO $J^R$</th>
        <th>CPO $J^C$</th>
        <th>PCPO $J^R$</th>
        <th>PCPO $J^C$</th>
        <th>CUP $J^R$</th>
        <th>CUP $J^C$</th>
        <th>FOCOPS $J^R$</th>
        <th>FOCOPS $J^C$</th>
    </tr>
    <tr>
        <td>ANTButton1</td>
        <td>38.70</td>
        <td>110.60</td>
        <td>3.63</td>
        <td>21.60</td>
        <td>4.06</td>
        <td>17.45</td>
        <td>8.93</td>
        <td>48.70</td>
        <td>6.16</td>
        <td>51.70</td>
        <td>4.50</td>
        <td>100.30</td>
        <td>1.27</td>
        <td>25.35</td>
        <td>1.26</td>
        <td>4.25</td>
        <td>0.22</td>
        <td>11.55</td>
    </tr>
    <tr>
        <td>ANTButton2</td>
        <td>36.15</td>
        <td>95.00</td>
        <td>2.72</td>
        <td>14.85</td>
        <td>2.86</td>
        <td>28.70</td>
        <td>8.66</td>
        <td>49.45</td>
        <td>8.66</td>
        <td>37.40</td>
        <td>4.63</td>
        <td>35.60</td>
        <td>3.04</td>
        <td>27.50</td>
        <td>1.60</td>
        <td>32.90</td>
        <td>-0.04</td>
        <td>6.80</td>
    </tr>
    <tr>
        <td>ANTCircle1</td>
        <td>94.04</td>
        <td>420.30</td>
        <td>74.31</td>
        <td>63.90</td>
        <td>64.90</td>
        <td>47.50</td>
        <td>61.02</td>
        <td>26.30</td>
        <td>59.42</td>
        <td>26.00</td>
        <td>43.74</td>
        <td>26.80</td>
        <td>26.47</td>
        <td>46.85</td>
        <td>56.77</td>
        <td>20.50</td>
        <td>2.27</td>
        <td>30.50</td>
    </tr>
    <tr>
        <td>ANTCircle2</td>
        <td>84.80</td>
        <td>736.00</td>
        <td>65.72</td>
        <td>22.45</td>
        <td>64.49</td>
        <td>39.85</td>
        <td>66.75</td>
        <td>22.75</td>
        <td>63.04</td>
        <td>19.00</td>
        <td>53.74</td>
        <td>43.90</td>
        <td>16.41</td>
        <td>15.85</td>
        <td>42.65</td>
        <td>10.80</td>
        <td>4.78</td>
        <td>66.30</td>
    </tr>
    <tr>
        <td>ANTGoal1</td>
        <td>82.02</td>
        <td>45.30</td>
        <td>21.33</td>
        <td>23.60</td>
        <td>38.79</td>
        <td>48.55</td>
        <td>20.64</td>
        <td>18.50</td>
        <td>23.38</td>
        <td>19.60</td>
        <td>15.35</td>
        <td>13.80</td>
        <td>7.31</td>
        <td>10.50</td>
        <td>27.98</td>
        <td>33.25</td>
        <td>6.99</td>
        <td>16.75</td>
    </tr>
    <tr>
        <td>ANTGoal2</td>
        <td>86.14</td>
        <td>165.60</td>
        <td>1.01</td>
        <td>0.00</td>
        <td>0.10</td>
        <td>0.00</td>
        <td>4.44</td>
        <td>13.45</td>
        <td>6.27</td>
        <td>54.00</td>
        <td>0.85</td>
        <td>4.60</td>
        <td>0.02</td>
        <td>0.00</td>
        <td>0.76</td>
        <td>1.15</td>
        <td>0.08</td>
        <td>1.15</td>
    </tr>
    <tr>
        <td>ANTPush1</td>
        <td>0.46</td>
        <td>47.55</td>
        <td>0.06</td>
        <td>0.00</td>
        <td>0.06</td>
        <td>0.00</td>
        <td>0.14</td>
        <td>0.00</td>
        <td>0.15</td>
        <td>0.00</td>
        <td>0.08</td>
        <td>0.00</td>
        <td>0.03</td>
        <td>0.00</td>
        <td>0.09</td>
        <td>0.00</td>
        <td>-0.14</td>
        <td>0.70</td>
    </tr>
    <tr>
        <td>ANTPush2</td>
        <td>0.77</td>
        <td>139.20</td>
        <td>0.01</td>
        <td>0.02</td>
        <td>0.02</td>
        <td>0.00</td>
        <td>0.01</td>
        <td>0.00</td>
        <td>0.10</td>
        <td>0.00</td>
        <td>0.05</td>
        <td>0.00</td>
        <td>0.02</td>
        <td>0.00</td>
        <td>0.02</td>
        <td>0.10</td>
        <td>0.07</td>
        <td>0.20</td>
    </tr>
    <tr>
        <td>CARButton1</td>
        <td>15.74</td>
        <td>398.81</td>
        <td>0.11</td>
        <td>11.87</td>
        <td>-1.70</td>
        <td>10.03</td>
        <td>-0.66</td>
        <td>26.90</td>
        <td>-3.16</td>
        <td>43.20</td>
        <td>1.30</td>
        <td>43.73</td>
        <td>0.27</td>
        <td>47.60</td>
        <td>0.68</td>
        <td>137.47</td>
        <td>0.60</td>
        <td>30.23</td>
    </tr>
    <tr>
        <td>CARButton2</td>
        <td>19.32</td>
        <td>333.82</td>
        <td>1.23</td>
        <td>46.14</td>
        <td>-1.83</td>
        <td>26.55</td>
        <td>-2.23</td>
        <td>17.98</td>
        <td>-0.02</td>
        <td>27.09</td>
        <td>-0.10</td>
        <td>36.97</td>
        <td>0.49</td>
        <td>38.54</td>
        <td>0.80</td>
        <td>154.50</td>
        <td>0.07</td>
        <td>53.49</td>
    </tr>
    <tr>
        <td>CARCircle1</td>
        <td>21.92</td>
        <td>208.73</td>
        <td>17.91</td>
        <td>20.62</td>
        <td>35.71</td>
        <td>44.87</td>
        <td>37.42</td>
        <td>69.30</td>
        <td>37.78</td>
        <td>77.77</td>
        <td>37.10</td>
        <td>78.23</td>
        <td>31.37</td>
        <td>49.80</td>
        <td>16.89</td>
        <td>25.88</td>
        <td>18.63</td>
        <td>27.98</td>
    </tr>
    <tr>
        <td>CARCircle2</td>
        <td>19.75</td>
        <td>401.83</td>
        <td>16.27</td>
        <td>29.88</td>
        <td>30.80</td>
        <td>40.37</td>
        <td>33.23</td>
        <td>54.20</td>
        <td>33.74</td>
        <td>42.17</td>
        <td>33.42</td>
        <td>78.97</td>
        <td>27.93</td>
        <td>70.40</td>
        <td>14.74</td>
        <td>15.46</td>
        <td>15.60</td>
        <td>31.20</td>
    </tr>
    <tr>
        <td>CARGoal1</td>
        <td>32.57</td>
        <td>58.91</td>
        <td>14.57</td>
        <td>9.84</td>
        <td>1.00</td>
        <td>61.71</td>
        <td>27.49</td>
        <td>27.28</td>
        <td>18.49</td>
        <td>21.45</td>
        <td>26.23</td>
        <td>40.71</td>
        <td>20.64</td>
        <td>35.41</td>
        <td>6.38</td>
        <td>15.67</td>
        <td>17.58</td>
        <td>23.22</td>
    </tr>
    <tr>
        <td>CARGoal2</td>
        <td>31.59</td>
        <td>215.74</td>
        <td>0.59</td>
        <td>16.81</td>
        <td>0.12</td>
        <td>23.09</td>
        <td>3.27</td>
        <td>47.18</td>
        <td>2.61</td>
        <td>25.45</td>
        <td>3.55</td>
        <td>32.63</td>
        <td>1.83</td>
        <td>57.82</td>
        <td>2.45</td>
        <td>125.80</td>
        <td>3.28</td>
        <td>23.01</td>
    </tr>
    <tr>
        <td>CARPush1</td>
        <td>1.13</td>
        <td>181.04</td>
        <td>0.49</td>
        <td>19.60</td>
        <td>0.03</td>
        <td>11.83</td>
        <td>1.48</td>
        <td>17.60</td>
        <td>1.19</td>
        <td>35.50</td>
        <td>0.89</td>
        <td>28.50</td>
        <td>0.68</td>
        <td>59.03</td>
        <td>0.34</td>
        <td>23.86</td>
        <td>0.31</td>
        <td>8.96</td>
    </tr>
    <tr>
        <td>CARPush2</td>
        <td>1.03</td>
        <td>46.87</td>
        <td>0.54</td>
        <td>43.32</td>
        <td>0.57</td>
        <td>37.37</td>
        <td>0.43</td>
        <td>38.63</td>
        <td>0.12</td>
        <td>27.57</td>
        <td>0.15</td>
        <td>19.03</td>
        <td>0.29</td>
        <td>60.10</td>
        <td>0.41</td>
        <td>82.20</td>
        <td>-0.28</td>
        <td>40.42</td>
    </tr>
    <tr>
        <td>DOGGOButton1</td>
        <td>27.23</td>
        <td>189.30</td>
        <td>0.33</td>
        <td>0.80</td>
        <td>0.22</td>
        <td>1.67</td>
        <td>0.01</td>
        <td>31.75</td>
        <td>0.30</td>
        <td>2.25</td>
        <td>0.03</td>
        <td>3.70</td>
        <td>-0.06</td>
        <td>6.20</td>
        <td>0.67</td>
        <td>11.17</td>
        <td>1.52</td>
        <td>91.90</td>
    </tr>
    <tr>
        <td>DOGGOButton2</td>
        <td>29.84</td>
        <td>194.60</td>
        <td>0.10</td>
        <td>1.00</td>
        <td>0.16</td>
        <td>2.70</td>
        <td>-0.05</td>
        <td>17.05</td>
        <td>0.07</td>
        <td>0.00</td>
        <td>0.03</td>
        <td>1.40</td>
        <td>0.01</td>
        <td>8.01</td>
        <td>0.35</td>
        <td>43.37</td>
        <td>0.22</td>
        <td>2.10</td>
    </tr>
    <tr>
        <td>DOGGOCircle2</td>
        <td>41.90</td>
        <td>442.70</td>
        <td>30.13</td>
        <td>14.20</td>
        <td>34.82</td>
        <td>62.03</td>
        <td>21.97</td>
        <td>46.75</td>
        <td>20.68</td>
        <td>37.35</td>
        <td>20.41</td>
        <td>32.55</td>
        <td>15.41</td>
        <td>24.05</td>
        <td>33.08</td>
        <td>58.33</td>
        <td>28.91</td>
        <td>122.80</td>
    </tr>
    <tr>
        <td>DOGGOCircle1</td>
        <td>41.61</td>
        <td>828.50</td>
        <td>32.03</td>
        <td>11.50</td>
        <td>34.26</td>
        <td>53.93</td>
        <td>27.86</td>
        <td>34.20</td>
        <td>22.93</td>
        <td>32.90</td>
        <td>27.65</td>
        <td>30.55</td>
        <td>12.94</td>
        <td>13.70</td>
        <td>33.45</td>
        <td>50.97</td>
        <td>30.29</td>
        <td>112.20</td>
    </tr>
    <tr>
        <td>DOGGOGOAL1</td>
        <td>43.10</td>
        <td>57.10</td>
        <td>2.00</td>
        <td>0.00</td>
        <td>0.13</td>
        <td>0.00</td>
        <td>7.88</td>
        <td>17.25</td>
        <td>6.82</td>
        <td>52.05</td>
        <td>12.73</td>
        <td>12.40</td>
        <td>0.14</td>
        <td>0.00</td>
        <td>0.16</td>
        <td>22.47</td>
        <td>1.88</td>
        <td>31.80</td>
    </tr>
    <tr>
        <td>DOGGOGOAL2</td>
        <td>42.04</td>
        <td>123.30</td>
        <td>0.06</td>
        <td>0.00</td>
        <td>0.09</td>
        <td>0.00</td>
        <td>0.02</td>
        <td>0.00</td>
        <td>0.06</td>
        <td>0.00</td>
        <td>0.03</td>
        <td>0.00</td>
        <td>0.06</td>
        <td>0.00</td>
        <td>0.28</td>
        <td>3.33</td>
        <td>0.08</td>
        <td>0.00</td>
    </tr>
    <tr>
        <td>DOGGOPUSH2</td>
        <td>0.82</td>
        <td>32.70</td>
        <td>-0.02</td>
        <td>0.00</td>
        <td>0.08</td>
        <td>0.00</td>
        <td>0.16</td>
        <td>0.00</td>
        <td>0.18</td>
        <td>0.00</td>
        <td>0.54</td>
        <td>39.08</td>
        <td>0.14</td>
        <td>0.00</td>
        <td>0.22</td>
        <td>52.70</td>
        <td>0.52</td>
        <td>0.00</td>
    </tr>
    <tr>
        <td>DOGGOPUSH1</td>
        <td>0.90</td>
        <td>32.70</td>
        <td>0.08</td>
        <td>0.00</td>
        <td>0.29</td>
        <td>11.03</td>
        <td>0.48</td>
        <td>19.40</td>
        <td>0.49</td>
        <td>38.80</td>
        <td>0.41</td>
        <td>0.00</td>
        <td>0.32</td>
        <td>0.00</td>
        <td>0.27</td>
        <td>17.10</td>
        <td>0.58</td>
        <td>85.10</td>
    </tr>
    <tr>
        <td>POINTButton1</td>
        <td>26.10</td>
        <td>151.38</td>
        <td>5.83</td>
        <td>32.98</td>
        <td>-0.12</td>
        <td>20.88</td>
        <td>7.13</td>
        <td>32.31</td>
        <td>3.01</td>
        <td>28.14</td>
        <td>3.20</td>
        <td>40.16</td>
        <td>2.18</td>
        <td>54.74</td>
        <td>4.70</td>
        <td>31.39</td>
        <td>6.60</td>
        <td>38.27</td>
    </tr>
    <tr>
        <td>POINTButton2</td>
        <td>27.96</td>
        <td>166.74</td>
        <td>0.27</td>
        <td>31.49</td>
        <td>0.44</td>
        <td>30.87</td>
        <td>4.87</td>
        <td>24.94</td>
        <td>7.90</td>
        <td>53.82</td>
        <td>5.58</td>
        <td>47.68</td>
        <td>1.12</td>
        <td>41.49</td>
        <td>3.52</td>
        <td>61.98</td>
        <td>1.29</td>
        <td>26.13</td>
    </tr>
    <tr>
        <td>POINTCircle1</td>
        <td>54.57</td>
        <td>202.54</td>
        <td>47.00</td>
        <td>23.28</td>
        <td>93.84</td>
        <td>52.23</td>
        <td>90.87</td>
        <td>33.83</td>
        <td>90.65</td>
        <td>35.53</td>
        <td>92.10</td>
        <td>43.50</td>
        <td>72.81</td>
        <td>56.53</td>
        <td>44.98</td>
        <td>15.50</td>
        <td>46.06</td>
        <td>22.36</td>
    </tr>
    <tr>
        <td>POINTCircle2</td>
        <td>54.39</td>
        <td>397.54</td>
        <td>41.60</td>
        <td>19.92</td>
        <td>83.67</td>
        <td>45.27</td>
        <td>82.62</td>
        <td>6.63</td>
        <td>83.39</td>
        <td>7.40</td>
        <td>85.22</td>
        <td>21.20</td>
        <td>79.22</td>
        <td>22.67</td>
        <td>41.45</td>
        <td>30.98</td>
        <td>42.38</td>
        <td>20.96</td>
    </tr>
    <tr>
        <td>POINTGoal1</td>
        <td>26.32</td>
        <td>48.20</td>
        <td>12.46</td>
        <td>37.62</td>
        <td>8.15</td>
        <td>26.31</td>
        <td>18.99</td>
        <td>22.87</td>
        <td>13.90</td>
        <td>24.66</td>
        <td>20.52</td>
        <td>27.44</td>
        <td>18.79</td>
        <td>20.48</td>
        <td>11.99</td>
        <td>18.15</td>
        <td>14.77</td>
        <td>32.95</td>
    </tr>
    <tr>
        <td>POINTGoal2</td>
        <td>26.43</td>
        <td>159.28</td>
        <td>0.59</td>
        <td>59.43</td>
        <td>-0.56</td>
        <td>60.37</td>
        <td>4.18</td>
        <td>26.80</td>
        <td>1.84</td>
        <td>29.19</td>
        <td>2.65</td>
        <td>42.40</td>
        <td>1.32</td>
        <td>37.66</td>
        <td>1.00</td>
        <td>162.97</td>
        <td>2.71</td>
        <td>18.63</td>
    </tr>
    <tr>
        <td>POINTPUSH1</td>
        <td>0.82</td>
        <td>57.80</td>
        <td>0.80</td>
        <td>33.18</td>
        <td>0.29</td>
        <td>8.87</td>
        <td>0.70</td>
        <td>24.93</td>
        <td>4.35</td>
        <td>23.47</td>
        <td>1.82</td>
        <td>19.90</td>
        <td>1.41</td>
        <td>31.33</td>
        <td>1.90</td>
        <td>19.98</td>
        <td>0.93</td>
        <td>62.64</td>
    </tr>
    <tr>
        <td>POINTPUSH2</td>
        <td>1.39</td>
        <td>42.82</td>
        <td>0.52</td>
        <td>25.90</td>
        <td>1.01</td>
        <td>25.87</td>
        <td>1.05</td>
        <td>56.07</td>
        <td>0.54</td>
        <td>29.83</td>
        <td>1.50</td>
        <td>29.17</td>
        <td>0.59</td>
        <td>27.57</td>
        <td>1.26</td>
        <td>56.08</td>
        <td>0.44</td>
        <td>39.24</td>
    </tr>
    <tr>
        <td>RACECARButton1</td>
        <td>8.48</td>
        <td>343.15</td>
        <td>-0.05</td>
        <td>48.55</td>
        <td>-1.37</td>
        <td>51.57</td>
        <td>-0.18</td>
        <td>44.25</td>
        <td>-0.63</td>
        <td>29.70</td>
        <td>0.02</td>
        <td>60.95</td>
        <td>0.13</td>
        <td>45.45</td>
        <td>0.04</td>
        <td>130.63</td>
        <td>-0.88</td>
        <td>84.20</td>
    </tr>
    <tr>
        <td>RACECARButton2</td>
        <td>5.77</td>
        <td>284.15</td>
        <td>-0.58</td>
        <td>22.35</td>
        <td>-0.64</td>
        <td>31.80</td>
        <td>0.19</td>
        <td>65.00</td>
        <td>0.38</td>
        <td>18.45</td>
        <td>0.01</td>
        <td>32.90</td>
        <td>0.04</td>
        <td>51.95</td>
        <td>-0.40</td>
        <td>72.57</td>
        <td>-0.40</td>
        <td>57.65</td>
    </tr>
    <tr>
        <td>RACECARCircle1</td>
        <td>81.62</td>
        <td>396.80</td>
        <td>67.49</td>
        <td>47.55</td>
        <td>47.66</td>
        <td>33.13</td>
        <td>65.54</td>
        <td>54.55</td>
        <td>67.39</td>
        <td>51.75</td>
        <td>64.77</td>
        <td>20.20</td>
        <td>18.05</td>
        <td>71.65</td>
        <td>60.68</td>
        <td>88.33</td>
        <td>62.77</td>
        <td>52.85</td>
    </tr>
    <tr>
        <td>RACECARCircle2</td>
        <td>82.61</td>
        <td>831.00</td>
        <td>46.85</td>
        <td>26.05</td>
        <td>28.04</td>
        <td>47.37</td>
        <td>60.83</td>
        <td>45.65</td>
        <td>61.40</td>
        <td>33.00</td>
        <td>59.17</td>
        <td>48.30</td>
        <td>8.81</td>
        <td>35.05</td>
        <td>41.50</td>
        <td>16.13</td>
        <td>52.38</td>
        <td>35.10</td>
    </tr>
    <tr>
        <td>RACECARGOAL1</td>
        <td>11.29</td>
        <td>106.40</td>
        <td>2.90</td>
        <td>12.70</td>
        <td>-0.42</td>
        <td>26.87</td>
        <td>13.40</td>
        <td>19.20</td>
        <td>9.89</td>
        <td>20.70</td>
        <td>13.30</td>
        <td>64.50</td>
        <td>3.72</td>
        <td>5.90</td>
        <td>1.47</td>
        <td>30.57</td>
        <td>3.47</td>
        <td>15.40</td>
    </tr>
    <tr>
        <td>RACECARGOAL2</td>
        <td>9.61</td>
        <td>158.25</td>
        <td>0.08</td>
        <td>54.40</td>
        <td>-0.85</td>
        <td>30.50</td>
        <td>0.40</td>
        <td>14.30</td>
        <td>0.55</td>
        <td>16.80</td>
        <td>1.19</td>
        <td>109.85</td>
        <td>0.69</td>
        <td>41.90</td>
        <td>-0.09</td>
        <td>62.33</td>
        <td>0.17</td>
        <td>93.05</td>
    </tr>
    <tr>
        <td>RACECARPush1</td>
        <td>0.50</td>
        <td>58.45</td>
        <td>-0.20</td>
        <td>0.00</td>
        <td>-0.42</td>
        <td>71.83</td>
        <td>0.37</td>
        <td>44.75</td>
        <td>0.29</td>
        <td>48.00</td>
        <td>0.47</td>
        <td>3.30</td>
        <td>-0.08</td>
        <td>4.50</td>
        <td>-0.03</td>
        <td>94.70</td>
        <td>0.15</td>
        <td>51.00</td>
    </tr>
    <tr>
        <td>RACECARPush2</td>
        <td>0.58</td>
        <td>213.95</td>
        <td>0.37</td>
        <td>43.85</td>
        <td>-0.08</td>
        <td>24.07</td>
        <td>-0.12</td>
        <td>5.50</td>
        <td>-0.03</td>
        <td>0.00</td>
        <td>0.23</td>
        <td>9.55</td>
        <td>-0.51</td>
        <td>49.75</td>
        <td>-1.54</td>
        <td>101.50</td>
        <td>-0.54</td>
        <td>56.00</td>
    </tr>
    <tr>
        <td>**Safety Velocity**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
        <td>**$J^R$**</td>
        <td>**$J^C$**</td>
    </tr>
    <tr>
        <td>ANTVEL</td>
        <td>5899.64</td>
        <td>943.57</td>
        <td>3221.90</td>
        <td>5.43</td>
        <td>3070.67</td>
        <td>10.23</td>
        <td>3157.40</td>
        <td>3.63</td>
        <td>3087.03</td>
        <td>14.12</td>
        <td>3116.77</td>
        <td>14.10</td>
        <td>2276.19</td>
        <td>10.18</td>
        <td>3297.29</td>
        <td>23.56</td>
        <td>3291.30</td>
        <td>15.07</td>
    </tr>
    <tr>
        <td>HALFCHEETAHVEL</td>
        <td>7013.92</td>
        <td>933.18</td>
        <td>3025.42</td>
        <td>0.00</td>
        <td>3336.80</td>
        <td>1.09</td>
        <td>2952.08</td>
        <td>25.23</td>
        <td>2520.50</td>
        <td>13.95</td>
        <td>2738.36</td>
        <td>5.68</td>
        <td>1743.71</td>
        <td>15.64</td>
        <td>2765.42</td>
        <td>4.28</td>
        <td>2873.14</td>
        <td>2.88</td>
    </tr>
    <tr>
        <td>HOPPERVEL</td>
        <td>2378.23</td>
        <td>543.14</td>
        <td>1347.98</td>
        <td>22.30</td>
        <td>1709.13</td>
        <td>11.11</td>
        <td>1377.89</td>
        <td>17.67</td>
        <td>1355.69</td>
        <td>14.85</td>
        <td>1713.22</td>
        <td>12.12</td>
        <td>1519.59</td>
        <td>12.79</td>
        <td>1716.35</td>
        <td>5.37</td>
        <td>1538.79</td>
        <td>7.43</td>
    </tr>
    <tr>
        <td>HUMANOIDVEL</td>
        <td>9117.61</td>
        <td>959.76</td>
        <td>6586.70</td>
        <td>18.95</td>
        <td>6620.69</td>
        <td>0.00</td>
        <td>6552.06</td>
        <td>59.85</td>
        <td>6236.18</td>
        <td>20.57</td>
        <td>6486.40</td>
        <td>0.22</td>
        <td>5863.98</td>
        <td>0.18</td>
        <td>6181.80</td>
        <td>19.88</td>
        <td>6502.90</td>
        <td>23.23</td>
    </tr>
    <tr>
        <td>SWIMMERVEL</td>
        <td>121.23</td>
        <td>171.21</td>
        <td>68.10</td>
        <td>27.68</td>
        <td>109.34</td>
        <td>22.92</td>
        <td>79.63</td>
        <td>20.98</td>
        <td>64.73</td>
        <td>22.56</td>
        <td>61.49</td>
        <td>20.46</td>
        <td>60.48</td>
        <td>17.31</td>
        <td>70.86</td>
        <td>23.93</td>
        <td>55.87</td>
        <td>32.62</td>
    </tr>
    <tr>
        <td>WALKER2DVEL</td>
        <td>6312.27</td>
        <td>899.82</td>
        <td>2756.61</td>
        <td>4.90</td>
        <td>1704.06</td>
        <td>8.90</td>
        <td>3209.78</td>
        <td>19.18</td>
        <td>3072.07</td>
        <td>3.72</td>
        <td>2440.82</td>
        <td>20.15</td>
        <td>1698.31</td>
        <td>17.73</td>
        <td>2739.50</td>
        <td>4.39</td>
        <td>3116.08</td>
        <td>3.93</td>
    </tr>
</table>## (b) The performance of SafePO multi-agent algorithms on Safety-Gymnasium.

<table>
    <tr>
        <th>Safety Velocity</th>
        <th>MAPPO $J^R$</th>
        <th>MAPPO $J^C$</th>
        <th>HAPPO $J^R$</th>
        <th>HAPPO $J^C$</th>
        <th>MAPPO-Lag $J^R$</th>
        <th>MAPPO-Lag $J^C$</th>
        <th>MACPO $J^R$</th>
        <th>MACPO $J^C$</th>
    </tr>
    <tr>
        <td>2x4ANTVEL</td>
        <td>4259.52</td>
        <td>894.06</td>
        <td>5368.61</td>
        <td>978.06</td>
        <td>2423.47</td>
        <td>0.00</td>
        <td>2169.23</td>
        <td>3.39</td>
    </tr>
    <tr>
        <td>4x2ANTVEL</td>
        <td>4309.05</td>
        <td>950.33</td>
        <td>4613.69</td>
        <td>858.50</td>
        <td>2171.40</td>
        <td>0.00</td>
        <td>2172.31</td>
        <td>0.17</td>
    </tr>
    <tr>
        <td>2x3HALFCHEETAHVEL</td>
        <td>5057.63</td>
        <td>975.50</td>
        <td>5605.98</td>
        <td>942.56</td>
        <td>1750.96</td>
        <td>0.33</td>
        <td>2470.29</td>
        <td>32.06</td>
    </tr>
    <tr>
        <td>6x1HALFCHEETAHVEL</td>
        <td>5061.53</td>
        <td>980.67</td>
        <td>5540.57</td>
        <td>943.56</td>
        <td>1439.38</td>
        <td>0.61</td>
        <td>1830.65</td>
        <td>9.33</td>
    </tr>
    <tr>
        <td>3x1HOPPERVEL</td>
        <td>2115.35</td>
        <td>564.56</td>
        <td>2207.50</td>
        <td>551.33</td>
        <td>1002.01</td>
        <td>0.00</td>
        <td>461.25</td>
        <td>25.78</td>
    </tr>
    <tr>
        <td>9x8HUMANOIDVEL</td>
        <td>974.50</td>
        <td>158.61</td>
        <td>2718.48</td>
        <td>429.61</td>
        <td>526.69</td>
        <td>21.00</td>
        <td>512.29</td>
        <td>32.50</td>
    </tr>
    <tr>
        <td>2x1SWIMMERVEL</td>
        <td>39.88</td>
        <td>101.89</td>
        <td>51.95</td>
        <td>267.00</td>
        <td>27.89</td>
        <td>59.73</td>
        <td>-4.02</td>
        <td>20.83</td>
    </tr>
    <tr>
        <td>2x3WALKER2DVEL</td>
        <td>2691.41</td>
        <td>574.72</td>
        <td>4183.34</td>
        <td>841.83</td>
        <td>1618.98</td>
        <td>0.33</td>
        <td>714.18</td>
        <td>30.22</td>
    </tr>
</table>## Experimental Results Analysis.

During the observation of the experimental results, we have discovered some Insightful findings that are presented below:

*   The Lagrangian method is a promising yet constrained baseline approach, successfully optimizing rewards while adhering to constraints. However, its effectiveness heavily relies on hyperparameters configuration, as discussed in Table A.1. Consequently, despite being a dependable baseline, the Lagrangian method is not exempt from limitations.
*   Second-order algorithms perform worse in achieving higher rewards in the MuJoCo velocity series but better in navigation series tasks that require higher safety standards, i.e., achieving similar or approximate rewards while minimizing the number and smoothness of cost violations.

15

# B Details Documentation of Gymnasium-based Learning Environments

## B.1 Single-agent Specification


(a) Point: front
(b) Point: back
(c) Point: left
(d) Point: right

Figure 9: A different view of the robot: Point.

Table 6: The overall information of Point
<table>
    <tr>
        <td>Specific Action Space</td>
        <td>Box(-1.0, 1.0, (2,), float64)</td>
    </tr>
    <tr>
        <td>Specific Observation Space</td>
        <td>(12, )</td>
    </tr>
    <tr>
        <td>Observation High</td>
        <td>inf</td>
    </tr>
    <tr>
        <td>Observation Low</td>
        <td>-inf</td>
    </tr>
</table>Table 7: The specific observation space of Point
<table>
    <tr>
        <th>Size</th>
        <th>Observation</th>
        <th>Min</th>
        <th>Max</th>
        <th>Name (in XML file)</th>
        <th>Joint/Site</th>
        <th>Unit</th>
    </tr>
    <tr>
        <td>3</td>
        <td>accelerometer</td>
        <td>-inf</td>
        <td>inf</td>
        <td>accelerometer</td>
        <td>site</td>
        <td>acceleration (m/s^2)</td>
    </tr>
    <tr>
        <td>3</td>
        <td>velocimeter</td>
        <td>-inf</td>
        <td>inf</td>
        <td>velocimeter</td>
        <td>site</td>
        <td>velocity (m/s)</td>
    </tr>
    <tr>
        <td>3</td>
        <td>gyro</td>
        <td>-inf</td>
        <td>inf</td>
        <td>gyro</td>
        <td>site</td>
        <td>anglular velocity (rad/s)</td>
    </tr>
    <tr>
        <td>3</td>
        <td>magnetometer</td>
        <td>-inf</td>
        <td>inf</td>
        <td>magnetometer</td>
        <td>site</td>
        <td>magnetic flux (Wb)</td>
    </tr>
</table>Table 8: The specific action space of Point
<table>
    <tr>
        <th>Num</th>
        <th>Action</th>
        <th>Control Min</th>
        <th>Control Max</th>
        <th>Name (in XML file)</th>
        <th>Joint/Site</th>
        <th>Unit</th>
    </tr>
    <tr>
        <td>0</td>
        <td>force applied on the agent to move forward or backward</td>
        <td>-1</td>
        <td>1</td>
        <td>x</td>
        <td>site</td>
        <td>force (N)</td>
    </tr>
    <tr>
        <td>1</td>
        <td>velocity of the agent, which is around the z-axis</td>
        <td>-1</td>
        <td>1</td>
        <td>z</td>
        <td>hinge</td>
        <td>velocity (m/s)</td>
    </tr>
</table>**Point:** As shown in Figure 9, Point operating within a 2D plane is equipped with two distinct actuators: one for rotation and another for forward/backward movement. This decomposed control system greatly facilitates the navigation of the robot. Moreover, there is a small square positioned in front of the robot, aiding in the visual identification of its orientation. Additionally, this square plays a crucial role in assisting the robot, named Point, to effectively push any boxes encountered during its tasks. The overall information of Point, the specific action and observation space of Point is shown in Table 6, Table 8, Table 7.


(a) Car: front
(b) Car: back
(c) Car: left
(d) Car: right

Figure 10: A different view of the robot: Car.

Table 9: The overall information of Car
<table>
    <tr>
        <td>Specific Action Space</td>
        <td>Box(-1.0, 1.0, (2,), float64)</td>
    </tr>
    <tr>
        <td>Specific Observation Space</td>
        <td>(24, )</td>
    </tr>
    <tr>
        <td>Observation High</td>
        <td>inf</td>
    </tr>
    <tr>
        <td>Observation Low</td>
        <td>-inf</td>
    </tr>
</table>16

Table 10: The specific action space of Car
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint/Site</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>force to applied on left wheel</td>
      <td>-1</td>
      <td>1</td>
      <td>left</td>
      <td>hinge</td>
      <td>force (N)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>force to applied on right wheel</td>
      <td>-1</td>
      <td>1</td>
      <td>right</td>
      <td>hinge</td>
      <td>force (N)</td>
    </tr>
  </tbody>
</table>

Table 11: The specific observation space of Car
<table>
  <tbody>
    <tr>
      <td>Size</td>
      <td>Observation</td>
      <td>Min</td>
      <td>Max</td>
      <td>Name (in XML file)</td>
      <td>Joint/Site</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Quaternions of the rear wheel which are turned into 3x3 rotation matrices</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ballquat_rear</td>
      <td>ball</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Angle velocity of the rear wheel</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ballangvel_rear</td>
      <td>ball</td>
      <td>anglular velocity (rad/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>accelerometer</td>
      <td>-inf</td>
      <td>inf</td>
      <td>accelerometer</td>
      <td>site</td>
      <td>acceleration (m/s^{}2)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>velocimeter</td>
      <td>-inf</td>
      <td>inf</td>
      <td>velocimeter</td>
      <td>site</td>
      <td>velocity (m/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>gyro</td>
      <td>-inf</td>
      <td>inf</td>
      <td>gyro</td>
      <td>site</td>
      <td>anglular velocity (rad/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>magnetometer</td>
      <td>-inf</td>
      <td>inf</td>
      <td>magnetometer</td>
      <td>site</td>
      <td>magnetic flux (Wb)</td>
    </tr>
  </tbody>
</table>

**Car:** As shown in Figure 10, the robot in question operates in three dimensions and features two independently driven parallel wheels, along with a freely rolling rear wheel. This design requires coordinated operation of the two drives for both steering and forward/backward movement. While the robot shares similarities with a basic Point robot, it possesses added complexity. The overall information of Car, the specific action and observation space of Car is shown in Table 9, Table 10, Table 11.


(a) Racecar: front
(b) Racecar: back
(c) Racecar: left
(d) Racecar: right

Figure 11: A different view of the robot: Racecar.

Table 12: The overall information of Racear
<table>
  <tbody>
    <tr>
      <td>Specific Action Space</td>
      <td>Box([-20. -0.785], [20. 0.785], (2,), float64)</td>
    </tr>
    <tr>
      <td>Specific Observation Space</td>
      <td>(12, )</td>
    </tr>
    <tr>
      <td>Observation High</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>Observation Low</td>
      <td>-inf</td>
    </tr>
  </tbody>
</table>

Table 13: The specific action space of Racecar
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint/Site</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Velocity of the rear wheels.</td>
      <td>-20</td>
      <td>20</td>
      <td>diff_ring</td>
      <td>hinge</td>
      <td>velocity (m/s)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Angle of the front wheel.</td>
      <td>-0.785</td>
      <td>0.785</td>
      <td>steering_hinge</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
  </tbody>
</table>

Table 14: The specific observation space of Racecar
<table>
  <tbody>
    <tr>
      <td>Size</td>
      <td>Observation</td>
      <td>Min</td>
      <td>Max</td>
      <td>Name (in XML file)</td>
      <td>Joint/Site</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>3</td>
      <td>accelerometer</td>
      <td>-inf</td>
      <td>inf</td>
      <td>accelerometer</td>
      <td>site</td>
      <td>acceleration (m/s^{}2)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>velocimeter</td>
      <td>-inf</td>
      <td>inf</td>
      <td>velocimeter</td>
      <td>site</td>
      <td>velocity (m/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>gyro</td>
      <td>-inf</td>
      <td>inf</td>
      <td>gyro</td>
      <td>site</td>
      <td>anglular velocity (rad/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>magnetometer</td>
      <td>-inf</td>
      <td>inf</td>
      <td>magnetometer</td>
      <td>site</td>
      <td>magnetic flux (Wb)</td>
    </tr>
  </tbody>
</table>

**Racecar.** As shown in Figure 11, the robot is closer to realistic car dynamics, moving in three dimensions, it has one velocity servo and one position servo, one to adjusts the rear wheel speed to

17

the target speed and the other to adjust the front wheel steering angle to the target angle. Racecar references the widely known MIT Racecar project’s dynamics model. For it to accomplish the specified goal, it must coordinate the relationship between the steering angle of the tires and the speed, just like a human driving a car. The overall information of Racecar, the specific action and observation space of Racecar is shown in Table 12, Table 13, Table 14.

Table 15: The overall information of Ant
<table>
    <tr>
        <th>Specific Action Space</th>
        <th>Box(-1.0, 1.0, (8,), float64)</th>
    </tr>
    <tr>
        <td>Specific Observation Space</td>
        <td>(40, )</td>
    </tr>
    <tr>
        <td>Observation High</td>
        <td>inf</td>
    </tr>
    <tr>
        <td>Observation Low</td>
        <td>-inf</td>
    </tr>
</table>Table 16: The specific action space of Ant
<table>
    <tr>
        <th>Num</th>
        <th>Action</th>
        <th>Control Min</th>
        <th>Control Max</th>
        <th>Name (in XML file)</th>
        <th>Joint/Site</th>
        <th>Unit</th>
    </tr>
    <tr>
        <td>0</td>
        <td>torque applied on the rotor between the torso and front left hip</td>
        <td>-1</td>
        <td>1</td>
        <td>hip_1 (front_left_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>1</td>
        <td>torque applied on the rotor between the front left two links</td>
        <td>-1</td>
        <td>1</td>
        <td>angle_1 (front_left_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>2</td>
        <td>torque applied on the rotor between the torso and front right hip</td>
        <td>-1</td>
        <td>1</td>
        <td>hip_2 (front_right_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>3</td>
        <td>torque applied on the rotor between the front right two links</td>
        <td>-1</td>
        <td>1</td>
        <td>angle_2 (front_right_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>4</td>
        <td>torque applied on the rotor between the torso and back left hip</td>
        <td>-1</td>
        <td>1</td>
        <td>hip_3 (back_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>5</td>
        <td>torque applied on the rotor between the back left two links</td>
        <td>-1</td>
        <td>1</td>
        <td>angle_3 (back_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>6</td>
        <td>torque applied on the rotor between the torso and back right hip</td>
        <td>-1</td>
        <td>1</td>
        <td>hip_4 (right_back_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
    <tr>
        <td>7</td>
        <td>torque applied on the rotor between the back right two links</td>
        <td>-1</td>
        <td>1</td>
        <td>angle_4 (right_back_leg)</td>
        <td>hinge</td>
        <td>torque (N m)</td>
    </tr>
</table>**Ant.** As depicted in Figure 12, the quadrupedal robot, inspired by the model proposed in [45]. It consists of a torso and four interconnected legs. Each leg is composed of two hinged connecting limbs, which, in turn, are connected to the torso via hinges. To achieve movement in the desired direction, coordination of the four legs is required by applying moments to the eight hinge drivers. For a comprehensive understanding of the robot, please refer to Table 15, Table 16, and Table 17, which provide an overview of the Ant robot, its specific action space, and observation space, respectively.


(a) Ant: front (b) Ant: back (c) Ant: left (d) Ant: right
Figure 12: A different view of the robot: Ant.

18

Table 17: The specific observation space of Ant
<table>
  <tbody>
    <tr>
      <td>Size</td>
      <td>Observation</td>
      <td>Min</td>
      <td>Max</td>
      <td>Name (in XML file)</td>
      <td>Joint/Site</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>3</td>
      <td>accelerometer</td>
      <td>-inf</td>
      <td>inf</td>
      <td>accelerometer</td>
      <td>site</td>
      <td>acceleration (m/s^2)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>velocimeter</td>
      <td>-inf</td>
      <td>inf</td>
      <td>velocimeter</td>
      <td>site</td>
      <td>velocity (m/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>gyro</td>
      <td>-inf</td>
      <td>inf</td>
      <td>gyro</td>
      <td>site</td>
      <td>anglular velocity (rad/s)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>magnetometer</td>
      <td>-inf</td>
      <td>inf</td>
      <td>magnetometer</td>
      <td>site</td>
      <td>magnetic flux (Wb)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and front left link</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of the angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between front left links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and front right link</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of the angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between front right links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_3 (back_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and back left link</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of the angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_3 (back_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between back left links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and back right link</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>angular velocity of the angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>angle (rad)</td>
    </tr>
    <tr>
      <td></td>
      <td>between back right links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>z-coordinate of the torso</td>
      <td>-inf</td>
      <td>inf</td>
      <td>torso</td>
      <td>site</td>
      <td>position (m)</td>
    </tr>
    <tr>
      <td></td>
      <td>(centre).</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>3</td>
      <td>xyz-coordinate angular</td>
      <td>-inf</td>
      <td>inf</td>
      <td>torso</td>
      <td>site</td>
      <td>angular velocity (rad/s)</td>
    </tr>
    <tr>
      <td></td>
      <td>velocity of the tors.</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_3 (back_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_3 (back_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>hip_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>sin() and cos() of angle</td>
      <td>-inf</td>
      <td>inf</td>
      <td>ankle_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>unitless</td>
    </tr>
    <tr>
      <td></td>
      <td>between torso and first link on front left</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

## B.2 Multi-agents Specification


(a) 2-ant: Render (b) 2-ant-diag: Render (c) 4-ant: Render (d) ant: Dynamics
Figure 13: A different view of the MA-Ant.

**2-ant.** The Ant is partitioned into 2 parts, the front part (containing the front legs) and the back part (containing the back legs). The action space of agent-0 and agent-1 as shown in Table 18 and Table 19.

19

Table 18: The specific action space of 2-ant: agent-0
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor between the torso and front left hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor between the front left two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Torque applied on the rotor between the torso and front right hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Torque applied on the rotor between the front right two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
  </tbody>
</table>

Table 19: The specific action space of 2-ant: agent-1
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor between the torso and front left hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor between the front left two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Torque applied on the rotor between the torso and front right hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Torque applied on the rotor between the front right two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
  </tbody>
</table>

**2-ant-diag.** The Ant is partitioned into 2 parts, split diagonally, the front part (containing the front legs) and the back part (containing the back legs). The action space of agent-0 and agent-1 as shown in Table 20 and Table 21.

Table 20: The specific action space of 2-ant-diag: agent-0
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor between the torso and front left hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor between the front left two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Torque applied on the rotor between the torso and back right hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Torque applied on the rotor between the back right two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
  </tbody>
</table>

Table 21: The specific action space of 4-ant: agent-1
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor between the torso and front right hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor between the front right two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Torque applied on the rotor between the torso and back left hip</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_3 (back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Torque applied on the rotor between the back left two links</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_3 (back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
  </tbody>
</table>

**4-ant.** The Ant is partitioned into 4 parts, with each part corresponding to a leg of the ant. The action space of agent-0, agent-1, agent-2, and agent-3 as shown in Table 22, Table 23, Table 24 and Table 25.

20

Table 22: The specific action space of 4-ant: agent-0
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the torso and front left hip</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_1 (front_left_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the front left two links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

Table 23: The specific action space of 2-ant-diag: agent-1
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the torso and front right hip</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_2 (front_right_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the front right two links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

Table 24: The specific action space of 4-ant: agent-2
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_3 (back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the torso and back left hip</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_3 (back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the back left two links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

Table 25: The specific action space of 4-ant: agent-3
<table>
  <tbody>
    <tr>
      <td>Num</td>
      <td>Action</td>
      <td>Control Min</td>
      <td>Control Max</td>
      <td>Name (in XML file)</td>
      <td>Joint</td>
      <td>Unit</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>hip_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the torso and back right hip</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>Torque applied on the rotor</td>
      <td>-1</td>
      <td>1</td>
      <td>angle_4 (right_back_leg)</td>
      <td>hinge</td>
      <td>torque (N m)</td>
    </tr>
    <tr>
      <td></td>
      <td>between the back right two links</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

In addition to the robots mentioned in this paper, we also provide other multi-agent versions of robots. Due to space constraints, we did not elaborate on them extensively in the paper. However, you can refer to https://www.safety-gymnasium.com/ for more detailed information.

## B.3 Task Representation


Figure 14: Tasks of Gymnasium-based Environments.

As shown in Figure 14, the Gymnasium-based learning environments support the following tasks:

**Velocity:** the robot aims to facilitate coordinated leg movement of the robot in the forward (right) direction by exerting torques on the hinges.

**Run:** the robot starts with a random initial direction and a specific initial speed as it embarks on a journey to reach the opposite side of the map.

**Circle:** the reward is maximized by moving along the green circle, and the agent is not allowed to enter the outside of the red region, so its optimal constrained path follows the line segments $AD$ and $BC$. The reward function: $R(s) = \frac{v^T [-y, x]}{1 + ||[x, y]||_2 - d}$, the cost function is $C(s) = 1 [|x| > x_{lim}]$, where $x, y$ are the coordinates in the plane, $v$ is the velocity, and $d, x_{lim}$ are environmental parameters.

21

**Goal:** the robot navigates to multiple goal positions. After successfully reaching a goal, its location is randomly reset while maintaining the overall layout. Achieving a goal position, indicated by entering the goal circle, yields a sparse reward. Additionally, a dense reward encourages the robot’s progress by rewarding proximity to the goal.

**Push:** the objective is to move a box to a series of goal positions. Like the goal task, a new random goal location is generated after each successful achievement. The sparse reward is earned when the yellow box enters the designated goal circle. The dense reward consists of two components: one for moving the agent closer to the box and another for bringing the box closer to the final goal.

**Button:** the objective is to activate a series of goal buttons distributed throughout the environment. The agent’s goal is to navigate towards and make contact with the currently highlighted button, known as the goal button. Once the correct button is pressed, a new goal button is selected and highlighted while preserving the rest of the environment. The sparse reward is earned upon successfully pressing the current goal button, while the dense reward component provides a bonus for progressing toward the highlighted goal button.

## B.4 Constraint Specification


Figure 15: Constraints of Gymnasium-based Environments.

**Velocity-Constraint** consists of a series of safety tasks based on MuJoCo agents [23]. In these tasks, agents, such as Ant, HalfCheetah, and Humanoid, are trained to move faster for higher rewards, while also being imposed a velocity constraint for safety considerations. Formally, for an agent moving on a two-dimensional plane, the velocity is calculated as $v(s, a) = \sqrt{v_x^2 + v_y^2}$; for an agent moving along a straight line, the velocity is calculated as $v(s, a) = |v_x|$, where $v_x$, $v_y$ are the velocities of the agent in the $x$ and $y$ directions respectively. Then, $cost(s, a) = [v(s, a) > v_{limit}]$, Here, $[P]$ denotes a notation where the value is 1 if the proposition $P$ is true, and 0 otherwise.

**Pillars** are employed to represent large cylindrical obstacles within the environment. In the general setting, contact with a pillar incurs costs.

**Hazards** are utilized to model areas within the environment that pose a risk, resulting in costs when an agent enters such areas.

**Sigwalls** are designed specifically for Circle tasks. They serve as visual representations of two or four solid walls, which limit the circular area to a smaller region. Crossing the wall from inside the safe area to the outside incurs costs.

**Vases** are specifically designed for Goal tasks. They represent static and fragile objects within the environment. Touching or displacing these objects incurs costs for the agent.

**Gremlins** are specifically employed in the Button tasks. They represent moving objects within the environment that can interact with the agent.

## B.5 Vision-only Tasks

In recent years, vision-only SafeRL has gained significant attention as a focal point of research, primarily due to its applicability in real-world contexts [40; 41]. While the initial iteration of Safety Gym offered rudimentary visual input support, there is room for enhancing the realism and complexity of its environment. To effectively evaluate vision-based safe reinforcement learning algorithms, we have devised some more realistic visual tasks utilizing MuJoCo. This enhanced environment facilitates the incorporation of both RGB and RGB-d inputs. More details can be

22

referred to our online documentation: https://www.safety-gymnasium.com/en/latest/environments/safe_vision.html.


(a) BuildingButton0 (b) BuildingButton1 (c) BuildingButton2
Figure 16: Overview of BuildingButton tasks.

**The Level 0 of BuildingButton** requires the agent to operate multiple machines within a construction site.

**The Level 1 of BuildingButton** requires the agent to proficiently and accurately operate multiple machines within a construction site, while concurrently evading other robots and obstacles present in the area.

**The Level 2 of BuildingButton** requires the agent to proficiently and accurately operate multiple machines within a construction site, while concurrently evading a heightened number of other robots and obstacles in the area.


(a) BuildingGoal0 (b) BuildingGoal1 (c) BuildingGoal2
Figure 17: Overview of BuildingGoal tasks.

**The Level 0 of BuildingGoal** requires the agent to dock at designated positions within a construction site.

**The Level 1 of BuildingGoal** requires the agent to dock at designated positions within a construction site while ensuring to avoid entry into hazardous areas.

**The Level 2 of BuildingGoal** requires the agent to dock at designated positions within a construction site, while ensuring to avoid entry into hazardous areas and circumventing the site’s exhaust fans.

23

(a) BuildingPush0 (b) BuildingPush1 (c) BuildingPush2
Figure 18: Overview of BuildingPush tasks.

**The Level 0 of BuildingPush** requires the agent to relocate the box to designated locations within a construction site.
**The Level 1 of BuildingPush** requires the agent to relocate the box to designated locations within a construction site while avoiding areas demarcated as restricted.
**The Level 2 of BuildingPush** requires the agent to relocate the box to designated locations within a construction while avoiding numerous hazardous fuel drums and areas demarcated as restricted.


(a) Race0 (b) Race1 (c) Race2
Figure 19: Overview of Race tasks.

**The Level 0 of Race** requires the agent to reach the goal position.
**The Level 1 of Race** requires the agent to reach the goal position while ensuring it avoids straying into the grass and prevents collisions with roadside objects.
**The Level 2 of Race** requires the agent to reach the goal position from a distant starting point while ensuring it avoids straying into the grass and prevents collisions with roadside objects.

24

(a) FormulaOne0 (b) FormulaOne1 (c) FormulaOne2
Figure 20: Overview of FormulaOne tasks.

**The Level 0 of FormulaOne** requires the agent to maximize its reach to the goal position. For each episode, the agent is randomly initialized at one of the seven checkpoints.
**The Level 1 of FormulaOne** requires the agent to maximize its reach to the goal position while circumventing barriers and racetrack fences. For each episode, the agent is randomly initialized at one of the seven checkpoints.
**The Level 2 of FormulaOne** requires the agent to maximize its reach to the goal position while circumventing barriers and racetrack fences. For each episode, the agent is randomly initialized at one of the seven checkpoints. Notably, the barriers surrounding the checkpoints are denser.

## B.6 Some Issues about Safety Gym

(a) Safety-Gymnasium
(b) Safety-Gym
Figure 21: The difference between Safety-Gymnasium and Safety Gym.

25

**The bug of Natural Lidar.** As shown in Figure 21, the original Natural Lidar in Safe-Gym<sup>7</sup> has a problem of not being able to detect low-lying objects, which may affect comprehensive environmental observations.

**The problem of observation space.** In Safety Gym, by default, the observation space is presented as a one-dimensional array. The implementation leads to all ranges in observation space to be $[-\infty, +\infty]$, as shown in the following code:

```python
if self.observation_flatten:
    self.obs_flat_size = sum([np.prod(i.shape) for i in
        self.obs_space_dict.values()])
    self.observation_space = gym.spaces.Box(-np.inf, np.inf,
        (self.obs_flat_size,), dtype=np.float32)
```

While this representation does not lead to behavioral errors in the environment, it can be somewhat misleading for users. To address this issue, we have implemented the Gymnasium’s flatten mechanism in the Safety Gym to handle the representation of the observation space. This mechanism reorganizes the observation space into a more intuitive and easily understandable format, enabling users to process and analyze the observation data more effectively.

```python
self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)

if self.observation_flatten:
    self.observation_space = gymnasium.spaces.utils.flatten_space(
        self.obs_info.obs_space_dict
    )
else:
    self.observation_space = self.obs_info.obs_space_dict
assert self.obs_info.obs_space_dict.contains(
    obs
), f'Bad obs {obs} {self.obs_info.obs_space_dict}'

if self.observation_flatten:
    obs =
        gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict,
        obs)
    return obs
```

**Missing cost information.** In Safety Gym, by default, there are only two possible outputs for the cost: 0 and 1, representing whether a cost is incurred or not.

```python
# Optionally remove shaping from reward functions.
if self.constrain_indicator:
    for k in list(cost.keys()):
        cost[k] = float(cost[k] > 0.0)  # Indicator function
```

We believe that this representation method loses some information. For example, when the robot collides with a vase and causes the vase to move at different velocities, there should be different cost values associated with it to indicate subtle differences in violating constraint behaviors. Additionally, these costs incurred by the actions are accumulated into the total cost. In typical cases, algorithms use the total cost to update the policy if the total cost generated by different obstacles is limited to only two states 0 and 1, the learning potential for multiple constraints is lost when multiple costs are triggered simultaneously.

**Neglected dependency maintenance leads to conflicts.**

The **numpy =1.17.4** will cause the following problems:

```text
ValueError: numpy.ndarray size changed, may indicate binary
    incompatibility. Expected 96 from C header, got 80 from PyObject

AttributeError: module 'numpy' has no attribute 'complex'.
```

---
<sup>7</sup>https://github.com/openai/safety-gym

26

# C Details of Isaac Gym-based Learning Environments

## C.1 Supported Agents

Safety-DexteroudsHand is based on Bi-DexHands (refer to [42] for more details). Bi-DexHands aims to establish a comprehensive learning framework for two Shadow Hands, enabling them to possess a wide range of skills similar to those of humans. The Shadow Hand’s joint limitations are as follows (refer to Table 26). The thumb exhibits 5 degrees of freedom with 5 joints, while the other fingers have 3 degrees of freedom and 4 joints each. The joints located at the fingertips are not controllable. Similar to human fingers, the distal joints of the fingers are interconnected, ensuring that the angle of the middle joint is always greater than or equal to that of the distal joint. This design allows the middle phalange to be curved while the distal phalange remains straight. Additionally, an extra joint (LF5) is located at the end of the little finger, enabling it to rotate in the same direction as the thumb. The wrist comprises two joints, facilitating a complete 360-degree rotation of the entire hand.

Table 26: Finger range of motion.
<table>
<tbody>
<tr>
<td>Joints</td>
<td>Corresponds to the number of Figure 22</td>
<td>Min</td>
<td>Max</td>
</tr>
<tr>
<td>Finger Distal (FF1,MF1,RF1,LF1)</td>
<td>15, 11, 7, 3</td>
<td>0°</td>
<td>90°</td>
</tr>
<tr>
<td>Finger Middle (FF2,MF2,RF2,LF2)</td>
<td>16, 12, 8, 4</td>
<td>0°</td>
<td>90°</td>
</tr><tr>
<td>Finger Base Abduction (FF3,MF3,RF3,LF3)</td>
<td>17, 13, 9, 5</td>
<td>-15°</td>
<td>90°</td>
</tr><tr>
<td>Finger Base Lateral (FF4,MF4,RF4,LF4)</td>
<td>18, 14, 10, 6</td>
<td>-20°</td>
<td>20°</td>
</tr><tr>
<td>Little Finger Rotation(LF5)</td>
<td>19</td>
<td>0°</td>
<td>45°</td>
</tr><tr>
<td>Thumb Distal (TH1)</td>
<td>20</td>
<td>-15°</td>
<td>90°</td>
</tr><tr>
<td>Thumb Middle (TH2)</td>
<td>21</td>
<td>-30°</td>
<td>30°</td>
</tr><tr>
<td>Thumb Base Abduction (TH3)</td>
<td>22</td>
<td>-12°</td>
<td>12°</td>
</tr><tr>
<td>Thumb Base Lateral (TH4)</td>
<td>23</td>
<td>0°</td>
<td>70°</td>
</tr>
<tr>
<td>Thumb Base Rotation (TH5)</td>
<td>24</td>
<td>-60°</td>
<td>60°</td>
</tr>
<tr>
<td>Hand Wrist Abduction (WR1)</td>
<td>1</td>
<td>-40°</td>
<td>28°</td>
</tr>
<tr>
<td>Hand Wrist Lateral (WR2)</td>
<td>2</td>
<td>-28°</td>
<td>8°</td>
</tr>
</tbody>
</table>

Stiffness, damping, friction, and armature are also important physical parameters in robotics. For each Shadow Hand joint, we show our DoF properties in Table 27. This part can be adjusted in the Isaac Gym simulator.

Table 27: DoF properties of Shadow Hand.
<table>
<tbody>
<tr>
<td>Joints</td>
<td>Stiffness</td>
<td>Damping</td>
<td>Friction</td>
<td>Armature</td>
</tr>
<tr>
<td>WR1</td>
<td>100</td>
<td>4.78</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>WR2</td>
<td>100</td>
<td>2.17</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>FF2</td>
<td>100</td>
<td>3.4e+38</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>FF3</td>
<td>100</td>
<td>0.9</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>FF4</td>
<td>100</td>
<td>0.725</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>MF2</td>
<td>100</td>
<td>3.4e+38</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>MF3</td>
<td>100</td>
<td>0.9</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>MF4</td>
<td>100</td>
<td>0.725</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>RF2</td>
<td>100</td>
<td>3.4e+38</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>RF3</td>
<td>100</td>
<td>0.9</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>RF4</td>
<td>100</td>
<td>0.725</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>LF2</td>
<td>100</td>
<td>3.4e+38</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>LF3</td>
<td>100</td>
<td>0.9</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>LF4</td>
<td>100</td>
<td>0.725</td>
<td>0</td>
<td>0</td>
</tr><tr>
<td>TH2</td>
<td>100</td>
<td>3.4e+38</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>TH3</td>
<td>100</td>
<td>0.99</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>TH4</td>
<td>100</td>
<td>0.99</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>TH5</td>
<td>100</td>
<td>0.81</td>
<td>0</td>
<td>0</td>
</tr>
</tbody>
</table>

27

Figure 22: Degree-of-Freedom (DOF) configuration of the Shadow Hand similar to the skeleton of a human hand.

Table 28: Observation space of dual Shadow Hands.

<table>
    <tr>
        <th>Index</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>0 - 23</td>
        <td>right Shadow Hand dof position</td>
    </tr>
    <tr>
        <td>24 - 47</td>
        <td>right Shadow Hand dof velocity</td>
    </tr>
    <tr>
        <td>48 - 71</td>
        <td>right Shadow Hand dof force</td>
    </tr>
    <tr>
        <td>72 - 136</td>
        <td>right Shadow Hand fingertip pose, linear velocity, angle velocity (5 x 13)</td>
    </tr>
    <tr>
        <td>137 - 166</td>
        <td>right Shadow Hand fingertip force, torque (5 x 6)</td>
    </tr>
    <tr>
        <td>167 - 169</td>
        <td>right Shadow Hand base position</td>
    </tr>
    <tr>
        <td>170 - 172</td>
        <td>right Shadow Hand base rotation</td>
    </tr>
    <tr>
        <td>173 - 198</td>
        <td>right Shadow Hand actions</td>
    </tr>
    <tr>
        <td>199 - 222</td>
        <td>left Shadow Hand dof position</td>
    </tr>
    <tr>
        <td>223 - 246</td>
        <td>left Shadow Hand dof velocity</td>
    </tr>
    <tr>
        <td>247 - 270</td>
        <td>left Shadow Hand dof force</td>
    </tr>
    <tr>
        <td>271 - 335</td>
        <td>left Shadow Hand fingertip pose, linear velocity, angle velocity (5 x 13)</td>
    </tr>
    <tr>
        <td>336 - 365</td>
        <td>left Shadow Hand fingertip force, torque (5 x 6)</td>
    </tr>
    <tr>
        <td>366 - 368</td>
        <td>left Shadow Hand base position</td>
    </tr>
    <tr>
        <td>369 - 371</td>
        <td>left Shadow Hand base rotation</td>
    </tr>
    <tr>
        <td>372 - 397</td>
        <td>left Shadow Hand actions</td>
    </tr>
</table>## C.2 Task Representation

### Hand Over

This scenario encompasses a specific environment comprising two Shadow Hands positioned opposite each other, with their palms facing upwards. The objective is to pass an object between these hands. Initially, the object will randomly descend within the area of the Shadow Hand on the right side. The hand on the right side then grasps the object and transfers it to the other hand. It is important to note that the base of each hand remains fixed throughout the process. Furthermore, the hand initially holding the object cannot directly make contact with the target hand or roll the object towards it. Hence, the object must be thrown into the air, maintaining its trajectory until it reaches the target hand.

28

In this task, there are 398-dimensional observations and 40-dimensional actions. The reward function is closely tied to the positional discrepancy between the object and the target. As the pose error diminishes, the reward increases significantly. The detailed observation space for each agent can be found in Table 29, while the corresponding action space is outlined in Table 30.

**Observations** The observational space for the Hand Over task consists of 398 dimensions, as indicated in Table 29. However, it is important to highlight that in this particular task, the base of the dual hands remains fixed. Therefore, the observation for the dual hands is compared to a reduced 24-dimensional space, as described in Table 28.

Table 29: Observation space of Hand Over.
<table>
  <tbody>
    <tr>
      <td>Index</td>
      <td>Description</td>
    </tr>
    <tr>
      <td>0 - 373</td>
      <td>dual hands observation shown in Table 28</td>
    </tr>
    <tr>
      <td>374 - 380</td>
      <td>object pose</td>
    </tr>
    <tr>
      <td>381 - 383</td>
      <td>object linear velocity</td>
    </tr>
    <tr>
      <td>384 - 386</td>
      <td>object angle velocity</td>
    </tr>
    <tr>
      <td>387 - 393</td>
      <td>goal pose</td>
    </tr>
    <tr>
      <td>394 - 397</td>
      <td>goal rot - object rot</td>
    </tr>
  </tbody>
</table>

**Actions** The action space for a single hand in the Hand Over task comprises 40 dimensions, as illustrated in Table 30.

Table 30: Action space of Hand Over.
<table>
  <tbody>
    <tr>
      <td>Index</td>
      <td>Description</td>
    </tr>
    <tr>
      <td>0 - 19</td>
      <td>right Shadow Hand actuated joint</td>
    </tr>
    <tr>
      <td>20 - 39</td>
      <td>left Shadow Hand actuated joint</td>
    </tr>
  </tbody>
</table>

**Rewards** Let the positions of the object and the goal be denoted as $x_o$ and $x_g$ respectively. The translational position difference between the object and the goal, represented as $d_t$, can be computed as $d_t = \|x_o - x_g\|_2$. Similarly, we define the angular position difference between the object and the goal as $d_a$. The rotational difference, denoted as $d_r$, is then calculated as $d_r = 2 \arcsin(clamp(\|d_a\|_2, max = 1.0))$.

The rewards for the Hand Over task are determined using the following formula:
$$r = \exp(-0.2(\alpha d_t + d_r)) \quad (2)$$
Here, $\alpha$ represents a constant that balances the rewards between translational and rotational aspects.

## Hand Over Catch

This environment is made up of a half Hand Over, and Catch Underarm [42], the object needs to be thrown from the vertical hand to the palm-up hand.

**Observations** The observational space for this combined task encompasses 422 dimensions, as illustrated in Table 31.

Table 31: Observation space of Hand Over Catch.
<table>
  <tbody>
    <tr>
      <td>Index</td>
      <td>Description</td>
    </tr>
    <tr>
      <td>0 - 397</td>
      <td>dual hands observation shown in Table 28</td>
    </tr>
    <tr>
      <td>398 - 404</td>
      <td>object pose</td>
    </tr>
    <tr>
      <td>405 - 407</td>
      <td>object linear velocity</td>
    </tr>
    <tr>
      <td>408 - 410</td>
      <td>object angle velocity</td>
    </tr>
    <tr>
      <td>411 - 417</td>
      <td>goal pose</td>
    </tr>
    <tr>
      <td>418 - 421</td>
      <td>goal rot - object rot</td>
    </tr>
  </tbody>
</table>

**Actions** The action space, consisting of 52 dimensions, is illustrated in Table 32, providing a comprehensive representation of the available actions.

29

Table 32: Action space of Hand Over Catch.
<table>
  <tbody>
    <tr>
      <td>Index</td>
      <td>Description</td>
    </tr>
    <tr>
      <td>0 - 19</td>
      <td>right Shadow Hand actuated joint</td>
    </tr>
    <tr>
      <td>20 - 22</td>
      <td>right Shadow Hand base translation</td>
    </tr>
    <tr>
      <td>23 - 25</td>
      <td>right Shadow Hand base rotation</td>
    </tr>
    <tr>
      <td>26 - 45</td>
      <td>left Shadow Hand actuated joint</td>
    </tr>
    <tr>
      <td>46 - 48</td>
      <td>left Shadow Hand base translation</td>
    </tr>
    <tr>
      <td>49 - 51</td>
      <td>left Shadow Hand base rotation</td>
    </tr>
  </tbody>
</table>

**Rewards** Let's denote the positions of the object and the goal as $x_o$ and $x_g$, respectively. The translational position difference between the object and the goal denoted as $d_t$, can be calculated as $d_t = \|x_o - x_g\|_2$. Additionally, we define the angular position difference between the object and the goal as $d_a$. The rotational difference, denoted as $d_r$, is given by the formula $d_r = 2 \arcsin(clamp(\|d_a\|_2, max = 1.0))$. Finally, the rewards are determined using the specific formula:
$$r = \exp[-0.2(\alpha d_t + d_r)] \quad (3)$$
Here, $\alpha$ represents a constant that balances the translational and rotational rewards.

### C.3 Constraint Specification


Figure 23: Tasks of Safety-DexterousHands.

**Safety Joint** constrains the freedom of joint ④ of the forefinger (please refer to Figure 23 (c) and (d)). Without the constraint, joint ④ has freedom of $[-20^\circ, 20^\circ]$. The safety tasks restrict joint ④ within $[-10^\circ, 10^\circ]$. Let ang\_4 be the angle of joint ④, and the cost is defined as:
$$c_t = I(ang\_4 \notin [-10^\circ, 10^\circ]). \quad (4)$$

**Safety Finger** constrains the freedom of joints ②, ③ and ④ of forefinger (please refer to Figure 23 (c) and (e)). Without the constraint, joints ② and ③ have freedom of $[0^\circ, 90^\circ]$ and joint ④ of $[-20^\circ, 20^\circ]$. The safety tasks restrict joints ②, ③, and ④ within $[22.5^\circ, 67.5^\circ]$, $[22.5^\circ, 67.5^\circ]$, and $[-10^\circ, 10^\circ]$ respectively. Let ang\_2, ang\_3, ang\_4 be the angles of joints ②, ③, ④, and the cost is defined as:
$$c_t = I(ang\_2 \notin [22.5^\circ, 67.5^\circ],  or  ang\_3 \notin [22.5^\circ, 67.5^\circ],  or  ang\_4 \notin [-10^\circ, 10^\circ]). \quad (5)$$

30