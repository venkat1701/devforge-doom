# References: primary-source citations for the DOOM RL agent

Every quote below was copied from the source text retrieved on 2026-11-26 (arXiv HTML, a PDF run through `pdftotext`, or raw source code). Quotes are verbatim and at most 125 characters. In equations, `$...$` is the LaTeX alt-text from the arXiv HTML. Where the arXiv HTML leaves an equation unnumbered, the entry says so and gives no number.

## URLs read

| # | Source | URL actually read | Form |
|---|---|---|---|
| 1 | PPO (Schulman et al., 2017), arXiv:1707.06347v2 | https://arxiv.org/pdf/1707.06347 | PDF via pdftotext. The arXiv HTML (https://arxiv.org/html/1707.06347) has no body text, only the line "See pages 1-last of ppo-min.pdf". |
| 2 | GAE (Schulman et al., 2016), arXiv:1506.02438v6 | https://arxiv.org/html/1506.02438 | HTML |
| 3 | Sample Factory (Petrenko et al., 2020), arXiv:2006.11751v2 | https://arxiv.org/html/2006.11751 | HTML |
| 4 | Asymmetric Actor Critic (Pinto et al., 2017), arXiv:1710.06542 | https://arxiv.org/html/1710.06542 | HTML |
| 5 | Neural Map (Parisotto & Salakhutdinov, 2017), arXiv:1702.08360v1 | https://arxiv.org/html/1702.08360 | HTML |
| 6 | Active Neural SLAM (Chaplot et al., 2020), arXiv:2004.05155v1 | https://arxiv.org/html/2004.05155 | HTML |
| 7 | CleanRL `ppo_atari_lstm.py` (master) | https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py | raw source, 375 lines |
| 8 | Wu & Tian (ICLR 2017) | https://research.facebook.com/file/249939083525401/paper_camera_ready_small-1.pdf | ICLR 2017 camera-ready PDF via pdftotext. https://openreview.net/forum?id=Hk3mPK5gg and `/pdf?id=Hk3mPK5gg` both returned HTTP 403 or a browser challenge. |
| 9 | DFP (Dosovitskiy & Koltun, 2017), arXiv:1611.01779v2 | https://arxiv.org/html/1611.01779 | HTML |
| 10a | Arnold (Lample & Chaplot), arXiv:1609.05521v2 | https://arxiv.org/html/1609.05521 | HTML |
| 10b | DRQN (Hausknecht & Stone), arXiv:1507.06527v4 | https://arxiv.org/html/1507.06527 | HTML |
| 10c | Double DQN (van Hasselt, Guez, Silver), arXiv:1509.06461v3 | https://arxiv.org/html/1509.06461v3 | HTML |
| 10d | Acme R2D2 learner | https://raw.githubusercontent.com/google-deepmind/acme/master/acme/agents/jax/r2d2/learning.py | raw source |
| 10e | Acme R2D2 config | https://raw.githubusercontent.com/google-deepmind/acme/master/acme/agents/jax/r2d2/config.py | raw source |
| 10f | rlax transforms | https://raw.githubusercontent.com/google-deepmind/rlax/master/rlax/_src/transforms.py | raw source |
| 10g | rlax nonlinear Bellman (defines `SIGNED_HYPERBOLIC_PAIR`) | https://raw.githubusercontent.com/google-deepmind/rlax/master/rlax/_src/nonlinear_bellman.py | raw source |

---

## 1. PPO: Proximal Policy Optimization Algorithms

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI), 2017. arXiv:1707.06347v2 (28 Aug 2017). Read from https://arxiv.org/pdf/1707.06347.

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 3 "Clipped Surrogate Objective", Eq. (6) | "TRPO maximizes a “surrogate” objective" (the extracted line is broken by the subscript "old" of Eq. 6) | `L^CPI(θ) = Ê_t[ π_θ(a_t|s_t) / π_θold(a_t|s_t) · Â_t ] = Ê_t[ r_t(θ) Â_t ]` (6) |
| Sec. 3, Eq. (7) | "The main objective we propose is the following:" | `L^CLIP(θ) = Ê_t[ min( r_t(θ) Â_t , clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]` (7) |
| Sec. 3, after Eq. (7) | "where epsilon is a hyperparameter, say, ε = 0.2." | ε = 0.2 (example value). pdftotext drops the ε glyph; the PDF prints the symbol here. |
| Sec. 3, after Eq. (7) | "so the final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective." | n/a |
| Sec. 5 "Algorithm", Eq. (9) | "This objective can further be augmented by adding an entropy bonus to ensure sufficient exploration" | `L_t^{CLIP+VF+S}(θ) = Ê_t[ L_t^CLIP(θ) − c1 L_t^VF(θ) + c2 S[π_θ](s_t) ]` (9) |
| Sec. 5, after Eq. (9) | "where c1 , c2 are coefficients, and S denotes an entropy bonus, and LVt F is a squared-error loss" | `L_t^VF = (V_θ(s_t) − V_t^targ)^2` |
| Sec. 5, Eq. (11)–(12) | "we can use a truncated version of generalized advantage estimation, which reduces to Equation (10) when λ = 1" | `Â_t = δ_t + (γλ)δ_{t+1} + ··· + (γλ)^{T−t+1} δ_{T−1}` (11); `δ_t = r_t + γV(s_{t+1}) − V(s_t)` (12) |
| Sec. 5, Algorithm 1 "PPO, Actor-Critic Style" | "Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ N T" | see full Algorithm 1 below |
| Sec. 5, before Algorithm 1 | "Each iteration, each of N (parallel) actors collect T timesteps of data." | n/a |
| Appendix A, Table 5 caption | "Table 5: PPO hyperparameters used in Atari experiments. α is linearly annealed from 1 to 0 over the course of learning." | see table below |

Algorithm 1 (Sec. 5), verbatim:

```
Algorithm 1 PPO, Actor-Critic Style
 for iteration=1, 2, . . . do
    for actor=1, 2, . . . , N do
        Run policy πθold in environment for T timesteps
        Compute advantage estimates Â1 , . . . , ÂT
    end for
    Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ N T
    θold ← θ
 end for
```

**Table 5: Atari hyperparameters (Appendix A), all rows exactly as printed**

| Hyperparameter | Value |
|---|---|
| Horizon (T) | 128 |
| Adam stepsize | 2.5 × 10^−4 × α |
| Num. epochs | 3 |
| Minibatch size | 32 × 8 |
| Discount (γ) | 0.99 |
| GAE parameter (λ) | 0.95 |
| Number of actors | 8 |
| Clipping parameter | 0.1 × α |
| VF coeff. c1 (9) | 1 |
| Entropy coeff. c2 (9) | 0.01 |

The appendix also contains Table 3 (Mujoco 1M timesteps: T=2048, Adam 3×10^−4, 10 epochs, minibatch 64, γ=0.99, λ=0.95) and Table 4 (Roboschool).

---

## 2. GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel, 2016 (ICLR). arXiv:1506.02438v6. Read from https://arxiv.org/html/1506.02438.

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 3 "Advantage function estimation", TD residual definition (inline, unnumbered) | "i.e., the TD residual of $V$ with discount $\gamma$ (Sutton & Barto, 1998)." | `δ^V_t = r_t + γ V(s_{t+1}) − V(s_t)` |
| Sec. 3, Eq. (10) | "if we have the correct value function $V=V^{\pi,\gamma}$ , then it is a $\gamma$ -just advantage estimator" | Eq. (10) is `E_{s_{t+1}}[δ_t^{V^{π,γ}}]` = A^{π,γ}(s_t,a_t) |
| Sec. 3, Eq. (14) | (k-step estimator) | `Â_t^(k) := Σ_{l=0}^{k−1} γ^l δ^V_{t+l} = −V(s_t) + r_t + γ r_{t+1} + ... + γ^{k−1} r_{t+k−1} + γ^k V(s_{t+k})` (14) |
| Sec. 3, Eq. (16) | "The generalized advantage estimator $\mathrm{GAE(\gamma,\lambda)}$ is defined as the exponentially-weighted average" | `Â_t^{GAE(γ,λ)} := (1−λ)(Â_t^(1) + λ Â_t^(2) + λ^2 Â_t^(3) + ...) = Σ_{l=0}^{∞} (γλ)^l δ^V_{t+l}` (16) |
| Sec. 3, special cases | "$\mathrm{GAE}(\gamma,1)$ is $\gamma$ -just regardless of the accuracy of $V$ , but it has high variance" | λ=1: `Σ γ^l δ_{t+l}`; λ=0: `δ_t` |
| Sec. 3, γ vs λ trade-off | "Taking $\gamma<1$ introduces bias into the policy gradient estimate, regardless of the value function’s accuracy." | n/a |
| Sec. 3, γ vs λ trade-off (next sentence) | "On the other hand, $\lambda<1$ introduces bias only when the value function is inaccurate." | n/a |
| Sec. 3, empirical note | "Empirically, we find that the best value of $\lambda$ is much lower than the best value of $\gamma$" | n/a |
| Sec. 2 "Preliminaries" | "it will appear below as an algorithm parameter that adjusts a bias-variance tradeoff." | γ treated as a variance-reduction parameter |

---

## 3. Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning

Aleksei Petrenko, Zhehui Huang, Tushar Kumar, Gaurav Sukhatme, Vladlen Koltun, 2020 (ICML). arXiv:2006.11751v2. Read from https://arxiv.org/html/2006.11751.

### 3a. VizDoom model architecture

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| App. A.1.3 "Model architectures" | "In our implementation the 3-layer convolutional head is followed by a fully-connected layer, an LSTM core," | simplified model: conv×3 → FC → LSTM → FC heads |
| App. A.1.3 | "in contrast to the full architecture used in Battle, Deathmatch, and Duel experiments," | n/a |
| App. A.1.3, GRU vs LSTM | "in our large-scale VizDoom experiments with the full model we chose to use GRU RNN cells (Cho et al., 2014) instead of LSTM" | GRU 512 |
| App. A.1.3, GRU vs LSTM rationale | "Empirically we find that GRU cells exhibit similar sample efficiency to LSTM cells and require slightly less computation." | n/a |
| Figure A.1 caption | "additional low-dimensional game information input (health, armor, ammunition, etc.) and $L$ independent action heads." | full model outputs `μ_1,...,μ_L` and `V^π(x_t,h_t)` |
| Figure A.1, embedded figure text (full model, right) | "/255 \| Conv2D , stride \| 8\times 8 \| 4 \| ReLU \| Conv2D , stride \| 4\times 4 \| 2 \| ... \| FC 512 \| ReLU \| GRU 512" | Input 128×72×3, /255. Conv 8×8 stride 4; Conv 4×4 stride 2; Conv 3×3 stride 2; ReLU after each; FC 512, ReLU; GRU 512. Filter labels in the figure read 32, 64, 128. Pairing those to layers 1–3 is inferred from their order in the figure. Game info 23×1 → FC 128, ReLU → FC 128, ReLU. |
| App. A.3 "VizDoom experiments", game-variable input | "We only used information available to a human player through in-game UI." | n/a |
| App. A.3, game-variable list | "health and armor, current score, number of players in a match, selected weapon index," | continued: "possession of different types of weapons, and amount of ammunition available for each weapon" |
| App. A.3, no reward input | "We do not use previous rewards as a policy input," | n/a |
| App. A.3, independent heads | "We decompose the set of possible actions into seven independent action distributions," | 7 heads (Table A.4) |
| App. A.3, aim discretization | "in this work we discretize it with $1.25^{\circ}$ step, which empirically leads to faster learning." | 1.25° |
| Sec. 4.3, Battle action heads | "five independent discrete action heads for moving, aiming, strafing, shooting, and sprinting." | 5 heads (Battle/Battle2) |
| Sec. 4.3, Duel/Deathmatch action space | "This brings the total number of possible actions to $\sim 1.2\times 10^{4}$" | ~1.2×10^4 (Table A.4 total: 12096) |

**Table A.4: action space used in the VizDoom multi-agent experiments (all rows)**

| Action head | Number of actions | Comment |
|---|---|---|
| Moving | 3 | no-action / forward / backward |
| Strafing | 3 | no-action / left / right |
| Attacking | 2 | no-action / attack |
| Sprinting | 2 | no-action / sprint |
| Object interaction | 2 | no-action / interact |
| Weapon selection | 8 | no-action / select weapon slot 1..7 |
| Horizontal aim | 21 | no-action / turning between −12.5° and 12.5° in 1.25° steps |
| Total number of possible actions | 12096 | |

### 3b. VizDoom hyperparameters

**Table A.5: Hyperparameters for VizDoom experiments (App. A.3), all rows as printed**

| Parameter | Value |
|---|---|
| Learning rate | 10^−4 |
| Action repeat (frameskip) | 2/4 |
| Framestack | No |
| Discount γ | 0.995/0.99 |
| Optimizer | Adam (Kingma & Ba, 2015) |
| Optimizer settings | β1=0.9, β2=0.999, ε=10^−6 |
| Gradient norm clipping | 4.0 |
| Rollout length T | 32 |
| Batch size, samples | 2048 |
| Number of training epochs | 1 |
| V-trace parameters | ρ̄ = c̄ = 1 |
| PPO clipping range | [1.1^−1, 1.1] |
| Entropy coefficient | 0.003 |
| Critic loss coefficient | 0.5 |

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| App. A.3, explaining the 2/4 entries | "We deviate from these parameters only in Deathmatch and Duel experiments where we used action repeat (frameskip) of two" | Duel/Deathmatch: frameskip 2, γ = 0.995; other scenarios: frameskip 4, γ = 0.99 |
| App. A.3 | "Consequently, we adjusted the discount factor to $0.995$ to account for this change." | γ = 0.995 |
| Sec. 3.4 "Policy lag" | "Thus we implemented both V-trace and PPO clipping in Sample Factory." | n/a |
| App. A.3, policy lag | "In all our experiments the policy lag was on average between 5 and 10 SGD steps, which results in stable training." | 5–10 SGD steps |

### 3c. Frameskip and resolution

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| App. A.1.2 "Environments" | "rendered with native resolution of $160\times 120\times 3$ which is downsampled to $128\times 72\times 3$" | 160×120×3 → 128×72×3 |
| App. A.1.2 | "For VizDoom and DMLab we used 4-frameskip and no framestacking." | frameskip 4, no framestack |
| Sec. 4.3 | "when we repeat actions for two consecutive frames instead of the traditional four (Bellemare et al., 2013)," | frameskip 2 (Duel, Deathmatch) |
| Sec. 4.3 | "allowing the agents to develop precise movement and aim." | n/a |
| App. A.3 | "We observe that in these environments repeating actions fewer times led to better final performance of the agents." | n/a |

### 3d. Population-based training

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 4.3 | "we randomly mutate the bottom 70% of the population every $5\times 10^{6}$ environment frames," | bottom 70%, every 5×10^6 frames |
| Sec. 4.3 | "altering hyperparameters such as learning rate, entropy coefficient, and reward weights." | n/a |
| Sec. 4.3 | "If the win rate of the policy is less than half of the best-performing agent’s win rate, we simply copy the model weights" | copy weights + hyperparameters when win rate < 0.5 × best |
| App. A.3.1 "Population-based training" | "we used System $\#3$ to train a population of 8 agents in parallel." | population 8 |
| App. A.3.1 | "The mutation rate is $15\%$ for each hyperparameter." | 15% |
| App. A.3.1 | "learning rate, entropy loss coefficient, Adam $\beta_{1}$ , and individual reward shaping coefficients" | perturbation factor: "by a factor of $1.2$" |
| App. A.3.1 | "every $5M$ frames we replace the policy weights for the worst $30\%$ of agents with weights of the policy randomly sampled" | worst 30% replaced from best 30% |
| App. A.3.1 (Duel) | "(the difference in win rate is less than $0.35$ ), which helps to increase the diversity of the population." | threshold 0.35 |
| Sec. 4.3, result | "The final agents beat the in-game bots on the highest difficulty in 100% of the matches in both environments." | Deathmatch 80.5 vs 12.6; Duel 34.7 to 3.6 frags |

---

## 4. Asymmetric Actor Critic for Image-Based Robot Learning

Lerrel Pinto, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, Pieter Abbeel, 2017. arXiv:1710.06542. Read from https://arxiv.org/html/1710.06542. The base algorithm in the paper is DDPG (with HER), not PPO.

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Abstract | "employing an actor-critic training algorithm in which the critic is trained on full states while the actor (or policy) gets" | continues: "rendered images as input" |
| Sec. I "Introduction" | "the actor takes visual partial observations as input while the critic takes the underlying full state as input." | n/a |
| Sec. I, why this is valid (critic used only in training) | "During testing, the actor is employed on the partial observations and does not depend on the full state" | continues: "(the full state is only used during training)." |
| Sec. I, why it helps | "Since the critic works on full state, it learns the state-action value function much faster," | continues: "which also allows for better updates for the actor." |
| Sec. IV-A "Asymmetric Actor Critic" | "by using the full state $s_{t}\in\mathcal{S}$ to train the critic, while using partial observation $o_{t}\in\mathcal{O}$" | critic Q(s_t, a_t); actor π(o_t) |
| Sec. IV-A | "Note that $s_{t}$ is the underlying full state for the observation $o_{t}$ ." | n/a |
| Sec. IV-A, Algorithm 1 text | "Since the critic takes full states as input, it is trained on $\{s,a,r,s^{{}^{\prime}},g\}^{B}_{0}$ ." | actor trained on `{o, a, r, o', g^o}` |
| Fig. 2 caption | "Having asymmetric inputs, i.e. full states for the critic and partial observations for the actor improves training." | n/a |
| Sec. V-C "Does asymmetric inputs to actor critic help?" | "asymmetric input versions of both DDPG and HER perform much better than their symmetric counterparts." | baseline: images for both actor and critic |

---

## 5. Neural Map: Structured Memory for Deep Reinforcement Learning

Emilio Parisotto, Ruslan Salakhutdinov (CMU), 2017. arXiv:1702.08360v1. Read from https://arxiv.org/html/1702.08360.

### 5a. Core operations (Sec. 3)

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 3 "Neural Map", map definition | "let the neural map $M$ be a $C\times H\times W$ feature block," | M ∈ C×H×W |
| Sec. 3, Eq. (2)–(7) | "The Neural Map is defined by the following set of equations:" | `r_t = read(M_t)` (2); `c_t = context(M_t, s_t, r_t)` (3); `w_{t+1}^{(x_t,y_t)} = write(s_t, r_t, c_t, M_t^{(x_t,y_t)})` (4); `M_{t+1} = update(M_t, w_{t+1}^{(x_t,y_t)})` (5); `o_t = [r_t, c_t, w_{t+1}^{(x_t,y_t)}]` (6); `π_t(a|s) = Softmax(f(o_t))` (7) |
| Sec. 3.1 "Global Read Operation" | "The $read$ operation passes the current neural map $M_{t}$ through a deep convolutional network" | continues: "and produces a $C$ -dimensional feature vector $r_{t}$" |
| Sec. 3.2 "Context Read Operation", Eq. (8)–(11) | "The context read operation allows the neural map to operate as an associative memory:" | `q_t = W[s_t, r_t]` (8); `a_t^{(x,y)} = q_t · M_t^{(x,y)}` (9); `α_t^{(x,y)} = e^{a_t^{(x,y)}} / Σ_{(w,z)} e^{a_t^{(w,z)}}` (10); `c_t = Σ_{(x,y)} α_t^{(x,y)} M_t^{(x,y)}` (11) |
| Sec. 3.3 "Local Write Operation", Eq. (12) | "This vector functions as the new local write candidate vector at the current position $(x_{t},y_{t})$ :" | `w_{t+1}^{(x_t,y_t)} = f([s_t, r_t, c_t, M_t^{(x_t,y_t)}])` (12) |
| Sec. 3.4 "Map Update Operation" (unnumbered in HTML) | "The new neural map $M_{t+1}$ is equal to the old neural map $M_{t}$ , except at the current agent position" | `M_{t+1}^{(a,b)} = w_{t+1}^{(x_t,y_t)} if (a,b)=(x_t,y_t); M_t^{(a,b)} otherwise` |
| Sec. 3.5.2 "Key-Value Context Read", Eq. (15)–(19) | "splitting each feature of the neural map into two parts" | `M_t^{(x,y)} = [k_t^{(x,y)}, v_t^{(x,y)}]` (16); `a_t^{(x,y)} = q_t · k_t^{(x,y)}` (17); `c_t = Σ α_t^{(x,y)} v_t^{(x,y)}` (19) |
| Sec. 3.5.3 "GRU-based Local Write Operation" (unnumbered in HTML) | "we can use a gated write operation based on the recurrent update equations of the Gated Recurrent Unit (GRU)" | `r = σ(W_r[s_t,r_t,c_t,M_t^{(x_t,y_t)}])`; `ŵ = tanh(W_ĥ[s_t,r_t,c_t] + U_ĥ(r ⊙ M_t^{(x_t,y_t)}))`; `z = σ(W_z[s_t,r_t,c_t,M_t^{(x_t,y_t)}])`; `w_{t+1} = (1−z) ⊙ M_t^{(x_t,y_t)} + z ⊙ ŵ` |

In the HTML, equation numbering jumps from (12) to (15). The map-update and GRU-write equations carry no visible number there.

### 5b. Ego-centric Neural Map (Sec. 5 "Extension: Ego-centric Neural Map")

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 5, motivation | "it requires some oracle to provide the current $(x,y)$ position of the agent." | n/a |
| Sec. 5, mechanism | "whenever the agent moves between time steps with some velocity $(u,v)$ , the map is counter-transformed by $(-u,-v)$" | shift in H and W |
| Sec. 5 | "the agent’s position will stay stationary in the center of the neural map while the world as defined by the map moves" | center (pw, ph) |
| Sec. 5, Eq. (20) (group) | "The ego-centric neural map operations are shown below:" | `M̄_t = CounterTransform(M_t, (u_t,v_t))`; `r_t = read(M̄_t)`, `c_t = context(M̄_t, s_t, r_t)`; `w_{t+1}^{(pw,ph)} = write(s_t, r_t, c_t, M̄_t^{(pw,ph)})`; `M_{t+1} = egoupdate(M̄_t, w_{t+1}^{(pw,ph)})`; `o_t = [r_t, c_t, w_{t+1}^{(pw,ph)}]`; `π_t = Softmax(f(o_t))` (20) |
| Sec. 5, CounterTransform (unnumbered) | "transforms the current neural map $M_{t}$ by the inverse of the agent’s current velocity $(u_{t},v_{t})$" | as printed: `M̄_t^{(a,b)} = M_{t+1}^{(a−u,b−v)} if (a−u)∈{1..W} ∧ (b−v)∈{1..H}; 0 else` |
| Sec. 5, rotations | "While here we only deal with reverse translation, it is possible to handle rotations as well" | n/a |
| Sec. 5, egoupdate (unnumbered) | "only the center position $(pw,ph)$ is only ever written to:" | `M_{t+1}^{(a,b)} = w_{t+1}^{(pw,ph)} if (a,b)=(pw,ph); M̄_t^{(a,b)} otherwise` |

The paper describes the ego-centric variant only. No experimental results for it were found in the HTML.

### 5c. 3D Doom maze experiment (Sec. 4.2 "3D Doom Environment")

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 4.2, task | "the indicator is a torch of either red or green color that is always at a fixed location in view of the player’s" | n/a |
| Sec. 4.2, task | "The goals are red/green towers that are randomly positioned throughout the maze." | n/a |
| Fig. 5 caption, reward | "receives a positive reward if it reached the correct tower, while a negative reward otherwise." | episode limit 420 steps |
| Sec. 4.2, inputs | "The state observations were the 5 previous $100\times 75$ RGB colour images." | 5 × 100×75 RGB; "We used an action repeat of 5." |
| Sec. 4.2, init | "pre-initialized with the weights from a network trained to play Doom taken from Lample & Chaplot (2016)." | n/a |
| Sec. 4.2, maze and map | "The mazes consisted of $3\times 3$ rooms," | "we used an internal map size of $32\times 9\times 9$ ." |
| Sec. 4.2, agents | "The agents we tested on are an LSTM baseline and an LSTM + Neural Map with GRU-based updates." | MemNN memory size 16 |
| Sec. 4.2, why LSTM on top | "the neural map agents that lacked an LSTM added to the top often repeated the same actions." | no stand-alone Neural Map results |
| Sec. 4.2, protocol | "We trained all agents on a single training maze" | up to 7 million frames; test on 6 held-out mazes |
| Sec. 4.2, main result | "We can see LSTM + Neural Map (GRU) surpasses all other methods on both the training map and on the 6 unseen maps." | see Table 2 |

**Table 2 (Sec. 4.2): percentage of 1000 episodes that found the correct goal within 420 steps**

| Agent | Training Map | Unseen Maps |
|---|---|---|
| Random | 20.9% | 22.1% |
| MemNN | 68.2% | 60.3% |
| LSTM | 69.2% | 52.4% |
| LSTM+Neural Map (GRU) | 78.3% | 66.6% |

---

## 6. Learning To Explore Using Active Neural SLAM

Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, Ruslan Salakhutdinov, 2020 (ICLR). arXiv:2004.05155v1. Read from https://arxiv.org/html/2004.05155.

### 6a. Map representation (Sec. 4 "Methods", paragraph "Map Representation")

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 4, Map Representation | "The spatial map, $m_{t}$ , is a ${2\times M\times M}$ matrix where $M\times M$ denotes the map size" | m_t ∈ 2×M×M, cell = 5cm×5cm |
| Sec. 4, channel 0 | "Each element in the first channel denotes the probability of an obstacle at the corresponding location" | channel 0 = obstacle probability |
| Sec. 4, channel 1 | "each element in the second channel denotes the probability of that location being explored." | channel 1 = explored probability |
| Sec. 4, explored definition | "A cell is considered to be explored when it is known to be free space or an obstacle." | n/a |
| Sec. 4, init and pose | "The agent always starts at the center of the map facing east at the beginning of the episode," | `m_0 = [0]^{2×M×M}`, `x_0 = (M/2, M/2, 0.0)`, `x_t ∈ R^3` |

### 6b. Egocentric to geocentric transformation (Sec. 4 "Neural SLAM Module"; Appendix C)

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 4, Mapper | "The Mapper ( $f_{Map}$ ) outputs a egocentric top-down 2D spatial map, $p_{t}^{ego}\in[0,1]^{2\times V\times V}$" | V = vision range |
| Sec. 4 | "The egocentric map from the Mapper is transformed to a geocentric map based on the pose estimate given by the Pose Estimator" | then aggregated with m_{t−1} |
| App. C "Neural SLAM module implementation details" | "Then we use a Spatial Transformation [Jaderberg et al., 2015] on the egocentric map prediction at the last frame" | `p'_{t−1} = f_ST(p^ego_{t−1} | dx)` |
| App. C | "Note that the parameters of this Spatial Transformation are not learnt, but calculated using the pose change ( $dx$ )." | n/a |
| App. C, pose update | "The predicted relative pose change is then added to the last pose estimate to get the final pose estimate" | `d̂x_t = f_PE(p'_{t−1}, p^ego_t | θ_P)`; `x̂_t = x̂_{t−1} + d̂x_t` |
| App. C, geocentric aggregation | "the egocentric spatial map prediction is transformed to the geocentric frame using the current pose prediction" | `m_t = m_{t−1} + f_ST(p^ego_t | x̂_t)` ("using Channel-wise Pooling operation") |
| App. E, sizes | "the vision range for the SLAM module is $V=64$ , i.e. $3.2m$ (each cell is $5cm$ in length)." | V = 64; M = 480 (train, 24m), M = 960 (eval, 48m) |

### 6c. How the global policy consumes the map (Sec. 4 "Global Policy"; Appendix D, E)

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 4, Global Policy input | "The Global Policy takes $h_{t}\in[0,1]^{4\times M\times M}$ as input," | 4 channels: m_t (2 ch), current position, visited locations |
| Sec. 4, channels | "the third channel represents the current agent position estimated by the SLAM module, the fourth channel represents" | continues: "the visited locations" |
| Sec. 4, channel equations | (unnumbered) | `h_t[c,i,j] = m_t[c,i,j] ∀c∈{0,1}`; `h_t[2,i,j] = 1 if i = x̂_t[0] and j = x̂_t[1]`; `h_t[3,i,j] = 1 if (i,j) ∈ [(x̂_k[0], x̂_k[1])]_{k∈{0..t}}` |
| Sec. 4, crop | "The first transformation subsamples a window of size $4\times G\times G$ around the agent from $h_{t}$ ." | local crop 4×G×G |
| Sec. 4, global view | "The second transformation performs max pooling operations to get an output of size $4\times G\times G$ from $h_{t}$ ." | max-pooled 4×G×G |
| Sec. 4, stacking | "Both the transformations are stacked to form a tensor of size $8\times G\times G$" | input 8×G×G |
| Sec. 4, output | "The Global Policy uses a convolutional neural network to predict a long-term goal, $g_{t}^{l}$ in $G\times G$ space" | `g^l_t = π_G(h_t | θ_G)` |
| App. E, G | "The size of the Global Policy input is constant, $G=240$ ," | G = 240: "we downscale map by 2 times during training and 4 times during evaluation" |
| App. D, architecture | "The Global Policy is a 5 layer convolutional network followed by 3 fully connected layers." | orientation via an Embedding layer, added to the FC input |
| App. E, global policy RL | "The Global policy samples a new goal every 25 timesteps." | PPO; reward = coverage increase in m² × 0.02; 72 threads; horizon 40; 36 minibatches; 4 epochs; lr 0.000025; γ=0.99; entropy 0.001; value coef 0.5 |

---

## 7. CleanRL recurrent PPO: `cleanrl/ppo_atari_lstm.py` (master)

Read from https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py (375 lines). Line numbers refer to that file as retrieved. The code below is copied exactly.

### 7a. `Args` dataclass defaults (lines 44–86)

| Field | Default (verbatim) |
|---|---|
| env_id | `"BreakoutNoFrameskip-v4"` |
| total_timesteps | `10000000` |
| learning_rate | `2.5e-4` |
| num_envs | `8` |
| num_steps | `128` |
| anneal_lr | `True` |
| gamma | `0.99` |
| gae_lambda | `0.95` |
| num_minibatches | `4` |
| update_epochs | `4` |
| norm_adv | `True` |
| clip_coef | `0.1` |
| clip_vloss | `True` |
| ent_coef | `0.01` |
| vf_coef | `0.5` |
| max_grad_norm | `0.5` |
| target_kl | `None` |
| batch_size / minibatch_size / num_iterations | `0` (computed at runtime: `args.batch_size = int(args.num_envs * args.num_steps)`, `args.minibatch_size = int(args.batch_size // args.num_minibatches)`) |

Related values from the same file: `optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)` (line 212); `self.lstm = nn.LSTM(512, 128)` (line 131); LSTM biases set to 0 and weights orthogonal(1.0) (lines 132–136); actor head `std=0.01`, critic head `std=1` (lines 137–138).

### 7b. LSTM state reset on done inside the forward pass (`Agent.get_states`, lines 140–158)

```python
    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
```

Rollout storage of the initial state (line 234): `initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())`. During rollout, `dones[step] = next_done` is stored before the step (line 244). The `done` passed with obs at step t therefore marks that obs as the first of a new episode, and the state is zeroed before that obs is processed.

### 7c. GAE loop (lines 266–284)

```python
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
```

Done is computed as `next_done = np.logical_or(terminations, truncations)` (line 255), so truncation and termination are treated the same way here.

### 7d. Recurrent minibatches by environment, keeping full sequences (lines 295–313)

```python
        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
```

### 7e. Loss terms (lines 323–353)

```python
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
```

Ratio (lines 314–315): `logratio = newlogprob - b_logprobs[mb_inds]`, `ratio = logratio.exp()`. Early stop (lines 355–356): `if args.target_kl is not None and approx_kl > args.target_kl: break`. Learning-rate anneal (lines 236–239): `frac = 1.0 - (iteration - 1.0) / args.num_iterations`, `lrnow = frac * args.learning_rate`.

---

## 8. Wu & Tian 2017: Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning

Yuxin Wu (CMU), Yuandong Tian (Facebook AI Research), ICLR 2017 conference paper. No arXiv copy was used. OpenReview (https://openreview.net/forum?id=Hk3mPK5gg) returned 403 or a browser challenge. Read from the camera-ready PDF at https://research.facebook.com/file/249939083525401/paper_camera_ready_small-1.pdf ("Published as a conference paper at ICLR 2017").

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 4.3 "Curriculum Learning" | "A direct training with A3C on the map CIGTrack1 with 8 built-in bots does not yield sensible performance." | n/a |
| Sec. 4.3 | "trains an agent with a sequence of progressively more difficult environments." | n/a |
| Sec. 4.3, Curriculum Design | "we design curricula by varying the strength of built-in bots, i.e., their moving speed, initial health and initial weapon." | see Table 3 |
| Sec. 4.3, Curriculum Design | "We train the bot on FlatMap that contains a simple square with a few pillars (Fig. 2(a)) with several curricula (Tbl. 3)," | then "proceed to CIGTrack1" |
| Sec. 4.3, Curriculum Design | "Our agent always uses RocketLauncher as its only weapon." | n/a |
| Sec. 4.3, CIGTrack1 adjustments | "for CIGTrack1 we increase dist penalty thres to keep the agent moving," | and "increase num bots" |
| Sec. 4.3, Adaptive Curriculum | "by assigning a probability distribution on different levels for each thread that runs a Doom instance." | n/a |
| Sec. 4.3, Adaptive Curriculum | "The probability distribution shifts towards more difficult curriculum when the agent performs well" | continues: "and shifts towards easier level otherwise." |
| Sec. 4.3, Adaptive Curriculum | "We consider the agent to perform well if its frag count is greater than 10 points." | threshold: frags > 10 |
| Sec. 4.2 "Training Pipeline" | "Batch size is 128, discount factor γ = 0.99, learning rate α = 10−4 and the policy learning rate β = 0.08α." | Adam ε = 10^−3 |
| Abstract, competition result | "our agent won 10 out of the 11 attended games and the champion of Track1 in ViZDoom AI Competition 2016 by a large margin," | continues: "35% higher score than the second place." |
| Sec. 5.3 "Competition" | "We have achieved 559 frags, 35.4% higher than 413 frags achieved by Arnold [Lample & Chaplot (2016)]," | 559 vs 413 (Arnold) vs 393 (CLYDE), Table 6 |
| Sec. 1 "Introduction" | "our AI won the champion of Track1 (limited deathmatch with known map), and IntelAct won the champion of Track2" | Track2 = full deathmatch, unknown maps |

**Table 3 (Sec. 4.3): curriculum design for FlatMap.** Caption: "Note that enemy uses RocketLauncher except for Class 0 (Pistol)."

| | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 |
|---|---|---|---|---|---|---|---|---|
| Speed | 0.2 | 0.2 | 0.4 | 0.4 | 0.6 | 0.8 | 0.8 | 1.0 |
| Health | 40 | 40 | 40 | 60 | 60 | 60 | 80 | 100 |

---

## 9. DFP: Learning to Act by Predicting the Future

Alexey Dosovitskiy, Vladlen Koltun (Intel Labs), 2017 (ICLR). arXiv:1611.01779v2. Read from https://arxiv.org/html/1611.01779.

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 1 "Introduction" | "The measurement stream has lower dimensionality and constitutes a set of data that pertain to the agent’s current state." | n/a |
| Sec. 3 "Model", observation definition | "We assume that the observations have the following structure:" | `o_t = ⟨s_t, m_t⟩`, s_t = raw sensory input (image), m_t = measurements |
| Sec. 3 | "The measurements $\mathbf{m}_{t}$ can indicate the attitude, supply levels, and structural integrity in a physical system," | continues: "or health, ammunition, and score in a computer game." |
| Sec. 3, role of measurements | "First, the measurement vector is the part of the observation that the agent will aim to predict." | n/a |
| Sec. 3, future-measurement target | "let $\tau_{1},\ldots,\tau_{n}$ be a set of temporal offsets" | `f = ⟨m_{t+τ_1} − m_t, ..., m_{t+τ_n} − m_t⟩` |
| Sec. 3, Eq. (1), goal | "Our experiments use goals that are expressed as linear combinations of future measurements:" | `u(f; g) = g^T f` (1) |
| Sec. 3, Eq. (1) | "the scalar reward signal can be viewed as a measurement," | n/a |
| Sec. 3.2 "Architecture" | "The network has three input modules: a perception module $S(\mathbf{s})$ , a measurement module $M(\mathbf{m})$" | plus "a goal module $G(\mathbf{g})$"; measurement and goal modules are fully-connected |
| Sec. 4.1 "Setup" | "We set the temporal offsets $\tau_{1},\ldots,\tau_{n}$ of predicted future measurements to 1, 2, 4, 8, 16, and 32 steps" | only the last three offsets count in the objective, with coefficients (0.5, 0.5, 1) |
| Sec. 4.2, Table 3 text | "Goals are given by the weights of the three measurements (ammo, health, and frags) in the objective function." | D3/D4 measurements: ammo, health, frags |
| Abstract | "A model trained using the presented approach won the Full Deathmatch track of the Visual Doom AI Competition," | continues: "which was held in previously unseen environments." |
| Sec. 4.2 "Visual Doom AI Competition" | "The Full Deathmatch track evaluated generalization to previously unseen environments" | "We only enrolled in the Full Deathmatch track." |
| Sec. 4.2 "Visual Doom AI Competition" | "Our model won, outperforming the second best submission by more than 50%." | second best = Lample & Chaplot (2016) |
| Sec. 4.2 | "Our model was trained using a variant of the D4-tx-L regime." | n/a |

---

## 10. Items re-confirmed this session

### 10a. Arnold: Playing FPS Games with Deep Reinforcement Learning

Guillaume Lample, Devendra Singh Chaplot (equal contribution). arXiv:1609.05521v2. Read from https://arxiv.org/html/1609.05521.

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Sec. 5.1 "Hyperparameters" | "The replay memory contained the one million most recent frames." | 1,000,000 frames |
| Sec. 5.1 | "where $\epsilon$ was linearly decreased from $1$ to $0.1$ over the first million steps, and then fixed to $0.1$ ." | ε: 1 → 0.1 linear over 1M steps, then 0.1 |
| Sec. 5.1 | "All networks were trained using the RMSProp algorithm and minibatches of size $32$ ." | RMSProp, batch 32 |
| Sec. 5.1 | "Network weights were updated every $4$ steps, so experiences are sampled on average 8 times during the training" | update every 4 steps |
| Sec. 5.1 | "The discount factor was set to $\gamma=0.99$ ." | γ = 0.99 |
| Sec. 5.1, resolution | "we used a 16/9 resolution of 440x225 which we resized to 108x60." | 108×60, colour |
| Sec. 4.2 "Frame skip" | "A frame skip of $k=4$ turned out to be the best trade-off." | k = 4 (screen every k+1 frames) |
| Sec. 4.3 "Sequential updates" | "In all our experiments, we set the minimum history size to $4$ , and we perform the updates on $5$ states." | h = 4, updates on 5 states |
| Sec. 4.3 | "errors from states $o_{1}...o_{h}$ , where $h$ is the minimum history size for a state to be updated, are not backpropagated" | n/a |
| Sec. 3.1 "Game feature augmentation" | "At training time, the cost of the network is a combination of the normal DRQN cost and the cross-entropy loss." | FC 512 → FC k on the CNN output |
| Sec. 3.1 | "Note that the LSTM only takes as input the CNN output, and is never directly provided with the game features." | n/a |
| Sec. 3.1 | "we only used an indicator about the presence of enemies on the current frame." | n/a |
| Sec. 3.2 "Divide and conquer", switching rule | "If no enemies are detected in the current frame, or if the agent does not have any ammo left, the navigation network" | continues: "Otherwise, the decision is given to the action network." |
| Sec. 3.2 | "We used a DRQN augmented with game features for the action network, and a simple DQN for the navigation network." | navigation: 3 actions (forward, turn left, turn right) |

### 10b. DRQN: Deep Recurrent Q-Learning for Partially Observable MDPs

Matthew Hausknecht, Peter Stone. arXiv:1507.06527v4. Read from https://arxiv.org/html/1507.06527. Sections in this HTML are unnumbered, so they are cited by title. The HTML labels the last appendix "Appendix D Appendix C: Flickering Results" as printed; the experimental details appear under "Appendix C Appendix C: Experimental Details".

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| "Stable Recurrent Updates", Bootstrapped Random Updates | "updates begin at random points in the episode and proceed for only unroll iterations timesteps" | n/a |
| "Stable Recurrent Updates" | "The RNN’s initial state is zeroed at the start of the update." | n/a |
| "Stable Recurrent Updates" | "Therefore, to limit complexity, all results herein use the randomized update strategy." | n/a |
| "Flickering Atari Games" | "DRQN is trained using backpropagation through time for the last ten timesteps." | unroll 10 |
| Appendix A "Alternative Architectures" | "The LSTM portion of the architecture remains the same and is unrolled over the last 10 timesteps." | unroll 10 |
| Appendix C "Experimental Details" | "Networks were trained for 10 million iterations and used a replay memory of size 400,000." | 10M iterations; replay 400,000 |
| Appendix C | "all networks used ADADELTA [...] optimizer with a learning rate of 0.1 and momentum of 0.95." | ADADELTA, lr 0.1, momentum 0.95 (citation bracket elided) |
| Appendix C | "LSTM’s gradients were clipped to a value of ten to ensure learning stability." | clip 10 |
| Appendix C | "All other settings were identical to those given in [...]" | Mnih et al. 2015 (citation bracket elided) |
| Appendix C | "Policies were evaluated every 50,000 iterations by playing 10 episodes and averaging the resulting scores." | n/a |

### 10c. Double DQN: Deep Reinforcement Learning with Double Q-learning

Hado van Hasselt, Arthur Guez, David Silver. arXiv:1509.06461v3 (08 Dec 2015). Read from https://arxiv.org/html/1509.06461v3.

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| Section "Double DQN" (unnumbered section; the target equation is unnumbered in v3 HTML) | "We therefore propose to evaluate the greedy policy according to the online network, but using the target network" | `Y_t^{DoubleDQN} ≡ R_{t+1} + γ Q(S_{t+1}, argmax_a Q(S_{t+1}, a; θ_t), θ_t^−)` |
| Section "Double DQN" | "Its update is the same as for DQN, but replacing the target $Y^{\text{DQN}}_{t}$ with" | n/a |
| Section "Double DQN" | "The update to the target network stays unchanged from DQN, and remains a periodic copy of the online network." | n/a |

### 10d. Acme R2D2 (`acme/agents/jax/r2d2/learning.py`, `config.py`, master)

**Loss code (learning.py, lines 77–159), verbatim excerpts:**

```python
      # Maybe burn the core state in.
      if burn_in_length:
        burn_obs = jax.tree.map(lambda x: x[:burn_in_length], data.observation)
        key_grad, key1, key2 = jax.random.split(key_grad, 3)
        _, online_state = networks.unroll(params, key1, burn_obs, online_state)
        _, target_state = networks.unroll(target_params, key2, burn_obs,
                                          target_state)

      # Only get data to learn on from after the end of the burn in period.
      data = jax.tree.map(lambda seq: seq[burn_in_length:], data)
```

```python
      # Get value-selector actions from online Q-values for double Q-learning.
      selector_actions = jnp.argmax(online_q, axis=-1)
      ...
      batch_td_error = batch_td_error_fn(
          online_q[:-1],
          data.action[:-1],
          target_q[1:],
          selector_actions[1:],
          rewards[:-1],
          discounts[:-1])
      batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)

      # Importance weighting.
      probs = sample.info.probability
      importance_weights = (1. / (probs + 1e-6)).astype(online_q.dtype)
      importance_weights **= importance_sampling_exponent
      importance_weights /= jnp.max(importance_weights)
      mean_loss = jnp.mean(importance_weights * batch_loss)

      # Calculate priorities as a mixture of max and mean sequence errors.
      abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
      max_priority = max_priority_weight * jnp.max(abs_td_error, axis=0)
      mean_priority = (1 - max_priority_weight) * jnp.mean(abs_td_error, axis=0)
      priorities = (max_priority + mean_priority)
```

(The `importance_weights` line in the file ends with a trailing `# pyrefly: ignore[missing-attribute]` comment, omitted above. `...` marks omitted lines.)

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| learning.py, TD fn | `rlax.transformed_n_step_q_learning,` with `n=bootstrap_n,` `tx_pair=tx_pair),` | n-step transformed double-Q TD error |
| learning.py, burn-in gradient | no `stop_gradient` call anywhere in the file (searched: 0 matches) | after burn-in, the burned-in `online_state` goes straight into `networks.unroll(params, key1, data.observation, online_state)` |
| learning.py, priority mix | `max_priority = max_priority_weight * jnp.max(abs_td_error, axis=0)` | `p = η·max_t|δ_t| + (1−η)·mean_t|δ_t|`, η = `max_priority_weight` |
| learning.py, IS normalization | `importance_weights /= jnp.max(importance_weights)` | `w_i = (1/(P_i+1e-6))^β / max_j w_j` |
| rlax nonlinear_bellman.py (`transformed_n_step_q_learning`) | `return target_tm1 - q_a_tm1` | target from `transformed_n_step_returns(tx_pair, r_t, discount_t, v_t, n, stop_target_gradients=...)`; default `stop_target_gradients: bool = True` |

**`R2D2Config` defaults (config.py, lines 22–53), verbatim:**

| Field | Default |
|---|---|
| discount | `0.997` |
| target_update_period | `2500` |
| evaluation_epsilon | `0.` |
| num_epsilons | `256` |
| variable_update_period | `400` |
| burn_in_length | `40` |
| trace_length | `80` |
| sequence_period | `40` |
| learning_rate | `1e-3` |
| bootstrap_n | `5` |
| clip_rewards | `False` |
| tx_pair | `rlax.SIGNED_HYPERBOLIC_PAIR` |
| samples_per_insert_tolerance_rate | `0.1` |
| samples_per_insert | `4.0` |
| min_replay_size | `50_000` |
| max_replay_size | `100_000` |
| batch_size | `64` |
| prefetch_size | `2` |
| num_parallel_calls | `16` |
| replay_table_name | `adders_reverb.DEFAULT_PRIORITY_TABLE` |
| importance_sampling_exponent | `0.6` |
| priority_exponent | `0.9` |
| max_priority_weight | `0.9` |

config.py's module docstring reads `"""PPO config."""` (line 15), even though the file defines `R2D2Config`.

### 10e. rlax value transforms (`rlax/_src/transforms.py`, master)

| Source location | Verbatim quote | Equation/value |
|---|---|---|
| transforms.py L60–63 | `def signed_hyperbolic(x: Array, eps: float = 1e-3) -> Array:` | `return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x`, i.e. `h(x) = sign(x)(√(|x|+1) − 1) + εx`, ε = 1e-3 |
| transforms.py L61 docstring | `"""Signed hyperbolic transform, inverse of signed_parabolic."""` | n/a |
| transforms.py L78–82 | `def signed_parabolic(x: Array, eps: float = 1e-3) -> Array:` | `z = jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps - 1 / 2 / eps`; `return jnp.sign(x) * (jnp.square(z) - 1)`, i.e. `h^{-1}(x) = sign(x)(((√(1+4ε(|x|+1+ε)) − 1)/(2ε))^2 − 1)` |
| nonlinear_bellman.py L47–48 | `SIGNED_HYPERBOLIC_PAIR = TxPair(` `transforms.signed_hyperbolic, transforms.signed_parabolic)` | (forward, inverse) pair used by R2D2Config |
