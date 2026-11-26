# Agent design and its sources

Every component below lists **what we implement** (file), **the source** (paper + section/equation/
table, quoted and checked in [REFERENCES.md](REFERENCES.md), item number in brackets), **why we
believe the implementation is correct**, and **deviations from the source**. "Verified here" means we
have a test or probe in this repo for it. Anything not backed by a source or a test is marked
**ASSUMPTION**.

Design standard (user decision): follow docs/RULES.md §1 literally. Every action comes from the trained
network. Deterministic preprocessing of observations (odometry, the ray-built map) is allowed:
"Allowed: preprocessing and encoding observations, action masking from observation fields, frame
stacking, learned memory modules".

---

## 1. Observation encoder (`submission/doom_model/network.py: Encoder`)

| Part | Implementation | Source and justification |
|---|---|---|
| Game variables as a separate input | `vitals` (17 values, scaled) + previous action → MLP, concatenated with the other features | Sample Factory [3a], App. A.1.3 / Fig. A.1: the full VizDoom model adds a separate "game info" input (23×1 → FC 128 → FC 128) to the conv features. DFP [9], Sec. 3, treats measurements as their own input stream. Both are proven on ViZDoom. |
| Rays | 64 rays × (distance + 6-way hit one-hot) → **circular** 1D conv | ASSUMPTION (no paper uses this sensor). Circular padding is used because ray 63 is adjacent to ray 0 (docs/CONTRACT.md §2.2: 64 rays over 360°). |
| Entities | Up to 32 entities × (5 features + type embedding) → shared MLP → masked mean and max pool | ASSUMPTION. A permutation-invariant pool is needed because the slots are sorted by distance, not identity. With no entity visible the output is exactly 0 (verified here: no NaN, unlike the attention pool in `probes/latency.py` flagged by the review). |
| Previous action as input | Embedding of the previous discrete action | CONTRACT §3: `USE` and `SELECT_NEXT_WEAPON` fire on the button-down edge, so the policy must know what it pressed last. |

## 2. Recurrent core: GRU, width 512 (`network.py: AgentNet.core`)

* Source: Sample Factory [3a], App. A.1.3: "in our large-scale VizDoom experiments with the full model we
  chose to use GRU RNN cells … instead of LSTM". Fig. A.1 shows FC 512 → GRU 512.
* Hidden state is reset where an episode starts, inside the sequence forward pass. This matches CleanRL
  `ppo_atari_lstm.py` [7b] (`(1.0 - d) * state` in `get_states`).
* Evidence for needing more than a GRU (probe, provisional): plain-GRU revisit accuracy drops from 0.67
  to 0.58 after 250 steps (memory probe v2). The review rightly notes this probe uses surrogate tasks.
  The decisive test is the RL ablation in §9.

## 3. Spatial memory: ray-built map (`submission/doom_model/features.py: Odometry, SpatialMap`)

* **What:** a 192×192 grid of 48-unit cells in an episode frame that starts at the agent (no global
  coordinates; CONTRACT §1). Channels are free / wall / door / locked door / visited, written from the
  64 rays using the dead-reckoned pose. The map is read as two ego-centric rotated crops (24×24 at 1
  cell per pixel, 24×24 at 6 cells per pixel) through a small CNN.
* **Sources:**
  * Active Neural SLAM [6]: the map is 2×M×M, with obstacle and explored channels (Sec. 4 "Map
    Representation"). The ego-centric projection is accumulated into a geocentric map using pose (Sec. 4
    "Neural SLAM Module", App. C: `m_t = m_{t−1} + f_ST(p^ego_t | x̂_t)`). The global policy reads a
    local crop stacked with a max-pooled global map (App. E). **Our two crops follow that
    local + coarse-global layout.**
  * Neural Map [5]: global read = "passes the current neural map M_t through a deep convolutional
    network" (Sec. 3.1). In ViZDoom (Sec. 4.2, Table 2), LSTM+Neural Map scored **78.3% / 66.6%** vs
    LSTM **69.2% / 52.4%** (training maze / 6 unseen mazes). That is the peer-reviewed evidence that
    explicit spatial memory helps in Doom, most of all on unseen maps.
* **Deviations, stated plainly:**
  1. Neural Map writes a *learned* vector at the agent position (Eq. 12). Our write is *geometric*
     (ANS-style), because our rays are already geometry. ANS learns its projection only because its
     input is RGB.
  2. Neural Map's Doom experiment gives the agent its position. ANS estimates pose with a learned
     model. We dead-reckon pose with Doom's own movement rules (next item).
  3. Their maps are 9×9 (Neural Map) or G=240 (ANS). Our 192 cells × 48 units is our own choice
     (ASSUMPTION), sized so a 9,216-unit span covers Freedoom maps. Writes outside the grid are dropped,
     not clipped (review fix: clipping drew fake border walls).
* **Odometry** (deterministic preprocessing; RULES §1 allows it; user decision "rules-compliant"):
  * heading: vanilla turn schedule; displacement: vanilla thrust/friction physics from the observed ego
    velocity, with a collision fallback, and zero translation when the ray scan is unchanged.
  * Verified here: `probes/odometry.py` and `odometry_learned.py`. Median drift on MAP17–24 is 28 / 54
    / 138 units after 250 / 500 / 1,000 steps, against 475 at 1,000 steps from velocity alone.
    `tests/test_features.py` checks that the online code reproduces the batch version (|Δ| < 4e-6
    units).
  * **Known limits (review, accepted):** the slow-turn length (5 tics) and the stillness threshold were
    fitted on probe data. Random exploration under-samples doors, lifts and teleports. Re-test pending
    (§9).
* **Crop orientation:** row 0 = ahead, columns left to right = left to right. Verified here at three
  headings (the probe version was reversed; review fix).

## 4. Optional episodic memory (`network.py: EpisodicReader`), PPO only

* Attention over GRU outputs stored every 4 steps. Keys and values get a learned projection of the
  memory's pose relative to the agent's current pose, plus its age.
* Sources: Neural Map key-value context read, Sec. 3.5.2, Eq. (15)–(19)
  (`M = [k, v]`, `a = q·k`, `c = Σ α v`). That is the same attention form, applied to stored embeddings
  instead of map cells. The stored states are detached and reused at training time, like R2D2's stored
  recurrent state (Acme learner [10d] replays `core_state` from replay).
* Status: an **ablation arm** (`configs/ppo_map_episodic.json`), not the default. No paper we verified
  shows it beats the map in Doom.

## 5. Policy and action space (`submission/doom_model/actions.py`)

* One categorical distribution over 56 button combinations: move × strafe × turn × attack, plus USE and
  NEXT_WEAPON as separate actions. Opposing buttons are never pressed together.
* Deviation from Sample Factory [3a] Sec. 4.3, which uses "independent discrete action heads" (5 heads;
  12,096 joint actions for Duel and Deathmatch). We use a joint categorical so that **DRQN can use the
  same action set**: a Q-network needs one value per discrete action. Arnold (DRQN in ViZDoom) also used
  a small discrete action set [10a]. If PPO wins §9, factorised heads are the follow-up that follows the
  source most closely.

## 6. PPO learner (`training/doomrl/ppo.py`, `configs/ppo_*.json`)

| Element | Source | Correctness evidence |
|---|---|---|
| Clipped surrogate `L^CLIP` | PPO [1] Sec. 3, Eq. (7) | Same formula, written as `max(−A·r, −A·clip(r))` (CleanRL [7e]) |
| Combined objective (policy − c1·value + c2·entropy) | PPO [1] Sec. 5, Eq. (9) | `loss = pg − ent_coef·H + vf_coef·vloss` |
| GAE | GAE [2] Sec. 3, Eq. (16); recursion as in CleanRL [7c] | `targets.gae` matches a CleanRL port exactly (`probes/test_targets.py`, max error 0.0) |
| Recurrent minibatching: a subset of envs, full rollout, from the stored initial state | CleanRL `ppo_atari_lstm.py` [7d] (`initial_lstm_state`, per-env minibatches) | Implemented the same way (`h0[:, e]`, `starts[:, e]`) |
| Clipped value loss, advantage normalisation | CleanRL [7e], defaults `clip_vloss=True`, `norm_adv=True` | As in the reference |
| Hyperparameters | CleanRL Args defaults [7a]: lr 2.5e-4 (annealed), num_steps 128, γ 0.99, λ 0.95, 4 minibatches, 4 epochs, clip 0.1, ent 0.01, vf 0.5, grad clip 0.5, Adam eps 1e-5 | We use exactly these, except **num_envs 32** (CleanRL: 8), for throughput |
| Why not Sample Factory's numbers | SF Table A.5 [3b] (lr 1e-4, 1 epoch, T=32, V-trace) belongs to *asynchronous* APPO | Mixing APPO settings into synchronous PPO would be incoherent, so we keep one reference for the algorithm |
| **Asymmetric critic** | Pinto et al. [4] Sec. I: "the actor takes visual partial observations as input while the critic takes the underlying full state as input"; valid because "During testing, the actor is employed on the partial observations and does not depend on the full state" | Critic gets 8 privileged scalars (time used, explored cells, kills, keys, health, doors, unlocks, exit). The actor never does; the saved weights strip the critic. **Deviation:** Pinto's method is DDPG/HER, not PPO. We rely on the same validity argument (the critic is only used in training). Time used also tells the critic how close the tick cap is, which reduces the bias from treating the cap as terminal (shared deviation, §8) |
| **Auxiliary heads** | Arnold [10a]: game features trained jointly, "the cost of the network is a combination of the normal DRQN cost and the cross-entropy loss" | Revisit (balanced BCE, a review fix) and the ego vector home (L1), weight 0.1. **ASSUMPTION:** the choice of targets and the 0.1 weight (Arnold's weight isn't stated in the text) |

## 7. DRQN learner (`training/doomrl/drqn.py`, `configs/drqn_map.json`)

| Element | Source | Correctness evidence |
|---|---|---|
| Stored-state sequence replay with burn-in; gradients flow through the online burn-in | Acme R2D2 `learning.py` [10d]: unrolls from `core_state`, burn-in with `params`, **no `stop_gradient`** | Same structure |
| n-step double-Q target with value rescaling | Double DQN [10c]: online argmax, target net evaluates; rlax transforms [10e] (eps 1e-3); Acme calls `transformed_n_step_q_learning` | `targets.nstep_double_q_target` matches an rlax port to 2e-13; an off-by-one mutant is caught (`probes/test_targets.py`) |
| Loss: 0.5·Σ_t td², IS-weighted batch mean, IS weights / max | Acme [10d] | Same |
| Replay storage | Acme stores sequences of `trace_length` 80 every `sequence_period` 40 [10d] | **Each step is stored once** in a per-env ring of preallocated arrays; a sequence is a window plus the recurrent state stored at its start; it is invalidated as soon as any of its steps is overwritten. Same sampling semantics as overlapping sequences, half the memory, no per-sequence Python objects (review fix). 6,872 B/step with map crops, so 1M steps ≈ 6.4 GiB, checked against `replay_ram_fraction` × RAM **before** allocating. `tests/test_replay.py` checks contiguity, h0, wrap-around invalidation and priority updates; a mutant without invalidation fails it |
| Priorities: 0.9·max\|td\| + 0.1·mean\|td\|, exponent 0.9, IS exponent 0.6 | Acme config [10d]: `max_priority_weight 0.9`, `priority_exponent 0.9`, `importance_sampling_exponent 0.6` | Same |
| seq 80, burn-in 40, period 40, n 5, target update 2,500 | Acme config [10d] | Same |
| Replay ~1M steps, γ 0.99, each step sampled ~8×, ε 1→0.1 over the first 1M steps | Arnold [10a] Sec. 4.2/4.3/5.1 | `replay_ratio: 8`; `epsilon: "arnold"` |
| Gradient clip 10 | DRQN [10b] App. C: "LSTM's gradients were clipped to a value of ten" | Same |
| Learning rate 1e-4, Adam eps 1e-3 | **ASSUMPTION**: Acme's default is 1e-3; the R2D2 paper could not be fetched (OpenReview bot check) | Flagged |
| Per-actor ε (`"epsilon": "apex"`, 0.4^(1+7i/(N−1))) | **ASSUMPTION**: the Ape-X formula, not verified this session. Arnold's shared schedule is the default | Flagged |
| Not reproduced | Arnold's separate navigation network and enemy-presence loss (our input already contains entities); R2D2's 256 actors; Arnold's RMSProp | Stated |

## 8. Environment, reward, curriculum, evaluation (`training/doomrl/`)

* **Shaped reward** (`envs.py`): exit +10, plus up to +2 for health and +2 for time left; key +2; door
  +0.5; unlock +3; new 64-unit cell +0.02 **capped at +4 per episode**; kill +0.5; −0.002 per HP lost;
  death −2. These mirror the official categories (docs/EVALUATION.md: exit largest, then progression,
  exploration, combat). The values are **ASSUMPTION** and the obvious candidates for Population-Based
  Training: Sample Factory [3d] Sec. 4.3 / App. A.3.1 mutates "individual reward shaping coefficients"
  (population 8, 15% mutation, ×1.2, every 5M frames).
* **Curriculum** (`maps.py`): scenario drills plus small maps, then more maps, then all training maps.
  Source: Wu & Tian [8] Sec. 4.3 (curriculum over enemy speed and health) won ViZDoom 2016 Track 1 (Sec.
  5.3: 559 vs 413 frags). **Deviation:** our stages are map pools by step count, not Wu & Tian's
  adaptive switch (theirs moves on when frags > 10). The stage boundaries are an ASSUMPTION.
* **Episode length** (review fix): the hidden maps use 2,100–4,200 ticks (CONTRACT §5), so training
  samples the tick cap per episode from {2,100, 4,200} and evaluation runs both by default. Behaviour
  after decision 525 is now trained and measured.
* **Map suites** (review fix): train = Freedoom 2 MAP01–16 + Freedoom 1 E1M1–E4M9 + 5 scenarios; dev =
  Freedoom 2 MAP17–24; **test = MAP25–32, untouched**. `evaluate.py` refuses `test` without `--final`,
  allows it **once** (a second look needs `--allow-test-reuse`), and logs every use with the checkpoint's
  sha256 in `training/docs/TEST_SUITE_USAGE.log`, which is committed so reviewers can see any reuse.
  `maps.assert_disjoint()` runs at every training start.
* **Evaluation** (`evaluate.py`): runs the **real `DoomAgent` on the official wrapper**. The agent
  samples with its own generator, reseeded at every `reset()`, so scores do not depend on how episodes
  are split across workers (verified: 6 vs 2 workers give identical episodes). It uses 5 seeds per map
  × both tick caps, bootstrap 95% CIs, and agent ms/step against the 40 ms budget.
* **Fail closed:** `DoomAgent` raises if `model.pt` or `model_config.json` is missing (it never submits
  a random network), and refuses weights whose core layers don't match the config.
  `ModelConfig` defaults are the production model (GRU 512, FC 512, map).
* **Validating the intended model:** `training/tools/validate_candidate.py --weights <ckpt> [--docker]`
  copies the working tree to a temp dir, installs the checkpoint as `submission/weights`, and runs the
  official `validate.py --repo <copy> --ticks 4200`. Production-shaped checkpoints (smoke-trained): map
  model p50/p99 0.7/1.4 ms, map+episodic 1.0/1.4 ms, 0.8–1.0 s of a 42 s budget. **Still to do on the
  VM with `--docker`:** 1-CPU pinning and peak RAM (macOS can't measure either).
* **RAM during training:** learner and worker RSS are logged with every update (`rss_learner_gb`,
  `rss_workers_gb`, Linux). The engine cache is 4 per worker (was 8). The first VM run must confirm the
  measured peak before scaling up `num_envs`.
* **PPO iteration count:** CleanRL rule `num_iterations = total_steps // (num_envs × num_steps)`, lr
  annealed by `1 − it/num_iterations`, so the last update has lr > 0 and the budget is never overshot
  (the actual step count is logged in `progress.json`).
* **Shared deviation:** reaching the tick cap is treated as terminal, in PPO (as CleanRL does: `done =
  terminations | truncations`) and in DRQN. The critic's time-used input reduces the resulting bias.
* **Fast training env** (`training/fastenv/`, numba, `uv run --group train`): training only;
  evaluation never uses it. `test_equivalence.py` shows **bit-identical** observations, reward, info
  and privileged state vs the official wrapper: 4 maps × 1,000 steps, plus 2,400 fuzzed edge cases
  (vertices, the 24/56-unit door thresholds, 32+ entities, locked-door hits). Max float difference: 0.
  Measured speedup is only **1.2–1.5× per step**: the remaining time is vizdoom's Python binding
  building and freeing sector/line objects on every `get_state()`. More throughput therefore needs more
  CPU cores, not code changes.

## 8b. DeepMind Acme: how it is used

Acme (github.com/google-deepmind/acme) is the **reference implementation**, not a dependency. Its
`setup.py` pins `jax==0.4.3`, `tensorflow==2.8.0`, `dm-reverb==0.7.2` (wheels for cp39–cp311 only) and
`dm-launchpad` (≤ cp39), while `doom_env` requires Python ≥ 3.13 and numpy 2. So Acme agents cannot
run in the same interpreter as the official wrapper. We port the specific pieces and test them against
Acme's code: the R2D2 learner now ([10d], `tests` above). Candidates to port the same way if needed:
`agents/jax/impala` (alternative to recurrent PPO) and `agents/jax/rnd` (exploration bonus).

## 9. What the code does NOT yet establish (open, in order)

1. **PPO vs DRQN** at equal budget on the dev suite (`ppo_map` vs `drqn_map`). The earlier probes used
   different code and the reused MAP17–24, so they don't count.
2. **Map vs no map vs map+episodic** (`ppo_gru`, `ppo_map`, `ppo_map_episodic`). This replaces the
   surrogate memory probe as the deciding test, matching Neural Map's [5] own Doom protocol: train,
   then evaluate on held-out maps.
3. Odometry re-test on scripted door/lift/teleport routes and the untouched test maps.
4. ≥3 seeds per arm, and decisions only where 95% CIs separate. If an ablation adds parameters (e.g.
   SEM), it needs a parameter-matched control (review).
5. Final number: the chosen checkpoint, once, on `test` with `--final`.
