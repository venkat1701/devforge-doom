# Hypothesis probes

Small, cheap experiments that check the design assumptions before we build the agent. Ground truth
comes from `env.privileged_state()`, which is training only. Every agent-side estimate uses only
what the agent can observe.

Data comes from a random explorer that picks a macro action and holds it for 1–12 steps
(`collect.py`). The maps are Freedoom 2: MAP01–16 for fitting, MAP17–24 held out as **unseen**.

```bash
uv run python training/probes/collect.py --out <dir> --maps MAP01 MAP02 ...
uv run python training/probes/odometry.py <dir>
uv run --with torch==2.15.0 python training/probes/odometry_learned.py <train dir> <test dir>
uv run --with torch==2.15.0 python training/probes/latency.py
```

## Results (2026-11-26, Apple M-series, 1 env per core)

| Hypothesis | Result | Verdict |
| --- | --- | --- |
| **H-thru**: RL on structured observations is cheap | 400–1,300 env steps/s per core. Rendering 160×120 pixels adds no measurable cost | ✅ ~5–10k steps/s on 10 cores |
| **H-odo-1**: heading can be tracked from own turn buttons | Vanilla turn schedule (Doom's `turnheld` counts tics with *any* turn key down; 5 slow tics at 320/65536 of a full turn, then 640) matches ground truth on **99.96%** of 11,301 steps. The only misses are death steps | ✅ exact |
| **H-scan**: heading change is visible in the ray scan | Circular shift of the 64-ray scan recovers the turn within one 5.6° bin on **100%** of 315 pure-turn steps | ✅ use as a check / correction |
| **H-odo-2**: displacement follows from vitals velocity | Vanilla per-tic physics (thrust 0.78125 forward / 0.75 strafe, friction 0.90625) is exact in free space. **But Doom keeps full velocity while you walk into a wall**, so velocity alone drifts badly (33–44% of explorer steps are wall contact) | ⚠️ velocity alone is not enough |
| **H-odo-3**: dead-reckoned pose is good enough for a grid map | Physics + "ray scan unchanged → not moving" rule, on **unseen** maps: median drift **11 / 28 / 54 / 138** map units after 100 / 250 / 500 / 1,000 steps (max 86 / 88 / 91 / 208). Velocity only: 28 / 95 / 212 / 475 | ✅ ~1 cell (64 u) over 500 steps; 2 cells over a full episode |
| **H-odo-L**: a learned odometry net beats the rule | 49k-param conv net trained on 35k steps (MAP01–16), residual on top of the rule: **worse** on unseen maps (drift 59 / 119 / 215 / 315) | ❌ not as tried. Use the rule; revisit with more data |
| **H-lat**: the full candidate agent fits 40 ms/step on 1 CPU thread | Ray CNN + entity attention + vitals MLP + numpy map write + 2 rotated 32×32 crops + map CNN + GRU-256 + 64-slot memory attention: **0.45 ms** mean, 0.56 ms p99, 1.2M params (4.6 MB) | ✅ ~88× headroom |

## Implications for the architecture

- **An explicit map is feasible.** Drift stays around one 64-unit cell for most of an episode,
  and the map can use 64-unit cells. Over a full 1,000-step episode expect ~2 cells of error, so
  the map should be tolerant: coarse cells, and older observations given less weight.
- **Odometry uses fixed rules, not learned weights.** It is preprocessing of observations; actions still come from the
  network. Confirm with the organizers that this reading of Rules §1 holds.
- **The compute budget is not a constraint** for this architecture class. Model size can be chosen for
  learning quality, not latency.
- **Still open:** does the explicit map actually beat a plain GRU for exploration and backtracking?
  That needs a memory probe or an RL ablation. See the next steps in the team notes.

## H-sem: Simplicial Embeddings

**Hypothesis:** adding Simplicial Embeddings (SEM; [Obando-Ceron et al., ICLR 2026](https://arxiv.org/abs/2510.13704))
to the recurrent PPO arm of `rl_algos.py` improves sample efficiency and final performance on unseen maps. Only the
architecture changes. The RL objective stays the same.

**Setup:** `rl_algos.py --algo ppo --sem` (opt-in; the default path is unchanged). We follow the paper's PPO/Atari
setup: SEM sits in the **actor only**, and the critic and shared trunk stay the same. The actor head goes from
`Linear(256→56)` to `Linear(256→512) → LayerNorm → softmax within each of L=128 groups of V=4 (τ=1) → Linear(512→56)`.
L=128 and V=4 are the paper's ALE values. The paper gives no default for τ, and its code uses a plain softmax. The
`Linear+LayerNorm+SimNorm` block matches the authors' `SimNormLinear`. The block adds about 146k parameters to the
actor head, so it is a confound. Budget is the same as the baseline: 600k steps, 2 seeds, both arms on the same Azure
VM at the same time. Eval is on unseen MAP17–24 (16 episodes). Every cell shows mean (seed 0 / seed 1).

| Steps | Arm | cells | kills | dead | return |
| --- | --- | --- | --- | --- | --- |
| 200k | ppo | **46.2** (48.9 / 43.6) | **1.41** (1.25 / 1.56) | 0.44 (0.69 / 0.19) | **5.18** (4.94 / 5.42) |
| 200k | ppo+SEM | 40.4 (40.4 / 40.5) | 1.09 (1.12 / 1.06) | **0.31** (0.25 / 0.38) | 4.51 (4.66 / 4.37) |
| 400k | ppo | **49.0** (58.1 / 39.8) | **2.00** (2.38 / 1.62) | 0.44 (0.56 / 0.31) | **6.04** (7.11 / 4.98) |
| 400k | ppo+SEM | 38.5 (33.9 / 43.1) | 1.28 (0.75 / 1.81) | **0.34** (0.38 / 0.31) | 4.43 (3.41 / 5.45) |
| 600k | ppo | **60.0** (64.8 / 55.3) | **2.34** (3.19 / 1.50) | 0.53 (0.38 / 0.69) | **7.34** (8.91 / 5.76) |
| 600k | ppo+SEM | 54.5 (46.1 / 62.9) | 1.78 (1.12 / 2.44) | **0.25** (0.19 / 0.31) | 6.65 (5.29 / 8.02) |
| final (same weights as 600k, re-evaluated) | ppo | 50.0 (52.6 / 47.4) | 1.59 (1.81 / 1.38) | 0.59 (0.38 / 0.81) | 5.49 (6.27 / 4.71) |
| final (same weights as 600k, re-evaluated) | ppo+SEM | **57.7** (47.8 / 67.5) | **2.22** (2.12 / 2.31) | **0.44** (0.31 / 0.56) | **7.12** (6.26 / 7.99) |

The "600k" and "final" rows evaluate the **same weights** twice. Baseline PPO scores 60.0 cells on one pass and 50.0
on the other, so a 16-episode eval has about ±5–10 cells of noise. Pooling both passes at 600k gives PPO 55.0 cells,
1.97 kills, 0.56 dead, return 6.42. PPO+SEM gets 56.1 cells, 2.00 kills, 0.34 dead, return 6.89.

On the **training** maps, the last 64 episodes at 563k steps favour the baseline. PPO reaches 75.7 cells, 3.47 kills
and return 9.74. PPO+SEM reaches 62.2 cells, 1.98 kills and return 7.21. SEM dies less often there too (0.52 vs 0.71).

**Runtime:** no measurable cost. Wall-clock time is 49.4–49.7 min for SEM vs 49.5 min for PPO. Throughput,
including evals, is about 200 vs 206 steps/s. That gap is within the noise from the 8 runs sharing the VM.

**Verdict:** ❌ SEM does **not** improve sample efficiency. It trails PPO on unseen maps at 200k and 400k, and
learns more slowly on the training maps. Final performance on unseen maps is a tie within eval noise. The one
consistent signal is fewer deaths (lower at every checkpoint, both seeds at 600k). With 2 seeds and noisy evals,
treat that as a trend, not a result. Not worth adopting as is.
