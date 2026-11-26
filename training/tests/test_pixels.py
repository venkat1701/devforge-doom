"""Pixel branch + warm start.

  1. A pixel model loaded from a no-pixel checkpoint computes BIT-IDENTICAL policy logits (zero-initialised
     projection), for any pixel input
  2. the branch is live: once the projection moves, logits depend on the pixels
  3. pixels() on the official env: (4, 60, 80) uint8 with a real image (not all zero)
  4. DoomAgent with a pixel config runs on the official env (contract: valid actions)
  5. critic warm-up: a warm-started PPO run with only warm-up updates leaves every non-critic weight unchanged

    uv run --group train python training/tests/test_pixels.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doom_model import AgentNet, ModelConfig  # noqa: E402
from doom_model.features import CROP, MAP_CH, PIX_H, PIX_W, pixels  # noqa: E402


def rand_obs(T=3, B=4):
    return {"dist": torch.rand(T, B, 64), "hit": torch.randint(0, 6, (T, B, 64)), "ent": torch.randn(T, B, 32, 5),
            "etype": torch.randint(0, 12, (T, B, 32)), "emask": (torch.rand(T, B, 32) < 0.2).float(),
            "vit": torch.randn(T, B, 17), "prev": torch.randint(0, 56, (T, B)),
            "crops": torch.randint(0, 255, (T, B, 2 * MAP_CH, CROP, CROP), dtype=torch.uint8),
            "pose": torch.randn(T, B, 3),
            "pix": torch.randint(0, 255, (T, B, 4, PIX_H, PIX_W), dtype=torch.uint8)}


def main() -> int:
    fails = []
    torch.manual_seed(0)
    old = AgentNet(ModelConfig(aux=False)).eval()
    state = {k: v for k, v in old.state_dict().items() if not k.startswith(("v.", "aux."))}   # as saved
    new = AgentNet(ModelConfig(use_pixels=True)).eval()
    missing, unexpected = new.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("v.", "aux.", "enc.pix")) for k in missing):
        fails.append(f"load mismatch: {missing} {unexpected}")
    st = torch.zeros(3, 4); st[0] = 1
    with torch.no_grad():
        o = rand_obs()
        a = old(o, old.init_state(4), st)["logits"]
        b1 = new(o, new.init_state(4), st)["logits"]
        o2 = dict(o, pix=torch.randint(0, 255, o["pix"].shape, dtype=torch.uint8))
        b2 = new(o2, new.init_state(4), st)["logits"]
    if not (torch.equal(a, b1) and torch.equal(a, b2)):
        fails.append(f"warm start not identical: max diff {(a - b1).abs().max():.2e}")
    # 2) branch is live after the projection moves
    with torch.no_grad():
        new.enc.pix_proj.weight.normal_(0, 0.01)
        c1 = new(o, new.init_state(4), st)["logits"]; c2 = new(o2, new.init_state(4), st)["logits"]
    if torch.equal(c1, c2):
        fails.append("pixel branch has no effect after its projection changes")
    # 3) real pixels from the official env
    import vizdoom as vd
    from doom_env import make_env
    env = make_env(os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"), "MAP01", episode_ticks=200)
    obs, _ = env.reset(seed=1)
    px = pixels(obs)
    if px.shape != (4, PIX_H, PIX_W) or px.dtype != np.uint8 or px[:3].max() == 0:
        fails.append(f"pixels(): shape {px.shape} dtype {px.dtype} max {px[:3].max()}")
    # 4) agent with a pixel config on the official env
    import agent as agent_mod
    wdir = Path(tempfile.mkdtemp())
    ModelConfig(use_pixels=True, aux=False).save(wdir / "model_config.json")
    torch.save({k: v for k, v in AgentNet(ModelConfig(use_pixels=True, aux=False)).state_dict().items()
                if not k.startswith("v.")}, wdir / "model.pt")
    agent_mod.WEIGHTS_DIR = wdir
    ag = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    ag.reset(); n = 0; done = False
    while not done:
        act = ag.act(obs)
        if act.shape != (9,) or set(np.unique(act)) - {0, 1}:
            fails.append(f"invalid action {act}"); break
        obs, _, te, tr, _ = env.step(act); n += 1; done = te or tr
    env.close()
    # 5) critic warm-up leaves the policy untouched
    tmp = Path(tempfile.mkdtemp())
    init = tmp / "init.pt"
    torch.save(state, init)
    cfg = json.loads((ROOT / "training/configs/ppo_map_pixels_big.json").read_text())
    cfg["ppo"]["critic_warmup_updates"] = 2
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "doomrl.train", "--config", str(tmp / "cfg.json"), "--out", str(tmp / "run"),
                        "--num_envs", "2", "--init_from", str(init), "--start_step", "3000000",
                        "--total_steps", str(3_000_000 + 2 * 2 * 128)],
                       cwd=ROOT / "training", env=dict(os.environ, PYTHONPATH="training:submission".replace("training", str(ROOT / "training")).replace("submission", str(ROOT / "submission"))),
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        fails.append("warm-start training run failed: " + r.stderr[-400:])
    else:
        after = torch.load(tmp / "run" / "model.pt", weights_only=True)
        changed = [k for k, v in after.items() if k in state and not torch.equal(v, state[k])]
        pix_moved = not torch.equal(after["enc.pix_proj.weight"], torch.zeros_like(after["enc.pix_proj.weight"]))
        if changed or pix_moved:
            fails.append(f"critic warm-up changed policy weights: {changed[:5]} pix_proj moved={pix_moved}")
        prog = json.loads((tmp / "run" / "progress.json").read_text())
        if prog["steps_actual"] != 3_000_000 + 2 * 2 * 128:
            fails.append(f"start_step not honoured: steps_actual={prog['steps_actual']}")
    print(f"pixels: warm-start identical, branch live, real image {px.shape}, agent {n} steps, "
          f"warm-up freeze checked | failures: {len(fails)}")
    for f in fails:
        print("  ", f)
    print("PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
