"""AgentNet.core (segmented cuDNN GRU) must equal the per-step reset-then-step reference loop.

    uv run --group train python training/tests/test_core_segments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission")]
from doom_model import AgentNet, ModelConfig  # noqa: E402


def reference(net, x, h, starts):
    outs = []
    for t in range(x.shape[0]):
        h = h * (1 - starts[t])[None, :, None]
        o, h = net.gru(x[t:t + 1], h)
        outs.append(o)
    return torch.cat(outs), h


def main() -> int:
    torch.manual_seed(0)
    net = AgentNet(ModelConfig(hidden=64, embed=32)).double()
    worst = 0.0
    for trial in range(50):
        T, B = 128, 8
        x = torch.randn(T, B, 32, dtype=torch.float64)
        h = torch.randn(1, B, 64, dtype=torch.float64)
        p = [0.0, 0.002, 0.02, 0.2][trial % 4]                       # none / rare / some / many resets
        starts = (torch.rand(T, B) < p).double()
        if trial % 5 == 0:
            starts[0] = 1.0
        a, ha = net.core(x, h, starts)
        b, hb = reference(net, x, h, starts)
        worst = max(worst, (a - b).abs().max().item(), (ha - hb).abs().max().item())
    # mutant: forgetting the reset at segment starts must be detected
    starts = torch.zeros(128, 8, dtype=torch.float64); starts[40, 3] = 1
    x = torch.randn(128, 8, 32, dtype=torch.float64); h = torch.randn(1, 8, 64, dtype=torch.float64)
    no_reset, _ = net.gru(x, h)
    ref, _ = reference(net, x, h, starts)
    caught = (no_reset - ref).abs().max().item() > 1e-6
    print(f"segmented core vs per-step reference: max |diff| over 50 trials = {worst:.2e}; mutant caught: {caught}")
    ok = worst < 1e-10 and caught
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
