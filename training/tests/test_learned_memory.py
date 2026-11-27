"""Opt-in latent memory: checkpoint, sequence, gradient and artifact contracts."""
import dataclasses
import json

import pytest
import torch
from torch import nn

from doom_model import AgentNet, ModelConfig
from doom_model.actions import N_ACTIONS
from doom_model.artifacts import export_policy, load_policy
from doom_model.features import PIX_H, PIX_W
from doom_model.network import Encoder
from doomrl.migrate import transfer_policy


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        yield
    torch.set_num_threads(threads)


def config(**kwargs):
    return ModelConfig(**dict(dict(hidden=16, embed=12, use_pixels=False,
                                   memory_slots=3, memory_dim=8), **kwargs))


def observations(t=5, n=2, use_pixels=False):
    result = {
        "dist": torch.rand(t, n, 64),
        "hit": torch.randint(6, (t, n, 64)),
        "ent": torch.randn(t, n, 32, 5),
        "etype": torch.randint(12, (t, n, 32)),
        "emask": torch.ones(t, n, 32),
        "vit": torch.randn(t, n, 17),
        "prev": torch.randint(N_ACTIONS, (t, n)),
    }
    if use_pixels:
        result["pix"] = torch.randint(256, (t, n, 4, PIX_H, PIX_W), dtype=torch.uint8)
    return result


class OriginalNetwork(nn.Module):
    """Independent pre-memory parameter construction, including initialization order."""

    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(cfg.embed, cfg.use_pixels)
        self.gru = nn.GRU(cfg.embed, cfg.hidden)
        if cfg.head == "actor_critic":
            self.pi = nn.Linear(cfg.hidden, N_ACTIONS)
            self.v = nn.Sequential(nn.Linear(cfg.hidden + cfg.privileged_dim, 256),
                                   nn.ReLU(), nn.Linear(256, 1))
        else:
            self.adv = nn.Sequential(nn.Linear(cfg.hidden, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
            self.val = nn.Sequential(nn.Linear(cfg.hidden, 256), nn.ReLU(), nn.Linear(256, 1))


@pytest.mark.parametrize("head", ["actor_critic", "q"])
def test_disabled_exact_legacy_state_dict_and_core(head, tmp_path):
    cfg = config(memory_slots=0, use_pixels=True, head=head)
    torch.manual_seed(3)
    original = OriginalNetwork(cfg)
    torch.manual_seed(3)
    net = AgentNet(cfg)
    assert not hasattr(net, "memory")
    assert original.state_dict().keys() == net.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(net.state_dict()[name], value), name
    net.load_state_dict(original.state_dict(), strict=True)
    legacy = dataclasses.asdict(cfg)
    del legacy["memory_slots"], legacy["memory_dim"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    loaded_cfg = ModelConfig.load(path)
    assert loaded_cfg.memory_slots == 0 and loaded_cfg.memory_dim == 64
    AgentNet(loaded_cfg).load_state_dict(original.state_dict(), strict=True)
    assert net.init_state(2).shape == (1, 2, cfg.hidden)
    x = torch.randn(5, 2, cfg.embed)
    starts = torch.zeros(5, 2)
    starts[2, 1] = 1
    initial = torch.randn_like(net.init_state(2))
    outputs, state = net.core(x, initial, starts)
    expected, h = [], initial
    for t in range(5):
        o, h = original.gru(x[t:t + 1], h * (1 - starts[t])[None, :, None])
        expected.append(o)
    torch.testing.assert_close(outputs, torch.cat(expected))
    torch.testing.assert_close(state, h)


@pytest.mark.parametrize("legacy_config", [True, False])
@pytest.mark.parametrize("head", ["actor_critic", "q"])
@pytest.mark.parametrize("use_pixels", [True, False])
def test_base_migration_preserves_logits_and_reports_fresh_memory(legacy_config, head, use_pixels):
    base = AgentNet(config(memory_slots=0, head=head, use_pixels=use_pixels))
    target = AgentNet(config(head=head, use_pixels=use_pixels))
    # Transfer also restores the zero residual when reusing a target instance.
    with torch.no_grad():
        target.memory.read_projection.weight.fill_(1)
    source_cfg = dataclasses.asdict(base.cfg)
    if legacy_config:
        del source_cfg["memory_slots"], source_cfg["memory_dim"]
    report = transfer_policy(target, base.state_dict(), source_cfg)
    memory_names = {name for name in target.state_dict() if name.startswith("memory.")}
    assert memory_names <= set(report["fresh"])
    assert report["requires_retraining"] is True
    assert report["function_preserving"] is True
    assert not torch.count_nonzero(target.memory.read_projection.weight)
    o, starts = observations(use_pixels=use_pixels), torch.zeros(5, 2)
    starts[0, 0] = starts[3, 1] = 1
    h = torch.randn_like(base.init_state(2))
    packed = torch.randn_like(target.init_state(2))
    packed[:, :, :base.cfg.hidden] = h
    key = "logits" if head == "actor_critic" else "q"
    expected = base.policy(o, h, starts)
    actual = target.policy(o, packed, starts)
    torch.testing.assert_close(actual[key], expected[key], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual["h"][:, :, :base.cfg.hidden], expected["h"])


def test_projection_learns_first_then_attention_write_and_slot_embeddings():
    net = AgentNet(config())
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    o, starts = observations(), torch.zeros(5, 2)
    net.policy(o, net.init_state(2), starts)["logits"].square().sum().backward()
    assert net.memory.read_projection.weight.grad.abs().sum() > 0
    for name, parameter in net.memory.named_parameters():
        if name != "read_projection.weight":
            assert parameter.grad is not None
            assert not torch.count_nonzero(parameter.grad), name
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    h = net.init_state(2).requires_grad_()
    net.policy(o, h, starts)["logits"].square().sum().backward()
    for name, parameter in net.memory.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert h.grad[:, :, net.cfg.hidden:].abs().sum() > 0


@pytest.mark.parametrize("head", ["actor_critic", "q"])
@pytest.mark.parametrize("n,slots,dim", [(1, 1, 1), (3, 4, 7)])
def test_sequence_matches_steps_and_chunked_minibatch_state(head, n, slots, dim):
    net = AgentNet(config(head=head, memory_slots=slots, memory_dim=dim))
    nn.init.normal_(net.memory.read_projection.weight)
    o, starts = observations(t=7, n=n), torch.zeros(7, n)
    starts[0, 0] = starts[3, -1] = starts[5, 0] = 1
    h = torch.randn_like(net.init_state(n))
    whole = net(o, h, starts)
    assert whole["core"].shape == (7, n, net.cfg.hidden)
    assert whole["h"].shape == (1, n, net.cfg.hidden + slots * dim)
    step_outputs, state = [], h
    for t in range(7):
        result = net({k: v[t:t + 1] for k, v in o.items()}, state, starts[t:t + 1])
        step_outputs.append(result)
        state = result["h"]
    for key in whole.keys() - {"h"}:
        torch.testing.assert_close(whole[key], torch.cat([result[key] for result in step_outputs]))
    torch.testing.assert_close(whole["h"], state)
    first = net.policy({k: v[:3] for k, v in o.items()}, h, starts[:3])
    envs = torch.arange(n - 1, -1, -1)
    h0 = first["h"].detach().clone()[:, envs]
    rest = net.policy({k: v[3:, envs] for k, v in o.items()}, h0, starts[3:, envs])
    torch.testing.assert_close(rest["core"], whole["core"][3:, envs])
    torch.testing.assert_close(rest["h"], whole["h"][:, envs])


def test_episode_resets_clear_both_states_at_initial_and_interior_boundaries():
    net = AgentNet(config()).double()
    nn.init.normal_(net.memory.read_projection.weight)
    x = torch.randn(6, 2, net.cfg.embed, dtype=torch.float64)
    zero = net.init_state(2)
    assert zero.dtype == torch.float64 and not torch.count_nonzero(zero)
    starts = torch.zeros(6, 2, dtype=torch.bool)
    starts[0, 0] = starts[3, 1] = True
    dirty = torch.randn_like(zero).requires_grad_()
    output, state = net.core(x, dirty, starts)
    reference, _ = net.core(x[:, :1], zero[:, :1], torch.zeros(6, 1, dtype=torch.bool))
    torch.testing.assert_close(output[:, :1], reference)
    reference, final = net.core(x[3:, 1:], zero[:, 1:], torch.zeros(3, 1, dtype=torch.bool))
    torch.testing.assert_close(output[3:, 1:], reference)
    torch.testing.assert_close(state[:, 1:], final)
    output[3:, 1:].sum().backward()
    assert not torch.count_nonzero(dirty.grad)
    # A reset also discards nonfinite state rather than multiplying NaNs by zero.
    clean, h = net.core(x[:1], torch.full_like(zero, float("nan")), torch.ones(1, 2))
    expected, expected_h = net.core(x[:1], zero, torch.ones(1, 2))
    torch.testing.assert_close(clean, expected)
    torch.testing.assert_close(h, expected_h)


def test_learned_slots_change_policy_without_changing_gru_initial_state():
    net = AgentNet(config())
    nn.init.normal_(net.memory.read_projection.weight)
    x, starts = torch.randn(2, 2, net.cfg.embed), torch.zeros(2, 2)
    state = net.init_state(2)
    altered = state.clone()
    altered[:, :, net.cfg.hidden:] = torch.randn_like(altered[:, :, net.cfg.hidden:])
    original, _ = net.core(x, state, starts)
    changed, _ = net.core(x, altered, starts)
    assert not torch.allclose(original, changed)


def test_slot_updates_preserve_gradients_across_time_until_reset():
    net = AgentNet(config())
    x = torch.randn(5, 2, net.cfg.embed, requires_grad=True)
    starts = torch.zeros(5, 2)
    starts[3, 1] = 1
    _, state = net.core(x, net.init_state(2), starts)
    state[:, :, net.cfg.hidden:].square().sum().backward()
    assert x.grad[0, 0].abs().sum() > 0
    assert not torch.count_nonzero(x.grad[:3, 1])
    assert x.grad[3, 1].abs().sum() > 0


@pytest.mark.parametrize("head", ["actor_critic", "q"])
def test_existing_artifact_export_load_roundtrip(head, tmp_path):
    net = AgentNet(config(head=head, use_pixels=True,
                          privileged_dim=3 if head == "actor_critic" else 0))
    nn.init.normal_(net.memory.read_projection.weight)
    path = export_policy(net, net.cfg, tmp_path / "memory", {"purpose": "memory-test"})
    loaded, cfg, manifest = load_policy(path)
    assert cfg == dataclasses.replace(net.cfg, privileged_dim=0)
    assert manifest["provenance"]["purpose"] == "memory-test"
    o, starts = observations(use_pixels=True), torch.zeros(5, 2)
    starts[3, 1] = 1
    before = net.policy(o, net.init_state(2), starts)
    after = loaded.policy(o, loaded.init_state(2), starts)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


@pytest.mark.parametrize("kwargs", [
    {"memory_slots": -1}, {"memory_slots": 1.5}, {"memory_slots": True}, {"memory_slots": None},
    {"memory_dim": 0}, {"memory_dim": -1}, {"memory_dim": 1.5}, {"memory_dim": True},
])
def test_bad_memory_config_rejected(kwargs):
    with pytest.raises(ValueError, match="memory"):
        config(**kwargs)


@pytest.mark.parametrize("target", [
    {"memory_slots": 0}, {"memory_slots": 4}, {"memory_dim": 9},
])
def test_memory_removal_or_layout_change_rejected(target):
    source = AgentNet(config())
    with pytest.raises(ValueError, match="memory"):
        transfer_policy(AgentNet(config(**target)), source.state_dict(), dataclasses.asdict(source.cfg))


def test_complete_memory_transfer_preserves_learned_projection():
    source, target = AgentNet(config()), AgentNet(config())
    nn.init.normal_(source.memory.read_projection.weight)
    report = transfer_policy(target, source.state_dict(), dataclasses.asdict(source.cfg))
    assert report["requires_retraining"] is False
    assert not any(name.startswith("memory.") for name in report["fresh"])
    for name, value in source.state_dict().items():
        if not name.startswith("v."):
            assert torch.equal(target.state_dict()[name], value)


def test_incomplete_memory_checkpoint_rejected_without_mutating_target():
    source, target = AgentNet(config()), AgentNet(config())
    initial = {name: value.clone() for name, value in target.state_dict().items()}
    source_cfg = dataclasses.asdict(source.cfg)
    for missing in [name for name in source.state_dict() if name.startswith("memory.")]:
        incomplete = dict(source.state_dict())
        del incomplete[missing]
        with pytest.raises(ValueError, match="missing required"):
            transfer_policy(target, incomplete, source_cfg)
        for name, value in target.state_dict().items():
            assert torch.equal(value, initial[name])


@pytest.mark.parametrize("mutation", [
    {"memory_slots": -1}, {"memory_slots": True}, {"memory_dim": 0}, {"memory_dim": False},
    {"memory_slots": 0},
])
def test_invalid_source_memory_declaration_rejected(mutation):
    source = AgentNet(config())
    cfg = dict(dataclasses.asdict(source.cfg), **mutation)
    with pytest.raises(ValueError, match="memory"):
        transfer_policy(AgentNet(config()), source.state_dict(), cfg)


@pytest.mark.parametrize("removed", [("memory_slots",), ("memory_dim",),
                                    ("memory_slots", "memory_dim")])
def test_missing_memory_source_config_rejected(removed):
    source = AgentNet(config())
    cfg = dataclasses.asdict(source.cfg)
    for key in removed:
        del cfg[key]
    with pytest.raises(ValueError, match="memory"):
        transfer_policy(AgentNet(config()), source.state_dict(), cfg)


def test_unpacked_state_rejected():
    net = AgentNet(config())
    with pytest.raises(ValueError, match="packed memory state"):
        net.core(torch.randn(2, 1, net.cfg.embed), torch.zeros(1, 1, net.cfg.hidden),
                 torch.zeros(2, 1))
