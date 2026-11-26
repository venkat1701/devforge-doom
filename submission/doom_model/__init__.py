"""Model code shared by the evaluated agent (submission/agent.py) and the training pipeline.

Everything here uses only agent-visible observations (docs/CONTRACT.md). Privileged state is used
only by training-time heads (critic input, auxiliary targets), which the agent never evaluates.
Design and paper references: training/docs/DESIGN.md.
"""

from .actions import ACTIONS, N_ACTIONS
from .config import ModelConfig
from .features import AgentSideFeatures

__all__ = ["ACTIONS", "N_ACTIONS", "ModelConfig", "AgentSideFeatures", "AgentNet"]


def __getattr__(name):
    # AgentNet (and torch) load lazily: env worker processes import doom_model.features (numpy only) and
    # must not pay torch's memory per process (measured ~22 GB RSS over 32 workers when imported eagerly).
    if name == "AgentNet":
        from .network import AgentNet
        return AgentNet
    raise AttributeError(name)
