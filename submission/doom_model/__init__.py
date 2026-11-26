"""Model code shared by the evaluated agent (submission/agent.py) and the training pipeline.

Everything here uses only agent-visible observations (docs/CONTRACT.md). Privileged state is used
only by training-time heads (critic input, auxiliary targets), which the agent never evaluates.
Design and paper references: training/docs/DESIGN.md.
"""

from .actions import ACTIONS, N_ACTIONS
from .config import ModelConfig
from .features import AgentSideFeatures
from .network import AgentNet

__all__ = ["ACTIONS", "N_ACTIONS", "ModelConfig", "AgentSideFeatures", "AgentNet"]
