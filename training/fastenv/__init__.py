"""FastDoomEnv: training-only drop-in for doom_env.DoomEnv with accelerated observation code.

Observations, info and privileged_state() are identical to the official wrapper (see
test_equivalence.py). Evaluation always uses the official wrapper; use this only for training.

    import sys; sys.path.insert(0, "training")
    from fastenv import make_fast_env
    env = make_fast_env("maps/two_rooms.wad", render_buffers=False)

Needs numba (`uv sync --group train`); without it the env works but falls back to the official
sensor code.
"""

from __future__ import annotations

from doom_env.wrapper import DoomEnv

from .sensors_fast import HAVE_NUMBA, FastLevelGeometry, FastSensors

__all__ = ["FastDoomEnv", "make_fast_env", "HAVE_NUMBA", "FastLevelGeometry", "FastSensors"]


class FastDoomEnv(DoomEnv):
    """DoomEnv whose geometry update and sensor computation run as numba kernels.

    The official reset() builds the official LevelGeometry once per env; on the first
    observation it is swapped for a FastLevelGeometry carrying the same static data.
    Everything else (_observe, vitals, KeyTracker, step/reset logic, info, privileged_state) is
    inherited unchanged.
    """

    def _observe(self, state, first: bool = False) -> dict:
        if not isinstance(self._sensors, FastSensors):
            self._geometry = FastLevelGeometry.from_official(self._geometry)
            self._sensors = FastSensors(self._geometry)
        return super()._observe(state, first)


def make_fast_env(wad: str, map_name: str = "MAP01", **kwargs) -> FastDoomEnv:
    """Mirror of doom_env.make_env, returning a FastDoomEnv."""
    return FastDoomEnv(wad, map_name, **kwargs)
