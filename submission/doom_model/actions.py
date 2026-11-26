"""Discrete action set over the contract's MultiBinary(9) buttons (docs/CONTRACT.md §3).

56 combos = move{-,F,B} x strafe{-,L,R} x turn{-,TL,TR} x attack{0,1}, plus USE alone and
SELECT_NEXT_WEAPON alone. Opposing buttons are never pressed together. USE and NEXT_WEAPON fire on the
button-down edge (CONTRACT §3), so they are separate actions and the previous action is a network
input: the policy can see whether the button is still held.

A single categorical over joint combos is used by both PPO and DRQN, because DQN-style learners
need one discrete Q per action (Arnold, Lample & Chaplot 2017, use a discrete set of combos too).
"""

from __future__ import annotations

import itertools

import numpy as np

BUTTON_INDEX = {"F": 0, "B": 1, "L": 2, "R": 3, "TL": 4, "TR": 5, "ATK": 6, "USE": 7, "WPN": 8}


def _build() -> np.ndarray:
    acts = []
    for mv, st, tn, atk in itertools.product((None, 0, 1), (None, 2, 3), (None, 4, 5), (0, 1)):
        a = np.zeros(9, np.int8)
        for i in (mv, st, tn):
            if i is not None:
                a[i] = 1
        a[6] = atk
        acts.append(a)
    for i in (7, 8):
        a = np.zeros(9, np.int8)
        a[i] = 1
        acts.append(a)
    return np.stack(acts)


ACTIONS = _build()
N_ACTIONS = len(ACTIONS)
