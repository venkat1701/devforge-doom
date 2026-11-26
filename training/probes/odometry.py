"""Probe: can the agent dead-reckon its pose from observations alone?

H-odo-1  Heading change per step is exactly predictable from the turn-button history
         (vanilla Doom: slow turn for the first tics a turn key is held, then fast).
H-odo-2  Ego-frame displacement per step is predictable from vitals velocity + buttons using
         Doom's per-tic physics (thrust, move, friction 0.90625), except when the player collides.
H-odo-3  Integrated pose drifts little enough over an episode for a grid map.
H-scan   Heading change can also be recovered from consecutive ray scans (circular shift).

Fits on even episodes, evaluates on odd ones. Ground truth = privileged_state (training only).

    uv run python training/probes/odometry.py <data dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F, B, L, R, TL, TR = range(6)
TICS = 4
FRICTION = 0xE800 / 0x10000           # 0.90625
SLOW_TURN, FAST_TURN = 320 * 360 / 65536, 640 * 360 / 65536   # degrees per tic (angleturn)
SLOW_TICS = 5                          # fitted from H-odo-1 v1 output: 4 slow, then 1 slow + 3 fast


def load(d: Path):
    eps = []
    for f in sorted(d.glob("*.npz")):
        z = dict(np.load(f))
        z["name"] = f.stem
        eps.append(z)
    return eps


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def tic_turns(actions):
    """(n, 4) degrees turned in each tic. Doom's turnheld counts tics with ANY turn key down."""
    out = np.zeros((len(actions), TICS))
    held = 0
    for t, a in enumerate(actions):
        sign = int(a[TL]) - int(a[TR])
        for k in range(TICS):
            if a[TL] or a[TR]:
                out[t, k] = sign * (SLOW_TURN if held < SLOW_TICS else FAST_TURN)
                held += 1
            else:
                held = 0
    return out


def physics_features(z, turns):
    """Free-space displacement over a step, in the ego frame at t, as linear features.

    Per tic k: theta_k += turn_k; mom += thrust * dir(theta_k); pos += mom; mom *= FRICTION.
    disp = v_now * sum_k f^k ... is linear in v_now and in the (unknown) forward/strafe thrusts.
    Returns (feat (n, 2, 3), heading offsets per tic). feat[:, axis, :] @ [1, thrust_f, thrust_s].
    """
    a = z["action"][:-1].astype(float)
    v = z["vel"][:-1]
    n = len(a)
    theta = np.radians(np.cumsum(turns, 1))          # heading offset (rad) during tic k, rel. to t
    fwd_in, side_in = a[:, F] - a[:, B], a[:, R] - a[:, L]
    # momentum carried from t: contributes v * f^k at tic k (k=0 uses it before friction).
    carry = sum(FRICTION ** k for k in range(TICS))
    feat = np.zeros((n, 2, 3))
    feat[:, :, 0] = v * carry
    for k in range(TICS):
        w = sum(FRICTION ** j for j in range(TICS - k))    # thrust at tic k persists for TICS-k moves
        c, s = np.cos(theta[:, k]), np.sin(theta[:, k])
        # unit forward of tic-k heading, in frame t (fwd, right): (cos, -sin); unit right: (sin, cos)
        feat[:, 0, 1] += w * fwd_in * c
        feat[:, 1, 1] += w * fwd_in * -s
        feat[:, 0, 2] += w * side_in * s
        feat[:, 1, 2] += w * side_in * c
    return feat


def predicted_vnext(z, turns, thrust):
    """Free-space momentum at t+1 in the ego frame at t+1, to detect collisions."""
    a = z["action"][:-1].astype(float)
    m = z["vel"][:-1].copy()
    theta = np.radians(np.cumsum(turns, 1))
    fwd_in, side_in = a[:, F] - a[:, B], a[:, R] - a[:, L]
    for k in range(TICS):
        c, s = np.cos(theta[:, k]), np.sin(theta[:, k])
        m = m + thrust[0] * fwd_in[:, None] * np.stack([c, -s], 1) + thrust[1] * side_in[:, None] * np.stack([s, c], 1)
        m = m * FRICTION
    tot = theta[:, -1]                                   # rotate frame t -> frame t+1
    c, s = np.cos(tot), np.sin(tot)
    return np.stack([m[:, 0] * c - m[:, 1] * s, m[:, 0] * s + m[:, 1] * c], 1)


def truth(z):
    """Heading change (deg) and displacement in the ego frame at t."""
    h = np.radians(z["angle"])
    dx, dy = np.diff(z["x"]), np.diff(z["y"])
    c, s = np.cos(h[:-1]), np.sin(h[:-1])
    fwd, right = dx * c + dy * s, dx * s - dy * c   # same convention as doom_env.sensors.ego_rotate
    return wrap_deg(np.diff(z["angle"])), np.stack([fwd, right], 1)


def scan_residual(z, turns):
    """Median |ray distance change| (map units) after undoing the step's turn by a fractional
    circular shift. ~0 means the scan did not change, i.e. the player did not translate."""
    d0, d1 = z["dist"][:-1] * 1024.0, z["dist"][1:] * 1024.0
    shift = turns.sum(1) / (360.0 / 64)                 # bins; turning left moves walls to lower idx
    idx = (np.arange(64)[None, :] + shift[:, None]) % 64
    lo = np.floor(idx).astype(int); fr = idx - lo
    rows = np.arange(len(d1))[:, None]
    d1s = (1 - fr) * d1[rows, lo] + fr * d1[rows, (lo + 1) % 64]
    return np.median(np.abs(d1s - d0), 1)   # median: robust to a few rays hitting moving monsters/doors


def estimate_disp(z, turns, thrust, collide_thresh, use_rays=False, still_thresh=0.25):
    """Physics estimate; on detected collision fall back to the mean of observed momenta x TICS.
    use_rays: if the ray scan is unchanged (after undoing the turn), the player is blocked -> 0."""
    feat = physics_features(z, turns)
    free = feat @ np.array([1.0, thrust[0], thrust[1]])
    vn = z["vel"][1:]
    tot = np.radians(turns.sum(1))
    c, s = np.cos(tot), np.sin(tot)
    vn_t = np.stack([vn[:, 0] * c + vn[:, 1] * s, -vn[:, 0] * s + vn[:, 1] * c], 1)  # frame t+1 -> t
    collided = np.linalg.norm(predicted_vnext(z, turns, thrust) - vn, axis=1) > collide_thresh
    fallback = 0.5 * (z["vel"][:-1] + vn_t) * TICS
    est = np.where(collided[:, None], fallback, free)
    if use_rays:
        est = np.where((scan_residual(z, turns) < still_thresh)[:, None], 0.0, est)
    return est, collided


def main() -> None:
    eps = load(Path(sys.argv[1]))
    train, test = eps[0::2], eps[1::2]

    # ---- H-odo-1
    print("H-odo-1  heading: Doom turn schedule vs ground truth (all episodes)")
    errs = np.concatenate([tic_turns(z["action"][:-1]).sum(1) - truth(z)[0] for z in eps])
    print(f"   exact (<0.01°) on {np.mean(np.abs(errs) < 0.01):.2%} of {len(errs)} steps; "
          f"max |err| = {np.abs(errs).max():.3f}°; steps with |err|>1°: {(np.abs(errs) > 1).sum()}")

    # ---- H-odo-2: vanilla thrust constants (forwardmove 25, sidemove 24, x2048/65536), not fitted:
    # a least-squares fit is dominated by wall-contact steps, which a random explorer has plenty of.
    thrust = np.array([25 * 2048 / 65536, 24 * 2048 / 65536])
    print(f"\nH-odo-2  thrust per tic: forward={thrust[0]:.4f}, strafe={thrust[1]:.4f} units")
    for (z_set, label), use_rays in [(p, r) for r in (False, True) for p in ((train, "train"), (test, "test "))]:
        label = f"{label} {'vel+rays' if use_rays else 'vel only'}"
        e_all, e_free, ncol, n = [], [], 0, 0
        for z in z_set:
            turns = tic_turns(z["action"][:-1])
            est, col = estimate_disp(z, turns, thrust, collide_thresh=0.5, use_rays=use_rays)
            err = np.linalg.norm(est - truth(z)[1], axis=1)
            e_all.append(err); e_free.append(err[~col]); ncol += col.sum(); n += len(col)
        e_all, e_free = np.concatenate(e_all), np.concatenate(e_free)
        print(f"   {label}  mean |err| all={e_all.mean():.3f}  free-space={e_free.mean():.4f} units/step  "
              f"collision steps={ncol / n:.1%}")

    # ---- H-odo-3
    for use_rays in (False, True):
      print(f"\nH-odo-3  dead-reckoning drift, held-out episodes, {'vel + ray stillness' if use_rays else 'velocity only'}"
            " (map units; 64 = one grid cell)")
      print("   episode       steps  heading@end  pos@100  pos@250  pos@500  pos@1000  pos@end   path")
      for z in test:
        n = len(z["x"]) - 1
        turns = tic_turns(z["action"][:-1])
        th = np.radians(z["angle"][0] + np.concatenate([[0], np.cumsum(turns.sum(1))]))
        disp, _ = estimate_disp(z, turns, thrust, collide_thresh=0.5, use_rays=use_rays)
        c, s = np.cos(th[:-1]), np.sin(th[:-1])
        wx = disp[:, 0] * c + disp[:, 1] * s
        wy = disp[:, 0] * s - disp[:, 1] * c
        px = z["x"][0] + np.concatenate([[0], np.cumsum(wx)])
        py = z["y"][0] + np.concatenate([[0], np.cumsum(wy)])
        perr = np.hypot(px - z["x"], py - z["y"])
        herr = abs(wrap_deg(np.degrees(th[-1]) - z["angle"][-1]))
        path = np.hypot(np.diff(z["x"]), np.diff(z["y"])).sum()
        at = lambda k: f"{perr[k]:7.1f}" if k <= n else "      -"
        print(f"   {z['name']:12s} {n:5d}  {herr:9.2f}°  {at(100)}  {at(250)}  {at(500)}  {at(1000)}  "
              f"{perr[-1]:7.1f}  {path:6.0f}")

    # ---- H-scan
    print("\nH-scan   heading change from consecutive ray scans (pure-turn steps)")
    hits = total = 0
    for z in test:
        dth, disp = truth(z)
        d = z["dist"]
        for t in range(len(dth)):
            if np.hypot(*disp[t]) > 0.5 or abs(dth[t]) < 1:
                continue
            errs = [np.abs(np.roll(d[t + 1], k) - d[t]).mean() for k in range(-8, 9)]
            est = (int(np.argmin(errs)) - 8) * 360.0 / 64
            hits += abs(est - dth[t]) <= 360.0 / 64
            total += 1
    print(f"   within one ray bin (5.6°): {hits / max(total, 1):.1%} of {total} pure-turn steps")


if __name__ == "__main__":
    main()
