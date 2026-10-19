"""Choose the grid resolution, rather than guessing it.

This is not a cosmetic knob. On real data it changes the answer by orders of
magnitude, and it is the single most important free parameter in the whole
grid-based route:

  too fine   -- barrier bins are empty, so basins DISCONNECT. The generator
                then silently deletes the slow process entirely: on the
                alanine dipeptide dataset a 60x60 grid drops the alphaL basin
                (21k frames of real data) and reports a slowest timescale of
                ~120 ps instead of nanoseconds. Nothing errors. The number is
                just wrong.
  too coarse -- the barrier is under-resolved and the landscape is smeared.
  in between -- there are also near-disconnected grids, joined by a single
                weak edge, whose smallest eigenvalue is numerical noise
                (~1e13 ps timescales). These must be rejected explicitly.

Policy: take the FINEST grid that keeps every well-populated basin inside one
connected component and yields a numerically sane spectrum. Report the choice
and the rejected candidates, so the number on the slide carries its own
provenance.

This is also, concretely, the curse of dimensionality that motivates the PHNN:
the viable window between "disconnected" and "smeared" narrows as dimension
grows, while a learned continuous H has no bins to leave empty.
"""

from __future__ import annotations

import numpy as np

from . import fes
from .grid import TorusGrid
from .thermo import committor as cm
from .thermo.smoluchowski import build_generator

# A timescale longer than this is taken to be a numerical artefact of a
# near-disconnected generator rather than physics.
MAX_SANE_TIMESCALE_RATIO = 1e4


def evaluate_grid(X, n, D_const, temperature=298.0, min_count=1,
                  min_basin_frames=5000):
    """Diagnostics for one candidate resolution."""
    g = TorusGrid(n, n)
    F, counts, mask = fes.histogram_fes(X, g, temperature, min_count)
    try:
        L, keep = build_generator(F, D_const, g, temperature, mask)
    except Exception as e:
        return {"n": n, "ok": False, "reason": f"generator failed: {e}"}

    lab, names = fes.assign_basins(g, mask)
    kept = np.zeros(g.n_bins, dtype=bool)
    kept[keep] = True

    missing = []
    for k, nm in enumerate(names):
        frames = counts[lab == k].sum()
        if frames >= min_basin_frames and not np.any(kept & (lab == k)):
            missing.append((nm, int(frames)))

    ts = cm.timescales(L, k=3)
    sane = True
    reason = ""
    if ts.size >= 2 and ts[1] > 0 and ts[0] / ts[1] > MAX_SANE_TIMESCALE_RATIO:
        sane, reason = False, "near-disconnected: spectrum is numerically degenerate"
    if missing:
        sane, reason = False, f"basins severed: {missing}"

    return {
        "n": n, "ok": bool(sane), "reason": reason,
        "n_sampled": int(mask.sum()), "n_kept": int(len(keep)),
        "n_dropped": int(mask.sum() - len(keep)),
        "timescales_ps": ts.tolist(),
        "severed_basins": missing,
    }


def choose_grid(X, D_const, candidates=(20, 24, 30, 36, 45, 60),
                temperature=298.0, min_count=1, min_basin_frames=5000,
                verbose=True):
    """Finest viable grid, with the full audit trail."""
    log = print if verbose else (lambda *a, **k: None)
    reports = []
    for n in sorted(candidates):
        r = evaluate_grid(X, n, D_const, temperature, min_count, min_basin_frames)
        reports.append(r)
        status = "ok" if r["ok"] else f"REJECTED ({r['reason']})"
        ts = r.get("timescales_ps") or []
        ts_s = ", ".join(f"{t:.1f}" for t in ts[:2]) if ts else "-"
        log(f"     {n:3d}x{n:<3d} sampled={r.get('n_sampled', 0):5d} "
            f"dropped={r.get('n_dropped', 0):4d}  ts=[{ts_s}]  {status}")

    viable = [r for r in reports if r["ok"]]
    if not viable:
        # Fall back on FEWEST severed basins, then coarsest. Choosing by
        # "most bins kept" is actively wrong: the finest grid keeps the most
        # bins precisely because it has fragmented the landscape, so that rule
        # picks the most severed candidate. Coarser grids merge bins and are
        # more likely to bridge a thin barrier.
        best = min(reports, key=lambda r: (len(r.get("severed_basins") or []),
                                           r["n"]))
        log(f"     no fully viable grid; falling back to {best['n']}x{best['n']} "
            f"(fewest severed basins: {best.get('severed_basins')}) "
            "-- kinetics from it are NOT trustworthy and are flagged as such")
        return TorusGrid(best["n"], best["n"]), {"chosen": best["n"],
                                                 "viable": False,
                                                 "reports": reports}

    chosen = max(viable, key=lambda r: r["n"])
    log(f"     chosen: {chosen['n']}x{chosen['n']} "
        f"(finest grid keeping all populated basins connected)")
    return TorusGrid(chosen["n"], chosen["n"]), {"chosen": chosen["n"],
                                                 "viable": True,
                                                 "reports": reports}
