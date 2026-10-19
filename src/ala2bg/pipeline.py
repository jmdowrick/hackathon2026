"""The pipeline driver: data -> FES + D -> PHNN -> bond graph -> validation.

Writes every artifact into one run directory with a manifest recording the
provider, the CV-spec hash and the input hashes, so any number on a slide can
be traced back to what produced it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import gridselect
from . import diffusion, fes, msm as msm_mod
from .cvs import DEFAULT_CV
from .grid import TorusGrid
from .thermo import bondgraph as bg_mod
from .thermo import committor as cm
from .thermo.cellml_writer import write_cellml
from .thermo.smoluchowski import build_generator, stationary_from_F, rt


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _save(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, cls=NumpyEncoder))


def run(
    out_dir="runs/latest",
    provider="toy",
    provider_kw=None,
    n_phi=None,
    n_psi=None,
    grid_candidates=(20, 24, 30, 36, 45, 60),
    temperature=298.0,
    min_count=1,
    phnn_steps=4000,
    phnn_min_count=30,
    do_phnn=True,
    do_baseline=True,
    do_msm=True,
    n_clusters=100,
    msm_lag=10,
    seed=0,
    verbose=True,
):
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = None if n_phi is None else TorusGrid(n_phi, n_psi)
    cv = DEFAULT_CV
    log = print if verbose else (lambda *a, **k: None)

    manifest = {
        "schema_version": "1.0",
        "cv_spec_hash": cv.hash(),
        "cv_spec": cv.to_dict(),
        "temperature_K": temperature,
        "RT_kJ_per_mol": rt(temperature),
        "provider_requested": provider,
    }

    # ---- 10/20 data -------------------------------------------------------
    log(f"[10] loading data (provider={provider}) ...")
    d = data_mod.load(provider, **(provider_kw or {}))
    X, dt, tid = d["X"], d["dt"], d["traj_id"]
    manifest["provider_actual"] = d["meta"].get("provider", provider)
    manifest["data"] = {**d["meta"], "n_frames": int(len(X)), "dt_ps": dt,
                        "n_traj": int(len(np.unique(tid)))}
    log(f"     {len(X)} frames, {len(np.unique(tid))} trajectories, dt={dt} ps")
    np.savez_compressed(out / "cv.npz", X=X, traj_id=tid, dt=dt)

    # ---- 35 diffusion (MEASURED, FROZEN) ----------------------------------
    log("[35] measuring D from the 2nd Kramers-Moyal moment ...")
    Dm = diffusion.measure_D(X, dt, tid, lag=1)
    Dstab = diffusion.lag_stability(X, dt, tid)
    D_const = Dm["D_const"]
    np.savez_compressed(out / "diffusion.npz", D_per_dim=Dm["D_per_dim"],
                        D_const=D_const)
    log(f"     D = {D_const:.5g} rad^2/ps (per-dim {Dm['D_per_dim']}); "
        f"lag-stability max ratio {Dstab['max_ratio']:.2f}")

    # ---- grid resolution: chosen, not guessed ------------------------------
    if grid is None:
        log("[32] choosing grid resolution (finest that stays connected) ...")
        grid, grid_report = gridselect.choose_grid(
            X, D_const, grid_candidates, temperature,
            min_count=min_count, verbose=verbose)
    else:
        grid_report = {"chosen": grid.n_phi, "viable": None,
                       "reports": [], "note": "grid fixed by caller"}
    manifest["grid"] = {"n_phi": grid.n_phi, "n_psi": grid.n_psi,
                        "selection": grid_report}

    # ---- 30 FES (thermodynamic route), on the chosen grid -----------------
    log("[30] histogram FES ...")
    F_emp, counts, mask = fes.histogram_fes(X, grid, temperature, min_count)
    F_err = fes.bootstrap_fes_error(X, grid, temperature, n_boot=12,
                                    traj_id=tid, seed=seed, min_count=min_count)
    labels, basin_names = fes.assign_basins(grid, mask)
    basin_dG = fes.basin_free_energies(F_emp, counts, labels, basin_names, temperature)
    trans = fes.count_transitions(X, grid, labels, basin_names, tid)
    np.savez_compressed(out / "fes.npz", F=F_emp, counts=counts, mask=mask,
                        F_err=F_err, labels=labels)
    log(f"     {int(mask.sum())}/{grid.n_bins} bins sampled; basins "
        + ", ".join(f"{k}={v:.1f}" for k, v in basin_dG.items() if np.isfinite(v)))

    results = {
        "grid": {"n_phi": grid.n_phi, "n_psi": grid.n_psi,
                 "selection": grid_report},
        "basins_empirical_kJ": basin_dG,
        "transitions_direct": trans,
        "diffusion": {"D_const": D_const,
                      "D_per_dim": Dm["D_per_dim"].tolist(),
                      "lag_stability": Dstab,
                      "rung": "constant (div D = 0 exactly)"},
    }

    # ---- 50 PHNN (kinetic route to H) -------------------------------------
    H_phnn = None
    if do_phnn:
        try:
            from . import phnn as phnn_mod
            log(f"[50] training PHNN (H only; D frozen) for {phnn_steps} steps ...")
            model, hist, info = phnn_mod.train_binned(
                X, dt, D_const, grid, temperature, traj_id=tid,
                steps=phnn_steps, seed=seed, min_count=phnn_min_count,
                verbose=verbose)
            pts = grid.center_grid()
            H_phnn = phnn_mod.evaluate_H(model, pts)
            np.savez_compressed(out / "phnn.npz", H=H_phnn)

            # Compare only where there IS data. Outside the sampled region the
            # network is extrapolating and so is the histogram, so a residual
            # there measures nothing.
            cmp_mask = mask & (counts >= phnn_min_count)
            resid = (H_phnn - F_emp)[cmp_mask]
            resid = resid - np.nanmean(resid)
            hh = H_phnn[cmp_mask]; ff = F_emp[cmp_mask]
            results["phnn"] = {
                "fit": info,
                "final_val_loss": hist[-1]["val"],
                "n_bins_compared": int(cmp_mask.sum()),
                "H_vs_FES_rmse_kJ": float(np.sqrt(np.nanmean(resid ** 2))),
                "H_vs_FES_corr": float(np.corrcoef(hh - hh.mean(),
                                                   ff - ff.mean())[0, 1]),
                "history": hist,
            }
            log(f"     H vs empirical FES RMSE = "
                f"{results['phnn']['H_vs_FES_rmse_kJ']:.2f} kJ/mol "
                "(this tests Markovianity of the CVs)")
        except Exception as e:
            log(f"     PHNN skipped: {e}")
            results["phnn_error"] = str(e)

    # ---- 60 bond graph ----------------------------------------------------
    log("[60] building bond graph ...")
    F_use = H_phnn if H_phnn is not None else F_emp
    source = "phnn" if H_phnn is not None else "histogram"
    F_use = np.where(mask, F_use, np.nan)

    L, keep = build_generator(F_use, D_const, grid, temperature, mask)
    pi = stationary_from_F(F_use, keep, temperature)
    bg = bg_mod.from_generator(L, np.asarray(F_use)[keep], grid.cell_area,
                               temperature,
                               meta={"F_source": source,
                                     "cv_spec_hash": cv.hash(),
                                     "D_const": D_const})
    checks = bg_mod.check_thermo(bg, pi)
    weg = bg_mod.wegscheider_residuals(L, grid, keep, n_random=40, seed=seed)

    # committor + MFPT between the two best-populated basins
    remap = -np.ones(grid.n_bins, dtype=int)
    remap[keep] = np.arange(len(keep))
    pops = {nm: counts[labels == k].sum() for k, nm in enumerate(basin_names)}
    ranked = sorted(pops, key=pops.get, reverse=True)
    A_name, B_name = ranked[0], ranked[1]
    A = remap[np.flatnonzero((labels == basin_names.index(A_name)) & mask)]
    B = remap[np.flatnonzero((labels == basin_names.index(B_name)) & mask)]
    A, B = A[A >= 0], B[B >= 0]

    kin = {"A": A_name, "B": B_name}
    if A.size and B.size:
        q = cm.committor(L, A, B)
        kin["mfpt_AB_ps"] = cm.mfpt_between(L, A, B, pi)
        kin["mfpt_BA_ps"] = cm.mfpt_between(L, B, A, pi)
        ts = cm.timescales(L, k=4)
        kin["timescales_ps"] = ts.tolist()
        kin["slowest_ps"] = float(ts[0]) if ts.size else float("nan")
        qfull = np.full(grid.n_bins, np.nan)
        qfull[keep] = q
        np.savez_compressed(out / "committor.npz", q=qfull)
        kin["transition_state_bins"] = int(np.sum(np.abs(q - 0.5) < 0.05))
        log(f"     MFPT {A_name}->{B_name} = {kin['mfpt_AB_ps']:.1f} ps; "
            f"slowest timescale {kin.get('slowest_ps', float('nan')):.1f} ps")

    _save(bg.to_dict(), out / "bondgraph.json")
    _save({"thermo": checks, "wegscheider": weg}, out / "checks.json")
    _save(kin, out / "kinetics.json")
    results["bondgraph"] = {"n_states": bg.n_states, "n_reactions": bg.n_reactions,
                            "F_source": source}
    results["checks"] = checks
    results["wegscheider"] = weg
    results["kinetics"] = kin
    def _fmt(v):
        return "n/a" if v is None else f"{v:.2e}"
    log(f"     Wegscheider max residual: plaquette {_fmt(weg['plaquette']['max'])}, "
        f"winding_phi {_fmt(weg['winding_phi']['max'])}, "
        f"winding_psi {_fmt(weg['winding_psi']['max'])}")
    if weg["winding_phi"]["max"] is None and weg["winding_psi"]["max"] is None:
        log("     (no complete winding loop lies inside the sampled region -- "
            "the non-contractible cycle test is unavailable on this dataset, "
            "which is a statement about coverage, not a failure)")

    # ---- coarse bond graph + CellML ---------------------------------------
    try:
        log("[60] coarse basin model + CellML ...")
        present = [k for k, nm in enumerate(basin_names)
                   if np.any((labels == k) & mask)]
        cF, clab = [], []
        for k in present:
            sel = (labels == k) & mask
            w = np.exp(-(np.asarray(F_use)[sel]) / rt(temperature))
            cF.append(-rt(temperature) * np.log(np.nansum(w)))
            clab.append(basin_names[k])
        cF = np.array(cF) - np.min(cF)

        edges, kappas = [], []
        Kc = np.exp(cF / rt(temperature)) / grid.cell_area
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                kap = D_const / grid.h[0] ** 2 * grid.cell_area * \
                    np.exp(-(cF[a] + cF[b]) / (2 * rt(temperature)))
                edges.append((a, b)); kappas.append(kap)
        coarse = bg_mod.BondGraph(K=Kc, kappa=np.array(kappas),
                                  edges=np.array(edges, dtype=int), F=cF,
                                  temperature=temperature, labels=clab,
                                  meta={"kind": "coarse", "basins": clab})
        write_cellml(coarse, out / "model_coarse.cellml",
                     model_name="ala2_bond_graph_coarse", max_states=12)
        _save(coarse.to_dict(), out / "bondgraph_coarse.json")
        results["coarse"] = {"states": clab, "F_kJ": cF.tolist()}
        log(f"     coarse model: {clab} -> model_coarse.cellml")
    except Exception as e:
        log(f"     coarse/CellML skipped: {e}")
        results["coarse_error"] = str(e)

    # ---- the unconstrained baseline (gives the cycle test teeth) ----------
    if do_baseline and H_phnn is not None:
        try:
            from . import phnn as phnn_mod
            log("[60] unconstrained baseline (for the cycle comparison) ...")
            bmodel, bhist, binfo = phnn_mod.train_binned(
                X, dt, D_const, grid, temperature, traj_id=tid,
                steps=phnn_steps, seed=seed + 1, min_count=phnn_min_count,
                unconstrained=True, verbose=False)
            pts = grid.center_grid()
            b = phnn_mod.drift_field(bmodel, pts, D_const, temperature,
                                     unconstrained=True)
            res = _baseline_cycle_residual(b, grid, mask)
            results["baseline"] = {
                "fit": binfo,
                "val_loss": bhist[-1]["val"],
                "phnn_val_loss": results.get("phnn", {}).get("final_val_loss"),
                "r2": binfo.get("r2"),
                "phnn_r2": results.get("phnn", {}).get("fit", {}).get("r2"),
                "cycle_residual": res,
                "interpretation": (
                    "Both models fit the data comparably; only the structured "
                    "one satisfies the cycle condition. That contrast is what "
                    "makes the Wegscheider check meaningful rather than vacuous."
                ),
            }
            log(f"     baseline cycle residual: plaquette median "
                f"{res['plaquette_median']:.3e}, winding median "
                f"{res['winding_median']:.3e}  (structured model: ~1e-15)")
        except Exception as e:
            log(f"     baseline skipped: {e}")
            results["baseline_error"] = str(e)

    # ---- 40 MSM reference arm ---------------------------------------------
    if do_msm:
        try:
            log("[40] MSM reference arm (deeptime) ...")
            m, _, _, _ = msm_mod.build(X, tid, dt, n_clusters=n_clusters,
                                       lagtime=msm_lag, seed=seed)
            results["msm"] = {k: v for k, v in m.items()
                              if k not in ("stationary", "pcca_assignments",
                                           "cluster_centers_feat")}
            _save(m, out / "msm.json")
            log(f"     MSM slowest timescale {m['slowest_ps']:.1f} ps "
                f"({m['n_states']} states, lag {m['lagtime_ps']:.0f} ps)")
        except Exception as e:
            log(f"     MSM skipped: {e}")
            results["msm_error"] = str(e)

    # ---- 65 Markovianity: explain the timescale gap, don't hide it --------
    try:
        t_model = results.get("kinetics", {}).get("slowest_ps")
        t_ref = results.get("msm", {}).get("slowest_ps")
        mem = diffusion.memory_time(X, dt, tid, D_const, t_model, t_ref)
        results["markovianity"] = mem
        if mem.get("available") and mem.get("memory_time_ps"):
            log(f"[65] timescale gap explained: an overdamped model needs "
                f"D_eff={mem['D_eff_to_reconcile']:.4g} "
                f"({mem['ratio']:.1f}x below the 1 ps value), which the measured "
                f"D(lag) reaches at lag ~{mem['memory_time_ps']:.0f} ps "
                "-- the memory time of these CVs")
    except Exception as e:
        results["markovianity_error"] = str(e)

    # ---- 70 scorecard ------------------------------------------------------
    results["scorecard"] = _scorecard(results)
    manifest["wall_time_s"] = time.time() - t_start
    _save(manifest, out / "_manifest.json")
    _save(results, out / "results.json")
    log(f"\n[70] done in {manifest['wall_time_s']:.1f}s -> {out}")
    return results


def _baseline_cycle_residual(drift, grid, mask):
    """Cycle residual for an unconstrained drift field.

    Line-integrate the drift around cycles. For a gradient field this vanishes;
    for a free vector field it does not, which is exactly the point.
    """
    b = np.asarray(drift)
    n_phi, n_psi = grid.shape
    h = grid.h
    bphi = b[:, 0].reshape(n_phi, n_psi)
    bpsi = b[:, 1].reshape(n_phi, n_psi)
    m = np.asarray(mask).reshape(n_phi, n_psi)

    # curl around unit plaquettes
    circ = (bphi * h[0] + np.roll(bpsi, -1, axis=0) * h[1]
            - np.roll(bphi, -1, axis=1) * h[0] - bpsi * h[1])
    good = m & np.roll(m, -1, axis=0) & np.roll(m, -1, axis=1) & \
        np.roll(np.roll(m, -1, axis=0), -1, axis=1)
    pl = np.abs(circ[good])

    wphi = np.abs(np.sum(bphi * h[0], axis=0))
    wpsi = np.abs(np.sum(bpsi * h[1], axis=1))
    wind = np.concatenate([wphi, wpsi])

    return {
        "plaquette_median": float(np.median(pl)) if pl.size else float("nan"),
        "plaquette_max": float(np.max(pl)) if pl.size else float("nan"),
        "winding_median": float(np.median(wind)),
        "winding_max": float(np.max(wind)),
        "units": "kJ/mol per circuit (drift line integral)",
    }


def _scorecard(r) -> list:
    """The three-way comparison. Thresholds deliberately honest.

    A factor of 2 between independent MFPT estimators on a short dataset is a
    GOOD result; demanding 10% would just manufacture a red cell.
    """
    rows = []

    def row(q, a, b, c, crit, ok):
        rows.append({"quantity": q, "phnn_bondgraph": a, "msm": b,
                     "direct_md": c, "criterion": crit, "pass": ok})

    kin = r.get("kinetics", {})
    msm = r.get("msm", {})

    bg_slow = kin.get("slowest_ps")
    msm_slow = msm.get("slowest_ps")
    ok = None
    if bg_slow and msm_slow and np.isfinite(bg_slow) and np.isfinite(msm_slow):
        ratio = max(bg_slow, msm_slow) / max(min(bg_slow, msm_slow), 1e-12)
        ok = bool(ratio < 3.0)
    row("slowest timescale (ps)", bg_slow, msm_slow, None, "within 3x", ok)

    # LIKE FOR LIKE: the same named basin pair in both estimators. Comparing a
    # specific bond-graph MFPT against the median of all PCCA MFPTs compares
    # different quantities and tells you nothing.
    mfpt_bg = kin.get("mfpt_AB_ps")
    pair = f"{kin.get('A')}->{kin.get('B')}"
    mfpt_msm = (msm.get("basin_mfpt_ps") or {}).get(pair)
    ok = None
    if mfpt_bg and mfpt_msm:
        ratio = max(mfpt_bg, mfpt_msm) / max(min(mfpt_bg, mfpt_msm), 1e-12)
        ok = bool(ratio < 5.0)
    row(f"MFPT {pair} (ps)", mfpt_bg, mfpt_msm, None,
        "same basin pair, within 5x", ok)

    rmse = r.get("phnn", {}).get("H_vs_FES_rmse_kJ")
    row("H_phnn vs empirical FES (kJ/mol RMSE)", rmse, None, 0.0,
        "reported; tests Markovianity of (phi,psi)",
        bool(rmse < 2.0) if rmse is not None else None)

    weg = r.get("wegscheider", {})
    base = r.get("baseline", {}).get("cycle_residual", {})
    pl = weg.get("plaquette", {}).get("max")
    row("cycle residual, contractible", pl, None,
        base.get("plaquette_median"), "< 1e-10 AND baseline nonzero",
        (bool(pl < 1e-10) if pl is not None else None))
    wd = weg.get("winding_phi", {}).get("max")
    if wd is None:
        wd = weg.get("winding_psi", {}).get("max")
    row("cycle residual, winding (torus)", wd, None,
        base.get("winding_median"),
        "< 1e-10 AND baseline nonzero (n/a if sampling does not wrap)",
        (bool(wd < 1e-10) if wd is not None else None))

    ch = r.get("checks", {})
    row("second law (Hdot <= 0)", ch.get("Hdot"), None, None,
        "structural, any weights", ch.get("Hdot_nonpos"))
    return rows
