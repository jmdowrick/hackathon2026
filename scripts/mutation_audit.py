"""Mutation audit: are the tests actually capable of failing?

A test suite that passes is evidence of nothing unless the tests would FAIL on
broken code. This script breaks the code on purpose, one targeted mutation at
a time, and records whether the suite notices.

    SURVIVED  = the mutation was NOT caught  -> that coverage is vacuous
    KILLED    = the tests caught it          -> the coverage is real

Each mutation re-introduces a bug that was actually hit during development, or
removes a guarantee the project claims. If a mutation survives, either the
claim is untested or the test is decorative, and the entry says which test was
supposed to catch it so the gap is actionable rather than a number.

Run:  .venv/bin/python scripts/mutation_audit.py          (fast tests)
      .venv/bin/python scripts/mutation_audit.py --all    (includes slow)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ala2bg"
PY = str(ROOT / ".venv" / "bin" / "python")

# (name, file, find, replace, claim it breaks, test expected to catch it)
MUTATIONS = [
    ("torus wrapping removed", "cvs.py",
     "    return wrap(np.asarray(x1) - np.asarray(x0))",
     "    return np.asarray(x1) - np.asarray(x0)",
     "displacements use shortest path on the torus",
     "test_wrapped_delta_crossing_boundary"),

    ("detailed balance broken", "thermo/smoluchowski.py",
     "        k_ab = pref * np.exp(-0.5 * beta * (F[b] - F[a]))",
     "        k_ab = pref * np.exp(-0.35 * beta * (F[b] - F[a]))",
     "Bicout-Szabo gives exact detailed balance",
     "test_detailed_balance_structural / test_kappa_symmetric"),

    ("connected component not enforced", "thermo/smoluchowski.py",
     "    n_comp, comp = sp.csgraph.connected_components(L, directed=False)",
     "    n_comp, comp = 1, np.zeros(n, dtype=int)",
     "generator is restricted to one connected component",
     "test_generator_restricted_to_largest_connected_component"),

    ("barrier ignored in rates", "barrier.py",
     "    return (D / h ** 2) * np.exp(-beta * (np.asarray(F_saddle)\n"
     "                                          - np.asarray(F_start)))",
     "    return (D / h ** 2) * np.exp(-beta * 0.0 * (np.asarray(F_saddle)\n"
     "                                                - np.asarray(F_start)))",
     "site-to-site rates depend on the barrier",
     "test_rates_depend_on_the_barrier_not_just_the_endpoints"),

    ("Faraday where elementary charge belongs", "electro.py",
     'CHARGE_UNIT = {"single_channel": ELEMENTARY_CHARGE, "molar": FARADAY}',
     'CHARGE_UNIT = {"single_channel": FARADAY, "molar": FARADAY}',
     "single-channel current uses the elementary charge",
     "test_single_channel_and_molar_modes_differ_by_avogadro",),

    ("half-period shift reintroduced", "periodic.py",
     "    return origin + ((theta + np.pi) % TWO_PI) / TWO_PI * period",
     "    return origin + (theta % TWO_PI) / TWO_PI * period",
     "from_angle is the exact inverse of to_angle",
     "test_arbitrary_period_round_trip"),

    ("charge closure always passes", "electro.py",
     '        return {"sum_z": s, "expected": float(z_total),\n'
     '                "closes": bool(abs(s - z_total) < tol),',
     '        return {"sum_z": s, "expected": float(z_total),\n'
     '                "closes": True,',
     "charge closure is enforced, not assumed",
     "test_charge_closure_is_enforced_not_assumed"),

    ("chemostat affinity dropped", "electro.py",
     "        if self.transits:\n            A = A + self.transit_affinity()",
     "        if self.transits:\n            A = A + 0.0 * self.transit_affinity()",
     "concentration gradients drive the model",
     "test_reversal_potential_reproduces_nernst"),

    ("early stopping disabled", "phnn.py",
     '    if best["state"] is not None:\n        model.load_state_dict(best["state"])',
     '    if False:\n        model.load_state_dict(best["state"])',
     "the returned model is the best-validation one",
     "test_early_stopping_beats_running_to_the_end"),

    ("ion exit/entry pairs not excluded", "kcsa.py",
     "    inside = (z0 >= z_min) & (z0 <= z_max) & (z1 >= z_min) & (z1 <= z_max)",
     "    inside = np.ones_like(z0, dtype=bool)",
     "pairs spanning a filter exit are excluded from D",
     "test_ions_leaving_the_filter_are_excluded"),

    ("CellML validation skipped", "thermo/cellml_writer.py",
     "    if validate:\n        issues = validate_cellml(path)",
     "    if False:\n        issues = validate_cellml(path)",
     "emitted CellML is validated before being written",
     "test_writer_refuses_to_emit_invalid_cellml"),

    ("barrier parameter decoupled", "kcsa.py",
     "    F = 0.5 * max_barrier_kJ * (1.0 - np.cos(k * (z - centres[0])))",
     "    F = 0.5 * 10.5 * (1.0 - np.cos(k * (z - centres[0])))",
     "max_barrier_kJ equals the realised barrier",
     "test_barrier_parameter_equals_the_realised_barrier"),

    ("spectral gap always confident", "minimal.py",
     '        "confident": bool(ratio > 2.0 and ratio > 1.5 * second),',
     '        "confident": True,',
     "the gap diagnostic can report absence of structure",
     "test_spectral_gap_reports_absence_of_a_gap"),

    ("PHNN kappa symmetry broken", "phgnn.py",
     "        return torch.cat([hi + hj, torch.abs(hi - hj)], dim=-1)",
     "        return torch.cat([hi + 1.3 * hj, hi - hj], dim=-1)",
     "kappa_ij == kappa_ji by construction",
     "test_phgnn_kappa_is_symmetric_by_construction"),

    ("delta of the mean ion position", "delta.py",
     "    return d.sum(axis=1), inside.sum(axis=1)",
     "    zbar = np.nanmean(np.where(inside, Z, np.nan), axis=1)\n"
     "    return np.nan_to_num(delta_fn(zbar)), inside.sum(axis=1)",
     "a state's charge is the sum of delta over its ions, not delta(<z>)",
     "test_two_ion_state_carries_the_sum_not_the_mean_position"),

    ("bath species dropped from reaction charge", "delta.py",
     "        z[t.reaction] += (0.0 if ds is None else float(ds)) \\",
     "        z[t.reaction] += 0.0 * (0.0 if ds is None else float(ds)) \\",
     "charge closure follows from bath stoichiometry",
     "test_closure_follows_from_stoichiometry_for_any_state_charges"),

    ("WE weights ignored", "delta.py",
     "    w = np.ones(len(Qf)) if weights is None else np.asarray(weights, float)",
     "    w = np.ones(len(Qf))",
     "weighted-ensemble weights enter the state charge",
     "test_we_weights_enter_as_weights"),

    ("radial gate removed from features", "features.py",
     "         * _sigmoid((ch.r_max - np.where(np.isfinite(rr), rr, 1e9)) / w))",
     "         * 1.0)",
     "bulk ions at the right z are not channel ions (bug 2)",
     "test_radial_gate_excludes_bulk_ion_at_channel_z"),

    ("minimum image skipped", "features.py",
     "        d = d - L * np.round(d / L)",
     "        d = d",
     "ions across a periodic boundary are still at the pore",
     "test_minimum_image_keeps_an_ion_in_the_pore"),

    ("GA wall drops HETATM", "gramicidin.py",
     "    wall = pdb.non_solvent(s)",
     "    wall = ~s.is_hetatm",
     "GA's D-residues line the pore",
     "test_ga_wall_includes_d_residues"),

    ("GA origin off the helix axis", "gramicidin.py",
     "    origin = c + np.dot(junction - c, axis) * axis",
     "    origin = junction",
     "radii are measured from the helix axis",
     "test_ga_lumen_is_not_clipped_by_an_off_axis_origin"),

    ("formyl charge not neutralised", "gramicidin.py",
     '   <Atom name="HN"   type="protein-H5" charge="-0.0293"/>',
     '   <Atom name="HN"   type="protein-H5" charge="0.0293"/>',
     "every GA residue is neutral",
     "test_every_residue_is_neutral"),

    ("GA not rotated into the pore frame", "gramicidin.py",
     "    X = (X - g[\"origin\"]) @ R.T",
     "    X = (X - g[\"origin\"])",
     "the MD frame is the analysis frame (bug 3's class)",
     "test_ga_topology_is_complete_and_in_the_pore_frame"),

    ("runner origin off the helix axis", "../../scripts/hpc/run_ga.py",
     "    origin = c + np.dot(junction - c, axis) * axis",
     "    origin = junction",
     "the runner records in the same frame the package analyses in",
     "test_runner_frame_matches_the_package_frame"),

    ("runner lock ignores a live holder", "../../scripts/hpc/run_ga.py",
     "                os.kill(pid, 0)",
     "                raise ProcessLookupError",
     "two runs cannot write one checkpoint",
     "test_lock_refuses_a_live_holder_and_takes_a_stale_one"),

    ("resume keeps frames newer than the state", "../../scripts/hpc/run_ga.py",
     '        keep = old["step"] <= cur',
     '        keep = old["step"] >= 0',
     "a crash between writes cannot duplicate frames (SLOW tier only)",
     "test_preflight_runs_the_shipped_bundle_end_to_end"),
]


def run_tests(slow: bool, timeout=1800):
    cmd = [PY, "-m", "pytest", "tests/", "-q", "-x", "--no-header"]
    if not slow:
        cmd += ["-m", "not slow"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode == 0, r.stdout[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include slow tests")
    ap.add_argument("--only", help="substring filter on mutation name")
    args = ap.parse_args()

    muts = [m for m in MUTATIONS if not args.only or args.only in m[0]]

    print("Baseline (unmutated) ...", flush=True)
    ok, tail = run_tests(args.all)
    if not ok:
        print("BASELINE FAILS -- fix the suite before auditing it.\n", tail)
        return 1
    print("  baseline passes\n")

    results = []
    with tempfile.TemporaryDirectory() as backup:
        for name, rel, find, repl, claim, test in muts:
            path = SRC / rel
            bak = Path(backup) / rel.replace("/", "_")
            shutil.copy2(path, bak)
            text = path.read_text()
            if find not in text:
                results.append((name, "SKIP", claim, test,
                                "anchor not found -- source moved"))
                print(f"  SKIP     {name}  (anchor not found)", flush=True)
                continue
            try:
                path.write_text(text.replace(find, repl, 1))
                passed, tail = run_tests(args.all)
                verdict = "SURVIVED" if passed else "killed"
                results.append((name, verdict, claim, test, ""))
                mark = "!! SURVIVED" if passed else "   killed  "
                print(f"  {mark} {name}", flush=True)
            except subprocess.TimeoutExpired:
                results.append((name, "TIMEOUT", claim, test, ""))
                print(f"  TIMEOUT  {name}", flush=True)
            finally:
                shutil.copy2(bak, path)

    survived = [r for r in results if r[1] == "SURVIVED"]
    skipped = [r for r in results if r[1] in ("SKIP", "TIMEOUT")]
    killed = [r for r in results if r[1] == "killed"]

    print("\n" + "=" * 72)
    print(f"killed {len(killed)}/{len(results)}   survived {len(survived)}   "
          f"skipped {len(skipped)}")
    if survived:
        print("\nVACUOUS COVERAGE -- these breakages were NOT caught:")
        for name, _, claim, test, _ in survived:
            print(f"  * {name}")
            print(f"      claim: {claim}")
            print(f"      test that should have caught it: {test}")
    if skipped:
        print("\nNot exercised:")
        for name, v, _, _, why in skipped:
            print(f"  * {name}  [{v}] {why}")
    if not survived and not skipped:
        print("\nEvery targeted breakage was caught.")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
