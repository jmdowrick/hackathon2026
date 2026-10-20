# Handover

For another Claude instance picking this up. Read `README.md` for the
project itself; this file is what you need that the README does not say.

---

## 1. What this is

A pipeline that turns molecular simulation into a runnable physiological
model. The core is one mapping:

```
free-energy landscape F(x)  +  diffusivity D(x)
        -> Bicout-Szabo / Kramers rates
        -> Ce/Re port-Hamiltonian bond graph    K_i = e^{beta F_i}/V,  kappa_ij = k_ij/K_i
        -> CellML 2.0 (validated with libcellml)
```

Thermodynamic consistency is carried end to end as an architectural
invariant. Two tracks share the estimators:

- **alanine dipeptide** — validation, where ground truth exists. Solid.
- **KcsA** — application. Heavily caveated; see §4.

`src/ala2bg/` (the package name is historical).

| file | role |
|---|---|
| `cvs.py` | torus wrapping, sin/cos embedding, CVSpec hash. ONE implementation |
| `periodic.py` | arbitrary-period coordinates (ion z in a box) |
| `grid.py` / `gridselect.py` | the 2D grid; grid chosen, not guessed |
| `fes.py` | histogram FES, basins, transition counts |
| `diffusion.py` | D from the 2nd Kramers-Moyal moment; memory-time diagnostic |
| `phnn.py` | the PHNN. Learns H only; D is measured and frozen |
| `nd.py` | sparse occupied-voxel path; the part that scales past 2 CVs |
| `minimal.py` | MSM metastable states from the spectral gap |
| `barrier.py` | minimax saddle (union-find), Kramers rates |
| `surrogate.py` | simulate the learned SDE to generate unlimited statistics |
| `phgnn.py` | port-Hamiltonian GNN over the state graph |
| `electro.py` | electrogenic bond graph, Hunter et al. formulation |
| `delta.py` | state charge <sum delta(z_a)> and per-reaction z from stoichiometry |
| `features.py` | permutation-invariant ion features (soft occupancy, count, Q); the ONE radial gate |
| `gramicidin.py` | GA from 1JNO: pore frame, radius profile, channel window, ff14SB cap templates, torsion audit |
| `kcsa.py` | KcsA specifics: sites, landscape, D from filter MD |
| `pdb.py` | structure import, BIOMT assembly, pore frame, radius profile |
| `thermo/` | generator, committor/MFPT, bond graph, CellML writer |

---

## 2. Working practices the user requires

These were stated explicitly and enforced repeatedly. Follow them.

1. **Every test must be able to fail.** A test that passes on broken code is
   worse than none. Where it is not obvious, reconstruct the broken behaviour
   and assert the two differ. See `test_diffusion_estimate_survives_boundary_
   crossings`, which asserts the naive estimator *is* wrong by 10x.
2. **Find a bug, write a regression test.** `tests/test_regressions.py` has one
   per bug actually hit.
3. **`scripts/mutation_audit.py`** breaks the code on purpose and checks the
   suite notices. Run it after substantial changes. Last run: 12/14 killed on
   the fast tier.
4. **Do not overclaim provenance.** `kcsa.provenance_report` exists because an
   artifact once reported `fully_derived: True` while using a literal D and an
   analytic landscape. Check what actually produced a number.
5. **Verify, do not assume.** Nearly every bug below passed every other check.

---

## 3. The bug catalogue

Read this before changing anything. Six of these produced **plausible wrong
numbers with no error**.

| # | bug | symptom | silent? |
|---|---|---|---|
| 1 | naive angle difference | ~2pi jumps at the seam | yes |
| 2 | **no radial gate** (3 separate functions) | bulk ions counted as filter ions; D 29x high | yes |
| 3 | **frame mismatch, 5.89 A** | sites read off the landscape at the wrong site | yes |
| 4 | Faraday where elementary charge belongs | 1e21 pS | partly |
| 5 | sampling stride computed, never used | frames 20 ps apart; D meaningless | yes |
| 6 | non-periodic position restraints | NaN at step 1 | no |
| 7 | crystal frame vs box frame | 46% of atoms outside the box, NaN | no |
| 8 | CellML: `math:` prefix, protected units, kappa units, J vs kJ | unloadable, or 1000x | mixed |
| 9 | Bicout-Szabo between well-separated sites | the barrier was ignored entirely; every kappa identical | yes |
| 10 | guard `max(snap) > 2*gap` | unreachable; dead code | yes |
| 11 | asymmetric driving-force split | prefactor scaled as 1/c; conductance fell with [K+] | yes |
| 12 | empty state given an electrical position | bath steps carried +-1 charge | yes |

**Pattern worth internalising:** bug 2 appeared three times in three different
functions. The same conceptual error propagates wherever the same assumption
is made, and fixing one instance does not fix the others. When you find a
class of bug, grep for every place it could apply.

---

## 4. What is actually established

**Solid, defensible:**

- D recovered to **0.5%** on a toy with known D; learned H correlates
  **0.984** with a potential the network never saw (ala2 track).
- **ln kappa ~ ln D - beta F_dagger** — derived, then measured
  (saddle R2 0.871 vs 0.793 for endpoint features). This is why cheap-feature
  rate surrogates fail.
- Structure-derived geometry from 1K4C: site positions from resolved K+ ions
  (S0 sits 1.3 A further out than uniform spacing implies), pore radius
  profile, electrical distances from the resistive model.
- Electrogenic bond graph following Hunter et al.: reversal potential matches
  Nernst to **~1e-13 mV** across a 100-fold concentration range; physiological
  K+ gives -91.30 mV.
- CellML validates with libcellml, carries the electrical port.
- `nd.py` recovers a known 4D potential at corr > 0.9 with < 10k parameters.

**Negative results — the most reliable output, and genuine findings:**

- A confined-ion trajectory **cannot constrain barriers**. Measured: filter
  ions occupy 6 of 40 bins (15% of the window) and never cross between sites.
  F at the barriers is pure extrapolation, and `ln kappa` depends on exactly
  that.
- **kappa does not transfer** from endpoint free energies. Barrier-aware
  features rescue it (RMSE 1.804 -> 1.169), but only with a nonlinear model.
- **D is strongly lag-dependent** (factor 27 over 0.4-10 ps on KcsA; similar
  on ala2). That is confinement, not diffusion. A single D does not describe
  the coordinate; `D_lag_stability` reports it.
- **A single-ion CV is the wrong coordinate for KcsA.** Pooling ions into one
  1D series makes an MSM report ~3 metastable states when the true count is 1.
  An MSM cannot detect this: ITS/CK test self-consistency *within* the
  projection.

**NOT established — do not claim:**

- The KcsA landscape is **not derived**. It is an analytic surface constrained
  to a published 2-3 kcal/mol barrier. `provenance_report` says so.
- Conductance is ~25x below experiment. The model is **single-ion**; KcsA
  conducts by multi-ion knock-on, so it is structurally the wrong model.
  Fixed-charge force fields independently run 2-10x low.
- No production MD was ever run. The only trajectory is 1 ns, restrained
  protein, no membrane.
- **Wegscheider/detailed balance/second law are automatic** in any bond graph
  built from Ce potentials — they are correctness tests of the implementation,
  not findings. The user corrected an earlier overclaim on this. What is
  actually ours is the *mapping*, and the charge-closure construction.
- Novelty of "bond graph from a free-energy landscape" is **unverified**. One
  search found nothing; that is weaker than establishing it.

---

## 5. Open problems, in priority order

1. *(Removed 2026-10-20: the identifiability and novelty gates were
   descoped by the user.)*
2. **The delta mapping — built (`delta.py`).** Not "mean ion z per state":
   that is wrong for multi-ion states (sum, not mean), for the nonlinear
   geometric delta (Jensen), and for closure (a state function telescopes to
   0). Instead Q_i = <sum_a delta(z_a)>_i over ions in the field window, the
   linear-response gating charge; z_r = Q_b - Q_a + bath-species deltas.
   Permutation-invariant by construction, WE weights enter as plain weights,
   closure is structural. `permeation_graph` now uses it (bit-identical
   output). Still open: (a) `count_purity` must be ~1 per state, i.e. the
   states must resolve the window ion count -- a test of the CV, not of this
   code; (b) `permeation_graph` still renormalises delta over the SITES, so
   the bath steps carry zero charge by choice. Unrenormalised geometric delta
   gives the cavity-entry step real charge and Nernst still holds
   (`test_bath_steps_can_carry_charge_and_nernst_still_holds`) -- a modelling
   decision for the user, not made here.
3. **Multi-ion CV space — built (`features.py`).** Soft per-basis-centre
   occupancies (a partition of unity, so they sum to the channel ion count),
   the count itself, and Q. All sums over ions: relabelling is exactly
   nothing. Sorted positions were rejected -- they teleport when an ion
   enters. Basis centres, not sites: the sites are for the SRV to find.
   `in_channel` is the single radial gate; `delta.frame_charges` uses it too
   (it had none -- bug 2's class again).
4. **Target: gramicidin A — decided.** Single-file, predominantly
   single-ion, so the one-ion bond graph is the correct structure.
   `gramicidin.load_gramicidin()` reads 1JNO (SDS micelle NMR; no resolved
   ions, so sites are NOT taken from it). Two traps found on the way in:
   half of GA is HETATM (D-residues, caps), so `pdb.radius_profile` needs
   `wall_mask=non_solvent(s)`; and the formyl-valine midpoint is off the
   helix axis, so the origin is projected onto it. The dimer's two-fold
   symmetry (profile mirror-symmetric to 0.001 A) checks the frame.

---

## 6. Recommended direction, and why it changed

The session converged, then revised:

- **GREST** (Shmilovich & Ferguson 2023, Girsanov Reweighting Enhanced
  Sampling Technique) discovers slow CVs *and* samples along them in one loop,
  which solves the chicken-and-egg of needing CVs to bias and sampling to
  learn CVs.
- **But Girsanov path-weight variance grows exponentially** with bias strength
  and segment length. Measured here: at a bias able to cross a 10.5 kJ/mol
  barrier over 500 ps, effective sample size is **0.03%**. Mitigated by
  reweighting only over the MSM lag, but that bound bites exactly where the
  slow process lives.
- **So weighted ensemble is the more scalable route.** WE never biases the
  dynamics; weights are maintained combinatorially, with no exponential
  variance. It needs a progress coordinate, which **SRV** can supply from
  unbiased data — no Girsanov needed because nothing was biased.

**Current recommendation:** permutation-invariant features -> SRV progress
coordinate -> weighted ensemble (WESTPA) for barrier coverage -> PHNN for H
and D -> bond graph. Do **not** combine WE and Girsanov; they solve the same
problem by opposite means and stacking them corrects twice.

**Agreed (2026-10-20):** pivot to gramicidin A and build that stack. Done:
features, GA frame, delta, and the membrane system (`scripts/ga_md.py
--prepare`): 1JNO in DMPC, 25,388 atoms, 0.15 M KCl, Amber ff14SB/lipid17,
11 post-build checks incl. the frame RE-MEASURED after addMembrane. Force
field: D-residues on L templates are exact (ff14SB torsion phases are all
0/pi, no CMAP; mirror energy equal to 5e-13 kJ/mol). Cap CHARGES are by
analogy (ACE; NME + Ser), not RESP -- a declared choice at the two places
that matter most (junction, mouths). OpenMM silently drops unmatched
torsions; `unexplained_missing_torsions` separates those from ff14SB's
deliberate k=0 terms. ~0.6 s/ps locally.

Next: staged equilibration (NPT, MonteCarloMembraneBarostat, GA restraints
released in steps) and an HPC runner like run_kcsa.py; then unbiased runs
-> SRV (deeptime) -> WESTPA (not installed).

---

## 7. Running it

```bash
source .venv/bin/activate
make test                 # 109 fast tests, ~5 s. Run after every change.
python -m pytest tests/ -q        # includes slow training tests, ~3 min
python scripts/mutation_audit.py  # are the tests able to fail?
make toy                  # exact ground truth, ~20 s
make ala2                 # real alanine dipeptide, ~45 s
```

KcsA MD (built, pre-flighted, never run in production):

```bash
python scripts/kcsa_md.py --prepare          # 1K4C -> 61k-atom solvated system
python scripts/hpc/preflight.py --dir runs/kcsa_md/bundle --ns 0.1   # 32 checks
# runs/kcsa_md/kcsa_bundle.tar.gz is self-contained: openmm + numpy only
```

Scope note on that MD: protein restrained, no membrane. Valid for the local
diffusivity of K+ in the filter and **nothing else** — not gating, not
absolute conductance.
