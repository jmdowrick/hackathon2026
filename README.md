# ala2bg — trajectories → learned free energy → port-Hamiltonian bond graph

Learn a free-energy landscape and a dissipation field from molecular dynamics
with the second law built into the architecture, discretise into a bond graph
whose detailed balance is exact by construction, and check the result three
independent ways.

Output: a runnable **CellML 2.0** model derived from simulation rather than
fitted to phenomenological rate equations.

---

## TL;DR — run it

```bash
cd /Users/jdow403/Documents/_projects/hackathon2026
source .venv/bin/activate          # env already built; see "Environment" if missing

# 1. unit + regression tests (fast, no GPU, no network) — ALWAYS run after editing
python -m pytest tests/ -q -m "not slow"      # ~3 s
python -m pytest tests/ -q                    # ~10 s, includes training tests

# 2. toy system — exact ground truth, ~20 s, proves the instrument is correct
python -c "from ala2bg import pipeline; pipeline.run(out_dir='runs/toy', provider='toy',
    provider_kw=dict(n_steps=60_000, n_traj=8, dt=0.5, D=0.01, barrier=12.0))"

# 3. real alanine dipeptide — 750k frames, ~45 s (downloads 6 MB once)
python -c "from ala2bg import pipeline; pipeline.run(out_dir='runs/ala2',
    provider='mdshare', provider_kw=dict(cache_dir='data', dt=1.0))"
```

Results land in `runs/<name>/`: `results.json` (everything), `_manifest.json`
(provenance), `model_coarse.cellml`, `bondgraph.json`, `checks.json`,
`kinetics.json`, `msm.json`, plus `.npz` arrays.

---

## What it does, stage by stage

```
data ──► measure D ──► choose grid ──► FES ──► PHNN (learn H) ──► bond graph ──► scorecard
                │                       │            │                  │
         2nd KM moment          -RT ln P      1st KM moment      Bicout-Szabo
         MEASURED, FROZEN    (thermodynamic)    (kinetic)        Ce/Re/0/1 + CellML
                                                                        │
                                              MSM (deeptime) ───────────┘ independent check
```

**Why `D` is measured and only `H` is learned.** Learning both from
displacements fails: at leading order only the *product* `D·∇H` enters the
drift, so a network trades one against the other, fits perfectly, and means
nothing. The second Kramers–Moyal moment gives `D` **without reference to
`H`** — so we measure `D` first, freeze it, and `H` is then exactly
identified. The degeneracy is impossible, not merely avoided.

**The payoff.** With `D` fixed, `H` is over-determined — obtained kinetically
(from the drift) *and* thermodynamically (from the histogram). Nothing forces
them to agree, so their residual measures **whether (φ,ψ) is actually a good
Markovian coordinate**.

**Why consistency is structural.** The drift is `−D∇H` with `D` PSD and `H` a
scalar potential, so `Ḣ = −∇Hᵀ D ∇H ≤ 0` identically, for *any* weights, at
*any* point in training. Never a penalty term.

---

## Editing it without help

### Repo map — what to open for what

| You want to change… | File |
|---|---|
| φ/ψ definition, torus wrapping, embedding | `src/ala2bg/cvs.py` |
| grid, neighbours, bin indexing | `src/ala2bg/grid.py` |
| how the grid resolution is chosen | `src/ala2bg/gridselect.py` |
| where trajectories come from | `src/ala2bg/data.py` |
| histogram FES, basins, transition counts | `src/ala2bg/fes.py` |
| the `D` measurement | `src/ala2bg/diffusion.py` |
| the neural network, loss, training | `src/ala2bg/phnn.py` |
| **>2 CVs / scalability** | `src/ala2bg/nd.py` |
| MSM reference arm | `src/ala2bg/msm.py` |
| F,D → generator (Bicout–Szabo) | `src/ala2bg/thermo/smoluchowski.py` |
| committor, MFPT | `src/ala2bg/thermo/committor.py` |
| Ce/Re components, cycle tests | `src/ala2bg/thermo/bondgraph.py` |
| CellML output | `src/ala2bg/thermo/cellml_writer.py` |
| stage order, scorecard | `src/ala2bg/pipeline.py` |

### House rules

1. **Run the tests after every change.** `pytest tests/ -q -m "not slow"` is 3 s.
   There is no excuse not to.
2. **When you find a bug, add a regression test for it** in
   `tests/test_regressions.py`. Every test there corresponds to a real failure
   that reached a result before being caught.
3. **Tests must not be vacuous.** A test that passes on the broken code is
   worse than no test. Where it isn't obvious, reconstruct the broken behaviour
   and assert the two differ — see `test_diffusion_estimate_survives_boundary_crossings`,
   which asserts the naive estimator *is* wrong by 10×.
4. **Never use a bare `-` when differencing angles.** Use
   `cvs.wrapped_delta`. This is the single most destructive silent bug
   available in this codebase.
5. **Keep scalability in mind** — see below.

### Adding a data source

Add a function to `src/ala2bg/data.py` returning the contract, then register it
in `PROVIDERS`:

```python
{"X": (n, d) float64 radians, "dt": float ps, "traj_id": (n,) int, "meta": {...}}
```

Nothing downstream knows where trajectories came from. **This is the scaling
story**: swapping brute force for RETIS, metadynamics or a bigger system
changes this file and nothing else.

### Running your own MD (needs OpenMM, not in the venv)

```bash
mamba create -n ala2md -c conda-forge python=3.11 openmm openmmtools mdtraj -y
mamba activate ala2md && pip install -e .
python -c "from ala2bg import pipeline; pipeline.run(out_dir='runs/md', provider='openmm',
    provider_kw=dict(n_replicas=8, ns=5.0, variant='implicit'))"
```

Two things that will bite you, both verified:

- **`openmmtools` ships AMBER ff96**, not ff14SB/ff99SB-ILDN — so *no* modern
  force-field literature value validates it. Pass `forcefield="amber14"` to
  rebuild with `amber14-all.xml`, or accept ff96 and compare only qualitatively.
  Either way the choice is recorded in the manifest.
- **Never call `step(1)` in a Python loop.** Per-step GPU round-trips dominate.
  `run_openmm_replica` batches `step(10_000)`.

Also: Ala2 is ~22 atoms (vacuum/implicit) or 2,269 (explicit) and badly
underuses a modern GPU. Run **many concurrent replicas** (NVIDIA MPS helps a
lot at this size), not one long trajectory. That also gives independent blocks
for error bars, which one trajectory does not.

---

## Scalability — read before extending

The grid route costs `n^d` bins. At 2 CVs, 60×60 = 3,600 is fine. At 4 CVs,
60⁴ = 13 million is not — and it fails *worse* than it sounds, because once
most bins are empty the generator **disconnects and silently deletes the slow
process**. On real data a 60×60 grid severed the αL basin (21,000 frames) and
reported ~120 ps instead of nanoseconds. Nothing errored.

Three consequences shape the design:

1. **`gridselect.choose_grid` picks the finest grid that stays connected**, and
   rejects near-disconnected grids whose spectrum is numerically degenerate
   (~1e13 ps timescales). It reports every rejected candidate.
2. **`nd.py` is the scalable path.** Hash frames to integer voxel keys and keep
   only *occupied* voxels: cost is `O(n_frames)`, independent of dimension. The
   learned `H` is a continuous function, so it is defined everywhere without
   any bin existing. `test_phnn_recovers_known_potential_in_four_dimensions`
   asserts 4D accuracy (corr > 0.9 against a known potential) with < 10,000
   parameters.
3. **The bond graph stays small by construction** — it is built from *basins*
   (a handful of states), not from `n^d` bins. The expensive part scales; the
   discrete part doesn't.

**If you add CVs, use `nd.train_nd`, not `phnn.train_binned`.** The latter
needs a dense 2D grid.

---

## Interpreting the output

`results.json → scorecard` compares three **disjoint** estimators that share
only the raw trajectories:

| Quantity | Criterion |
|---|---|
| slowest timescale | bond graph vs MSM within 3× |
| MFPT, **same basin pair** | within 5× |
| `H_phnn` vs empirical FES | RMSE reported — this *is* the Markovianity test |
| cycle residual, contractible | < 1e-10 **and** baseline nonzero |
| cycle residual, winding | same, or `n/a` if sampling doesn't wrap |
| second law | `Ḣ ≤ 0`, structural |

**Thresholds are deliberately loose.** Independent estimators agreeing within
2× on a short dataset is a *good* result; demanding 10% just manufactures red
cells. Set thresholds before looking and report what comes out — a
disagreement you understand is worth more than a suspiciously perfect table.

### The cycle test needs its baseline

On a torus the grid wraps, admitting non-contractible cycles. Zero affinity
around them is required at equilibrium — but a potential-based model satisfies
this **automatically**, so checked alone the test is near-vacuous. It gets
teeth from the unconstrained neural-SDE baseline (`phnn.UnconstrainedDrift`),
which fits the data about as well and *does* violate it:

| Model | contractible | winding |
|---|---|---|
| PHNN (structured) | ~1e-15 | ~1e-15 |
| unconstrained | ~1e-4 | ~1e-2 |

Both fit; only one is physically admissible. **Don't present the cycle residual
without the baseline column.**

---

## Known limitations — state these, don't hide them

- **`D` has no plateau in lag on real Ala2** (0.112 → 0.002 rad²/ps from 1 ps
  to 500 ps). That is confined, not free, diffusion: within a basin ⟨Δx²⟩
  saturates so apparent `D` falls as 1/τ. The short-time value is used and the
  lag dependence is reported as `lag_stability`. It is a genuine statement
  that (φ,ψ) is imperfectly Markovian, not a bug to tune away.
- **Bond-graph and MSM slowest timescales differ ~17× on real data — and that
  gap is now a measurement, not a failure.** `results.json → markovianity`
  reports the effective diffusivity that would reconcile them
  (`D_eff = D · t_model / t_ref`) and the lag at which the measured `D(lag)`
  curve passes through it. On the alanine dipeptide dataset that is
  **~125 ps — the memory time of (φ,ψ)**. Say it that way: *not* "the model is
  wrong by 17×", but "an overdamped model in these coordinates needs a
  diffusivity measured on a ~125 ps lag".
  The diagnostic is checked both ways: on the toy, whose dynamics genuinely is
  overdamped Langevin, it returns ratio ≈ 1 (`test_memory_diagnostic_is_quiet_
  on_truly_markovian_dynamics`). A diagnostic that fired on everything would
  measure nothing.
- **C7ax is visited in ~0.2% of frames** of the explicit-water dataset, so any
  rate involving it is poorly determined.
- **Literature reference values are force-field dependent.** Quote the force
  field with every number; don't compare across force fields. The widely-cited
  "283 ps" for this dataset comes from a page whose printed units are
  internally inconsistent — our own MSM gives a clean ITS plateau at ~1.4 ns.
- `model_coarse.cellml` is validated by the master-equation round-trip test,
  **not** by OpenCOR. OpenCOR is presentation polish; don't let it block.

---

## Environment

Already built at `.venv/`. To rebuild:

```bash
uv venv --python 3.11 .venv
uv pip install numpy scipy torch deeptime matplotlib pytest
uv pip install -e .
```

OpenMM and mdtraj are **not** in this venv (conda-only in practice) and are
needed only for `provider="openmm"`. `mdshare` is deliberately not used — it
imports `pkg_resources`, which modern setuptools no longer ships; `data.py`
fetches the same file directly over HTTP.

---

## What is novel here, honestly

Learning free energies from MD is not new. MSMs on alanine dipeptide are a
textbook exercise. Port-Hamiltonian and GENERIC-style structure-preserving
networks exist.

**The novel join is the last one**: a learned continuous structure-preserving
model discretised into a bond-graph / CellML component, with thermodynamic
consistency carried end-to-end as an architectural invariant. Claim that join
and nothing more.
