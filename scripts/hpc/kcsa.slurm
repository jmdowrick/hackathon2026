#!/bin/bash
#SBATCH --job-name=kcsa-filter
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=kcsa-%j.out

# KcsA selectivity-filter K+ diffusivity.
#
# Run preflight.py FIRST, on this cluster:   python preflight.py --dir . --ns 0.1
#
# Two things to EDIT below: the environment activation, and nothing else.

set -uo pipefail          # NOT -e: a failed `module load` must not kill the job

# ---- EDIT: provide a Python with OpenMM built against CUDA -----------------
# Whichever of these your cluster uses. `conda activate` needs the shell hook
# sourced -- a bare `conda activate` in a batch script usually fails, which is
# why the hook is sourced explicitly rather than assumed.
module load Miniconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate openmm || echo "WARNING: could not activate 'openmm'"
fi
# If you built it yourself:
#   conda create -y -n openmm -c conda-forge python=3.11 openmm numpy
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY' || { echo "FATAL: OpenMM with CUDA not available"; exit 1; }
import sys, openmm
from openmm import Platform
names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print("openmm", openmm.__version__, "platforms", names, flush=True)
sys.exit(0 if "CUDA" in names else 1)
PY

# --max-hours stops cleanly and flushes rather than being killed at the wall
# clock; keep it comfortably under #SBATCH --time above.
# At 200-500 ns/day for this 61k-atom system, 7 h gives roughly 60-150 ns.
# D needs only tens of ns, so a short run is still a result.
python run_kcsa.py --ns 150 --platform CUDA --save-ps 0.4 \
                   --checkpoint-ns 10 --max-hours 7 --out filter_ions.npz

echo
echo "Copy filter_ions.npz back, put it at runs/kcsa_md/filter_ions.npz, then:"
echo "    python scripts/kcsa_md.py --analyse"
