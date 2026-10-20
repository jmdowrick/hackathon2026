#!/bin/bash
#SBATCH --job-name=kcsa-filter
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=kcsa-%j.out

# KcsA filter K+ diffusivity run. Edit the two lines marked EDIT.

set -euo pipefail

# EDIT: however your cluster provides a Python with OpenMM + CUDA.
#   module load Miniconda3 && conda activate openmm
# or, self-contained and usually fastest to set up:
#   conda create -y -n openmm -c conda-forge python=3.11 openmm cudatoolkit numpy
module load Miniconda3 2>/dev/null || true
source activate openmm 2>/dev/null || conda activate openmm

# EDIT: where the bundle was copied to
WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"

python -c "import openmm; from openmm import Platform; \
print('openmm', openmm.__version__); \
print('platforms', [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])"

# Checkpoints every 10 ns, so a job killed at the wall clock still returns data.
python run_kcsa.py --ns 100 --platform CUDA --save-ps 0.4 \
                   --checkpoint-ns 10 --out filter_ions.npz

echo "copy filter_ions.npz back and run:  python scripts/kcsa_md.py --analyse"
