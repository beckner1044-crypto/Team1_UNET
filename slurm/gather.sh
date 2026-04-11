#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Chimera — Gather / Report Job  (CPU-only, chained after job array)
#
# Reads all per-model output directories, prints a ranked table,
# writes combined_results.csv, and saves all three comparison plots.
#
# Normally submitted automatically by submit_benchmark.sh via --dependency.
# Can also be run manually after the array finishes:
#   sbatch slurm/gather.sh --export=BASE_OUT=<path>,PARTITION=<name>
#
# Or interactively (login node is fine — no GPU needed):
#   conda activate torch_gpu_nnunet
#   python gather_results.py --base chimera_results_pomplun
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=unet_gather
#SBATCH --account=cs_funda.durupinarbabur
#SBATCH --partition=pomplun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mem=8gb
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/%j_gather.out
#SBATCH --error=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/%j_gather.err

# BASE_OUT is passed in via --export when chained from submit_benchmark.sh.
# Fall back to pomplun results if not set.
BASE_OUT="${BASE_OUT:-/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/chimera_results_pomplun}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu_nnunet

PROJECT=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET
cd "$PROJECT"

echo "=============================================="
echo "  Gather job"
echo "  Base dir  : $BASE_OUT"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "=============================================="

python gather_results.py --base "$BASE_OUT"

echo ""
echo "=============================================="
echo "  Done: $(date)"
echo "=============================================="
