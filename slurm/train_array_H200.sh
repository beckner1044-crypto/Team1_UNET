#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Chimera — UNet Benchmark  |  Job Array  |  AICORE_H200 partition (chimera24)
#
# Spawns 4 independent GPU jobs (one per model) in parallel.
#
# Submit via submit_benchmark.sh (recommended) or manually:
#   sbatch slurm/train_array_H200.sh
#
# Array task → model mapping:
#   0 → unet
#   1 → unet++
#   2 → unet3++
#   3 → nnunet
#
# Tune EPOCHS / NNUNET_EPOCHS / BASE_OUT below as needed.
# Note: AICORE_H200 QOS wall-time cap is 2 hours — keep epochs reasonable.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=unet_bench
#SBATCH --account=impact
#SBATCH --partition=AICORE_H200
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/%j_%a.out
#SBATCH --error=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/%j_%a.err

# ── Settings ──────────────────────────────────────────────────────────────────
EPOCHS=50
NNUNET_EPOCHS=50
BASE_OUT=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/chimera_results_H200

# ── Model selection ───────────────────────────────────────────────────────────
MODELS=("unet" "unet++" "unet3++" "nnunet")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu_nnunet

PROJECT=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET
cd "$PROJECT"
mkdir -p slurm_logs "$BASE_OUT/$MODEL"

echo "=============================================="
echo "  Array task : $SLURM_ARRAY_TASK_ID / 3"
echo "  Model      : $MODEL"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  Partition  : AICORE_H200 (H200)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Start      : $(date)"
echo "=============================================="

python chimera_train.py \
    --models "$MODEL" \
    --epochs "$EPOCHS" \
    --nnunet-epochs "$NNUNET_EPOCHS" \
    --out "$BASE_OUT/$MODEL"

echo ""
echo "=============================================="
echo "  Done: $(date)"
echo "=============================================="
