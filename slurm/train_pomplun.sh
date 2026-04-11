#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Chimera — UNet Benchmark Job (pomplun / H200 partition)
#
# Submit:  sbatch slurm/train_pomplun.sh
# Monitor: squeue -u p.bendiksen001
# Logs:    tail -f /hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/<jobid>.out
#
# Customise the TRAIN_ARGS line below to select models / epochs.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=unet_bench_pomplun
#SBATCH --account=cs_funda.durupinarbabur
#SBATCH --partition=pomplun
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=04-00:00:00
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/%j.out
#SBATCH --error=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET/slurm_logs/%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu

PROJECT=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET
cd "$PROJECT"

mkdir -p slurm_logs

echo "=============================="
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  Partition: pomplun (H200)"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Start    : $(date)"
echo "=============================="

# ── Training arguments ────────────────────────────────────────────────────────
# Edit this line to customise the run.
# Examples:
#   --models all                          → train all 4 variants
#   --models unet unet++                  → train only UNet and UNet++
#   --epochs 100 --nnunet-epochs 100      → longer training
#   --out chimera_results_pomplun         → separate output dir

TRAIN_ARGS="--models all --epochs 50 --nnunet-epochs 50 --out chimera_results_pomplun"

python chimera_train.py $TRAIN_ARGS

echo ""
echo "=============================="
echo "  Done: $(date)"
echo "=============================="
