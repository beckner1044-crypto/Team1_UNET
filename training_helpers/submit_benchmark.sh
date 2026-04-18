#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_benchmark.sh — One-command benchmark launcher
#
# Submits the 4-task job array (one GPU per model, all in parallel),
# then chains a gather job that runs automatically when all 4 finish.
#
# Usage:
#   ./submit_benchmark.sh H200        # AICORE_H200 (chimera24) — recommended
#   ./submit_benchmark.sh pomplun     # pomplun (H200, cs account)
#   ./submit_benchmark.sh A30         # A30
#   ./submit_benchmark.sh             # defaults to H200
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PARTITION="${1:-H200}"

PROJECT=/hpcstor6/scratch01/p/p.bendiksen001/Team1_UNET
cd "$PROJECT"
mkdir -p slurm_logs

case "$PARTITION" in
  H200)
    ARRAY_SCRIPT=slurm/train_array_H200.sh
    BASE_OUT="$PROJECT/chimera_results_H200"
    GATHER_ACCOUNT=impact
    GATHER_PARTITION=AICORE_H200
    ;;
  pomplun)
    ARRAY_SCRIPT=slurm/train_array_pomplun.sh
    BASE_OUT="$PROJECT/chimera_results_pomplun"
    GATHER_ACCOUNT=cs_funda.durupinarbabur
    GATHER_PARTITION=pomplun
    ;;
  A30)
    ARRAY_SCRIPT=slurm/train_array_A30.sh
    BASE_OUT="$PROJECT/chimera_results_A30"
    GATHER_ACCOUNT=pi_funda.durupinarbabur
    GATHER_PARTITION=A30
    ;;
  *)
    echo "Unknown partition '$PARTITION'. Use 'H200', 'pomplun', or 'A30'."
    exit 1
    ;;
esac

echo "========================================"
echo "  UNet Benchmark Submission"
echo "  Partition : $PARTITION  (gather: $GATHER_PARTITION)"
echo "  Output    : $BASE_OUT"
echo "========================================"

# ── Submit the 4-task array ───────────────────────────────────────────────────
ARRAY_JOB_ID=$(sbatch --parsable "$ARRAY_SCRIPT")
echo "  Array job submitted  : $ARRAY_JOB_ID  (tasks 0-3)"

# ── Chain the gather job (runs only if ALL array tasks succeeded) ─────────────
# Override partition/account in gather.sh so it works for both A30 and pomplun.
GATHER_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --partition="$GATHER_PARTITION" \
    --account="$GATHER_ACCOUNT" \
    --time=00:15:00 \
    --export="BASE_OUT=$BASE_OUT" \
    slurm/gather.sh)
echo "  Gather job submitted : $GATHER_JOB_ID  (runs after all 4 tasks finish)"

echo ""
echo "  Monitor progress:"
echo "    squeue -u p.bendiksen001"
echo ""
echo "  Live logs (replace JOBID and TASKID):"
echo "    tail -f slurm_logs/${ARRAY_JOB_ID}_0.out   # unet"
echo "    tail -f slurm_logs/${ARRAY_JOB_ID}_1.out   # unet++"
echo "    tail -f slurm_logs/${ARRAY_JOB_ID}_2.out   # unet3++"
echo "    tail -f slurm_logs/${ARRAY_JOB_ID}_3.out   # nnunet"
echo ""
echo "  Results (after gather job finishes):"
echo "    $BASE_OUT/combined_results.csv"
echo "    $BASE_OUT/plots/"
echo "========================================"
