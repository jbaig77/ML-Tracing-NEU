#!/bin/bash
#SBATCH --job-name=gnn4_job
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --mail-user=baig.mi@northeastern.edu
#SBATCH --mail-type=END,FAIL
# (GPU toggle below will override gres/partition if set)

set -euo pipefail

# ---------------- USER TOGGLES ----------------
# Request a GPU? If yes, we also switch to a GPU partition name.
USE_GPU="${USE_GPU:-0}"        # 0 or 1
GPUS="${GPUS:-1}"              # number of GPUs if USE_GPU=1
GPU_PARTITION="${GPU_PARTITION:-gpu}"  # cluster-specific (e.g., 'gpu', 'p100', 'gpu-short')
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"

# SBATCH overrides for GPU runs (only at launcher)
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && "$USE_GPU" == "1" ]]; then
  # Re-submit ourselves with GPU resources
  # NOTE: discovery cluster names may differ; adjust GPU_PARTITION as needed.
  sbatch --job-name=gnn4_job \
         --nodes=1 --ntasks=1 --cpus-per-task=4 \
         --gres="gpu:${GPUS}" --mem=100G --time=48:00:00 \
         --partition="${GPU_PARTITION}" \
         --mail-user=baig.mi@northeastern.edu \
         --mail-type=END,FAIL \
         --export=ALL,PARAM_FILE="${PARAM_FILE:-train_gnn4_params.txt}",TOP_ROOT="${TOP_ROOT:-experiment_runs}",USE_GPU="$USE_GPU",GPUS="$GPUS",GPU_PARTITION="$GPU_PARTITION",CUDA_MODULE="$CUDA_MODULE" \
         "$0"
  exit 0
fi
# ---------------------------------------------

# --- User knobs (can be overridden via --export) ---
PARAM_FILE="${PARAM_FILE:-train_gnn4_params.txt}"
TOP_ROOT="${TOP_ROOT:-experiment_runs}"

# Detect launcher vs worker
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ ! -f "$PARAM_FILE" ]]; then
    echo "PARAM_FILE not found: $PARAM_FILE" >&2; exit 1
  fi
  N=$(grep -vE '^\s*#' "$PARAM_FILE" | sed '/^\s*$/d' | wc -l)
  if [[ "$N" -le 0 ]]; then
    echo "No experiments found in $PARAM_FILE" >&2; exit 1
  fi

  base=$(basename "$PARAM_FILE"); base="${base%.*}"
  OUT_ROOT="${TOP_ROOT}/${base}"
  mkdir -p "$OUT_ROOT"

  echo "Submitting $N jobs from $PARAM_FILE"
  echo "Batch OUT_ROOT: $OUT_ROOT"

  # Submit THIS SAME SCRIPT as an array with same toggles/env
  sbatch \
    --job-name=gnn4_job \
    --array=0-$((N-1)) \
    --export=ALL,PARAM_FILE="$PARAM_FILE",OUT_ROOT="$OUT_ROOT",TOP_ROOT="$TOP_ROOT",USE_GPU="$USE_GPU",GPUS="$GPUS",GPU_PARTITION="$GPU_PARTITION",CUDA_MODULE="$CUDA_MODULE" \
    "$0"
  exit 0
fi

# -------------------- WORKER PHASE --------------------
LINE=$(grep -vE '^\s*#' "$PARAM_FILE" | sed '/^\s*$/d' | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
if [[ -z "${LINE:-}" ]]; then
  echo "No line found for task $SLURM_ARRAY_TASK_ID in $PARAM_FILE"; exit 1
fi

# Core columns (your current file)
read -r EXP_ID DATA_ROOT T EPOCHS LR BATCH_GRAPHS LOSS START_MIN_T END_MIN_T VAL_STEPS VAL_DAMP \
    FREE_STEPS PIN_DROPOUT PIN_NOISE FREE_PROB RUN_PREFIX ARCH GRAPH_K \
    LOOP_TYPE LOOP_LAMBDA LOOP_K \
    ENDPOINT_TYPE ENDPOINT_LAMBDA ENDPOINT_SIGMA \
    LOOP_TAU LOOP_ESTIMATOR LOOP_HUTCH_R LOOP_NBT_WEIGHTING <<< "$LINE"

# Backward-compat defaults if param file doesn’t include the new fields
LOOP_TAU="${LOOP_TAU:-1.1}"
LOOP_ESTIMATOR="${LOOP_ESTIMATOR:-auto}"     # auto|exact|hutch
LOOP_HUTCH_R="${LOOP_HUTCH_R:-8}"
LOOP_NBT_WEIGHTING="${LOOP_NBT_WEIGHTING:-sqrt}"  # sqrt|child

# Sanity for VAL_STEPS
if [[ -z "${VAL_STEPS}" || "${VAL_STEPS}" == "-" ]]; then VAL_STEPS="$T"; fi

# Env
set +u
[[ -f ~/.bashrc ]] && source ~/.bashrc
set -u

# Activate your env and CUDA (if available)
conda activate traceGNN
module load "${CUDA_MODULE}" || true

SEED=$(( 1234 + SLURM_ARRAY_TASK_ID ))
RUN_NAME="${EXP_ID}"

mkdir -p "${OUT_ROOT}/logs"
LOG_OUT="${OUT_ROOT}/logs/output_${EXP_ID}_${SLURM_JOB_ID}.out"
LOG_ERR="${OUT_ROOT}/logs/error_${EXP_ID}_${SLURM_JOB_ID}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

echo "===== RUN $SLURM_ARRAY_TASK_ID | JOB $SLURM_JOB_ID ====="
echo "EXP_ID        : $EXP_ID"
echo "DATA_ROOT     : $DATA_ROOT"
echo "T / VAL_STEPS : $T / $VAL_STEPS"
echo "EPOCHS / LR   : $EPOCHS / $LR"
echo "BATCH_GRAPHS  : $BATCH_GRAPHS"
echo "LOSS          : $LOSS"
echo "START/END t   : $START_MIN_T / $END_MIN_T"
echo "FREE_STEPS    : $FREE_STEPS"
echo "PIN_DROPOUT   : $PIN_DROPOUT"
echo "PIN_NOISE     : $PIN_NOISE"
echo "FREE_PROB     : $FREE_PROB"
echo "RUN_PREFIX    : $RUN_PREFIX"
echo "ARCH / GRAPH_K: $ARCH / ${GRAPH_K:-}"
echo "LOOP          : ${LOOP_TYPE:-none}  λ=${LOOP_LAMBDA:-0}  K=${LOOP_K:-6}  τ=${LOOP_TAU}  est=${LOOP_ESTIMATOR}  R=${LOOP_HUTCH_R}  nbt_w=${LOOP_NBT_WEIGHTING}"
echo "ENDPOINT      : ${ENDPOINT_TYPE:-none}  λ=${ENDPOINT_LAMBDA:-0}  σ=${ENDPOINT_SIGMA:-}"
echo "OUT_ROOT      : $OUT_ROOT"
echo "---------------------------------------"

# Build python args safely and conditionally
PY_ARGS=(
  --data-root        "${DATA_ROOT}"
  --train-subdir     "train"
  --val-subdir       "validation"
  --run-prefix       "${RUN_PREFIX}"
  --T                "${T}"
  --epochs           "${EPOCHS}"
  --lr               "${LR}"
  --batch-graphs     "${BATCH_GRAPHS}"
  --loss-type        "${LOSS}"
  --use-curriculum   true
  --start-min-t      "${START_MIN_T}"
  --end-min-t        "${END_MIN_T}"
  --train-eval-frac  "0.25"
  --val-infer-steps  "${VAL_STEPS}"
  --val-damping      "${VAL_DAMP}"
  --free-run-steps   "${FREE_STEPS}"
  --p-in-dropout     "${PIN_DROPOUT}"
  --p-in-noise-std   "${PIN_NOISE}"
  --free-run-prob    "${FREE_PROB}"
  --save-every-steps 50
  --infer-every-steps 50
  --out-dir          "${OUT_ROOT}"
  --run-name         "${RUN_NAME}"
  --seed             "${SEED}"
  --arch             "${ARCH}"
  --loop-penalty     "${LOOP_TYPE}"
  --loop-lambda      "${LOOP_LAMBDA}"
  --loop-K           "${LOOP_K}"
  --endpoint-penalty "${ENDPOINT_TYPE}"
  --endpoint-lambda  "${ENDPOINT_LAMBDA}"
)

# Only pass graph-k if provided and non-empty (avoids setting it to "")
if [[ -n "${GRAPH_K:-}" ]]; then
  PY_ARGS+=( --graph-k "${GRAPH_K}" )
fi

# Endpoint sigma only matters for the bump penalty
if [[ "${ENDPOINT_TYPE:-none}" == "bump" && -n "${ENDPOINT_SIGMA:-}" ]]; then
  PY_ARGS+=( --endpoint-sigma "${ENDPOINT_SIGMA}" )
fi

# Loop extras: tau/estimator/hutch_R always safe for 'complex'; weighting used by 'nbt'
if [[ "${LOOP_TYPE:-none}" == "complex" ]]; then
  PY_ARGS+=( --loop-tau "${LOOP_TAU}" --loop-complex-estimator "${LOOP_ESTIMATOR}" --loop-hutch-R "${LOOP_HUTCH_R}" )
elif [[ "${LOOP_TYPE:-none}" == "nbt" ]]; then
  PY_ARGS+=( --loop-hutch-R "${LOOP_HUTCH_R}" --loop-K "${LOOP_K}" --loop-nbt-weighting "${LOOP_NBT_WEIGHTING}" )
fi

# Ensure we pick up your fixed 'quad' semantics in code (no CLI needed here)
# Launch
srun python train_gnn4.py "${PY_ARGS[@]}"
