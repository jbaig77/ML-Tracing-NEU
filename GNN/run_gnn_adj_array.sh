#!/bin/bash
#SBATCH --job-name=gnn_adj_job
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --mail-user=baig.mi@northeastern.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ------------ User knobs (override via --export) ------------
PARAM_FILE="${PARAM_FILE:-train_gnn_adj_params.txt}"
TOP_ROOT="${TOP_ROOT:-experiment_runs_gnn_adj}"
CONDA_ENV="${CONDA_ENV:-traceGNN}"
# -----------------------------------------------------------

# Validate param file
if [[ ! -f "$PARAM_FILE" ]]; then
  echo "PARAM_FILE not found: $PARAM_FILE" >&2
  exit 1
fi

# Count non-empty, non-comment lines
N=$(grep -vE '^\s*#' "$PARAM_FILE" | sed '/^\s*$/d' | wc -l)
if [[ "$N" -le 0 ]]; then
  echo "No experiments in $PARAM_FILE" >&2
  exit 1
fi

# If array range is still default 0-0, re-submit with correct size
# (So you can always do: sbatch ./run_gnn_adj_array.sh)
if [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -eq 1 && "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 && "$N" -gt 1 ]]; then
  base=$(basename "$PARAM_FILE"); base="${base%.*}"
  OUT_ROOT="${TOP_ROOT}/${base}"
  mkdir -p "$OUT_ROOT"

  echo "Resubmitting as array 0-$((N-1))"
  sbatch \
    --array=0-$((N-1)) \
    --export=ALL,PARAM_FILE="$PARAM_FILE",TOP_ROOT="$TOP_ROOT",CONDA_ENV="$CONDA_ENV" \
    "$0"
  exit 0
fi

# Derive OUT_ROOT
base=$(basename "$PARAM_FILE"); base="${base%.*}"
OUT_ROOT="${TOP_ROOT}/${base}"
mkdir -p "$OUT_ROOT/logs"

# Pick this task's line
LINE=$(grep -vE '^\s*#' "$PARAM_FILE" | sed '/^\s*$/d' | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
if [[ -z "${LINE:-}" ]]; then
  echo "No line found for task $SLURM_ARRAY_TASK_ID in $PARAM_FILE" >&2
  exit 1
fi

read -r EXP_ID DATA_ROOT EPOCHS LR NLAYERS LOOP_ON LOOP_LAMBDA <<< "$LINE"

SEED=$(( 4242 + SLURM_ARRAY_TASK_ID ))

LOG_OUT="${OUT_ROOT}/logs/output_${EXP_ID}_${SLURM_JOB_ID}.out"
LOG_ERR="${OUT_ROOT}/logs/error_${EXP_ID}_${SLURM_JOB_ID}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

echo "===== ARRAY TASK $SLURM_ARRAY_TASK_ID | JOB $SLURM_JOB_ID ====="
echo "EXP_ID      : $EXP_ID"
echo "DATA_ROOT   : $DATA_ROOT"
echo "EPOCHS / LR : $EPOCHS / $LR"
echo "NLAYERS     : $NLAYERS"
echo "LOOP_ON     : $LOOP_ON"
echo "LOOP_LAMBDA : $LOOP_LAMBDA"
echo "OUT_ROOT    : $OUT_ROOT"
echo "SEED        : $SEED"
echo "---------------------------------------"

# ---- Env ----
set +u
[[ -f ~/.bashrc ]] && source ~/.bashrc
set -u

# Conda activation that works in batch shells
if command -v conda >/dev/null 2>&1; then
  if ! conda info >/dev/null 2>&1; then
    [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
    [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh"  ]] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
else
  echo "conda not found; cannot activate env '$CONDA_ENV'." >&2
  exit 1
fi

python train_gnn_adj_from_mat.py \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --nlayers "${NLAYERS}" \
  --loop-penalty "${LOOP_ON}" \
  --loop-lambda "${LOOP_LAMBDA}" \
  --out-dir "${OUT_ROOT}" \
  --run-name "${EXP_ID}" \
  --seed "${SEED}" \
  --use-gpu false
