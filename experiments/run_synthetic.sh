#!/bin/bash
# run_synthetic.sh
# Per-job wrapper called by submit_synthetic.sh.
#
# Usage:
#   bash run_synthetic.sh <N_HIST> <N_NEW> <BETA> <DATA_MODEL> \
#        <K> <EPOCHS> <SEED> <FEATURE_SET> <OUT_DIR> \
#        [ALPHA] [P] [CAL_PROP] [METHODS] [SELECTION_THRESHOLD]

N_HIST=$1
N_NEW=$2
BETA=$3
DATA_MODEL=$4
K=$5
EPOCHS=$6
SEED=$7
FEATURE_SET=$8
OUT_DIR=$9
ALPHA=${10:-0.1}
P=${11:-100}
CAL_PROP=${12:-0.5}
METHODS=${13:-""}
SELECTION_THRESHOLD=${14:-0.7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the conda env's Python directly — more reliable than conda activate in Slurm
# PYTHON=

echo "=============================="
echo "Job parameters:"
echo "  n_hist=$N_HIST  n_new=$N_NEW  beta=$BETA  data_model=$DATA_MODEL"
echo "  K=$K  epochs=$EPOCHS  seed=$SEED  feature_set=$FEATURE_SET"
echo "  alpha=$ALPHA  p=$P  cal_prop=$CAL_PROP"
echo "  methods=$METHODS  selection_threshold=$SELECTION_THRESHOLD"
echo "  outdir=$OUT_DIR"
echo "=============================="

METHODS_FLAG=""
if [[ -n "$METHODS" ]]; then
  METHODS_FLAG="--methods $METHODS"
fi

SEL_THRESH_FLAG=""
if [[ -n "$SELECTION_THRESHOLD" ]]; then
  SEL_THRESH_FLAG="--selection_threshold $SELECTION_THRESHOLD"
fi

$PYTHON "$SCRIPT_DIR/run_synthetic_experiment.py" \
    --n_hist                "$N_HIST"               \
    --n_new                 "$N_NEW"                 \
    --beta                  "$BETA"                  \
    --data_model            "$DATA_MODEL"            \
    --K                     "$K"                     \
    --epochs                "$EPOCHS"                \
    --seed                  "$SEED"                  \
    --feature_set           "$FEATURE_SET"           \
    --outdir                "$OUT_DIR"               \
    --alpha                 "$ALPHA"                 \
    --p                     "$P"                     \
    --cal_prop              "$CAL_PROP"              \
    $SEL_THRESH_FLAG \
    $METHODS_FLAG
