#!/bin/bash
# run_real.sh
# Per-job wrapper for both Camelyon17 and CIFAR-10 real-data experiments.
# Dispatched by submit_real.sh (DATASET=<...> CONF=<n> bash submit_real.sh).
#
# Usage:
#   bash run_real.sh camelyon17 <N_NEW> <BETA> <SHIFT_CENTER> <SEED> <OUT_DIR> \
#        [N_HIST] [N_TEST] [N_EVAL] [EPOCHS] [KNN_K] [ALPHA] [WILDS_ROOT]
#
#   bash run_real.sh cifar10    <N_NEW> <BETA> <CORRUPTION> <SEVERITY> <SEED> <OUT_DIR> \
#        [N_HIST] [N_TEST] [N_EVAL] [EPOCHS] [KNN_K] [ALPHA] [DATA_ROOT] [CIFAR10C_DIR]

DATASET=$1
shift

if [[ "$DATASET" == "camelyon17" ]]; then
    N_NEW=$1
    BETA=$2
    SHIFT_CENTER=$3
    SEED=$4
    OUT_DIR=$5
    N_HIST=${6:-10000}
    N_TEST=${7:-1000}
    N_EVAL=${8:-5000}
    EPOCHS=${9:-10}
    KNN_K=${10:-50}
    ALPHA=${11:-0.1}
    WILDS_ROOT=${12:-"experiments/data"}

    DATASET_ARGS=(
        --dataset       camelyon17
        --wilds_root    "$WILDS_ROOT"
        --shift_center  "$SHIFT_CENTER"
    )

elif [[ "$DATASET" == "cifar10" ]]; then
    N_NEW=$1
    BETA=$2
    CORRUPTION=$3
    SEVERITY=$4
    SEED=$5
    OUT_DIR=$6
    N_HIST=${7:-10000}
    N_TEST=${8:-1000}
    N_EVAL=${9:-5000}
    EPOCHS=${10:-100}
    KNN_K=${11:-50}
    ALPHA=${12:-0.1}
    DATA_ROOT=${13:-"experiments/data/cifar10"}
    CIFAR10C_DIR=${14:-"experiments/data/CIFAR-10-C"}

    DATASET_ARGS=(
        --dataset       cifar10
        --cifar10_root  "$DATA_ROOT"
        --cifar10c_dir  "$CIFAR10C_DIR"
        --corruption    "$CORRUPTION"
        --severity      "$SEVERITY"
    )

else
    echo "ERROR: first arg must be 'camelyon17' or 'cifar10', got: '$DATASET'"
    echo "Usage: bash run_real.sh <camelyon17|cifar10> <args...>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the conda env's Python directly
# PYTHON=

echo "=============================="
echo "Job parameters:"
echo "  dataset=$DATASET  seed=$SEED"
if [[ "$DATASET" == "camelyon17" ]]; then
    echo "  shift_center=$SHIFT_CENTER  wilds_root=$WILDS_ROOT"
else
    echo "  corruption=$CORRUPTION  severity=$SEVERITY"
    echo "  data_root=$DATA_ROOT  cifar10c_dir=$CIFAR10C_DIR"
fi
echo "  n_new=$N_NEW  beta=$BETA"
echo "  n_hist=$N_HIST  n_test=$N_TEST  n_eval=$N_EVAL  epochs=$EPOCHS"
echo "  knn_k=$KNN_K  alpha=$ALPHA"
echo "  outdir=$OUT_DIR"
echo "=============================="

$PYTHON -u "$SCRIPT_DIR/run_real_experiment.py" \
    "${DATASET_ARGS[@]}"             \
    --beta          "$BETA"          \
    --n_hist        "$N_HIST"        \
    --n_new         "$N_NEW"         \
    --n_test        "$N_TEST"        \
    --n_eval        "$N_EVAL"        \
    --alpha         "$ALPHA"         \
    --epochs        "$EPOCHS"        \
    --seed          "$SEED"          \
    --knn_k         "$KNN_K"         \
    --outdir        "$OUT_DIR"
