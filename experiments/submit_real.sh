#!/bin/bash
# submit_real.sh
# Submits batches of real-data experiments (Camelyon17 or CIFAR-10/CIFAR-10-C)
# to a Slurm cluster. Per-job execution goes through run_real.sh.
#
# Usage:
#   DATASET=<camelyon17|cifar10> CONF=<n> bash submit_real.sh
#
# Configuration index guide:
#   camelyon17 CONF=0 — vary n_new (calibration sweep), beta=0.1, shift_center=2
#   camelyon17 CONF=1 — vary beta (OOD-fraction sweep), n_new=500
#   camelyon17 CONF=9 — quick smoke test
#
#   cifar10    CONF=0 — vary n_new (calibration sweep), contrast sev=5
#   cifar10    CONF=1 — vary beta, contrast sev=5
#   cifar10    CONF=2 — vary corruption type at severity=5
#   cifar10    CONF=3 — vary severity for gaussian_noise
#   cifar10    CONF=9 — quick smoke test (epochs=20)

# -----------------------------------------------------------------------
# REQUIRE DATASET AND CONF (e.g. DATASET=cifar10 CONF=0 bash submit_real.sh)
# -----------------------------------------------------------------------
if [[ -z "$DATASET" ]]; then
  echo "ERROR: DATASET is not set."
  echo "Run as:  DATASET=<camelyon17|cifar10> CONF=<n> bash submit_real.sh"
  exit 1
fi
if [[ -z "$CONF" ]]; then
  echo "ERROR: CONF is not set."
  echo "Run as:  DATASET=<camelyon17|cifar10> CONF=<n> bash submit_real.sh"
  exit 1
fi
if [[ "$DATASET" != "camelyon17" && "$DATASET" != "cifar10" ]]; then
  echo "ERROR: DATASET must be 'camelyon17' or 'cifar10', got: '$DATASET'"
  exit 1
fi

# -----------------------------------------------------------------------
# PER-DATASET / PER-CONF SWEEP GRIDS
# -----------------------------------------------------------------------
if [[ "$DATASET" == "camelyon17" ]]; then
  # Camelyon17 fixed defaults
  N_HIST=10000
  N_TEST=1000
  N_EVAL=5000
  EPOCHS=10
  KNN_K=50
  ALPHA=0.1
  SHIFT_CENTER=2
  # Placeholders so the unified 5-deep submission loop works
  CORRUPTION_LIST=("__na__")
  SEVERITY_LIST=(0)

  case "$CONF" in
    0)
      # Vary calibration set size
      N_NEW_LIST=(200 300 500 1000 2000)
      BETA_LIST=(0.1)
      SEED_LIST=$(seq 2000 2030)
      ;;
    1)
      # Vary OOD fraction
      N_NEW_LIST=(500)
      BETA_LIST=(0.10 0.15 0.20 0.25 0.30 0.35 0.40)
      SEED_LIST=$(seq 2000 2030)
      ;;
    9)
      # Quick smoke test
      N_NEW_LIST=(500)
      BETA_LIST=(0.1)
      SEED_LIST=(886)
      ;;
    *)
      echo "ERROR: Unknown camelyon17 CONF=$CONF. Valid: 0 1 9"
      exit 1
      ;;
  esac

elif [[ "$DATASET" == "cifar10" ]]; then
  # CIFAR-10 fixed defaults
  N_HIST=10000
  N_TEST=1000
  N_EVAL=5000
  EPOCHS=100
  KNN_K=50
  ALPHA=0.1
  SHIFT_CENTER=0    # unused for cifar10

  case "$CONF" in
    0)
      # Vary calibration set size
      N_NEW_LIST=(200 300 500 1000 2000)
      BETA_LIST=(0.1)
      CORRUPTION_LIST=("contrast")
      SEVERITY_LIST=(5)
      SEED_LIST=$(seq 2000 2030)
      ;;
    1)
      # Vary OOD fraction
      N_NEW_LIST=(500)
      BETA_LIST=(0.10 0.15 0.20 0.25 0.30 0.35 0.40)
      CORRUPTION_LIST=("contrast")
      SEVERITY_LIST=(5)
      SEED_LIST=$(seq 2000 2030)
      ;;
    2)
      # Vary corruption type at severity=5
      N_NEW_LIST=(2000)
      BETA_LIST=(0.1)
      CORRUPTION_LIST=("gaussian_noise" "shot_noise" "impulse_noise" "contrast" "fog" "frost" "snow")
      SEVERITY_LIST=(5)
      SEED_LIST=$(seq 2000 2010)
      ;;
    3)
      # Vary severity for a fixed corruption
      N_NEW_LIST=(2000)
      BETA_LIST=(0.1)
      CORRUPTION_LIST=("gaussian_noise")
      SEVERITY_LIST=(1 2 3 4 5)
      SEED_LIST=$(seq 2000 2010)
      ;;
    9)
      # Quick smoke test (shorter training)
      N_NEW_LIST=(500)
      BETA_LIST=(0.1)
      CORRUPTION_LIST=("contrast")
      SEVERITY_LIST=(5)
      SEED_LIST=(42)
      EPOCHS=100
      ;;
    *)
      echo "ERROR: Unknown cifar10 CONF=$CONF. Valid: 0 1 2 3 9"
      exit 1
      ;;
  esac
fi

# -----------------------------------------------------------------------
# DATA ROOTS
# -----------------------------------------------------------------------
WILDS_ROOT="experiments/data"
DATA_ROOT="experiments/data/cifar10"
CIFAR10C_DIR="experiments/data/CIFAR-10-C"

# -----------------------------------------------------------------------
# SLURM PARAMETERS
# -----------------------------------------------------------------------
MEMO=16G
TIME=00-8:00:00
CPUS=4
GPU_TYPE=p100
ORDP="sbatch --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=$CPUS --gres=gpu:${GPU_TYPE}:1 --time=$TIME --mem=$MEMO"

# -----------------------------------------------------------------------
# OUTPUT DIRECTORIES
#   camelyon17 results live under .../real/binary/...    (legacy layout)
#   cifar10    results live under .../real/cifar10/...
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d)

if [[ "$DATASET" == "camelyon17" ]]; then
  SUBDIR="binary"
else
  SUBDIR="cifar10"
fi
LOGS="$SCRIPT_DIR/../logs/real/$SUBDIR/conf_${CONF}"
OUT_DIR="$SCRIPT_DIR/../results/real/$SUBDIR/conf_${CONF}/${TIMESTAMP}"
mkdir -p "$LOGS" "$OUT_DIR"

echo "Dataset:    $DATASET"
echo "CONF:       $CONF"
echo "Results:    $OUT_DIR"
echo "Logs:       $LOGS"
echo ""

# -----------------------------------------------------------------------
# SUBMISSION LOOP
# -----------------------------------------------------------------------
N_SUBMITTED=0
N_SKIPPED=0

for SEED in $SEED_LIST; do
  for N_NEW in "${N_NEW_LIST[@]}"; do
    for BETA in "${BETA_LIST[@]}"; do
      for CORRUPTION in "${CORRUPTION_LIST[@]}"; do
        for SEVERITY in "${SEVERITY_LIST[@]}"; do

          if [[ "$DATASET" == "camelyon17" ]]; then
            JOBN="cam17_sc${SHIFT_CENTER}_nnew${N_NEW}_beta${BETA}_seed${SEED}"
            EXPECTED_OUT="$OUT_DIR/camelyon17_center${SHIFT_CENTER}_beta${BETA}_nhist${N_HIST}_nnew${N_NEW}_ntest${N_TEST}_seed${SEED}.csv"
            WRAP_CMD="bash $SCRIPT_DIR/run_real.sh camelyon17 \
              $N_NEW $BETA $SHIFT_CENTER $SEED '$OUT_DIR' \
              $N_HIST $N_TEST $N_EVAL $EPOCHS $KNN_K $ALPHA '$WILDS_ROOT'"
          else
            JOBN="cif10c_${CORRUPTION}_sev${SEVERITY}_nnew${N_NEW}_beta${BETA}_seed${SEED}"
            EXPECTED_OUT="$OUT_DIR/cifar10c_${CORRUPTION}_sev${SEVERITY}_beta${BETA}_nhist${N_HIST}_nnew${N_NEW}_ntest${N_TEST}_seed${SEED}.csv"
            WRAP_CMD="bash $SCRIPT_DIR/run_real.sh cifar10 \
              $N_NEW $BETA $CORRUPTION $SEVERITY $SEED '$OUT_DIR' \
              $N_HIST $N_TEST $N_EVAL $EPOCHS $KNN_K $ALPHA \
              '$DATA_ROOT' '$CIFAR10C_DIR'"
          fi

          # Skip if output CSV already exists
          if [[ -f "$EXPECTED_OUT" ]]; then
            (( N_SKIPPED++ ))
            continue
          fi

          OUTF="$LOGS/${JOBN}.out"
          ERRF="$LOGS/${JOBN}.err"

          $ORDP -J "$JOBN" -o "$OUTF" -e "$ERRF" --wrap="$WRAP_CMD"

          echo "Submitted: $JOBN"
          (( N_SUBMITTED++ ))

        done  # SEVERITY
      done    # CORRUPTION
    done      # BETA
  done        # N_NEW
done          # SEED

echo ""
echo "Done. Submitted: $N_SUBMITTED jobs, skipped: $N_SKIPPED (output already exists)."
echo "Results directory: $OUT_DIR"
