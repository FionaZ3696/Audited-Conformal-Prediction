#!/bin/bash
# submit_synthetic.sh
# Submits batches of synthetic multiclass experiments to a Slurm cluster.
#
# Usage:
#   CONF=<n> bash submit_synthetic.sh
#
# Configuration index guide:
#   0 — vary n_new (calibration size sweep), K in {5,10}, beta=0.1
#   1 — vary beta (distribution shift severity), K in {5,10}, n_new=2000
#   2 — vary K (number of classes), beta=0.1, n_new=2000
#   3 — adaptive methods only, vary n_new, K=5, beta=0.1
#   9 — quick test run (single seed, small grid)

# -----------------------------------------------------------------------
# REQUIRE CONF TO BE SET EXTERNALLY  (e.g.  CONF=9 bash Submit_...)
# -----------------------------------------------------------------------
if [[ -z "$CONF" ]]; then
  echo "ERROR: CONF is not set. Run as:  CONF=<n> bash submit_synthetic.sh"
  exit 1
fi

# -----------------------------------------------------------------------
# PER-CONFIGURATION SWEEP GRIDS
# Varying parameters: data_model, n_hist, n_new, beta, K, seed
# Fixed-per-conf parameters: epochs, feature_set, alpha, p, cal_prop,
#                             methods, selection_threshold
# -----------------------------------------------------------------------
NON_ADAPTIVE_METHODS="standard ts_fixed ts_adaptive trustscore retrain oracle condconf equalized combscore"

if [[ $CONF == 0 ]]; then
  # Vary calibration set size
  N_HIST_LIST=(10000)
  N_NEW_LIST=(200 500 1000 2000 5000)
  BETA_LIST=(0.1)
  DATA_MODEL_LIST=(3 5) 
  K_LIST=(5)
  SEED_LIST=$(seq 2000 2050)
  EPOCHS=30
  FEATURE_SET="full"
  ALPHA=0.1
  P=100
  CAL_PROP=0.5
  METHODS="$NON_ADAPTIVE_METHODS"
  SELECTION_THRESHOLD=""

elif [[ $CONF == 1 ]]; then
  # Vary distribution shift severity
  N_HIST_LIST=(10000)
  N_NEW_LIST=(2000)
  BETA_LIST=(0.15 0.20 0.25 0.30 0.35 0.40) 
  DATA_MODEL_LIST=(3 5) 
  K_LIST=(5)
  SEED_LIST=$(seq 2000 2050)
  EPOCHS=30
  FEATURE_SET="full"
  ALPHA=0.1
  P=100
  CAL_PROP=0.5
  METHODS="$NON_ADAPTIVE_METHODS"
  SELECTION_THRESHOLD=""

elif [[ $CONF == 2 ]]; then
  # Vary number of classes
  N_HIST_LIST=(10000)
  N_NEW_LIST=(2000)
  BETA_LIST=(0.1)
  DATA_MODEL_LIST=(3 5)
  K_LIST=(2 3 5 10 15 20) 
  SEED_LIST=$(seq 2000 2050)
  EPOCHS=30
  FEATURE_SET="full"
  ALPHA=0.1
  P=100
  CAL_PROP=0.5
  METHODS="$NON_ADAPTIVE_METHODS"
  SELECTION_THRESHOLD=""

elif [[ $CONF == 3 ]]; then
  # Adaptive methods only — vary calibration set size
  N_HIST_LIST=(10000)
  N_NEW_LIST=(100 200 300 500 1000 2000 5000) 
  BETA_LIST=(0.1)
  DATA_MODEL_LIST=(3 5)
  K_LIST=(5)
  SEED_LIST=$(seq 2000 2050)
  EPOCHS=30
  FEATURE_SET="full"
  ALPHA=0.1
  P=100
  CAL_PROP=0.5
  METHODS="standard ts_adaptive retrain condconf equalized combscore adaptive_combscore adaptive_equalized adaptive_condconf"
  SELECTION_THRESHOLD=0.7

elif [[ $CONF == 9 ]]; then
  # Adaptive methods only — vary calibration set size
  N_HIST_LIST=(10000)
  N_NEW_LIST=(2000) # 2000 5000
  BETA_LIST=(0.1)
  DATA_MODEL_LIST=(3)
  K_LIST=(5)
  SEED_LIST=(42) # $(seq 2000 2050)
  EPOCHS=30
  FEATURE_SET="full"
  ALPHA=0.1
  P=100
  CAL_PROP=0.5
  METHODS="standard ts_adaptive retrain condconf equalized combscore adaptive_combscore adaptive_equalized adaptive_condconf"
  SELECTION_THRESHOLD=0.7

else
  echo "ERROR: Unknown CONF=$CONF. Valid values: 0 1 2 3 9"
  exit 1
fi

# -----------------------------------------------------------------------
# SLURM PARAMETERS
# -----------------------------------------------------------------------
MEMO=16G
TIME=00-5:00:00
CPUS=4

ORDP="sbatch --nodes=1 --ntasks=1 --cpus-per-task=$CPUS --time=$TIME --mem=$MEMO"

# -----------------------------------------------------------------------
# OUTPUT DIRECTORIES
# Results: timestamped so each submission batch is clearly distinguished.
# Logs:    fixed path — overwritten by future experiments.
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d)

LOGS="$SCRIPT_DIR/../logs/synthetic/multiclass/conf_${CONF}"
mkdir -p "$LOGS"

OUT_DIR="$SCRIPT_DIR/../results/synthetic/multiclass/conf_${CONF}/${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "Results will be saved to: $OUT_DIR"
echo "Logs will be written to:  $LOGS"
echo ""

# -----------------------------------------------------------------------
# SUBMISSION LOOP
# -----------------------------------------------------------------------
N_SUBMITTED=0
N_SKIPPED=0

for SEED in $SEED_LIST; do
  for N_HIST in "${N_HIST_LIST[@]}"; do
    for N_NEW in "${N_NEW_LIST[@]}"; do
      for BETA in "${BETA_LIST[@]}"; do
        for K in "${K_LIST[@]}"; do
          for DATA_MODEL in "${DATA_MODEL_LIST[@]}"; do

            JOBN="nhist${N_HIST}_nnew${N_NEW}_beta${BETA}_dm${DATA_MODEL}_k${K}_seed${SEED}_ep${EPOCHS}_feat${FEATURE_SET}_alpha${ALPHA}_p${P}_calprop${CAL_PROP}"
            if [[ -n "$SELECTION_THRESHOLD" ]]; then
              JOBN="${JOBN}_selthresh${SELECTION_THRESHOLD}"
            fi
            EXPECTED_OUT="$OUT_DIR/${JOBN}.csv"

            # Skip if output CSV already exists in this timestamped batch
            if [[ -f "$EXPECTED_OUT" ]]; then
              (( N_SKIPPED++ ))
              continue
            fi

            OUTF="$LOGS/${JOBN}.out"
            ERRF="$LOGS/${JOBN}.err"

            $ORDP -J "$JOBN" -o "$OUTF" -e "$ERRF" \
              --wrap="bash $SCRIPT_DIR/run_synthetic.sh \
                $N_HIST $N_NEW $BETA $DATA_MODEL $K $EPOCHS $SEED \
                $FEATURE_SET '$OUT_DIR' $ALPHA $P $CAL_PROP \
                '$METHODS' $SELECTION_THRESHOLD"

            echo "Submitted: $JOBN"
            (( N_SUBMITTED++ ))

          done  # DATA_MODEL
        done  # K
      done  # BETA
    done  # N_NEW
  done  # N_HIST
done  # SEED

echo ""
echo "Done. Submitted: $N_SUBMITTED jobs, skipped: $N_SKIPPED (output already exists)."
echo "Results directory: $OUT_DIR"
