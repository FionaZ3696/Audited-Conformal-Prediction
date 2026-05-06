# Audited Conformal Prediction (ACP)

Code for the paper *"Audited Conformal Prediction for Classification under
Unknown Distribution Shift"*.

ACP builds on split conformal prediction by training an **audit model** that
estimates the per-sample correctness probability `r*(x) = P_T(f(x) = Y | X=x)`
of a pretrained legacy classifier under an arbitrary target distribution `P_T`.
The audit score is then folded into the conformal calibration step in three
ways, all of which preserve marginal coverage at level `1-α` while improving
**conditional coverage on hard / shifted samples**:

- **ACP-MC** — audited probabilistic classifier (combines the legacy model and
  audit estimate before applying APS-style conformal calibration).
- **ACP-AEC** — Mondrian-style equalized coverage with a single audit-score
  threshold.
- **ACP-ACC** — conditional-conformal calibration over multiple overlapping
  audit-score groups (Gibbs et al.).

An adaptive variant (**AACP**) uses a held-out subset to data-driven select
between an ACP method and a retraining baseline.

## Repository layout

```
src/
  audited_conformal.py    Audited_Model, Audited_Conformal_Prediction,
                          Adaptive_Audited_Conformal_Prediction, eta-method helpers
  baseline_conformal.py   Standard, AdaTS, Platt, retrain, trust-score baselines
  data_models.py          Synthetic data generators (concept shift, covariate shift)
  evaluation.py           Coverage / conditional-coverage / size metrics
  models.py               ClassNNet, BinaryClassification, ClassDenseNet,
                          ClassResNet18Cifar, Blackbox training wrapper
  thirdparty.py           CondConf (LP-based conditional conformal) — vendored
  utils.py                APS scores, Platt + temperature scaling, kNN r* helper,
                          WILDS + CIFAR data loaders, EmbeddingModel, etc.

experiments/
  run_synthetic_experiment.py    Synthetic data entry point
  run_real_experiment.py         Real-data entry point (camelyon17 OR cifar10)
  run_synthetic.sh               Per-job wrapper (synthetic)
  run_real.sh                    Per-job wrapper (real data)
  submit_synthetic.sh            Slurm submitter (synthetic)
  submit_real.sh                 Slurm submitter (real data)

results/    Output CSVs, organized as <conf>/<YYYYMMDD>/<job>.csv
logs/       Slurm stdout/stderr per job
models/     Optional model checkpoint dumps
```

## Setup

Python ≥ 3.9, with the following packages:

```bash
pip install torch torchvision torchmetrics numpy scipy scikit-learn pandas \
            matplotlib tqdm cvxpy faiss-cpu
# Optional, only if running the Camelyon17 experiments:
pip install wilds
```

LP solvers used by `ACP-ACC` / `CondConf`: MOSEK is preferred when available
(`pip install mosek`); otherwise the code falls back to OSQP / HiGHS.

## Quick start

### Synthetic experiment (one configuration, locally)

```bash
python experiments/run_synthetic_experiment.py \
    --data_model 3 --K 5 --p 100 \
    --n_hist 10000 --n_new 2000 --beta 0.1 \
    --epochs 100 --seed 123
```

`--data_model 3` is `Data_Model_ConceptShift_Uniform`; `--data_model 5` is
`Data_Model_CovariateShift_Uniform`. Output CSV goes to
`results/synthetic/multiclass/`.

### Real-data experiment (one configuration, locally)

Camelyon17:

```bash
python experiments/run_real_experiment.py \
    --dataset camelyon17 \
    --wilds_root experiments/data \
    --shift_center 2 --beta 0.1 \
    --n_hist 10000 --n_new 2000 --n_test 1000 --n_eval 5000 \
    --epochs 10 --seed 2006
```

CIFAR-10 / CIFAR-10-C:

```bash
python experiments/run_real_experiment.py \
    --dataset cifar10 \
    --cifar10_root experiments/data/cifar10 \
    --cifar10c_dir experiments/data/CIFAR-10-C \
    --corruption contrast --severity 5 --beta 0.1 \
    --n_hist 10000 --n_new 2000 --n_test 1000 --n_eval 5000 \
    --epochs 100 --seed 2006
```

Both datasets auto-download on first run (CIFAR-10 via `torchvision`,
CIFAR-10-C from Zenodo).

### Slurm sweeps

Each submitter script takes a `CONF=<n>` env var that selects a pre-defined
sweep grid. The real-data submitter additionally takes `DATASET=<...>`:

```bash
# Synthetic
CONF=0 bash experiments/submit_synthetic.sh    # main sweep
CONF=9 bash experiments/submit_synthetic.sh    # quick smoke test

# Real
DATASET=camelyon17 CONF=0 bash experiments/submit_real.sh    # n_new sweep
DATASET=camelyon17 CONF=1 bash experiments/submit_real.sh    # beta sweep
DATASET=cifar10    CONF=2 bash experiments/submit_real.sh    # corruption-type sweep
DATASET=cifar10    CONF=3 bash experiments/submit_real.sh    # severity sweep
DATASET=cifar10    CONF=9 bash experiments/submit_real.sh    # quick smoke test
```

Per-`CONF` grids are documented at the top of each submitter script. Outputs
land under `results/{synthetic|real/binary|real/cifar10}/conf_<n>/<YYYYMMDD>/`,
and submitters skip jobs whose expected CSV already exists.

## Method choices in code

| Paper name | Code argument |
|---|---|
| Standard CP | `--methods standard` |
| Temperature scaling (T=1.5) | `--methods ts_fixed` |
| AdaTS | `--methods ts_adaptive` |
| Trust-score conditional CP | `--methods trustscore` |
| Retraining + CP | `--methods retrain` |
| ACP-MC | `--methods combscore` |
| ACP-ACC | `--methods condconf` |
| ACP-AEC | `--methods equalized` |
| AACP-MC / AACP-ACC / AACP-AEC | `--methods adaptive_combscore`, `adaptive_condconf`, `adaptive_equalized` |

The ACP-MC `η` reallocation rule is selected via
`eta_params={'eta_method': X, ...}` where `X ∈ {renormalize, learned, counts}`;
default is `counts` for multiclass and not used for binary.

## Reproducibility notes

- Each job is seeded by `--seed` (legacy training, calibration split, audit
  model, eval-set sampling). The data-pool sampler uses three independent RNGs
  derived from `seed`, `seed+1`, `seed+2` to keep `hist`, `new ∪ test`, and
  `eval` provably disjoint; this is asserted at runtime.
- Real-data results in the paper are averaged over 30 seeds
  (`SEED_LIST=$(seq 2000 2030)`).
- Synthetic results are averaged over 100 seeds.
- The audit model is a calibrated random forest by default. See
  `audit_model_params` in `experiments/run_*_experiment.py`.

## Citation

```bibtex
@article{audited_conformal_prediction,
  title  = {Audited Conformal Prediction for Classification under Unknown Distribution Shift},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review.}
}
```
