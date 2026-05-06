import argparse
import os
import sys
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add parent dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_models import Data_Model_ConceptShift_Uniform, Data_Model_CovariateShift_Uniform
from src.models import ClassNNet, BinaryClassification, Blackbox
from src.baseline_conformal import BaselineConformalPrediction
from src.audited_conformal import (
    Audited_Model,
    Audited_Conformal_Prediction,
    Adaptive_Audited_Conformal_Prediction,
)
from src.utils import (
    apply_temperature_scaling,
    print_confidence_diagnostics,
    compute_binned_accuracy,
    add_final_model_acc,
)


# ==================== Available methods ====================
# Baselines
#   'standard'      — standard split conformal (no adjustment)
#   'ts_fixed'      — temperature scaling with fixed T=1.5
#   'ts_adaptive'   — temperature scaling with learned T
#   'trustscore'    — CondConf with trust-score features
#   'retrain'       — retrain model from scratch on new data
# Non-adaptive ACP
#   'oracle'        — APA with oracle r*(x) (synthetic only)
#   'condconf'      — ACondConf with audit model
#   'equalized'     — AEqualized with audit model
#   'combscore'     — APA with audit model
# Adaptive AACP
#   'adaptive_combscore'   — AACP: combscore vs retraining
#   'adaptive_equalized'   — AACP: equalized vs retraining
#   'adaptive_condconf'    — AACP: condconf vs retraining

ALL_METHODS = [
    'standard', 'ts_fixed', 'ts_adaptive', 'trustscore', 'retrain',
    'oracle', 'condconf', 'equalized', 'combscore',
    'adaptive_combscore', 'adaptive_equalized', 'adaptive_condconf',
]

BASELINE_METHODS = {'standard', 'ts_fixed', 'ts_adaptive', 'trustscore', 'retrain'}
ACP_METHODS = {'oracle', 'condconf', 'equalized', 'combscore'}
ADAPTIVE_METHODS = {'adaptive_combscore', 'adaptive_equalized', 'adaptive_condconf'}

# Methods that require the audit model
NEEDS_AUDIT = ACP_METHODS - {'oracle'} | ADAPTIVE_METHODS
# Methods that require the retraining model
NEEDS_RETRAIN = {'retrain'} | ADAPTIVE_METHODS


def parse_args():
    parser = argparse.ArgumentParser(description="Run Audited Conformal Prediction experiment")

    parser.add_argument("--data_model", type=int, default=3, choices=[3, 5],
                        help="Data-generating process: 3=ConceptShift_Uniform, "
                             "5=CovariateShift_Uniform")
    parser.add_argument("--n_hist", type=int, default=10000,
                        help="Number of historical data points")
    parser.add_argument("--n_new", type=int, default=2000,
                        help="Number of new data points")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Beta parameter controlling shift proportion")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of classes")
    parser.add_argument("--p", type=int, default=100,
                        help="Feature dimension")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Nominal miscoverage level")
    parser.add_argument("--cal_prop", type=float, default=0.5,
                        help="Proportion of calibration data among n_new")
    parser.add_argument("--outdir", type=str, default="results/synthetic/",
                        help="Output directory for results")
    parser.add_argument("--modeldir", type=str, default=None,
                        help="Directory to save model checkpoints (None = don't save)")
    parser.add_argument("--feature_set", type=str, default="full",
                        help="Audit model feature set")
    parser.add_argument("--selection_threshold", type=float, default=0.7,
                        help="Selection epsilon threshold for adaptive methods (Eq. A17)")
    parser.add_argument("--equalized_threshold", type=float, default=0.7,
                        help="Group threshold t for the equalized method (Eq. 7)")
    parser.add_argument("--methods", type=str, nargs='+', default=None,
                        help="List of methods to run (default: all). "
                             "E.g. --methods adaptive_combscore adaptive_equalized")
    return parser.parse_args()

def main():
    args = parse_args()

    np.set_printoptions(
        threshold=np.inf,
        linewidth=np.inf,
        formatter={'float': lambda x: "{0:0.9f}".format(x)}
    )

    #########################
    # Experiment parameters #
    #########################

    alpha = args.alpha
    n_test = 1000
    p = args.p
    cal_prop = args.cal_prop
    data_model = args.data_model
    n_hist = args.n_hist
    n_new = args.n_new
    beta = args.beta
    K = args.K
    epochs = args.epochs
    seed = args.seed
    selection_threshold = args.selection_threshold
    equalized_threshold = args.equalized_threshold

    n_bins = [3, 11]

    # Determine which methods to run
    if args.methods is not None:
        methods_to_run = set(args.methods)
        unknown = methods_to_run - set(ALL_METHODS)
        if unknown:
            print(f"WARNING: Unknown methods ignored: {unknown}")
            methods_to_run -= unknown
    else:
        methods_to_run = set(ALL_METHODS)

    print(f"Methods to run: {sorted(methods_to_run)}")

    audit_model_params = {"feature_set": args.feature_set, "base_correctness_model": None}

    # Define different eta configurations to test
    eta_params_configs = {
        'learned': {
            'eta_method': 'learned',
            'eta_adapter_C': 0.1,
            'eta_smoothing_alpha': 1
        },
        'renormalize': {
            'eta_method': 'renormalize',
            'eta_adapter_C': 0.1,
            'eta_smoothing_alpha': 1
        },
        'counts': {
            'eta_method': 'counts',
            'eta_adapter_C': 0.1,
            'eta_smoothing_alpha': 1000
        },
    }


    ##############
    # Output file #
    ###############
    feature_set_name = audit_model_params.get('feature_set', 'custom')
    outfile_name = ("nhist" + str(n_hist) + "_nnew" + str(n_new) + "_beta" + str(beta) +
                    "_datamodel" + str(data_model) + "_K" + str(K) + "_epochs" + str(epochs) +
                    "_seed" + str(seed) + "_audit" + str(feature_set_name))

    outdir = args.outdir
    if args.modeldir is not None:
        modeldir = args.modeldir
    else:
        modeldir = "models/synthetic/" + outfile_name + "/"

    if not outdir.endswith('/'):
        outdir += '/'
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir + outfile_name + ".csv"
    print("Output file: {:s}".format(outfile), end="\n")

    #################
    # Generate Data #
    #################

    if data_model == 3:
        generator = Data_Model_ConceptShift_Uniform(K=K, p=p, random_state=seed)
        hist_X, hist_y = generator.generate_historical_data(n_samples=n_hist)
        new_X, new_y, sample_types = generator.generate_new_data(n_samples=n_new, beta=beta)
        test_X, test_y, test_sample_types = generator.generate_new_data(n_samples=n_test, beta=beta)

    elif data_model == 5:
        generator = Data_Model_CovariateShift_Uniform(K=K, p=p, random_state=seed)
        hist_X, hist_y = generator.generate_historical_data(n_samples=n_hist, beta=beta)
        new_X, new_y, sample_types = generator.generate_new_data(n_samples=n_new, beta=beta, a=0.5)
        test_X, test_y, test_sample_types = generator.generate_new_data(n_samples=n_test, beta=beta, a=0.5)

    else:
        raise ValueError(f"Unsupported data_model={data_model}; only 3 (ConceptShift_Uniform) "
                         f"and 5 (CovariateShift_Uniform) are supported.")


    #########################################
    # Train Old Model Using Historical Data #
    #########################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Is CUDA available? {}".format(torch.cuda.is_available()))

    if K == 2:
        net = BinaryClassification(num_features=p, device=device, use_dropout=False)
        criterion = nn.BCEWithLogitsLoss()
    else:
        net = ClassNNet(num_features=p, num_classes=K, device=device, use_dropout=False)
        criterion = nn.CrossEntropyLoss()

    lr = 0.001
    batch_size = 10
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # Training the model
    old_model = Blackbox(
        net=net,
        device=device,
        X_train=hist_X,
        y_train=hist_y,
        X_val=None,
        y_val=None,
        batch_size=batch_size,
        max_epoch=epochs,
        learning_rate=lr,
        criterion=criterion,
        optimizer=optimizer,
        compute_accuracy=True,
        verbose=True
    )

    old_model.fit(save_dir=None)

    # Historical accuracy (doesn't depend on a)
    old_model_hist_acc = accuracy_score(hist_y, old_model.net.predict(hist_X))
    print(f"Old model's accuracy on historical data: {old_model_hist_acc:.3f}")

    old_model_test_preds = old_model.net.predict(test_X)
    old_model_test_acc = accuracy_score(test_y, old_model_test_preds)
    print(f"Old model's accuracy on test data (overall): {old_model_test_acc:.3f}")

    # Oracle r*(x) on test set — used for binned accuracy
    r_star_test = generator.calculate_c_star(test_X, old_model_test_preds)
    old_model_test_acc_dict = compute_binned_accuracy(test_y, old_model_test_preds, r_star_test, n_bins)
    print(f"Old model test acc by r* bin: { {k: f'{v:.3f}' for k,v in old_model_test_acc_dict.items()} }")



    #############################
    # Apply conformal inference #
    #############################

    # Header for results file
    def add_header(df):
        df["n_hist"] = n_hist
        df["n_new"] = n_new
        df["seed"] = seed
        df["beta"] = beta
        df["data_model"] = data_model
        df['n_test'] = n_test
        df['p'] = p
        df['K'] = K
        df['alpha'] = alpha
        df['old_model_hist_acc'] = old_model_hist_acc
        df['old_model_test_acc_overall'] = old_model_test_acc_dict['acc_overall']
        for k, v in old_model_test_acc_dict.items():
            if k != 'acc_overall':
                df[f'old_model_test_{k}'] = v
        return df

    def add_audit_info(df, audit_model_info):
        """Add audit model information to results dataframe"""
        for key, value in audit_model_info.items():
            df[f'audit_{key}'] = value
        return df

    # Initialize result data frame
    results = pd.DataFrame({})

    # Precompute old model test probabilities (reused by several methods)
    old_model_test_proba = old_model.net.predict_proba(test_X)

    #########################################
    # Train Retraining Model (if needed)    #
    #########################################

    need_retrain = bool(methods_to_run & NEEDS_RETRAIN)
    epoch_retraining = 30
    retraining_model = None

    if need_retrain:
        print("\n=== Training Retraining Model ===")
        if K == 2:
            net_retraining = BinaryClassification(num_features=p, device=device, use_dropout=False)
            criterion_retraining = nn.BCEWithLogitsLoss()
        else:
            net_retraining = ClassNNet(num_features=p, num_classes=K, device=device, use_dropout=False)
            criterion_retraining = nn.CrossEntropyLoss()
        optimizer_retraining = optim.Adam(net_retraining.parameters(), lr=0.001, weight_decay=0)
        retraining_model = Blackbox(
            net=net_retraining,
            device=device,
            X_train=None,
            y_train=None,
            X_val=None,
            y_val=None,
            batch_size=batch_size,
            max_epoch=epoch_retraining,
            learning_rate=lr,
            criterion=criterion_retraining,
            optimizer=optimizer_retraining,
            compute_accuracy=True,
            verbose=True
        )

    #########################################
    # Split new data (if needed by ACP/AACP)#
    #########################################

    NEEDS_SPLIT = {'ts_adaptive', 'platt', 'retrain'} | ACP_METHODS | ADAPTIVE_METHODS
    need_split = bool(methods_to_run & NEEDS_SPLIT)
    X_new_train = X_new_cal = y_new_train = y_new_cal = sample_types_cal = None

    if need_split:
        X_new_train, X_new_cal, y_new_train, y_new_cal, _, sample_types_cal = train_test_split(
            new_X, new_y, sample_types, test_size=cal_prop, random_state=seed
        )

    #------------ Baseline Methods ------------------#

    if 'standard' in methods_to_run:
        print('\n=== Standard Conformal Prediction ===')
        baseline_method = BaselineConformalPrediction(primary_model=old_model.net, alpha=alpha, random_state=seed, method="standard")
        results_baseline = baseline_method.run_baseline(X_cal=new_X, y_cal=new_y, X_test=test_X, y_test=test_y,
                                                        generator=generator, n_bins=n_bins, sample_types_test=test_sample_types)  # standard uses all new data
        results_baseline = add_final_model_acc(results_baseline, old_model_test_proba, test_y, r_star_test, n_bins)
        results = pd.concat([results, results_baseline])

    if 'ts_fixed' in methods_to_run:
        print('\n=== Temperature Scaling (Fixed T=1.5) ===')
        baseline_method_tsf = BaselineConformalPrediction(primary_model=old_model.net, alpha=alpha, random_state=seed, method="ts_fixed")
        results_ts_baseline = baseline_method_tsf.run_baseline(X_cal=new_X, y_cal=new_y, X_test=test_X, y_test=test_y,
                                                               generator=generator, n_bins=n_bins, sample_types_test=test_sample_types)
        tsf_proba_test = apply_temperature_scaling(old_model_test_proba, T=1.5)
        results_ts_baseline = add_final_model_acc(results_ts_baseline, tsf_proba_test, test_y, r_star_test, n_bins)
        results = pd.concat([results, results_ts_baseline])

    if 'ts_adaptive' in methods_to_run:
        print('\n=== Temperature Scaling (Adaptive T) ===')
        baseline_method_tsa = BaselineConformalPrediction(primary_model=old_model.net, alpha=alpha, random_state=seed, method="ts_adaptive")
        results_tsa_baseline = baseline_method_tsa.run_baseline(X_cal=X_new_cal, y_cal=y_new_cal, X_test=test_X, y_test=test_y,
                                                                generator=generator, n_bins=n_bins, sample_types_test=test_sample_types,
                                                                X_new_train=X_new_train, y_new_train=y_new_train)
        tsa_proba_test = apply_temperature_scaling(old_model_test_proba, T=baseline_method_tsa.T_optimal)
        results_tsa_baseline = add_final_model_acc(results_tsa_baseline, tsa_proba_test, test_y, r_star_test, n_bins)
        results = pd.concat([results, results_tsa_baseline])

    if 'trustscore' in methods_to_run:
        print('\n=== TrustScore ===')
        baseline_method_ts = BaselineConformalPrediction(primary_model=old_model.net, alpha=alpha, random_state=seed, method="trustscore")
        results_trustscore = baseline_method_ts.run_baseline(X_cal=new_X, y_cal=new_y, X_test=test_X, y_test=test_y,
                                                             generator=generator, n_bins=n_bins, sample_types_test=test_sample_types, X_train=hist_X, y_train=hist_y)
        results_trustscore = add_final_model_acc(results_trustscore, old_model_test_proba, test_y, r_star_test, n_bins)
        results = pd.concat([results, results_trustscore])

    if 'retrain' in methods_to_run:
        print('\n=== Retraining Baseline ===')
        baseline_method_retr = BaselineConformalPrediction(primary_model=old_model, retraining_model=retraining_model, alpha=alpha, random_state=seed, method="retrain", retraining_epoch=epoch_retraining)
        results_retrain_baseline = baseline_method_retr.run_baseline(X_cal=X_new_cal, y_cal=y_new_cal, X_test=test_X, y_test=test_y, generator=generator,
                                                                      n_bins=n_bins, sample_types_test=test_sample_types, epoch=epoch_retraining,
                                                                      X_new_train=X_new_train, y_new_train=y_new_train)
        retrain_proba_test = retraining_model.net.predict_proba(test_X)
        results_retrain_baseline = add_final_model_acc(results_retrain_baseline, retrain_proba_test, test_y, r_star_test, n_bins)
        results = pd.concat([results, results_retrain_baseline])

        print_confidence_diagnostics(
            label=f"Retrain (ep={epoch_retraining}, lr=0.001, wd=0)",
            proba=retrain_proba_test,
            test_sample_types=test_sample_types,
            K=K
        )

    #------------ Non-Adaptive ACP Methods ------------------#

    if 'oracle' in methods_to_run:
        print('\n=== Oracle APA ===')
        acp_oracle = Audited_Conformal_Prediction(
            primary_model=old_model.net,
            audit_model=None,
            K=K,
            alpha=alpha,
            random_state=seed
        )
        results_oracle = acp_oracle.run_audited_conformal_prediction(
            method='oracle',
            X_new_cal=new_X,
            y_new_cal=new_y,
            sample_types_cal=sample_types,
            n_bins=n_bins,
            X_test=test_X,
            y_test=test_y,
            sample_types_test=test_sample_types,
            data_generator=generator
        )
        _, oracle_p_tilde_test = acp_oracle._conformal_predict_oracle(test_X, generator)
        results_oracle = add_final_model_acc(results_oracle, oracle_p_tilde_test, test_y, r_star_test, n_bins)
        results = pd.concat([results, results_oracle])

    # ---- Train Audit Model (shared by condconf, equalized, combscore, adaptive_*) ----
    audit = None
    audit_model_info = None

    if methods_to_run & NEEDS_AUDIT:
        print("\n=== Training Audit Model ===")
        print(f"Configuration: {audit_model_params}")

        audit = Audited_Model(
            primary_model=old_model.net,
            audit_model_params=audit_model_params,
            random_state=seed,
            verbose=True
        )
        feature_importances, audit_acc = audit.train_evaluate_audit_model(
            hist_X, hist_y, X_new_train, y_new_train, X_new_cal, y_new_cal
        )
        print(f"Audit model accuracy: {audit_acc:.4f}")

        if sample_types_cal is not None:
            primary_correct_cal = (old_model.net.predict(X_new_cal) == y_new_cal).astype(int)
            audit_preds_cal = audit.audit_model.predict(X_new_cal)
            hard_mask_cal = sample_types_cal == 0
            easy_mask_cal = sample_types_cal == 1
            if hard_mask_cal.any():
                audit_acc_hard = accuracy_score(primary_correct_cal[hard_mask_cal], audit_preds_cal[hard_mask_cal])
                print(f"Audit model accuracy (hard samples): {audit_acc_hard:.4f}")
            if easy_mask_cal.any():
                audit_acc_easy = accuracy_score(primary_correct_cal[easy_mask_cal], audit_preds_cal[easy_mask_cal])
                print(f"Audit model accuracy (easy samples): {audit_acc_easy:.4f}")

        audit_model_info = {
            **audit_model_params,
            'audit_acc': audit_acc
        }
        if feature_importances is not None:
            audit_model_info.update(feature_importances)
        else:
            audit_model_info.update({
                'basic_feature_importance': None,
                'distributional_shift_feature_importance': None,
                'distance_feature_importance': None,
                'model_agg_feature_importance': None,
                'shortcut_feature_importance': None
            })

    if 'condconf' in methods_to_run:
        print("\n=== Running CondConf Method ===")
        try:
            acp_condconf = Audited_Conformal_Prediction(
                primary_model=old_model.net,
                audit_model=audit.audit_model,
                K=K,
                alpha=alpha,
                random_state=seed
            )
            results_condconf = acp_condconf.run_audited_conformal_prediction(
                method='condconf',
                X_new_cal=X_new_cal,
                y_new_cal=y_new_cal,
                sample_types_cal=sample_types_cal,
                n_bins=n_bins,
                X_test=test_X,
                y_test=test_y,
                sample_types_test=test_sample_types,
                threshold_list=[0.5, 0.6, 0.7, 0.8, 0.9],
                data_generator=generator
            )
            results_condconf = add_audit_info(results_condconf, audit_model_info)
            results_condconf = add_final_model_acc(results_condconf, old_model_test_proba, test_y, r_star_test, n_bins)
            results = pd.concat([results, results_condconf])
            print("CondConf method completed successfully")
        except Exception as e:
            print(f"CondConf method failed with error: {e}")
            print("Skipping CondConf method")

    if 'equalized' in methods_to_run:
        print("\n=== Running Equalized Method ===")
        for threshold_ in [0.5, 0.6, 0.7, 0.8, 0.9]:
            print(f"\nTesting threshold: {threshold_}")
            acp_equalized = Audited_Conformal_Prediction(
                primary_model=old_model.net,
                audit_model=audit.audit_model,
                K=K,
                alpha=alpha,
                random_state=seed
            )
            results_equalized = acp_equalized.run_audited_conformal_prediction(
                method='equalized',
                X_new_cal=X_new_cal,
                y_new_cal=y_new_cal,
                sample_types_cal=sample_types_cal,
                n_bins=n_bins,
                X_test=test_X,
                y_test=test_y,
                sample_types_test=test_sample_types,
                threshold=threshold_,
                data_generator=generator
            )
            results_equalized = add_audit_info(results_equalized, audit_model_info)
            results_equalized = add_final_model_acc(results_equalized, old_model_test_proba, test_y, r_star_test, n_bins)
            results = pd.concat([results, results_equalized])

    if 'combscore' in methods_to_run:
        print("\n=== Running CombScore Method ===")
        if K == 2:
            print("\n--- CombScore (binary APA, no eta) ---")
            acp_combscore = Audited_Conformal_Prediction(
                primary_model=old_model.net,
                audit_model=audit.audit_model,
                K=K,
                alpha=alpha,
                random_state=seed
            )
            results_combscore = acp_combscore.run_audited_conformal_prediction(
                method='combscore',
                X_new_cal=X_new_cal,
                y_new_cal=y_new_cal,
                sample_types_cal=sample_types_cal,
                n_bins=n_bins,
                X_test=test_X,
                y_test=test_y,
                sample_types_test=test_sample_types,
                eta_params=None,
                X_new_train=X_new_train,
                y_new_train=y_new_train,
                data_generator=generator
            )
            results_combscore = add_audit_info(results_combscore, audit_model_info)
            _, apa_p_tilde_test = acp_combscore._conformal_predict_combscore(test_X)
            results_combscore = add_final_model_acc(results_combscore, apa_p_tilde_test, test_y, r_star_test, n_bins)
            results = pd.concat([results, results_combscore])
        else:
            print("(Different Eta Configurations)")
            for eta_name, eta_params in eta_params_configs.items():
                print(f"\n--- CombScore with eta_method={eta_name} ---")
                acp_combscore = Audited_Conformal_Prediction(
                    primary_model=old_model.net,
                    audit_model=audit.audit_model,
                    K=K,
                    alpha=alpha,
                    random_state=seed
                )
                results_combscore = acp_combscore.run_audited_conformal_prediction(
                    method='combscore',
                    X_new_cal=X_new_cal,
                    y_new_cal=y_new_cal,
                    sample_types_cal=sample_types_cal,
                    n_bins=n_bins,
                    X_test=test_X,
                    y_test=test_y,
                    sample_types_test=test_sample_types,
                    eta_params=eta_params,
                    X_new_train=X_new_train,
                    y_new_train=y_new_train,
                    data_generator=generator
                )
                results_combscore = add_audit_info(results_combscore, audit_model_info)
                _, apa_p_tilde_test = acp_combscore._conformal_predict_combscore(test_X)
                results_combscore = add_final_model_acc(results_combscore, apa_p_tilde_test, test_y, r_star_test, n_bins)
                results = pd.concat([results, results_combscore])

            print_confidence_diagnostics(
                label=f"APA-{eta_name} (adjusted p_tilde)",
                proba=apa_p_tilde_test,
                test_sample_types=test_sample_types,
                K=K
            )

    ####################################################
    #              Adaptive Methods (AACP)             #
    ####################################################

    run_any_adaptive = bool(methods_to_run & ADAPTIVE_METHODS)
    sel_criterion_ = ['condcov'] #, 'size'


    if run_any_adaptive:
        # retraining_model is already trained on X_new_train (via baseline retrain),
        # which is disjoint from X_new_cal — no data contamination.
        assert retraining_model is not None, \
            "Adaptive methods require 'retrain' in methods_to_run (retraining model must be trained first)"
        print("\n=== Running Adaptive Methods ===")
        print(f"Selection epsilon: {selection_threshold}")
        print(f"Selection criterion: {sel_criterion_}")
    # --- Adaptive CombScore ---
    if 'adaptive_combscore' in methods_to_run:
        print("\n=== Adaptive CombScore vs Retraining ===")

        if K == 2:
            # Binary: single run, no eta
            for sel_criterion in sel_criterion_:
                print(f"\n--- epsilon={selection_threshold}, criterion={sel_criterion} ---")
                adaptive_acp = Adaptive_Audited_Conformal_Prediction(
                    primary_model=old_model.net,
                    audit_model=audit.audit_model,
                    retraining_model=retraining_model,
                    K=K,
                    alpha=alpha,
                    random_state=seed,
                    calib_cal_size=0.75,
                    selection_epsilon=selection_threshold,
                    retraining_epoch=epoch_retraining,
                    selection_criterion=sel_criterion
                )
                results_adaptive = adaptive_acp.run_adaptive_audited_conformal_prediction(
                    audited_method='combscore',
                    X_new_cal=X_new_cal,
                    y_new_cal=y_new_cal,
                    sample_types_cal=sample_types_cal,
                    n_bins=n_bins,
                    X_test=test_X,
                    y_test=test_y,
                    sample_types_test=test_sample_types,
                    eta_params=None,
                    X_new_train=X_new_train,
                    y_new_train=y_new_train,
                    data_generator=generator
                )
                results_adaptive = add_audit_info(results_adaptive, audit_model_info)
                results_adaptive = add_final_model_acc(results_adaptive, None, test_y, r_star_test, n_bins)
                results = pd.concat([results, results_adaptive])
        else:
            # Multiclass: loop over eta configs
            eta_params_configs_adapt = {
                'counts': {
                    'eta_method': 'counts',
                    'eta_adapter_C': 0.1,
                    'eta_smoothing_alpha': 1000
                }
            }

            for eta_name, eta_params_base in eta_params_configs_adapt.items():
                eta_params_adaptive = {**eta_params_base, 'eta_smoothing_alpha': 1000}

                for sel_criterion in sel_criterion_:
                    print(f"\n--- eta={eta_name}, epsilon={selection_threshold}, criterion={sel_criterion} ---")
                    adaptive_acp = Adaptive_Audited_Conformal_Prediction(
                        primary_model=old_model.net,
                        audit_model=audit.audit_model,
                        retraining_model=retraining_model,
                        K=K,
                        alpha=alpha,
                        random_state=seed,
                        calib_cal_size=0.75,
                        selection_epsilon=selection_threshold,
                        retraining_epoch=epoch_retraining,
                        selection_criterion=sel_criterion
                    )
                    results_adaptive = adaptive_acp.run_adaptive_audited_conformal_prediction(
                        audited_method='combscore',
                        X_new_cal=X_new_cal,
                        y_new_cal=y_new_cal,
                        sample_types_cal=sample_types_cal,
                        n_bins=n_bins,
                        X_test=test_X,
                        y_test=test_y,
                        sample_types_test=test_sample_types,
                        eta_params=eta_params_adaptive,
                        X_new_train=X_new_train,
                        y_new_train=y_new_train,
                        data_generator=generator
                    )
                    results_adaptive = add_audit_info(results_adaptive, audit_model_info)
                    results_adaptive = add_final_model_acc(results_adaptive, None, test_y, r_star_test, n_bins)
                    results = pd.concat([results, results_adaptive])

    # --- Adaptive Equalized ---
    if 'adaptive_equalized' in methods_to_run:
        print("\n=== Adaptive Equalized vs Retraining ===")
        for sel_criterion in sel_criterion_:
            print(f"\n--- epsilon={selection_threshold}, criterion={sel_criterion} ---")
            adaptive_acp = Adaptive_Audited_Conformal_Prediction(
                primary_model=old_model.net,
                audit_model=audit.audit_model,
                retraining_model=retraining_model,
                K=K,
                alpha=alpha,
                random_state=seed,
                calib_cal_size=0.75,
                selection_epsilon=selection_threshold,
                retraining_epoch=epoch_retraining,
                selection_criterion=sel_criterion
            )
            results_adaptive = adaptive_acp.run_adaptive_audited_conformal_prediction(
                audited_method='equalized',
                X_new_cal=X_new_cal,
                y_new_cal=y_new_cal,
                sample_types_cal=sample_types_cal,
                n_bins=n_bins,
                X_test=test_X,
                y_test=test_y,
                sample_types_test=test_sample_types,
                threshold=equalized_threshold,
                data_generator=generator,
                X_new_train=X_new_train,
                y_new_train=y_new_train
            )
            results_adaptive = add_audit_info(results_adaptive, audit_model_info)
            results_adaptive = add_final_model_acc(results_adaptive, None, test_y, r_star_test, n_bins)
            results = pd.concat([results, results_adaptive])

    # --- Adaptive CondConf ---
    if 'adaptive_condconf' in methods_to_run:
        print("\n=== Adaptive CondConf vs Retraining ===")
        for sel_criterion in sel_criterion_:
            print(f"\n--- epsilon={selection_threshold}, criterion={sel_criterion} ---")
            try:
                adaptive_acp = Adaptive_Audited_Conformal_Prediction(
                    primary_model=old_model.net,
                    audit_model=audit.audit_model,
                    retraining_model=retraining_model,
                    K=K,
                    alpha=alpha,
                    random_state=seed,
                    calib_cal_size=0.75,
                    selection_epsilon=selection_threshold,
                    retraining_epoch=epoch_retraining,
                    selection_criterion=sel_criterion
                )
                results_adaptive = adaptive_acp.run_adaptive_audited_conformal_prediction(
                    audited_method='condconf',
                    X_new_cal=X_new_cal,
                    y_new_cal=y_new_cal,
                    sample_types_cal=sample_types_cal,
                    n_bins=n_bins,
                    X_test=test_X,
                    y_test=test_y,
                    sample_types_test=test_sample_types,
                    threshold_list=[0.5, 0.6, 0.7, 0.8, 0.9],
                    data_generator=generator,
                    X_new_train=X_new_train,
                    y_new_train=y_new_train
                )
                results_adaptive = add_audit_info(results_adaptive, audit_model_info)
                results_adaptive = add_final_model_acc(results_adaptive, None, test_y, r_star_test, n_bins)
                results = pd.concat([results, results_adaptive])
                print(f"Adaptive CondConf (criterion={sel_criterion}) completed successfully")
            except Exception as e:
                print(f"Adaptive CondConf (criterion={sel_criterion}) failed with error: {e}")

    ################
    # Save Results #
    ################
    results = add_header(results)
    results.to_csv(outfile, index=False)
    print("\nResults written to {:s}\n".format(outfile))
    sys.stdout.flush()

    if args.modeldir is not None:
        shutil.rmtree(modeldir, ignore_errors=True)


if __name__ == "__main__":
    main()
