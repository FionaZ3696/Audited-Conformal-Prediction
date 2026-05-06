import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import ProbAccum, compute_nonconf_scores, calibrate_alpha, predict_set, romano_score_fn, romano_score_vector
from .thirdparty import CondConf_trustscore_modified, compute_trust_scores, build_phi, get_bin_edges
from .evaluation import evaluate_coverage, evaluate_coverage_with_auditing, evaluate_coverage_with_auditing_real
from .utils import (apply_temperature_scaling, find_optimal_temperature,
                          fit_platt_binary, predict_platt_binary,
                          fit_platt_multiclass, predict_platt_multiclass)


class BaselineConformalPrediction:
    def __init__(self, primary_model, alpha=0.1, generator = None, random_state=2025, verbose=True, method="standard", retraining_model = None, retraining_epoch = None):
        """
        Baseline conformal prediction using only the original model probabilities

        Args:
            alpha: Miscoverage rate (1-alpha is target coverage)
            method: str - "standard", "ts_fixed", or "ts_adaptive"
        """
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        self.primary_model = primary_model
        self.method = method
        self.T_optimal = None  # Store optimal temperature
        self.retraining_model = retraining_model
        self.retraining_epoch = retraining_epoch
        self.generator = generator

    def _conformal_calibrate(self, X_new_cal, y_new_cal, X_train = None, y_train = None,
                             X_new_train = None, y_new_train = None):
        """
        Calibrate conformal predictor using calibration set.

        Args:
            X_new_cal, y_new_cal: Calibration data for conformal score computation.
            X_train, y_train: Historical training data (used by trustscore).
            X_new_train, y_new_train: Pre-split training portion for methods
                that need it (ts_adaptive, platt, retrain). When provided,
                X_new_cal/y_new_cal is used directly for calibration without
                further splitting.
        """
        # Handle temperature scaling based on method

        if self.method == "standard":
            # No temperature scaling
            cal_proba = self.primary_model.predict_proba(X_new_cal)

        elif self.method == "oracle":
            assert self.generator is not None, "missing data generator for Oracle"
            cal_proba = self.generator.predict_proba(X_new_cal)

        elif self.method == "ts_fixed":
            # Fixed temperature T = 1.5
            cal_proba_raw = self.primary_model.predict_proba(X_new_cal)
            cal_proba = apply_temperature_scaling(cal_proba_raw, T=1.5)
            if self.verbose:
                print("Applied fixed temperature scaling with T = 1.5")

        elif self.method == "ts_adaptive":
            if X_new_train is not None:
                X_train_ts, y_train_ts = X_new_train, y_new_train
            else:
                X_train_ts, X_new_cal, y_train_ts, y_new_cal = train_test_split(
                    X_new_cal, y_new_cal,
                    test_size=0.5,
                    random_state=self.random_state,
                    stratify=y_new_cal
                )

            # Find optimal temperature using training split
            train_proba_raw = self.primary_model.predict_proba(X_train_ts)
            self.T_optimal = find_optimal_temperature(train_proba_raw, y_train_ts)

            # Apply temperature scaling to calibration set
            cal_proba_raw = self.primary_model.predict_proba(X_new_cal)
            cal_proba = apply_temperature_scaling(cal_proba_raw, self.T_optimal)

            if self.verbose:
                print(f"Found optimal temperature T = {self.T_optimal:.3f}")
                print(f"Using {len(X_train_ts)} samples for T optimization, {len(X_new_cal)} for conformal calibration")

        elif self.method == "platt":
            if X_new_train is not None:
                X_train_platt, y_train_platt = X_new_train, y_new_train
            else:
                X_train_platt, X_new_cal, y_train_platt, y_new_cal = train_test_split(
                    X_new_cal, y_new_cal,
                    test_size=0.5,
                    random_state=self.random_state,
                    stratify=y_new_cal
                )

            # Get probabilities for both splits
            train_proba = self.primary_model.predict_proba(X_train_platt)
            cal_proba_raw = self.primary_model.predict_proba(X_new_cal)
            n_classes = train_proba.shape[1]

            # Train Platt scaling on training split, apply to calibration set
            if n_classes == 2:
                platt_model = fit_platt_binary(train_proba[:,1], y_train_platt)
            else:
                platt_model = fit_platt_multiclass(train_proba, y_train_platt)

            self.platt_model = platt_model
            if n_classes == 2:
                cal_proba = predict_platt_binary(self.platt_model, cal_proba_raw[:,1])
            else:
                cal_proba = predict_platt_multiclass(self.platt_model, cal_proba_raw)

            if self.verbose:
                print(f"Using {len(X_train_platt)} samples for Platt training, {len(X_new_cal)} for conformal calibration")

        elif self.method == "retrain":
            assert self.retraining_model is not None, "retraining method requires model input"
            if X_new_train is not None:
                X_train_retrain, y_train_retrain = X_new_train, y_new_train
            else:
                X_train_retrain, X_new_cal, y_train_retrain, y_new_cal = train_test_split(
                    X_new_cal, y_new_cal,
                    test_size=0.5,
                    random_state=self.random_state,
                    stratify=y_new_cal
                )
            self.retraining_model.fit(X_train_retrain, y_train_retrain)

            try:
                cal_proba = self.retraining_model.predict_proba(X_new_cal)
            except:
                cal_proba = self.retraining_model.net.predict_proba(X_new_cal)

            if self.verbose:
                print(f"Using {len(X_train_retrain)} samples for retraining classifier, {len(X_new_cal)} for conformal calibration")
                y_hat_pred_temp = self.retraining_model.predict(X_new_cal)
                print(f"Accuracy of the retrained model = {np.mean(y_hat_pred_temp == y_new_cal)}")

        elif self.method == 'trustscore':
            self.num_bins = 3
            self.bin_strategy = "uniform"

            y_cal_pred   = self.primary_model.predict(X_new_cal)
            calib_scores = romano_score_fn(self.primary_model, X_new_cal, y_new_cal,
                                          random_state=self.random_state)
            calib_confidence = np.max(self.primary_model.predict_proba(X_new_cal), axis=1)

            # extract 2D embeddings for FAISS if raw images, else use as-is (tabular)
            X_train_feat   = self.primary_model.get_embeddings(X_train)   if X_train.ndim   > 2 else X_train
            X_new_cal_feat = self.primary_model.get_embeddings(X_new_cal) if X_new_cal.ndim > 2 else X_new_cal

            calib_trust_scores = compute_trust_scores(X_train_feat, y_train,
                                                      X_new_cal_feat, y_cal_pred)

            calib_conf_bins, calib_trust_bins, calib_final_conf_scores, calib_final_trust_scores = \
                get_bin_edges(calib_confidence, calib_trust_scores, self.num_bins, self.bin_strategy)

            phi_cal = build_phi(calib_final_conf_scores, calib_final_trust_scores,
                                calib_conf_bins, calib_trust_bins,
                                num_bins_conf=self.num_bins, num_bins_trust=self.num_bins)
            phi_cal = np.column_stack([np.ones((phi_cal.shape[0], 1)), phi_cal])

            self.cond_conf = CondConf_trustscore_modified(score_fn=None, Phi_fn=None)
            self.cond_conf.setup_problem_precomputed(None, phi_cal, calib_scores)
            return



        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'standard', 'ts_fixed', 'ts_adaptive', 'platt', 'retrain'")

        cal_scores = compute_nonconf_scores(cal_proba, y_new_cal, alpha=self.alpha, random_state=self.random_state)
        calibrated_threshold = calibrate_alpha(cal_scores, alpha=self.alpha)
        self.calibrated_threshold = calibrated_threshold

        if self.verbose:
            print(f'Calibrated threshold = {self.calibrated_threshold}')

    def _conformal_predict(self, X_test, y_test = None, X_train = None, y_train = None):
        """
        Make conformal prediction sets using baseline approach
        """
        # Apply same temperature scaling method as during calibration
        if self.method == "standard":
            test_proba = self.primary_model.predict_proba(X_test)

        elif self.method == "oracle":
            assert self.generator is not None, "missing data generator for Oracle"
            test_proba = self.generator.predict_proba(X_test)

        elif self.method == "ts_fixed":
            test_proba_raw = self.primary_model.predict_proba(X_test)
            test_proba = apply_temperature_scaling(test_proba_raw, T=1.5)

        elif self.method == "ts_adaptive":
            test_proba_raw = self.primary_model.predict_proba(X_test)
            test_proba = apply_temperature_scaling(test_proba_raw, self.T_optimal)

        elif self.method == "retrain":
            test_proba = self.retraining_model.predict_proba(X_test)

        elif self.method == "platt":

            test_proba_raw = self.primary_model.predict_proba(X_test)
            n_classes = test_proba_raw.shape[1]
            if n_classes == 2:
                test_proba = predict_platt_binary(self.platt_model, p_hat = test_proba_raw[:,1])
            else:
                test_proba = predict_platt_multiclass(self.platt_model, test_proba_raw)

        elif self.method == 'trustscore':
            S_hat = []

            test_y_pred      = self.primary_model.predict(X_test)
            test_confidence  = np.max(self.primary_model.predict_proba(X_test), axis=1)

            # extract 2D embeddings for FAISS if raw images, else use as-is (tabular)
            X_train_feat = self.primary_model.get_embeddings(X_train) if X_train.ndim > 2 else X_train
            X_test_feat  = self.primary_model.get_embeddings(X_test)  if X_test.ndim  > 2 else X_test

            test_trust_scores = compute_trust_scores(X_train_feat, y_train,
                                                    X_test_feat, test_y_pred)

            test_conf_bins, test_trust_bins, test_final_conf_scores, test_final_trust_scores = \
                get_bin_edges(test_confidence, test_trust_scores, self.num_bins, self.bin_strategy)

            phi_test = build_phi(test_final_conf_scores, test_final_trust_scores,
                                test_conf_bins, test_trust_bins,
                                num_bins_conf=self.num_bins, num_bins_trust=self.num_bins)

            scores_test = []
            for i in range(len(X_test)):
                x_i = X_test[i]
                test_score_i = romano_score_vector(self.primary_model, x_i, eps=None)
                scores_test.append(test_score_i)

            phi_test = np.column_stack([np.ones((phi_test.shape[0], 1)), phi_test])

            for i in range(len(X_test)):
                try:
                    S_hat_i = self.cond_conf.predict_input_adjusted(
                        quantile    = 1 - self.alpha,
                        x_test      = None,
                        phi_test    = phi_test[i, :],
                        scores_test = scores_test[i],
                        randomize   = False,
                        exact       = True,
                    )
                    S_hat.append(S_hat_i)
                except:
                    pass

                if i % 100 == 0:
                    print(f'finish processing {i}/{len(X_test)} test points')

            def convert_to_class_labels(prediction_sets):
                return [np.where(pred_set)[0] for pred_set in prediction_sets]

            S_hat = convert_to_class_labels(S_hat)
            return S_hat

        S_hat = predict_set(test_proba, self.calibrated_threshold, allow_empty=True, random_state=self.random_state)

        return S_hat, test_proba

    def run_baseline(self, X_cal, y_cal, X_test, y_test, generator, n_bins = 3, sample_types_test = None, epoch = None, X_train = None, y_train = None, r_hat_test = None,
                     X_new_train = None, y_new_train = None):

        # Perform conformal prediction
        if self.verbose:
            print(f"n_cal = {X_cal.shape[0]}, n_test = {X_test.shape[0]}")
            print(f"Calibrating method: {self.method}")

        # calibrating nonconformity scores
        if self.method == 'trustscore':
            assert X_train is not None and y_train is not None, "Trustscore method requires training data input"
            self._conformal_calibrate(X_cal, y_cal, X_train = X_train, y_train = y_train)
        else:
            self._conformal_calibrate(X_cal, y_cal, X_new_train=X_new_train, y_new_train=y_new_train)

        # predicting test labels from the primary model
        primary_model_preds = self.primary_model.predict(X_test)

        # building prediction sets
        if self.method == 'trustscore':
            prediction_sets = self._conformal_predict(X_test, X_train = X_train, y_train = y_train)
            if r_hat_test is None:
                results = evaluate_coverage_with_auditing(self.alpha, prediction_sets,
                                                          y_true = y_test, X_test = X_test,
                                                          primary_model_preds = primary_model_preds,
                                                          data_model = generator,
                                                          n_bins=n_bins,
                                                          sample_types=sample_types_test,
                                                          test_pred_prob=None)
            else:
                results = evaluate_coverage_with_auditing_real(self.alpha, prediction_sets,
                                                          y_true = y_test,
                                                          r_hat_test = r_hat_test,
                                                          n_bins = n_bins,
                                                          sample_types = sample_types_test,
                                                          test_pred_prob = None)

        else:
            prediction_sets, test_proba = self._conformal_predict(X_test)
            if r_hat_test is None:
                results = evaluate_coverage_with_auditing(self.alpha, prediction_sets,
                                                          y_true = y_test, X_test = X_test,
                                                          primary_model_preds = primary_model_preds,
                                                          data_model = generator,
                                                          n_bins=n_bins,
                                                          sample_types=sample_types_test,
                                                          test_pred_prob=test_proba)
            else:
                results = evaluate_coverage_with_auditing_real(self.alpha, prediction_sets,
                                                          y_true = y_test,
                                                          r_hat_test = r_hat_test,
                                                          n_bins = n_bins,
                                                          sample_types = sample_types_test,
                                                          test_pred_prob = test_proba)

        # Calculate elapsed time
        if self.method != 'retrain':
            results['Method'] = f'benchmark ({self.method})'
        elif self.method == 'retrain':
            results['Method'] = f'benchmark ({self.method}-{self.retraining_epoch})'

        results['eta_method'] = None
        results['equalized_threshold'] = None
        results['equalized_audit_est_acc'] = None
        results['condconf_threshold_list'] = None
        results['eta_adapter_C'] = None
        results['eta_smoothing_alpha'] = None

        if self.verbose:
            print(results)

        return pd.DataFrame([results])
