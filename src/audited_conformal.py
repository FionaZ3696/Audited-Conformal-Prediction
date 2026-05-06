import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings("ignore")

from .utils import ProbAccum, compute_nonconf_scores, calibrate_alpha, predict_set, romano_score_fn, romano_score_vector
from .thirdparty import CondConf, CondConf_trustscore_modified, compute_trust_scores, build_phi, get_bin_edges
from .utils import fit_platt_multiclass, predict_platt_multiclass, fit_platt_binary, predict_platt_binary, apply_temperature_scaling, find_optimal_temperature, indicator_matrix_overlapping_thresholds
from .evaluation import evaluate_coverage, evaluate_coverage_with_auditing, evaluate_coverage_with_auditing_real
from .models import BinaryClassification, Blackbox


class AuditClassifier:
    """
    Consolidated Audit Classifier.

    Important Features:
    1. Original feature extraction (Basic, Shift).
    2. Deep Embedding distances (replacing/enhancing PCA distances).
    3. MC Dropout Uncertainty (replacing/enhancing Tree variance).
    4. Trust Scores/Ratios.

    No inheritance. Standalone implementation.
    """

    def __init__(self, old_model, hist_X, hist_y,
                 base_correctness_model=None,
                 apply_pca=False,
                 apply_standardization=True,
                 use_embeddings = True,
                 use_basic_prediction_features=True,
                 use_distributional_shift_features=True,
                 use_distance_features=True,
                 use_class_conditional_features=True,
                 use_uncertainty_features=True,
                 use_density_features=True,
                 dnn_training_params=None,  # <-- NEW
                 random_state=42):

        self.old_model = old_model
        self.random_state = random_state
        self.apply_standardization = apply_standardization
        self.apply_pca = apply_pca
        self.base_correctness_model = base_correctness_model
        self.use_original_feature = True
        self.dnn_training_params = dnn_training_params or {}  # <-- NEW

        # Feature flags
        self.use_basic_prediction_features = use_basic_prediction_features
        self.use_distributional_shift_features = use_distributional_shift_features
        self.use_distance_features = use_distance_features
        self.use_class_conditional_features = use_class_conditional_features
        self.use_uncertainty_features = use_uncertainty_features
        self.use_density_features = use_density_features
        self.use_embeddings = use_embeddings

        # Determine number of classes
        self.n_classes = len(np.unique(hist_y))
        self.hist_y = hist_y

        print(f"Initializing Audit Classifier.")
        print(f"Features: Basic={self.use_basic_prediction_features}, Shift={self.use_distributional_shift_features}, "
              f"Dist={self.use_distance_features}, ClassCond={self.use_class_conditional_features}, "
              f"Uncertainty={self.use_uncertainty_features}")

        # 1. Standardize Raw Features
        if self.apply_standardization:
            self.scaler = StandardScaler()
            self.hist_X_scaled = self.scaler.fit_transform(hist_X)
        else:
            self.scaler = None
            self.hist_X_scaled = hist_X.copy()

        # 2. Fit Outlier Detector (on scaled raw features)
        if self.use_distributional_shift_features:
            self.outlier_detector = IsolationForest(
                contamination=0.5,
                random_state=random_state,
                n_estimators=100
            )
            self.outlier_detector.fit(self.hist_X_scaled)

        # 3. Prepare Latent Space for Distances (Embeddings vs PCA)
        # Check if we can get Deep Embeddings (NEW FEATURE)
        if self.use_embeddings:
            use_embeddings = hasattr(self.old_model, 'get_embeddings')

        if use_embeddings:
            print(">> Detected 'get_embeddings': Using Deep Representation for distance features.")
            self.hist_latent = self.old_model.get_embeddings(hist_X)
            # Normalize embeddings for better distance metrics (Cosine-like behavior in Euclidean space)
            self.embed_scaler = StandardScaler()
            self.hist_latent = self.embed_scaler.fit_transform(self.hist_latent)
        elif self.apply_pca:
            print(">> No embeddings found: Falling back to PCA.")
            self.n_pc = 5
            self.pca = PCA(n_components=min(self.n_pc, hist_X.shape[1]))
            self.hist_latent = self.pca.fit_transform(self.hist_X_scaled)
        else:
            print(">> No embeddings or PCA: Using Scaled Raw Features for distances.")
            self.hist_latent = self.hist_X_scaled

        # 4. Pre-compute Historical Statistics on the Latent Space
        self._compute_historical_stats()

        self.correctness_model = None
        self.last_feature_info = None

    def _compute_historical_stats(self):
        """Pre-compute centroids and fit KNN on the Latent Space (Embeddings or PCA)"""

        # Distance/Density logic (Uses Latent Space: Embeddings or PCA)
        if self.use_distance_features or self.use_class_conditional_features or self.use_density_features:
            # Overall Centroid
            self.hist_centroid = np.mean(self.hist_latent, axis=0)

            # Overall KNN
            self.nn_overall = NearestNeighbors(n_neighbors=5)
            self.nn_overall.fit(self.hist_latent)

            # Class-conditional stats
            if self.use_class_conditional_features:
                self.hist_class_centroids = {}
                self.nn_by_class = {}
                self.hist_latent_by_class = {}

                for k in range(self.n_classes):
                    mask = self.hist_y == k
                    if np.any(mask):
                        data_k = self.hist_latent[mask]
                        self.hist_latent_by_class[k] = data_k
                        self.hist_class_centroids[k] = np.mean(data_k, axis=0)

                        # Fit Class KNN
                        if len(data_k) > 0:
                            n_neighbors = min(5, len(data_k))
                            self.nn_by_class[k] = NearestNeighbors(n_neighbors=n_neighbors)
                            self.nn_by_class[k].fit(data_k)
                    else:
                        self.hist_class_centroids[k] = self.hist_centroid
                        self.hist_latent_by_class[k] = self.hist_latent

    def extract_correctness_features(self, X_new):
        """Extract all features for correctness prediction."""
        n_samples = X_new.shape[0]

        # 1. Preprocess X_new
        if self.apply_standardization:
            X_new_scaled = self.scaler.transform(X_new)
        else:
            X_new_scaled = X_new.copy()

        # 2. Get Latent Representation (Embedding or PCA)
        if self.use_embeddings:
            X_new_latent = self.old_model.get_embeddings(X_new)
            X_new_latent = self.embed_scaler.transform(X_new_latent)
        elif self.apply_pca:
            X_new_latent = self.pca.transform(X_new_scaled)
        else:
            X_new_latent = X_new_scaled

        feature_arrays = []
        feature_info = {'feature_names': [], 'feature_categories': []}

        # Get Predictions
        pred_proba = self.old_model.predict_proba(X_new)
        pred_labels = np.argmax(pred_proba, axis=1)

        # --- GROUP 1: Basic Prediction Features ---
        if self.use_basic_prediction_features:
            max_proba = np.max(pred_proba, axis=1)
            pred_entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-10), axis=1)
            sorted_proba = np.sort(pred_proba, axis=1)
            second_max_proba = sorted_proba[:, -2] if pred_proba.shape[1] > 1 else np.zeros(n_samples)
            confidence_margin = max_proba - second_max_proba
            pred_class_proba = pred_proba[np.arange(n_samples), pred_labels]

            basic_features = np.column_stack([
                max_proba, pred_entropy, confidence_margin, second_max_proba, pred_class_proba
            ])
            names = ['max_proba', 'pred_entropy', 'confidence_margin', 'second_max_proba', 'pred_class_proba']

            feature_arrays.append(basic_features)
            feature_info['feature_names'].extend(names)
            feature_info['feature_categories'].extend(['basic'] * len(names))

        # --- GROUP 2: Shift Features ---
        if self.use_distributional_shift_features:
            outlier_scores = self.outlier_detector.decision_function(X_new_scaled)
            shift_features = np.column_stack([
                outlier_scores,
                -outlier_scores,
                (outlier_scores < 0).astype(float)
            ])
            names = ['outlier_scores', 'outlier_severity', 'is_outlier']
            feature_arrays.append(shift_features)
            feature_info['feature_names'].extend(names)
            feature_info['feature_categories'].extend(['distributional_shift'] * len(names))

        # --- GROUP 3: Distance Features (Deep or PCA) ---
        if self.use_distance_features:
            dist_centroid = np.linalg.norm(X_new_latent - self.hist_centroid, axis=1)
            dists, _ = self.nn_overall.kneighbors(X_new_latent)
            min_dist = dists[:, 0]
            avg_dist = np.mean(dists, axis=1)

            dist_features = np.column_stack([dist_centroid, min_dist, avg_dist])
            names = ['dist_centroid', 'min_dist_hist', 'avg_dist_hist']

            if self.use_embeddings: names = ['emb_' + n for n in names]

            feature_arrays.append(dist_features)
            feature_info['feature_names'].extend(names)
            feature_info['feature_categories'].extend(['distance'] * len(names))

        # --- GROUP 4: Class Conditional & Trust Ratio (Deep or PCA) ---
        if self.use_class_conditional_features:
            # A. Dist to Predicted Centroid
            dist_to_pred_centroid = np.zeros(n_samples)
            dist_to_nearest_other_centroid = np.full(n_samples, np.inf)

            # Calculate distances to all centroids first
            all_dists = np.zeros((n_samples, self.n_classes))
            for k in range(self.n_classes):
                all_dists[:, k] = np.linalg.norm(X_new_latent - self.hist_class_centroids[k], axis=1)

            # Assign specific distances
            for i in range(n_samples):
                k_pred = pred_labels[i]
                dist_to_pred_centroid[i] = all_dists[i, k_pred]

                # Find nearest OTHER class
                other_dists = all_dists[i, :].copy()
                other_dists[k_pred] = np.inf
                dist_to_nearest_other_centroid[i] = np.min(other_dists)

            # B. Trust Ratio (Distance to Nearest Other / Distance to Predicted)
            # Low ratio = Predicted is far, Other is close -> Likely Wrong
            trust_ratio = dist_to_nearest_other_centroid / (dist_to_pred_centroid + 1e-10)

            # C. KNN Distances (Local Density per class)
            dist_to_pred_knn = np.zeros(n_samples)
            dist_to_nearest_other_knn = np.full(n_samples, np.inf)

            for i in range(n_samples):
                k_pred = pred_labels[i]

                # KNN to Predicted Class
                if k_pred in self.nn_by_class:
                    d, _ = self.nn_by_class[k_pred].kneighbors([X_new_latent[i]])
                    dist_to_pred_knn[i] = d[0,0]

                # KNN to Nearest Other Class
                min_d_other = np.inf
                for k in range(self.n_classes):
                    if k != k_pred and k in self.nn_by_class:
                        d, _ = self.nn_by_class[k].kneighbors([X_new_latent[i]])
                        if d[0,0] < min_d_other:
                            min_d_other = d[0,0]
                dist_to_nearest_other_knn[i] = min_d_other if min_d_other != np.inf else 0

            class_features = np.column_stack([
                dist_to_pred_centroid,
                dist_to_nearest_other_centroid,
                trust_ratio,
                dist_to_pred_knn,
                dist_to_nearest_other_knn
            ])
            names = ['dist_pred_centroid', 'dist_other_centroid', 'trust_ratio', 'dist_pred_knn', 'dist_other_knn']
            if self.use_embeddings: names = ['emb_' + n for n in names]

            # Add raw distances to all classes (legacy support)
            class_features = np.hstack([class_features, all_dists])
            for k in range(self.n_classes): names.append(f'dist_class_{k}')

            feature_arrays.append(class_features)
            feature_info['feature_names'].extend(names)
            feature_info['feature_categories'].extend(['class_conditional'] * len(names))

        # --- GROUP 5: Uncertainty (MC Dropout or Tree) ---
        if self.use_uncertainty_features:
            unc_features = self._extract_uncertainty_features(X_new, pred_labels)
            # Function handles returning 2 columns (MCD) or 3 columns (Tree)
            if unc_features.shape[1] == 2:
                names = ['mcd_variance', 'mcd_entropy']
            else:
                names = ['tree_variance', 'tree_agreement', 'tree_entropy']

            feature_arrays.append(unc_features)
            feature_info['feature_names'].extend(names)
            feature_info['feature_categories'].extend(['uncertainty'] * len(names))

        # --- GROUP 6: Density (Latent) ---
        if self.use_density_features:
            k = min(10, len(self.hist_latent) // 10)
            if k > 0:
                nn_density = NearestNeighbors(n_neighbors=k).fit(self.hist_latent)
                dists, _ = nn_density.kneighbors(X_new_latent)
                local_density = 1.0 / (dists[:, -1] + 1e-10)
                avg_density = 1.0 / (np.mean(dists, axis=1) + 1e-10)
            else:
                local_density = np.ones(n_samples); avg_density = np.ones(n_samples)

            names = ['local_density', 'avg_local_density']
            if self.use_embeddings: names = ['emb_' + n for n in names]

            feature_arrays.append(np.column_stack([local_density, avg_density]))
            feature_info['feature_names'].extend(names)
            feature_info['feature_categories'].extend(['density'] * len(names))

        # Combine
        if feature_arrays:
            features = np.column_stack(feature_arrays)
        else:
            features = np.ones((n_samples, 1))
            feature_info['feature_names'] = ['dummy']
            feature_info['feature_categories'] = ['dummy']

        self.last_feature_info = feature_info
        return features

    def _extract_uncertainty_features(self, X_new, pred_labels):
        """
        Extracts MC Dropout (if available) OR Tree Variance (fallback).
        """
        # 1. MC Dropout check
        if hasattr(self.old_model, 'predict_uncertainty_mcd'):
            mean_entropy, pred_variance = self.old_model.predict_uncertainty_mcd(X_new)
            return np.column_stack([pred_variance, mean_entropy])

        # 2. Tree Variance check
        elif hasattr(self.old_model, 'estimators_'):
            tree_preds = np.array([tree.predict(X_new) for tree in self.old_model.estimators_])
            n_samples = len(pred_labels)
            var, agree, ent = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)

            for i in range(n_samples):
                preds_i = tree_preds[:, i]
                agree[i] = np.mean(preds_i == pred_labels[i])
                uni, counts = np.unique(preds_i, return_counts=True)
                probs = counts / len(preds_i)
                ent[i] = -np.sum(probs * np.log(probs + 1e-10))
                var[i] = 1 - np.max(probs)

            return np.column_stack([var, agree, ent])

        # 3. Fallback
        else:
            return np.zeros((len(pred_labels), 3))

    def fit(self, X_new_train, y_new_train, calibrate=True):
        print("Training Improved Auditing Classifier...")

        features = self.extract_correctness_features(X_new_train)

        pred_labels = self.old_model.predict(X_new_train)
        correctness = (pred_labels == y_new_train).astype(int)

        print(f"Features: {features.shape[1]} dims. Correctness Rate: {np.mean(correctness):.3f}")

        if self.use_original_feature:
            features = np.hstack([features, X_new_train])

        if self.base_correctness_model is None:
            # Default: RandomForest (optionally calibrated) — unchanged behaviour
            base = RandomForestClassifier(
                n_estimators=300, max_depth=12,
                class_weight='balanced', random_state=self.random_state
            )
            if calibrate:
                self.correctness_model = CalibratedClassifierCV(base, method='isotonic', cv=3)
            else:
                self.correctness_model = base
            self.correctness_model.fit(features, correctness)

        elif isinstance(self.base_correctness_model, BinaryClassification):
            # NEW: DNN path — train via Blackbox wrapper
            dnn = self.base_correctness_model

            expected_in = dnn.layer_1.in_features
            actual_in   = features.shape[1]
            if expected_in != actual_in:
                raise ValueError(
                    f"BinaryClassification expects {expected_in} input features, "
                    f"but extracted feature matrix has {actual_in} columns. "
                    f"Re-instantiate BinaryClassification(num_features={actual_in}, device=...)."
                )

            trainer = Blackbox(
                net=dnn,
                criterion=self.dnn_training_params.get('criterion', None),
                max_epoch=self.dnn_training_params.get('max_epoch', 50),
                batch_size=self.dnn_training_params.get('batch_size', 64),
                lr=self.dnn_training_params.get('lr', 1e-3),
                device=dnn.device,
                verbose=self.dnn_training_params.get('verbose', True),
                task='binary',
                num_classes=2,
                compute_accuracy=self.dnn_training_params.get('compute_accuracy', True),
            )

            trainer.fit(
                X_train=features.astype('float32'),
                y_train=correctness.astype('int64'),
                save_dir=self.dnn_training_params.get('save_dir', None),
            )

            self.correctness_model = dnn
            print("DNN correctness classifier training complete.")

        else:
            # User-supplied sklearn-compatible model — unchanged behaviour
            self.correctness_model = self.base_correctness_model
            self.correctness_model.fit(features, correctness)

        self._print_feature_importance()

    def _print_feature_importance(self):
        base = None
        if hasattr(self.correctness_model, 'calibrated_classifiers_'):
             if len(self.correctness_model.calibrated_classifiers_) > 0:
                cal = self.correctness_model.calibrated_classifiers_[0]
                base = getattr(cal, 'estimator', None) or getattr(cal, 'base_estimator', None)
        elif hasattr(self.correctness_model, 'feature_importances_'):
            base = self.correctness_model

        if base and hasattr(base, 'feature_importances_') and self.last_feature_info:
            imps = base.feature_importances_
            names = self.last_feature_info['feature_names']
            cats = self.last_feature_info['feature_categories']

            print("\nTop 10 Feature Importances:")
            idxs = np.argsort(imps)[::-1][:10]
            for idx in idxs:
                if idx < len(names):
                    print(f"{names[idx]} ({cats[idx]}): {imps[idx]:.4f}")

    def predict_proba(self, X_new):
        features = self.extract_correctness_features(X_new)
        if self.use_original_feature:
            features = np.hstack([features, X_new])
        # BinaryClassification.predict_proba expects float32 numpy arrays
        if isinstance(self.correctness_model, BinaryClassification):
            features = features.astype('float32')
        return self.correctness_model.predict_proba(features)

    def predict(self, X_new):
        return (self.predict_proba(X_new)[:, 1] >= 0.5).astype(int)


def train_audit_model(old_model, hist_X, hist_y,
                        X_new_train, y_new_train,
                        base_correctness_model = None,
                        apply_pca=False,
                        apply_standardization=True,
                        use_embeddings = True,
                        use_basic_prediction_features=True,
                        use_distributional_shift_features=True,
                        use_distance_features=True,
                        use_class_conditional_features=True,
                        use_uncertainty_features=True,
                        use_density_features=True,
                        random_state=42):
    """
    Complete training and evaluation pipeline
    NOTE: sample_types only used for evaluation, NOT for training
    """

    correctness_classifier = AuditClassifier(
        old_model, hist_X, hist_y,
        base_correctness_model = base_correctness_model,
        apply_pca=apply_pca,
        apply_standardization=apply_standardization,
        use_embeddings = use_embeddings,
        use_basic_prediction_features=use_basic_prediction_features,
        use_distributional_shift_features=use_distributional_shift_features,
        use_distance_features=use_distance_features,
        use_class_conditional_features=use_class_conditional_features,
        use_uncertainty_features=use_uncertainty_features,
        use_density_features=use_density_features,
        random_state=random_state
    )

    correctness_classifier.fit(X_new_train, y_new_train)

    y_hat_old = old_model.predict(X_new_train)
    correctness_groundtruth = y_hat_old == y_new_train
    y_new_train_hat = correctness_classifier.predict(X_new_train)
    print(f'Correctness classifier accuracy (on train) = {np.mean(correctness_groundtruth == y_new_train_hat):.3f}')

    return correctness_classifier


class Audited_Model:

    def __init__(self, primary_model,
                 audit_model_params=None,
                 random_state=2025,
                 verbose=True):
        """
        Initialize audited conformal prediction (ACP) for multiclass-classification problems

        Args:
            primary_model: Pre-trained model from historical data
            audit_model_params: dict, configuration for audit model training
                - feature_set: 'full' or 'base' (default: 'full')
                - Or individual flags (apply_pca, use_basic_prediction_features, etc.)
            random_state: Random seed
            verbose: Print progress messages
        """
        self.random_state = random_state
        self.verbose = verbose
        self.primary_model = primary_model
        self.audit_model_params = audit_model_params or {}

    @staticmethod
    def _get_feature_config(feature_set):
        """Get feature configuration for a given preset."""
        presets = {
            'full': {
                'apply_pca': False,
                'apply_standardization': True,
                'use_embeddings': True,
                'use_basic_prediction_features': True,
                'use_distributional_shift_features': True,
                'use_distance_features': True,
                'use_class_conditional_features': True,
                'use_uncertainty_features': True,
                'use_density_features': True
            },
            'base': {
                'apply_pca': False,
                'apply_standardization': False,
                'use_embeddings': False,
                'use_basic_prediction_features': False,
                'use_distributional_shift_features': False,
                'use_distance_features': False,
                'use_class_conditional_features': False,
                'use_uncertainty_features': False,
                'use_density_features': False
            },
            'no_history': {
                'apply_pca': False,
                'apply_standardization': True,
                'use_embeddings': False,
                'use_basic_prediction_features': True,
                'use_distributional_shift_features': False,
                'use_distance_features': False,
                'use_class_conditional_features': False,
                'use_uncertainty_features': True,
                'use_density_features': False
            }
        }

        if feature_set not in presets:
            raise ValueError(f"feature_set must be 'full' or 'base', got '{feature_set}'")

        return presets[feature_set]


    def _train_audit_model(self, X_hist, y_hist, X_new_train, y_new_train):
        """
        Train correctness model using audit_model_params from __init__.

        audit_model_params can contain:
            - feature_set: 'full' or 'base' (uses preset)
            - Individual flags to override preset (e.g., apply_pca=True)
        """
        # Get feature set preset (default to 'full')
        feature_set = self.audit_model_params.get('feature_set', 'full')
        base_correctness_model = self.audit_model_params.get('base_correctness_model', None)
        config = self._get_feature_config(feature_set)

        # Allow individual overrides
        config.update({k: v for k, v in self.audit_model_params.items()
                      if k != 'feature_set'})

        # Extract final configuration
        apply_pca = config['apply_pca']
        apply_standardization = config['apply_standardization']
        use_embeddings = config['use_embeddings']
        use_basic_prediction_features = config['use_basic_prediction_features']
        use_distributional_shift_features = config['use_distributional_shift_features']
        use_distance_features = config['use_distance_features']
        use_class_conditional_features = config['use_class_conditional_features']
        use_uncertainty_features = config['use_uncertainty_features']
        use_density_features = config['use_density_features']

        # Your existing training logic here...
        if self.verbose:
            print(f"Training with feature_set='{feature_set}': {config}")

        self.audit_model = train_audit_model(
            self.primary_model, X_hist, y_hist,
            X_new_train, y_new_train,
            base_correctness_model=base_correctness_model,
            apply_pca=apply_pca,
            apply_standardization=apply_standardization,
            use_embeddings = use_embeddings,
            use_basic_prediction_features=use_basic_prediction_features,
            use_distributional_shift_features=use_distributional_shift_features,
            use_distance_features=use_distance_features,
            use_class_conditional_features=use_class_conditional_features,
            use_uncertainty_features=use_uncertainty_features,
            use_density_features=use_density_features,
            random_state=self.random_state
        )

    def _eval_audit_model(self, X_new_val, y_new_val):
        """Evaluate audit classifier accuracy"""
        pred_proba_cal = self.primary_model.predict_proba(X_new_val)
        yhat_cal = np.argmax(pred_proba_cal, axis=1)
        actual_correctness = (yhat_cal == y_new_val).astype(int)
        predicted_correctness = self.audit_model.predict(X_new_val)

        correctness_acc_cal = accuracy_score(actual_correctness, predicted_correctness)
        if self.verbose:
            print(f"Correctness classifier accuracy (on hold-out): {correctness_acc_cal:.4f}")
        return correctness_acc_cal

    def train_evaluate_audit_model(self, X_hist, y_hist, X_new_train, y_new_train, X_new_val, y_new_val):

        self._train_audit_model(X_hist, y_hist, X_new_train, y_new_train)

        audit_acc = self._eval_audit_model(X_new_val, y_new_val)
        feature_importances = getattr(self.audit_model, 'feature_importances', None)

        return feature_importances, audit_acc

class Audited_Conformal_Prediction:
    def __init__(self, primary_model,
                 audit_model,
                 K,
                 alpha=0.1,
                 random_state=2025,
                 verbose=True):
        self.primary_model = primary_model
        self.audit_model = audit_model
        self.K = K
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        self.eta_method = None
        self.eta_adapter_C = None
        self.eta_smoothing_alpha = None
        self.audit_estimate_of_hard_acc_cal = None

    def _conformal_calibrate(self, method,
                             X_new_cal, y_new_cal,
                             threshold, threshold_list, eta_params,
                             X_new_train = None, y_new_train = None, sample_types_cal = None, data_generator = None):
        """Calibrate conformal predictor using calibration set"""
        if method == "combscore":
            if eta_params is None and self.K > 2:
              eta_params = {'eta_method': 'learned',
                  'eta_adapter_C': 0.1,
                  'eta_smoothing_alpha': 1
              }

            if eta_params is not None:
                eta_method = eta_params['eta_method']
                eta_adapter_C = eta_params['eta_adapter_C']
                eta_smoothing_alpha = eta_params['eta_smoothing_alpha']
                self.eta_method = eta_method
                self.eta_adapter_C = eta_adapter_C
                self.eta_smoothing_alpha = eta_smoothing_alpha

                assert eta_method in ['renormalize', 'uniform', 'counts', 'learned'], \
                    "eta_method must be one of ['renormalize', 'uniform', 'counts', 'learned']"

            self._combscore_helper(X_new_train, y_new_train)
            self._conformal_calibrate_combscore(X_new_cal, y_new_cal)

        elif method == "equalized":
            self._conformal_calibrate_equalized(X_new_cal, y_new_cal, threshold, sample_types_cal)
        elif method == 'condconf':
            self._conformal_calibrate_auditcondconf(X_new_cal, y_new_cal, threshold_list)
        elif method == "oracle":
            self._conformal_calibrate_oracle(X_new_cal, y_new_cal, data_generator)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _conformal_calibrate_oracle(self, X_new_cal, y_new_cal, data_generator):
        """Calibrate using oracle correctness labels"""
        if self.verbose:
            print('Using oracle correctness label for calib samples')

        assert data_generator is not None, "Need oracle data distribution"

        p_hat = self.primary_model.predict_proba(X_new_cal)
        y_new_cal_hat = np.argmax(p_hat, axis=1)

        c_hat = data_generator.calculate_c_star(X_new_cal, y_new_cal_hat)
        if self.K == 2:
            p_tilde = self.update_pred_proba_binary(p_hat, c_hat)
        else:
            eta_hat = data_generator.calculate_eta_star(X_new_cal, y_new_cal_hat)
            p_tilde = self.update_pred_proba(p_hat, c_hat, eta_hat)

        cal_scores = compute_nonconf_scores(p_tilde, y_new_cal,
                                           alpha=self.alpha, random_state=self.random_state)
        self.calibrated_threshold = calibrate_alpha(cal_scores, alpha=self.alpha)

        if self.verbose:
            print(f'Calibrated threshold = {self.calibrated_threshold}')

    def _conformal_calibrate_equalized(self, X_new_cal, y_new_cal, threshold, sample_types_cal):
        """
        Group-conditional calibration based on predicted hardness
        Uses correctness classifier ONLY for grouping, not for adjusting probabilities
        """
        if self.verbose:
            print('=== Equalized Method: Separate calibration for hard and easy samples ===')

        # Step 1: Predict hardness using correctness classifier (for grouping only)
        c_hat_cal = self.audit_model.predict_proba(X_new_cal)[:, 1]

        # Note: Lower correctness probability means harder sample
        # So we use < threshold (not >) to identify hard samples
        hard_mask = c_hat_cal < threshold
        easy_mask = c_hat_cal >= threshold

        if sample_types_cal is not None:
            self.audit_estimate_of_hard_acc_cal = (easy_mask == sample_types_cal).mean()

        n_hard = np.sum(hard_mask)
        n_easy = np.sum(easy_mask)

        if self.verbose:
            print(f"Calibration set: {n_hard} predicted hard, {n_easy} predicted easy samples")

        # Step 2: Get old model predictions (use directly, no adjustment)
        p_hat_cal = self.primary_model.predict_proba(X_new_cal)

        # Step 3: Calibrate separately for hard and easy groups using old model predictions
        if n_hard > 0:
            # Hard samples calibration - use old model predictions directly
            cal_scores_hard = compute_nonconf_scores(p_hat_cal[hard_mask], y_new_cal[hard_mask],
                                                     alpha=self.alpha, random_state=self.random_state)
            self.calibrated_threshold_hard = calibrate_alpha(cal_scores_hard, alpha=self.alpha)

            if self.verbose:
                print(f'Calibrated threshold (hard): {self.calibrated_threshold_hard}')
        else:
            self.calibrated_threshold_hard = None
            if self.verbose:
                print("Warning: No hard samples in calibration set")

        if n_easy > 0:
            # Easy samples calibration - use old model predictions directly
            cal_scores_easy = compute_nonconf_scores(p_hat_cal[easy_mask], y_new_cal[easy_mask],
                                                     alpha=self.alpha, random_state=self.random_state)
            self.calibrated_threshold_easy = calibrate_alpha(cal_scores_easy, alpha=self.alpha)

            if self.verbose:
                print(f'Calibrated threshold (easy): {self.calibrated_threshold_easy}')
        else:
            self.calibrated_threshold_easy = None
            if self.verbose:
                print("Warning: No easy samples in calibration set")

    def _conformal_calibrate_auditcondconf(self, X_new_cal, y_new_cal, threshold_list):
        if self.verbose:
            print('=== Condconf Method: Simultaneous coverage across overlapped subgroups defined by audit scores ===')


        def score_fn(X, Y):
            return romano_score_fn(self.primary_model, X, Y, random_state=self.random_state)

        def Phi_fn_Audit(X):
            X = np.asarray(X)
            if X.ndim == 3:          # single image (C, H, W) → add batch dim
                X = X[np.newaxis]
            elif X.ndim == 1:
                X = X.reshape(1, -1)

            # now call predict_proba with 2D input
            audit_scores = self.audit_model.predict_proba(X)[:, 1]
            audit_scores = audit_scores.reshape(-1, 1)

            M = indicator_matrix_overlapping_thresholds(audit_scores, threshold_list)

            return M

        # Instantiate CondConf
        self.cond_conf = CondConf(
            score_fn=score_fn,
            Phi_fn=Phi_fn_Audit,
            infinite_params={},      # no RKHS in this example
            seed=self.random_state
        )

        # Fit conditional quantile model on calibration data
        self.cond_conf.setup_problem(X_new_cal, y_new_cal)
        return

    def _combscore_helper(self, X_new_train, y_new_train):
        assert y_new_train is not None and X_new_train is not None, "Need training data to train eta"

        # Binary case: no eta needed
        if self.K == 2:
            return

        # Build confusion matrix if using counts eta method
        if self.eta_method == 'counts':
            p_hat_train = self.primary_model.predict_proba(X_new_train)
            y_pred_train = np.argmax(p_hat_train, axis=1)
            n_classes = p_hat_train.shape[1]

            self.confusion_matrix = build_confusion_matrix(y_new_train, y_pred_train, n_classes)
            if self.verbose:
                print(f"Building confusion matrix from training data")
                print(f"Confusion matrix:\n{self.confusion_matrix}\n")

        # Train eta adapter if using learned method
        if self.eta_method == 'learned':
            p_hat_train = self.primary_model.predict_proba(X_new_train)
            n_classes = p_hat_train.shape[1]
            c_hat_train = self.audit_model.predict_proba(X_new_train)[:, 1]

            if self.verbose:
                print("Training learned eta adapter...")

            self.eta_adapter = EtaAdapterMultinomial(
                n_classes=n_classes,
                regularization='l2',
                C=self.eta_adapter_C,
                random_state=self.random_state
            )
            self.eta_adapter.fit(X_new_train, y_new_train, p_hat_train, c_hat_train)
            if self.verbose:
                print()

    def update_pred_proba_binary(self, p_hat, c_hat):
        """Binary APA adjustment: replaces primary model probs with audit-corrected vector."""
        y_hat = np.argmax(p_hat, axis=1)
        p_tilde_1 = np.where(y_hat == 1, c_hat, 1 - c_hat)
        p_tilde_0 = 1 - p_tilde_1
        return np.column_stack([p_tilde_0, p_tilde_1])

    def update_pred_proba(self, p_hat, c_hat, eta_hat):
        """
        Update predicted probabilities using correctness and eta
        Formula: p_tilde_k = c * I(k == y_hat) + (1 - c) * eta_k
        """
        n_samples, n_classes = p_hat.shape

        # Validate inputs
        assert c_hat.shape == (n_samples,), \
            f"c_hat shape mismatch: expected ({n_samples},), got {c_hat.shape}"
        assert eta_hat.shape == (n_samples, n_classes), \
            f"eta shape mismatch: expected {(n_samples, n_classes)}, got {eta_hat.shape}"

        # Get predicted class for each sample
        y_hat = np.argmax(p_hat, axis=1)

        # Create indicator matrix: indicator[i, k] = 1 if k == y_hat[i], else 0
        indicator = np.zeros((n_samples, n_classes))
        indicator[np.arange(n_samples), y_hat] = 1

        # Reshape c_hat for broadcasting
        c_hat_expanded = c_hat[:, np.newaxis]

        # Apply formula
        p_tilde = c_hat_expanded * indicator + (1 - c_hat_expanded) * eta_hat

        # Normalize to ensure probabilities sum to 1
        p_tilde = p_tilde / np.sum(p_tilde, axis=1, keepdims=True)

        return p_tilde

    def _estimate_eta(self, p_hat, X=None, correctness_probs=None):
        """Helper method to estimate eta based on selected method"""
        if self.eta_method == 'renormalize':
            return estimate_eta_renormalize(p_hat)
        elif self.eta_method == 'uniform':
            return estimate_eta_uniform(p_hat)
        elif self.eta_method == 'counts':
            if self.confusion_matrix is None:
                raise ValueError("Confusion matrix not computed.")
            return estimate_eta_confusion_prior(p_hat, self.confusion_matrix,
                                               smoothing_alpha=self.eta_smoothing_alpha)
        elif self.eta_method == 'learned':
            if self.eta_adapter is None:
                raise ValueError("Eta adapter not fitted.")
            return estimate_eta_learned(X, p_hat, self.eta_adapter, correctness_probs)
        else:
            raise ValueError(f"Unknown eta_method: {self.eta_method}")

    def _conformal_calibrate_combscore(self, X_new_cal, y_new_cal):
        """Calibrate using APA (binary formula for K=2, multiclass formula otherwise)"""
        if self.verbose:
            print('=== CombScore Method: adjust per-sample level scores ===')

        p_hat = self.primary_model.predict_proba(X_new_cal)
        c_hat = self.audit_model.predict_proba(X_new_cal)[:, 1]

        if self.K == 2:
            p_tilde = self.update_pred_proba_binary(p_hat, c_hat)
        else:
            eta_hat = self._estimate_eta(p_hat=p_hat, X=X_new_cal, correctness_probs=c_hat)
            p_tilde = self.update_pred_proba(p_hat, c_hat, eta_hat)

        cal_scores = compute_nonconf_scores(p_tilde, y_new_cal,
                                           alpha=self.alpha, random_state=self.random_state)
        self.calibrated_threshold = calibrate_alpha(cal_scores, alpha=self.alpha)

        if self.verbose:
            print(f'Calibrated threshold = {self.calibrated_threshold}')

    def _conformal_predict(self, method, X_test, threshold, data_generator):

        p_tilde_test = None

        if method == "oracle":
            S_hat, p_tilde_test = self._conformal_predict_oracle(X_test, data_generator)
        elif method == "combscore":
            S_hat, p_tilde_test = self._conformal_predict_combscore(X_test)
        elif method == "equalized":
            S_hat, p_tilde_test = self._conformal_predict_equalized(X_test, threshold)
        elif method == "condconf":
            S_hat = self._conformal_predict_auditcondconf(X_test)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return S_hat, p_tilde_test

    def _conformal_predict_auditcondconf(self, X_test):
        S_hat = []

        for i in range(len(X_test)):
            x_i = X_test[i]

            # Closure captures x_i directly — ignores CondConf's x_test.reshape(-1,1)
            # which mangles image (C,H,W) into (C*H*W, 1)
            def score_inv_fn(score_cutoff, _x_ignored, _x_orig=x_i):
                x = np.asarray(_x_orig)
                if x.ndim == 3:        # image (C, H, W) → (1, C, H, W)
                    x = x[np.newaxis]
                elif x.ndim == 1:
                    x = x.reshape(1, -1)
                scores_y = romano_score_vector(self.primary_model, x)
                return np.where(scores_y <= score_cutoff)[0]

            S_hat_i = self.cond_conf.predict(
                quantile     = 1 - self.alpha,
                x_test       = x_i,
                score_inv_fn = score_inv_fn,
                S_min        = None,
                S_max        = None,
                randomize    = False,
                exact        = True,
            )
            S_hat.append(S_hat_i)
            if i % 100 == 0:
                print(f'finish processing {i}/{len(X_test)} test points')

        return np.array(S_hat, dtype=object)


    def _conformal_predict_equalized(self, X_test, threshold):
        """Predict using group-conditional thresholds based on predicted hardness"""
        if self.verbose:
            print('Using group-conditional method for test prediction')

        # Step 1: Predict hardness for test samples (for grouping only)
        c_hat_test = self.audit_model.predict_proba(X_test)[:, 1]
        hard_mask_test = c_hat_test < threshold
        easy_mask_test = c_hat_test >= threshold

        n_hard_test = int(np.sum(hard_mask_test))
        n_easy_test = int(np.sum(easy_mask_test))

        if self.verbose:
            print(f"Test set: {n_hard_test} predicted hard, {n_easy_test} predicted easy samples")

        # Step 2: Get old model predictions (use directly, no adjustment)
        p_hat_test = self.primary_model.predict_proba(X_test)

        # Resolve the two per-group thresholds once, with explicit fallbacks.
        t_hard = self.calibrated_threshold_hard if self.calibrated_threshold_hard is not None \
                 else self.calibrated_threshold_easy
        t_easy = self.calibrated_threshold_easy if self.calibrated_threshold_easy is not None \
                 else self.calibrated_threshold_hard

        if self.verbose:
            if self.calibrated_threshold_hard is None and n_hard_test > 0:
                print("Warning: Using fallback (easy) threshold for hard test samples")
            if self.calibrated_threshold_easy is None and n_easy_test > 0:
                print("Warning: Using fallback (hard) threshold for easy test samples")

        # Step 3: Vectorized per-group prediction. One predict_set call per group gives
        # IID randomization ε's across that group's test points; distinct seeds across
        # groups keep the two batches independent. (Prior per-sample loop re-seeded the
        # RNG every iteration, collapsing all ε's to a single value and inflating the
        # seed-to-seed variance of realized coverage.)
        S_hat = [None] * len(X_test)
        if n_hard_test > 0 and t_hard is not None:
            sets_h = predict_set(p_hat_test[hard_mask_test], t_hard,
                                 allow_empty=True, random_state=self.random_state)
            for j, idx in enumerate(np.where(hard_mask_test)[0]):
                S_hat[idx] = sets_h[j]
        if n_easy_test > 0 and t_easy is not None:
            sets_e = predict_set(p_hat_test[easy_mask_test], t_easy,
                                 allow_empty=True, random_state=self.random_state + 1)
            for j, idx in enumerate(np.where(easy_mask_test)[0]):
                S_hat[idx] = sets_e[j]

        return S_hat, p_hat_test

    def _conformal_predict_combscore(self, X_test):
        """Predict using predicted correctness probability"""
        if self.verbose:
            print('Using predicted correctness probability for test samples')

        p_hat_test = self.primary_model.predict_proba(X_test)
        c_hat_test = self.audit_model.predict_proba(X_test)[:, 1]

        if self.K == 2:
            p_tilde_test = self.update_pred_proba_binary(p_hat_test, c_hat_test)
        else:
            if self.eta_method == 'learned':
                correctness_probs_test = c_hat_test
                eta_hat_test = self._estimate_eta(p_hat=p_hat_test, X=X_test,
                                                  correctness_probs=correctness_probs_test)
            else:
                eta_hat_test = self._estimate_eta(p_hat=p_hat_test)
            p_tilde_test = self.update_pred_proba(p_hat_test, c_hat_test, eta_hat_test)

        S_hat = predict_set(p_tilde_test, self.calibrated_threshold,
                           allow_empty=True, random_state=self.random_state)

        return S_hat, p_tilde_test

    def _conformal_predict_oracle(self, X_test, data_generator):
        """Predict using oracle correctness labels"""
        if self.verbose:
            print('Using oracle correctness label for test samples')

        p_hat_test = self.primary_model.predict_proba(X_test)
        y_test_hat = np.argmax(p_hat_test, axis=1)

        c_hat_test = data_generator.calculate_c_star(X_test, y_test_hat)
        if self.K == 2:
            p_tilde_test = self.update_pred_proba_binary(p_hat_test, c_hat_test)
        else:
            eta_hat_test = data_generator.calculate_eta_star(X_test, y_test_hat)
            p_tilde_test = self.update_pred_proba(p_hat_test, c_hat_test, eta_hat_test)

        S_hat = predict_set(p_tilde_test, self.calibrated_threshold,
                           allow_empty=True, random_state=self.random_state)

        return S_hat, p_tilde_test



    def run_audited_conformal_prediction(self, method,
                                         X_new_cal, y_new_cal, sample_types_cal,
                                         X_test, y_test,
                                         n_bins = 10, sample_types_test = None,
                                         threshold = 0.8, threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9], eta_params = None,
                                         X_new_train = None, y_new_train = None,
                                         data_generator = None, r_hat_test = None):

        assert method in ['equalized', 'condconf', 'combscore', 'oracle'], \
          "Method must be one of ['equalized', 'condconf', 'combscore','oracle]"

        # Calibrate and predict
        self._conformal_calibrate(method = method,
                                  X_new_cal = X_new_cal, y_new_cal = y_new_cal,
                                  threshold = threshold, threshold_list = threshold_list, eta_params = eta_params,
                                  X_new_train = X_new_train, y_new_train = y_new_train,
                                  sample_types_cal = sample_types_cal,
                                  data_generator = data_generator)
        prediction_sets, test_proba = self._conformal_predict(method = method,
                                                              X_test = X_test,
                                                              threshold = threshold,
                                                              data_generator = data_generator)

        primary_model_preds = self.primary_model.predict(X_test)

        if r_hat_test is None:
            results = evaluate_coverage_with_auditing(self.alpha, prediction_sets,
                                                      y_true = y_test, X_test = X_test,
                                                      primary_model_preds = primary_model_preds,
                                                      data_model = data_generator,
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

        # Add metadata
        results['Method'] = f'new ({method})'
        results['eta_method'] = self.eta_method
        results['equalized_threshold'] = threshold
        results['equalized_audit_est_acc'] = self.audit_estimate_of_hard_acc_cal
        results['condconf_threshold_list'] = threshold_list
        results['eta_adapter_C'] = self.eta_adapter_C
        results['eta_smoothing_alpha'] = self.eta_smoothing_alpha

        if self.verbose:
            print(f"\n{'='*60}")
            print("Final Results:")
            print(f"{'='*60}")
            print(results)
            print()

        return pd.DataFrame([results])


class Adaptive_Audited_Conformal_Prediction:
    def __init__(self, primary_model,
                 audit_model,
                 retraining_model,
                 K,
                 alpha=0.1,
                 random_state=2025,
                 verbose=True,
                 calib_cal_size=0.75,
                 selection_epsilon=0.5,
                 retraining_epoch=None,
                 selection_criterion='condcov',
                 selection_significance=0.1):
        """
        Initialize adaptive audited conformal prediction for multiclass classification.
        Adaptively selects between retraining and an audited method based on
        conditional coverage on predicted hard samples.

        Args:
            primary_model: Pre-trained model from historical data
            audit_model: Trained audit/correctness classifier
            retraining_model: Model retrained on new data
            K: Number of classes. When K=2, the combscore method uses the binary
                APA adjustment rule (Eq. 5) and does not require eta.
            alpha: Miscoverage rate (1-alpha is target coverage)
            random_state: Random seed
            verbose: Print progress messages
            calib_cal_size: Fraction of calibration data for final calibration (rest for selection)
            selection_epsilon: Threshold epsilon for identifying hard samples in the
                selection step (Eq. A17): H = {i : r_hat(X_i) <= epsilon}. This is
                distinct from the equalized method's group threshold t (Eq. 7).
            retraining_epoch: Number of epochs used to train the retraining model (for bookkeeping)
            selection_criterion: Criterion for method selection on the selection set.
                'condcov' (default): select method with higher conditional coverage on predicted hard samples.
                'size': select method with smaller average prediction set size on the full selection set.
            selection_significance: Significance level gamma for the one-sided
                t-test used in the 'condcov' selection criterion (Algorithm A5,
                Step 12). Defaults to 0.1.
        """
        assert selection_criterion in ('condcov', 'size'), \
            f"selection_criterion must be 'condcov' or 'size', got '{selection_criterion}'"
        self.primary_model = primary_model
        self.audit_model = audit_model
        self.retraining_model = retraining_model
        self.K = K
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        self.calib_cal_size = calib_cal_size
        self.selection_epsilon = selection_epsilon
        self.retraining_epoch = retraining_epoch
        self.selection_criterion = selection_criterion
        self.selection_significance = selection_significance

        # Will be set during calibration
        self.best_method = None  # 'retraining' or the audited method name
        self.eta_method = None
        self.eta_adapter_C = None
        self.eta_smoothing_alpha = None
        self.calibrated_threshold = None
        self.calibrated_threshold_hard = None  # For equalized method
        self.calibrated_threshold_easy = None  # For equalized method
        self.cond_conf = None  # For condconf method
        self.confusion_matrix = None
        self.eta_adapter = None

    def _conformal_calibrate_adaptive(self, audited_method,
                                      X_new_cal, y_new_cal, sample_types_cal,
                                      threshold, threshold_list, eta_params,
                                      X_new_train, y_new_train):
        """
        Adaptive calibration: select between audited method and retraining
        based on hard sample coverage.

        Args:
            audited_method: One of ['equalized', 'condconf', 'combscore']
            X_new_cal: Calibration features
            y_new_cal: Calibration labels
            sample_types_cal: Optional true sample types (1=easy, 0=hard)
            threshold: Threshold for equalized method
            threshold_list: List of thresholds for condconf method
            eta_params: Parameters for combscore method
            X_new_train: Training features (for combscore)
            y_new_train: Training labels (for combscore)
        """
        if self.verbose:
            print(f'=== Adaptive Method: Selecting between {audited_method} and retraining ===')

        # --- Three-way data split (Section 3.3) ---
        # D¹_cal     = X_new_train  (passed in)  — retrain primary model + train audit model
        # D²_cal-select = X_cal_select (1 - calib_cal_size fraction of X_new_cal)
        #                             — method selection only
        # D²_cal-calib  = X_cal_calib (calib_cal_size fraction of X_new_cal)
        #                             — final conformal calibration
        if sample_types_cal is not None:
            X_cal_select, X_cal_calib, y_cal_select, y_cal_calib, sample_types_select, sample_types_calib = train_test_split(
                X_new_cal, y_new_cal, sample_types_cal,
                test_size=self.calib_cal_size,
                random_state=self.random_state,
                stratify=y_new_cal
            )
        else:
            X_cal_select, X_cal_calib, y_cal_select, y_cal_calib = train_test_split(
                X_new_cal, y_new_cal,
                test_size=self.calib_cal_size,
                random_state=self.random_state,
                stratify=y_new_cal
            )
            sample_types_select = None
            sample_types_calib = None

        if self.verbose:
            print(f"Split: {len(X_cal_select)} for selection, {len(X_cal_calib)} for calibration")

        # Step 1: Identify hard and easy samples using audit model
        c_hat_select = self.audit_model.predict_proba(X_cal_select)[:, 1]
        hard_mask_predicted = c_hat_select < self.selection_epsilon
        easy_mask_predicted = c_hat_select >= self.selection_epsilon

        n_ph = int(hard_mask_predicted.sum())

        if self.verbose:
            print(f"Selection set (predicted): {np.sum(hard_mask_predicted)} hard, {np.sum(easy_mask_predicted)} easy samples")

        # If we have true sample types, compare predicted vs true
        if sample_types_select is not None:
            true_hard_mask = sample_types_select == 0
            true_easy_mask = sample_types_select == 1

            n_predicted_hard = np.sum(hard_mask_predicted)
            n_true_hard = np.sum(true_hard_mask)
            n_overlap = np.sum(hard_mask_predicted & true_hard_mask)

            if self.verbose:
                print(f"Selection set (true): {n_true_hard} hard, {np.sum(true_easy_mask)} easy samples")
                if n_predicted_hard > 0:
                    print(f"Overlap: {n_overlap}/{n_predicted_hard} predicted hard samples are truly hard ({n_overlap/n_predicted_hard*100:.1f}%)")
                if n_true_hard > 0:
                    print(f"Recall: {n_overlap}/{n_true_hard} true hard samples captured ({n_overlap/n_true_hard*100:.1f}%)")

        # Step 2a: Evaluate audited method on selection set
        if audited_method == 'combscore':
            coverage_audited_hard_predicted, size_audited = self._evaluate_combscore_on_selection(
                X_cal_select, y_cal_select, hard_mask_predicted, eta_params, X_new_train, y_new_train
            )
        elif audited_method == 'equalized':
            coverage_audited_hard_predicted, size_audited = self._evaluate_equalized_on_selection(
                X_cal_select, y_cal_select, hard_mask_predicted, threshold
            )
        elif audited_method == 'condconf':
            coverage_audited_hard_predicted, size_audited = self._evaluate_condconf_on_selection(
                X_cal_select, y_cal_select, hard_mask_predicted, threshold_list
            )
        else:
            raise ValueError(f"Unknown audited_method: {audited_method}")

        if self.verbose:
            print(f"{audited_method} method - Coverage on predicted hard samples: {coverage_audited_hard_predicted:.4f}, "
                  f"Avg set size: {size_audited:.4f}")

        # Calculate coverage on TRUE hard subset if available
        if sample_types_select is not None:
            true_hard_mask = sample_types_select == 0
            if np.sum(true_hard_mask) > 0:
                # Re-evaluate on true hard samples (for reporting only)
                if audited_method == 'combscore':
                    coverage_audited_hard_true, _ = self._evaluate_combscore_on_selection(
                        X_cal_select, y_cal_select, true_hard_mask, eta_params, X_new_train, y_new_train
                    )
                elif audited_method == 'equalized':
                    coverage_audited_hard_true, _ = self._evaluate_equalized_on_selection(
                        X_cal_select, y_cal_select, true_hard_mask, threshold
                    )
                elif audited_method == 'condconf':
                    coverage_audited_hard_true, _ = self._evaluate_condconf_on_selection(
                        X_cal_select, y_cal_select, true_hard_mask, threshold_list
                    )

                if self.verbose:
                    print(f"{audited_method} method - Coverage on TRUE hard samples: {coverage_audited_hard_true:.4f}")

        # Step 2b: Evaluate retraining model on selection set.
        # The retraining_model is expected to be pre-trained on D¹_cal (X_new_train)
        # before being passed to this class — consistent with how primary_model and
        # audit_model are handled. Training is the caller's responsibility.
        p_hat_select_retrain = self.retraining_model.predict_proba(X_cal_select)

        cal_scores_retrain = compute_nonconf_scores(p_hat_select_retrain, y_cal_select,
                                                     alpha=self.alpha, random_state=self.random_state)
        threshold_retrain = calibrate_alpha(cal_scores_retrain, alpha=self.alpha)

        S_hat_retrain = predict_set(p_hat_select_retrain, threshold_retrain, allow_empty=True,
                                     random_state=self.random_state)

        # Calculate coverage on predicted hard subset for retraining method
        if np.sum(hard_mask_predicted) > 0:
            coverage_retrain_hard_predicted = np.mean([y_cal_select[i] in S_hat_retrain[i]
                                             for i in range(len(y_cal_select)) if hard_mask_predicted[i]])
        else:
            coverage_retrain_hard_predicted = 0.0

        size_retrain = np.mean([len(s) for s in S_hat_retrain])

        if self.verbose:
            print(f"Retraining method - Coverage on predicted hard samples: {coverage_retrain_hard_predicted:.4f}, "
                  f"Avg set size: {size_retrain:.4f}")

        # Calculate coverage on TRUE hard subset if available
        if sample_types_select is not None and np.sum(true_hard_mask) > 0:
            coverage_retrain_hard_true = np.mean([y_cal_select[i] in S_hat_retrain[i]
                                                  for i in range(len(y_cal_select)) if true_hard_mask[i]])
            if self.verbose:
                print(f"Retraining method - Coverage on TRUE hard samples: {coverage_retrain_hard_true:.4f}")

        # Step 3: Select best method based on the chosen selection criterion
        if n_ph == 0 and self.selection_criterion == 'condcov':
            if self.verbose:
                print("No predicted-hard points in selection; defaulting to retraining.")
            self.best_method = "retraining"
        elif self.selection_criterion == 'condcov':
            p1 = coverage_audited_hard_predicted
            p2 = coverage_retrain_hard_predicted

            # One-sided two-sample t-test (Algorithm A5, Step 12):
            #   H0: delta_ACP <= delta_Retrain  vs  H1: delta_ACP > delta_Retrain
            # For Bernoulli indicators, the Welch t-statistic uses the
            # sample proportions and counts directly.
            eps = 1e-12
            var1 = max(p1 * (1.0 - p1), eps) / n_ph
            var2 = max(p2 * (1.0 - p2), eps) / n_ph
            se_diff = np.sqrt(var1 + var2)

            if se_diff < eps:
                # Both methods have identical (degenerate) coverage; default to retraining
                t_stat = 0.0
                p_value = 1.0
            else:
                t_stat = (p1 - p2) / se_diff
                # Welch–Satterthwaite degrees of freedom
                df = (var1 + var2) ** 2 / (var1 ** 2 / (n_ph - 1) + var2 ** 2 / (n_ph - 1)) if n_ph > 1 else 1.0
                p_value = 1.0 - stats.t.cdf(t_stat, df=df)

            if self.verbose:
                print(f"[condcov] n_pred_hard = {n_ph}, cov_{audited_method} = {p1:.3f}, cov_retrain = {p2:.3f}, "
                      f"t_stat = {t_stat:.3f}, p_value = {p_value:.4f}, gamma = {self.selection_significance}")

            # Reject H0 at significance level gamma => select ACP
            if p_value < self.selection_significance:
                self.best_method = audited_method
            else:
                self.best_method = "retraining"

        elif self.selection_criterion == 'size':
            if self.verbose:
                print(f"[size] size_{audited_method} = {size_audited:.3f}, size_retrain = {size_retrain:.3f}")

            # Select the method with the smaller average prediction set size
            if size_audited < size_retrain:
                self.best_method = audited_method
            else:
                self.best_method = "retraining"

        if self.verbose:
            print(f"✓ Selected method: {self.best_method}")

        # Step 4: Calibrate on remaining using selected method
        if self.best_method == "retraining":
            p_tilde_calib = self.retraining_model.predict_proba(X_cal_calib)
            cal_scores = compute_nonconf_scores(p_tilde_calib, y_cal_calib,
                                               alpha=self.alpha, random_state=self.random_state)
            self.calibrated_threshold = calibrate_alpha(cal_scores, alpha=self.alpha)
        else:
            # Calibrate using the selected audited method
            if audited_method == 'combscore':
                self._calibrate_combscore_final(X_cal_calib, y_cal_calib, eta_params, X_new_train, y_new_train)
            elif audited_method == 'equalized':
                self._calibrate_equalized_final(X_cal_calib, y_cal_calib, threshold)
            elif audited_method == 'condconf':
                self._calibrate_condconf_final(X_cal_calib, y_cal_calib, threshold_list)

        if self.verbose:
            if self.best_method == "retraining":
                print(f'Calibrated threshold (using {self.best_method}): {self.calibrated_threshold}')
            elif audited_method == 'equalized':
                print(f'Calibrated thresholds (using {self.best_method}): hard={self.calibrated_threshold_hard}, easy={self.calibrated_threshold_easy}')
            elif audited_method == 'combscore':
                print(f'Calibrated threshold (using {self.best_method}): {self.calibrated_threshold}')
            elif audited_method == 'condconf':
                print(f'Calibrated CondConf model (using {self.best_method})')

    # ==================== Helper methods for evaluation on selection set ====================

    def _evaluate_combscore_on_selection(self, X_select, y_select, hard_mask, eta_params, X_new_train, y_new_train):
        """Evaluate combscore method on selection set and return coverage on hard samples."""
        # Get predictions
        p_hat_select = self.primary_model.predict_proba(X_select)
        c_hat_select = self.audit_model.predict_proba(X_select)[:, 1]

        if self.K == 2:
            # Binary case: use binary APA rule (Eq. 5), no eta needed
            p_tilde_select = self.update_pred_proba_binary(p_hat_select, c_hat_select)
        else:
            # Multiclass case: train eta and use multiclass rule (Eq. A14)
            if eta_params is None:
                eta_params = {'eta_method': 'learned', 'eta_adapter_C': 0.1, 'eta_smoothing_alpha': 1}

            eta_method = eta_params['eta_method']
            eta_adapter_C = eta_params['eta_adapter_C']
            eta_smoothing_alpha = eta_params['eta_smoothing_alpha']

            # Train eta components if needed
            if eta_method == 'counts':
                p_hat_train = self.primary_model.predict_proba(X_new_train)
                y_pred_train = np.argmax(p_hat_train, axis=1)
                n_classes = p_hat_train.shape[1]
                confusion_matrix_temp = build_confusion_matrix(y_new_train, y_pred_train, n_classes)
            else:
                confusion_matrix_temp = None

            if eta_method == 'learned':
                p_hat_train = self.primary_model.predict_proba(X_new_train)
                n_classes = p_hat_train.shape[1]
                c_hat_train = self.audit_model.predict_proba(X_new_train)[:, 1]

                eta_adapter_temp = EtaAdapterMultinomial(
                    n_classes=n_classes,
                    regularization='l2',
                    C=eta_adapter_C,
                    random_state=self.random_state
                )
                eta_adapter_temp.fit(X_new_train, y_new_train, p_hat_train, c_hat_train)
            else:
                eta_adapter_temp = None

            # Estimate eta
            if eta_method == 'renormalize':
                eta_hat_select = estimate_eta_renormalize(p_hat_select)
            elif eta_method == 'uniform':
                eta_hat_select = estimate_eta_uniform(p_hat_select)
            elif eta_method == 'counts':
                eta_hat_select = estimate_eta_confusion_prior(p_hat_select, confusion_matrix_temp,
                                                              smoothing_alpha=eta_smoothing_alpha)
            elif eta_method == 'learned':
                eta_hat_select = estimate_eta_learned(X_select, p_hat_select, eta_adapter_temp, c_hat_select)

            p_tilde_select = self.update_pred_proba(p_hat_select, c_hat_select, eta_hat_select)

        cal_scores = compute_nonconf_scores(p_tilde_select, y_select,
                                           alpha=self.alpha, random_state=self.random_state)
        threshold_temp = calibrate_alpha(cal_scores, alpha=self.alpha)

        S_hat = predict_set(p_tilde_select, threshold_temp, allow_empty=True,
                           random_state=self.random_state)

        # Calculate coverage on hard subset
        if np.sum(hard_mask) > 0:
            coverage_hard = np.mean([y_select[i] in S_hat[i]
                                    for i in range(len(y_select)) if hard_mask[i]])
        else:
            coverage_hard = 0.0

        avg_size = np.mean([len(s) for s in S_hat])

        return coverage_hard, avg_size

    def _evaluate_equalized_on_selection(self, X_select, y_select, hard_mask, threshold):
        """Evaluate equalized method on selection set and return coverage on hard samples"""
        c_hat_select = self.audit_model.predict_proba(X_select)[:, 1]
        hard_mask_eq = c_hat_select < threshold
        easy_mask_eq = c_hat_select >= threshold

        p_hat_select = self.primary_model.predict_proba(X_select)

        # Calibrate on hard and easy separately
        if np.sum(hard_mask_eq) > 0:
            cal_scores_hard = compute_nonconf_scores(p_hat_select[hard_mask_eq], y_select[hard_mask_eq],
                                                     alpha=self.alpha, random_state=self.random_state)
            threshold_hard_temp = calibrate_alpha(cal_scores_hard, alpha=self.alpha)
        else:
            threshold_hard_temp = None

        if np.sum(easy_mask_eq) > 0:
            cal_scores_easy = compute_nonconf_scores(p_hat_select[easy_mask_eq], y_select[easy_mask_eq],
                                                     alpha=self.alpha, random_state=self.random_state)
            threshold_easy_temp = calibrate_alpha(cal_scores_easy, alpha=self.alpha)
        else:
            threshold_easy_temp = None

        # Resolve per-group thresholds with fallbacks
        t_hard = threshold_hard_temp if threshold_hard_temp is not None else threshold_easy_temp
        t_easy = threshold_easy_temp if threshold_easy_temp is not None else threshold_hard_temp

        # Vectorized per-group prediction (see _conformal_predict_equalized for rationale)
        n_hard_eq = int(np.sum(hard_mask_eq))
        n_easy_eq = int(np.sum(easy_mask_eq))
        S_hat = [None] * len(X_select)
        if n_hard_eq > 0 and t_hard is not None:
            sets_h = predict_set(p_hat_select[hard_mask_eq], t_hard,
                                 allow_empty=True, random_state=self.random_state)
            for j, idx in enumerate(np.where(hard_mask_eq)[0]):
                S_hat[idx] = sets_h[j]
        if n_easy_eq > 0 and t_easy is not None:
            sets_e = predict_set(p_hat_select[easy_mask_eq], t_easy,
                                 allow_empty=True, random_state=self.random_state + 1)
            for j, idx in enumerate(np.where(easy_mask_eq)[0]):
                S_hat[idx] = sets_e[j]

        # Calculate coverage on hard subset
        if np.sum(hard_mask) > 0:
            coverage_hard = np.mean([y_select[i] in S_hat[i]
                                    for i in range(len(y_select)) if hard_mask[i]])
        else:
            coverage_hard = 0.0

        avg_size = np.mean([len(s) for s in S_hat])

        return coverage_hard, avg_size

    def _evaluate_condconf_on_selection(self, X_select, y_select, hard_mask, threshold_list):
        """Evaluate condconf method on selection set and return coverage on hard samples"""
        # Setup condconf
        def score_fn(X, Y):
            return romano_score_fn(self.primary_model, X, Y, random_state=self.random_state)

        def Phi_fn_Audit(X):
            X = np.asarray(X)
            if X.ndim == 3:          # single image (C, H, W) → add batch dim
                X = X[np.newaxis]
            elif X.ndim == 1:
                X = X.reshape(1, -1)

            audit_scores = self.audit_model.predict_proba(X)[:, 1]
            audit_scores = audit_scores.reshape(-1, 1)

            M = indicator_matrix_overlapping_thresholds(audit_scores, threshold_list)
            return M

        cond_conf_temp = CondConf(
            score_fn=score_fn,
            Phi_fn=Phi_fn_Audit,
            infinite_params={},
            seed=self.random_state
        )

        cond_conf_temp.setup_problem(X_select, y_select)

        S_hat = []
        for i in range(len(X_select)):
            x_i = X_select[i]

            def score_inv_fn(score_cutoff, _x_ignored, _x_orig=x_i):
                x = np.asarray(_x_orig)
                if x.ndim == 3:
                    x = x[np.newaxis]
                elif x.ndim == 1:
                    x = x.reshape(1, -1)
                scores_y = romano_score_vector(self.primary_model, x)
                return np.where(scores_y <= score_cutoff)[0]

            S_hat_i = cond_conf_temp.predict(
                quantile=1-self.alpha,
                x_test=x_i,
                score_inv_fn=score_inv_fn,
                S_min=None,
                S_max=None,
                randomize=False,
                exact=True
            )
            S_hat.append(S_hat_i)

        # Calculate coverage on hard subset
        if np.sum(hard_mask) > 0:
            coverage_hard = np.mean([y_select[i] in S_hat[i]
                                    for i in range(len(y_select)) if hard_mask[i]])
        else:
            coverage_hard = 0.0

        avg_size = np.mean([len(s) for s in S_hat])

        return coverage_hard, avg_size

    # ==================== Helper methods for final calibration ====================

    def _calibrate_combscore_final(self, X_calib, y_calib, eta_params, X_new_train, y_new_train):
        """Final calibration using combscore method"""
        p_hat = self.primary_model.predict_proba(X_calib)
        c_hat = self.audit_model.predict_proba(X_calib)[:, 1]

        if self.K == 2:
            # Binary case: use binary APA rule (Eq. 5), no eta needed
            p_tilde = self.update_pred_proba_binary(p_hat, c_hat)
        else:
            # Multiclass case: train eta and use multiclass rule (Eq. A14)
            if eta_params is None:
                eta_params = {'eta_method': 'learned', 'eta_adapter_C': 0.1, 'eta_smoothing_alpha': 1}

            self.eta_method = eta_params['eta_method']
            self.eta_adapter_C = eta_params['eta_adapter_C']
            self.eta_smoothing_alpha = eta_params['eta_smoothing_alpha']

            # Train eta components
            if self.eta_method == 'counts':
                p_hat_train = self.primary_model.predict_proba(X_new_train)
                y_pred_train = np.argmax(p_hat_train, axis=1)
                n_classes = p_hat_train.shape[1]
                self.confusion_matrix = build_confusion_matrix(y_new_train, y_pred_train, n_classes)

            if self.eta_method == 'learned':
                p_hat_train = self.primary_model.predict_proba(X_new_train)
                n_classes = p_hat_train.shape[1]
                c_hat_train = self.audit_model.predict_proba(X_new_train)[:, 1]

                self.eta_adapter = EtaAdapterMultinomial(
                    n_classes=n_classes,
                    regularization='l2',
                    C=self.eta_adapter_C,
                    random_state=self.random_state
                )
                self.eta_adapter.fit(X_new_train, y_new_train, p_hat_train, c_hat_train)

            eta_hat = self._estimate_eta(p_hat=p_hat, X=X_calib, correctness_probs=c_hat)
            p_tilde = self.update_pred_proba(p_hat, c_hat, eta_hat)

        cal_scores = compute_nonconf_scores(p_tilde, y_calib,
                                           alpha=self.alpha, random_state=self.random_state)
        self.calibrated_threshold = calibrate_alpha(cal_scores, alpha=self.alpha)

    def _calibrate_equalized_final(self, X_calib, y_calib, threshold):
        """Final calibration using equalized method"""
        c_hat_calib = self.audit_model.predict_proba(X_calib)[:, 1]
        hard_mask = c_hat_calib < threshold
        easy_mask = c_hat_calib >= threshold

        p_hat_calib = self.primary_model.predict_proba(X_calib)

        if np.sum(hard_mask) > 0:
            cal_scores_hard = compute_nonconf_scores(p_hat_calib[hard_mask], y_calib[hard_mask],
                                                     alpha=self.alpha, random_state=self.random_state)
            self.calibrated_threshold_hard = calibrate_alpha(cal_scores_hard, alpha=self.alpha)
        else:
            self.calibrated_threshold_hard = None

        if np.sum(easy_mask) > 0:
            cal_scores_easy = compute_nonconf_scores(p_hat_calib[easy_mask], y_calib[easy_mask],
                                                     alpha=self.alpha, random_state=self.random_state)
            self.calibrated_threshold_easy = calibrate_alpha(cal_scores_easy, alpha=self.alpha)
        else:
            self.calibrated_threshold_easy = None

    def _calibrate_condconf_final(self, X_calib, y_calib, threshold_list):
        """Final calibration using condconf method"""
        def score_fn(X, Y):
            return romano_score_fn(self.primary_model, X, Y, random_state=self.random_state)

        def Phi_fn_Audit(X):
            X = np.asarray(X)
            if X.ndim == 3:          # single image (C, H, W) → add batch dim
                X = X[np.newaxis]
            elif X.ndim == 1:
                X = X.reshape(1, -1)

            audit_scores = self.audit_model.predict_proba(X)[:, 1]
            audit_scores = audit_scores.reshape(-1, 1)

            M = indicator_matrix_overlapping_thresholds(audit_scores, threshold_list)
            return M

        self.cond_conf = CondConf(
            score_fn=score_fn,
            Phi_fn=Phi_fn_Audit,
            infinite_params={},
            seed=self.random_state
        )

        self.cond_conf.setup_problem(X_calib, y_calib)

    # ==================== Utility methods ====================

    def update_pred_proba_binary(self, p_hat, c_hat):
        """Binary APA adjustment (Eq. 5): no eta needed."""
        y_hat = np.argmax(p_hat, axis=1)
        p_tilde_1 = np.where(y_hat == 1, c_hat, 1 - c_hat)
        p_tilde_0 = 1 - p_tilde_1
        return np.column_stack([p_tilde_0, p_tilde_1])

    def update_pred_proba(self, p_hat, c_hat, eta_hat):
        """Update predicted probabilities using correctness and eta"""
        n_samples, n_classes = p_hat.shape

        assert c_hat.shape == (n_samples,), f"c_hat shape mismatch: expected ({n_samples},), got {c_hat.shape}"
        assert eta_hat.shape == (n_samples, n_classes), f"eta shape mismatch: expected {(n_samples, n_classes)}, got {eta_hat.shape}"

        y_hat = np.argmax(p_hat, axis=1)
        indicator = np.zeros((n_samples, n_classes))
        indicator[np.arange(n_samples), y_hat] = 1

        c_hat_expanded = c_hat[:, np.newaxis]
        p_tilde = c_hat_expanded * indicator + (1 - c_hat_expanded) * eta_hat
        p_tilde = p_tilde / np.sum(p_tilde, axis=1, keepdims=True)

        return p_tilde

    def _estimate_eta(self, p_hat, X=None, correctness_probs=None):
        """Helper method to estimate eta based on selected method"""
        if self.eta_method == 'renormalize':
            return estimate_eta_renormalize(p_hat)
        elif self.eta_method == 'uniform':
            return estimate_eta_uniform(p_hat)
        elif self.eta_method == 'counts':
            if self.confusion_matrix is None:
                raise ValueError("Confusion matrix not computed.")
            return estimate_eta_confusion_prior(p_hat, self.confusion_matrix,
                                               smoothing_alpha=self.eta_smoothing_alpha)
        elif self.eta_method == 'learned':
            if self.eta_adapter is None:
                raise ValueError("Eta adapter not fitted.")
            return estimate_eta_learned(X, p_hat, self.eta_adapter, correctness_probs)
        else:
            raise ValueError(f"Unknown eta_method: {self.eta_method}")

    # ==================== Prediction methods ====================

    def _conformal_predict(self, audited_method, X_test, threshold, threshold_list):
        """Make conformal prediction sets using selected method"""
        if self.best_method == "retraining":
            p_tilde_test = self.retraining_model.predict_proba(X_test)
            S_hat = predict_set(p_tilde_test, self.calibrated_threshold,
                               allow_empty=True, random_state=self.random_state)
            return S_hat, p_tilde_test
        else:
            # Use the audited method
            if audited_method == 'combscore':
                return self._conformal_predict_combscore(X_test)
            elif audited_method == 'equalized':
                return self._conformal_predict_equalized(X_test, threshold)
            elif audited_method == 'condconf':
                S_hat = self._conformal_predict_condconf(X_test)
                return S_hat, None
            else:
                raise ValueError(f"Unknown audited_method: {audited_method}")

    def _conformal_predict_combscore(self, X_test):
        """Predict using combscore method"""
        p_hat_test = self.primary_model.predict_proba(X_test)
        c_hat_test = self.audit_model.predict_proba(X_test)[:, 1]

        if self.K == 2:
            p_tilde_test = self.update_pred_proba_binary(p_hat_test, c_hat_test)
        else:
            eta_hat_test = self._estimate_eta(p_hat=p_hat_test, X=X_test, correctness_probs=c_hat_test)
            p_tilde_test = self.update_pred_proba(p_hat_test, c_hat_test, eta_hat_test)

        S_hat = predict_set(p_tilde_test, self.calibrated_threshold,
                           allow_empty=True, random_state=self.random_state)

        return S_hat, p_tilde_test

    def _conformal_predict_equalized(self, X_test, threshold):
        """Predict using equalized method"""
        c_hat_test = self.audit_model.predict_proba(X_test)[:, 1]
        hard_mask_test = c_hat_test < threshold
        easy_mask_test = c_hat_test >= threshold

        p_hat_test = self.primary_model.predict_proba(X_test)

        t_hard = self.calibrated_threshold_hard if self.calibrated_threshold_hard is not None \
                 else self.calibrated_threshold_easy
        t_easy = self.calibrated_threshold_easy if self.calibrated_threshold_easy is not None \
                 else self.calibrated_threshold_hard

        # Vectorized per-group prediction (see Audited_Conformal_Prediction
        # ._conformal_predict_equalized for rationale).
        n_hard_test = int(np.sum(hard_mask_test))
        n_easy_test = int(np.sum(easy_mask_test))
        S_hat = [None] * len(X_test)
        if n_hard_test > 0 and t_hard is not None:
            sets_h = predict_set(p_hat_test[hard_mask_test], t_hard,
                                 allow_empty=True, random_state=self.random_state)
            for j, idx in enumerate(np.where(hard_mask_test)[0]):
                S_hat[idx] = sets_h[j]
        if n_easy_test > 0 and t_easy is not None:
            sets_e = predict_set(p_hat_test[easy_mask_test], t_easy,
                                 allow_empty=True, random_state=self.random_state + 1)
            for j, idx in enumerate(np.where(easy_mask_test)[0]):
                S_hat[idx] = sets_e[j]

        return S_hat, p_hat_test

    def _conformal_predict_condconf(self, X_test):
        """Predict using condconf method"""
        S_hat = []

        for i in range(len(X_test)):
            x_i = X_test[i]

            def score_inv_fn(score_cutoff, _x_ignored, _x_orig=x_i):
                x = np.asarray(_x_orig)
                if x.ndim == 3:
                    x = x[np.newaxis]
                elif x.ndim == 1:
                    x = x.reshape(1, -1)
                scores_y = romano_score_vector(self.primary_model, x)
                return np.where(scores_y <= score_cutoff)[0]

            S_hat_i = self.cond_conf.predict(
                quantile=1-self.alpha,
                x_test=x_i,
                score_inv_fn=score_inv_fn,
                S_min=None,
                S_max=None,
                randomize=False,
                exact=True
            )

            S_hat.append(S_hat_i)
            if i % 100 == 0 and self.verbose:
                print(f'Finished processing {i}/{len(X_test)} test points')

        S_hat = np.array(S_hat, dtype=object)
        return S_hat

    # ==================== Main entry point ====================

    def run_adaptive_audited_conformal_prediction(self, audited_method,
                                                  X_new_cal, y_new_cal, sample_types_cal,
                                                  X_test, y_test, data_generator, n_bins, sample_types_test = None,
                                                  threshold=0.8, threshold_list=[0.5, 0.6, 0.7, 0.8, 0.9],
                                                  eta_params=None,
                                                  X_new_train=None, y_new_train=None):
        """
        Run Adaptive Audited Conformal Prediction (AACP).

        Implements the three-way data split described in Section 3.3 of the paper:
          D¹_cal     = X_new_train / y_new_train  — used to (i) retrain the primary model
                       from scratch, and (ii) train the auditing classifier r̂ (done externally).
          D²_cal-select = a held-out fraction of X_new_cal (size = 1 - calib_cal_size)
                       — used exclusively for data-driven method selection: estimating which
                       of {ACP method, retraining} achieves higher conditional coverage (or
                       smaller set size) on predicted-hard samples.
          D²_cal-calib  = the remaining fraction of X_new_cal (size = calib_cal_size)
                       — used for the final conformal calibration step (computing nonconformity
                       scores and estimating the empirical quantile threshold).

        Args:
            audited_method: One of ['equalized', 'condconf', 'combscore']
            X_new_cal: D²_cal — calibration data not used for audit/retrain training.
                       Split internally into D²_cal-select and D²_cal-calib.
            y_new_cal: Labels for X_new_cal.
            sample_types_cal: Sample types for X_new_cal (1=easy, 0=hard); optional.
            X_test: Test features.
            y_test: Test labels.
            sample_types_test: Sample types for test set.
            threshold: Hard-sample threshold for equalized method.
            threshold_list: List of thresholds for condconf method.
            eta_params: Parameters for combscore eta estimator.
            X_new_train: D¹_cal features — required for all methods (used to train
                         the retraining model; also used as training data for the
                         combscore eta adapter).
            y_new_train: D¹_cal labels.

        Returns:
            DataFrame with results.
        """
        assert audited_method in ['equalized', 'condconf', 'combscore'], \
            "audited_method must be one of ['equalized', 'condconf', 'combscore']"

        assert X_new_train is not None and y_new_train is not None, \
            ("X_new_train and y_new_train (D¹_cal) are required for all adaptive methods: "
             "they are used to train the retraining model from scratch.")

        # Adaptive calibration
        self._conformal_calibrate_adaptive(
            audited_method=audited_method,
            X_new_cal=X_new_cal,
            y_new_cal=y_new_cal,
            sample_types_cal=sample_types_cal,
            threshold=threshold,
            threshold_list=threshold_list,
            eta_params=eta_params,
            X_new_train=X_new_train,
            y_new_train=y_new_train
        )

        # Predict on test set
        prediction_sets, test_proba = self._conformal_predict(
            audited_method=audited_method,
            X_test=X_test,
            threshold=threshold,
            threshold_list=threshold_list
        )

        # Evaluate coverage
        primary_model_preds = self.primary_model.predict(X_test)
        results = evaluate_coverage_with_auditing(self.alpha, prediction_sets,
                                                  y_true = y_test, X_test = X_test,
                                                  primary_model_preds = primary_model_preds,
                                                  data_model = data_generator,
                                                  n_bins=n_bins,
                                                  sample_types=sample_types_test,
                                                  test_pred_prob=test_proba)

        # results = evaluate_coverage(
        #     self.alpha, prediction_sets, y_test,
        #     test_pred_prob=test_proba,
        #     sample_types=sample_types_test
        # )


        # Add metadata
        results['Method'] = f'new A ({audited_method})'
        results['selected_method'] = self.best_method
        results['eta_method'] = self.eta_method
        results['selection_epsilon'] = self.selection_epsilon
        results['calib_cal_size'] = self.calib_cal_size
        results['equalized_threshold'] = threshold if audited_method == 'equalized' else None
        results['condconf_threshold_list'] = threshold_list if audited_method == 'condconf' else None
        results['equalized_audit_est_acc'] = 'skip'
        results['eta_adapter_C'] = self.eta_adapter_C
        results['eta_smoothing_alpha'] = self.eta_smoothing_alpha
        results['retraining_epoch'] = self.retraining_epoch
        results['selection_criterion'] = self.selection_criterion
        results['selection_significance'] = self.selection_significance

        if self.verbose:
            print(f"\n{'='*60}")
            print("Final Results:")
            print(f"{'='*60}")
            print(results)
            print()

        return pd.DataFrame([results])

# ============================================================================
# Eta-method helpers (moved from src/eta_methods.py).
# ============================================================================


def make_eta_features(X, p_hat, correctness_probs=None, include_raw_features=False):
    """
    Build feature vector for eta adapter with improved signals.

    Key improvements:
    - Top-one-hot encoding (per-class behavior)
    - Differences and log-ratios to top prediction
    - Optional raw features

    Args:
        X: raw features (n_samples, n_features) or None
        p_hat: predicted probabilities (n_samples, n_classes)
        correctness_probs: optional correctness estimates (n_samples,)
        include_raw_features: whether to include raw X features

    Returns:
        features: array of shape (n_samples, n_eta_features)
    """
    n_samples, n_classes = p_hat.shape
    eps = 1e-10

    # Get top prediction
    y_hat = np.argmax(p_hat, axis=1)

    # Convert probabilities to logits (with clipping to avoid inf)
    p_clipped = np.clip(p_hat, eps, 1 - eps)
    logits = np.log(p_clipped)

    # Get top probability and logit
    top_prob = np.max(p_hat, axis=1, keepdims=True)
    top_logit = logits[np.arange(n_samples), y_hat][:, np.newaxis]

    # Basic prediction features
    entropy = -np.sum(p_hat * np.log(p_clipped), axis=1, keepdims=True)

    # Probability margins
    sorted_probs = np.sort(p_hat, axis=1)
    second_max_prob = sorted_probs[:, -2:(-1)] if n_classes > 1 else np.zeros((n_samples, 1))
    margin = top_prob - second_max_prob

    # Gini impurity (alternative uncertainty measure)
    gini = 1 - np.sum(p_hat ** 2, axis=1, keepdims=True)

    # Top-one-hot encoding (indicator for which class is predicted)
    top_onehot = np.zeros((n_samples, n_classes))
    top_onehot[np.arange(n_samples), y_hat] = 1

    # Differences to top (p_k - p_top for all k)
    prob_diff_to_top = p_hat - top_prob

    # Log-ratios to top (log p_k - log p_top for all k)
    # These are strong predictors of relative likelihood
    log_ratio_to_top = logits - top_logit

    feature_list = [
        top_prob,                # Top probability
        entropy,                 # Prediction entropy
        margin,                  # Margin between top 2 classes
        second_max_prob,         # Second highest probability
        gini,                    # Gini impurity
        top_onehot,              # One-hot of predicted class (per-class behavior)
        p_hat,                   # All probabilities
        logits,                  # All logits
        prob_diff_to_top,        # p_k - p_top (strong signal!)
        log_ratio_to_top,        # log p_k - log p_top (strong signal!)
    ]

    # Add correctness probability if available
    if correctness_probs is not None:
        feature_list.append(correctness_probs[:, np.newaxis])

    # Add raw features if requested and available
    if include_raw_features and X is not None:
        feature_list.append(X)

    features = np.column_stack(feature_list)
    return features

class EtaAdapterMultinomial:
    """
    Learned adapter for eta that redistributes probability mass among non-top classes.

    Improvements:
    - Handles class coverage mismatch
    - Standardizes features
    - Gracefully handles edge cases (few wrong samples, no X features)
    - Better numerical stability
    """

    def __init__(self, n_classes, regularization='l2', C=0.1, max_iter=1000,
                 random_state=42, min_wrong_samples=10, include_raw_features=False):
        """
        Args:
            n_classes: number of classes
            regularization: 'l2' or 'l1'
            C: inverse regularization strength (smaller = stronger regularization, default: 0.1)
            max_iter: maximum iterations for solver
            random_state: random seed
            min_wrong_samples: minimum wrong samples required to fit (else fallback)
            include_raw_features: whether to use raw X features
        """
        self.n_classes = n_classes
        self.regularization = regularization
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.min_wrong_samples = min_wrong_samples
        self.include_raw_features = include_raw_features

        self.model = None
        self.scaler = None
        self.fitted_classes = None
        self.fallback_mode = False

    def fit(self, X, y, p_hat, correctness_probs=None):
        """
        Fit the eta adapter on samples where the model was wrong.

        Args:
            X: raw features (n_samples, n_features) or None
            y: true labels (n_samples,)
            p_hat: predicted probabilities (n_samples, n_classes)
            correctness_probs: optional correctness estimates (n_samples,)
        """
        # Get predictions
        y_hat = np.argmax(p_hat, axis=1)

        # Keep only samples where model was wrong
        wrong_mask = (y != y_hat)
        n_wrong = np.sum(wrong_mask)

        if n_wrong < self.min_wrong_samples:
            warnings.warn(
                f"Only {n_wrong} wrong predictions (< {self.min_wrong_samples} minimum). "
                f"Falling back to renormalization mode."
            )
            self.fallback_mode = True
            return self

        # Extract wrong samples (handle X=None case)
        X_wrong = X[wrong_mask] if X is not None else None
        y_wrong = y[wrong_mask]
        p_hat_wrong = p_hat[wrong_mask]
        correctness_wrong = correctness_probs[wrong_mask] if correctness_probs is not None else None

        # Build features
        features = make_eta_features(X_wrong, p_hat_wrong, correctness_wrong,
                                     include_raw_features=self.include_raw_features)

        # Standardize features for better LogReg performance
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Store which classes appear in training (for handling class coverage mismatch)
        self.fitted_classes = np.unique(y_wrong)

        if len(self.fitted_classes) < 2:
            warnings.warn(
                f"Only {len(self.fitted_classes)} unique class(es) in wrong samples. "
                f"Falling back to renormalization mode."
            )
            self.fallback_mode = True
            return self

        # Train multinomial logistic regression
        penalty = 'l2' if self.regularization == 'l2' else 'l1'
        solver = 'lbfgs' if penalty == 'l2' else 'saga'

        self.model = LogisticRegression(
            penalty=penalty,
            C=self.C,
            max_iter=self.max_iter,
            multi_class='multinomial',
            solver=solver,
            random_state=self.random_state,
            class_weight='balanced'  # Handle class imbalance
        )

        self.model.fit(features_scaled, y_wrong)

        print(f"Eta adapter fitted on {n_wrong} wrong predictions")
        print(f"Classes observed in errors: {sorted(self.fitted_classes)}")
        print(f"Training accuracy: {self.model.score(features_scaled, y_wrong):.3f}")

        return self

    def _expand_to_all_classes(self, scores_subset, kind='logits'):
        """
        Expand model outputs from fitted classes to all K classes.

        sklearn's LogisticRegression only outputs scores for classes in training.
        We need to expand back to all K classes, filling missing ones appropriately.

        Args:
            scores_subset: array of shape (n_samples, n_fitted_classes)
            kind: 'logits' or 'probs' - determines fill value
                  'logits': fill with -inf (zero probability after softmax)
                  'probs': fill with 0.0 (zero probability directly)

        Returns:
            scores_full: array of shape (n_samples, n_classes)
        """
        n_samples = scores_subset.shape[0]

        # Choose fill value based on kind
        if kind == 'logits':
            fill_value = -np.inf  # Will become 0 after softmax
        elif kind == 'probs':
            fill_value = 0.0      # Already a probability
        else:
            raise ValueError(f"kind must be 'logits' or 'probs', got {kind}")

        scores_full = np.full((n_samples, self.n_classes), fill_value)

        # Map fitted classes back to their indices
        for i, cls in enumerate(self.fitted_classes):
            scores_full[:, cls] = scores_subset[:, i]

        return scores_full

    def predict_logits(self, X, p_hat, correctness_probs=None):
        """
        Predict logits for all classes.

        Args:
            X: raw features (n_samples, n_features) or None
            p_hat: predicted probabilities (n_samples, n_classes)
            correctness_probs: optional correctness estimates (n_samples,)

        Returns:
            logits: array of shape (n_samples, n_classes)
        """
        if self.fallback_mode:
            # Fallback: return logits from renormalization
            eps = 1e-10
            y_hat = np.argmax(p_hat, axis=1)
            p_hat_clipped = np.clip(p_hat, eps, 1 - eps)
            logits = np.log(p_hat_clipped)
            # Don't return -inf for top class yet; masking happens in estimate_eta_learned
            return logits

        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Build features
        features = make_eta_features(X, p_hat, correctness_probs,
                                     include_raw_features=self.include_raw_features)
        features_scaled = self.scaler.transform(features)

        # Get decision function (logits before softmax)
        logits_subset = self.model.decision_function(features_scaled)

        # Handle binary case (sklearn returns 1D for binary classification)
        if len(logits_subset.shape) == 1:
            # Binary: expand to 2 classes
            logits_subset = np.column_stack([-logits_subset, logits_subset])

        # Expand to all K classes (fill missing with -inf)
        logits_full = self._expand_to_all_classes(logits_subset, kind='logits')

        return logits_full

    def predict_proba(self, X, p_hat, correctness_probs=None):
        """
        Predict probabilities for all classes (before masking).

        Args:
            X: raw features (n_samples, n_features) or None
            p_hat: predicted probabilities (n_samples, n_classes)
            correctness_probs: optional correctness estimates (n_samples,)

        Returns:
            probs: array of shape (n_samples, n_classes)
        """
        if self.fallback_mode:
            # Fallback: use renormalization
            return p_hat

        if self.model is None:
            raise ValueError("Model not fitted yet!")

        # Build features
        features = make_eta_features(X, p_hat, correctness_probs,
                                     include_raw_features=self.include_raw_features)
        features_scaled = self.scaler.transform(features)

        # Get probabilities (these are already probabilities, not logits!)
        probs_subset = self.model.predict_proba(features_scaled)

        # Expand to all K classes (fill missing with 0.0, not -inf!)
        probs_full = self._expand_to_all_classes(probs_subset, kind='probs')

        # Renormalize (some mass might have gone to missing classes)
        eps = 1e-10
        row_sums = np.sum(probs_full, axis=1, keepdims=True)
        row_sums = np.where(row_sums > eps, row_sums, 1.0)
        probs_full = probs_full / row_sums

        return probs_full


def estimate_eta_learned(X, p_hat, eta_adapter, correctness_probs=None):
    """
    Estimate eta using the learned adapter.

    Predicts full K-class probabilities, then masks the top-1 class and renormalizes.
    Improved numerical stability and edge-case handling.

    Args:
        X: raw features (n_samples, n_features) or None
        p_hat: predicted probabilities (n_samples, n_classes)
        eta_adapter: fitted EtaAdapterMultinomial
        correctness_probs: optional correctness estimates (n_samples,)

    Returns:
        eta: array of shape (n_samples, n_classes) with top-1 class set to 0
    """
    n_samples, n_classes = p_hat.shape
    eps = 1e-10

    # Get predicted class
    y_hat = np.argmax(p_hat, axis=1)

    # If in fallback mode, use renormalization
    if eta_adapter.fallback_mode:
        warnings.warn("Eta adapter in fallback mode, using renormalization", stacklevel=2)
        return estimate_eta_renormalize(p_hat)

    # Get logits from adapter
    logits = eta_adapter.predict_logits(X, p_hat, correctness_probs)

    # Mask top-1 class by setting to -inf
    logits_masked = logits.copy()
    logits_masked[np.arange(n_samples), y_hat] = -np.inf

    # Apply softmax to get probabilities (with numerical stability)
    # Subtract max for numerical stability before exp
    logits_max = np.max(logits_masked, axis=1, keepdims=True)
    # Handle case where all logits are -inf (shouldn't happen but be safe)
    logits_max = np.where(np.isfinite(logits_max), logits_max, 0)

    logits_stable = logits_masked - logits_max
    exp_logits = np.exp(logits_stable)

    # Sum and normalize
    row_sums = np.sum(exp_logits, axis=1, keepdims=True)
    row_sums = np.where(row_sums > eps, row_sums, 1.0)  # Avoid division by zero

    eta = exp_logits / row_sums

    # Ensure top-1 class is exactly 0 (handle numerical issues)
    eta[np.arange(n_samples), y_hat] = 0.0

    # Final renormalization to ensure sum to 1 (handle any numerical drift)
    row_sums = np.sum(eta, axis=1, keepdims=True)
    row_sums = np.where(row_sums > eps, row_sums, 1.0)
    eta = eta / row_sums

    # Sanity check: all values should be finite and non-negative
    if not np.all(np.isfinite(eta)) or not np.all(eta >= 0):
        warnings.warn("Numerical issues in eta estimation, falling back to renormalization")
        return estimate_eta_renormalize(p_hat)

    return eta


def estimate_eta_uniform(p_hat):
    """
    Estimate eta by distributing probability uniformly across non-top-1 classes.

    This is a simpler alternative to renormalization that doesn't rely on
    the pretrained model's probability distribution for non-top classes.

    Args:
        p_hat: array of shape (n_samples, n_classes)

    Returns:
        eta: array of shape (n_samples, n_classes) where:
            - eta[i, k] = 0 if k is the top-1 prediction for sample i
            - eta[i, k] = 1/(n_classes-1) otherwise
    """
    n_samples, n_classes = p_hat.shape

    # Get predicted class for each sample
    y_hat = np.argmax(p_hat, axis=1)

    # Initialize eta with uniform probability across all classes
    uniform_prob = 1.0 / (n_classes - 1) if n_classes > 1 else 0.0
    eta = np.full((n_samples, n_classes), uniform_prob)

    # Set predicted class probability to 0
    eta[np.arange(n_samples), y_hat] = 0.0

    return eta

def estimate_eta_renormalize(p_hat):
    """
    Vectorized version of estimate_eta for better performance.

    Same functionality as estimate_eta but uses numpy broadcasting
    instead of explicit loops.

    Args:
        p_hat: array of shape (n_samples, n_classes)

    Returns:
        eta: array of shape (n_samples, n_classes)
    """
    n_samples, n_classes = p_hat.shape

    # Get predicted class for each sample
    y_hat = np.argmax(p_hat, axis=1)

    # Get predicted class probabilities
    p_hat_y_hat = p_hat[np.arange(n_samples), y_hat]

    # Compute denominator: 1 - p_hat_{y_hat}
    denominator = 1.0 - p_hat_y_hat + 1e-10

    # Divide all probabilities by denominator (broadcast)
    eta = p_hat / denominator[:, np.newaxis]

    # Set predicted class probability to 0
    eta[np.arange(n_samples), y_hat] = 0.0

    return eta


def estimate_eta_confusion_prior(p_hat, confusion_matrix, smoothing_alpha=1.0):
    """
    Estimate eta using confusion prior from calibration data (vectorized).

    This method uses the empirical confusion matrix to estimate where probability
    mass should go when the top-1 prediction is wrong. For each top-1 predicted
    class j, we look at the distribution of true labels k when the prediction was
    wrong (Y ≠ j).

    Args:
        p_hat: array of shape (n_samples, n_classes) - predicted probabilities
        confusion_matrix: array of shape (n_classes, n_classes) where
                         confusion_matrix[j, k] = count of samples where
                         Y_hat = j and Y = k
        smoothing_alpha: Dirichlet/Laplace smoothing parameter to avoid zeros (default: 1.0)

    Returns:
        eta: array of shape (n_samples, n_classes) where:
            - eta[i, k] represents P(Y=k | Y_hat=j, Y≠j) for the predicted class j of sample i
            - eta[i, j] = 0 when j is the top-1 prediction (by definition)
    """
    n_samples, n_classes = p_hat.shape

    # Get predicted class for each sample
    y_hat = np.argmax(p_hat, axis=1)  # Shape: (n_samples,)

    # Pre-compute eta distribution for each possible predicted class
    eta_by_class = np.zeros((n_classes, n_classes))

    for j in range(n_classes):
        # Get the confusion row for predicted class j
        confusion_row = confusion_matrix[j, :].copy()

        # Zero out the diagonal BEFORE smoothing
        confusion_row[j] = 0

        # Add smoothing only to off-diagonal entries
        confusion_row = confusion_row + smoothing_alpha

        # Keep diagonal at zero
        confusion_row[j] = 0

        # Normalize to get probability distribution
        total = np.sum(confusion_row)
        if total > 0:
            eta_by_class[j, :] = confusion_row / total
        else:
            # Fallback to uniform over non-predicted classes if no errors observed
            uniform_prob = 1.0 / (n_classes - 1) if n_classes > 1 else 0.0
            eta_by_class[j, :] = uniform_prob
            eta_by_class[j, j] = 0.0

    # Vectorized lookup: for each sample i, get eta distribution for its predicted class
    eta = eta_by_class[y_hat]  # Shape: (n_samples, n_classes)

    return eta

def build_confusion_matrix(y_true, y_pred, n_classes):
    """
    Build confusion matrix from true labels and predictions.

    Args:
        y_true: array of true labels
        y_pred: array of predicted labels
        n_classes: number of classes

    Returns:
        confusion_matrix: array of shape (n_classes, n_classes) where
                         confusion_matrix[j, k] = #{(x_i, y_i) : Y_hat(x_i) = j, y_i = k}
    """
    confusion_matrix = np.zeros((n_classes, n_classes))

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[pred_label, true_label] += 1

    return confusion_matrix


def estimate_eta_vectorized(p_hat):
    """
    Vectorized version of estimate_eta for better performance.

    Same functionality as estimate_eta but uses numpy broadcasting
    instead of explicit loops.

    Args:
        p_hat: array of shape (n_samples, n_classes)

    Returns:
        eta: array of shape (n_samples, n_classes)
    """
    n_samples, n_classes = p_hat.shape

    # Get predicted class for each sample
    y_hat = np.argmax(p_hat, axis=1)

    # Get predicted class probabilities
    p_hat_y_hat = p_hat[np.arange(n_samples), y_hat]

    # Compute denominator: 1 - p_hat_{y_hat}
    denominator = 1.0 - p_hat_y_hat + 1e-10

    # Divide all probabilities by denominator (broadcast)
    eta = p_hat / denominator[:, np.newaxis]

    # Set predicted class probability to 0
    eta[np.arange(n_samples), y_hat] = 0.0

    return eta



