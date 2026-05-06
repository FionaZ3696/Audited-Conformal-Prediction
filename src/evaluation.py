import numpy as np
import pandas as pd

def evaluate_coverage_with_auditing(
    alpha,
    prediction_sets,
    y_true,
    X_test,
    primary_model_preds,
    data_model,
    n_bins=[3, 11],
    sample_types=None,
    test_pred_prob=None
):
    """
    Evaluates standard coverage metrics PLUS 'Audited Conditional Coverage'
    across multiple binning strategies.

    Args:
        n_bins: int or List[int]. Example: [3, 11].
                Calculates conditional coverage for each bin count strategy.
    """

    # --- 1. Compute Oracle r*(x) via the data model's calculate_c_star ---
    r_star_scores = data_model.calculate_c_star(X_test, primary_model_preds)
    # pdb.set_trace()

    # --- 2. Calculate Standard Metrics ---
    marginal_coverage = np.mean([y_true[i] in pred_set for i, pred_set in enumerate(prediction_sets)])
    avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])

    results = {
        'Marginal_Coverage': marginal_coverage,
        'Size': avg_size,
        'alpha': 1 - alpha
    }

    # Conditional (Binary: Original vs Shifted) - keeping existing logic
    if sample_types is not None:
        original_mask = sample_types == 1
        shifted_mask = sample_types == 0

        if np.any(original_mask):
            results['Conditional_Coverage_Original'] = np.mean([y_true[i] in prediction_sets[i] for i in np.where(original_mask)[0]])
            results['Size_Original'] = np.mean([len(prediction_sets[i]) for i in np.where(original_mask)[0]])

        if np.any(shifted_mask):
            results['Conditional_Coverage_Shifted'] = np.mean([y_true[i] in prediction_sets[i] for i in np.where(shifted_mask)[0]])
            results['Size_Shifted'] = np.mean([len(prediction_sets[i]) for i in np.where(shifted_mask)[0]])

    # --- 3. Audited Conditional Coverage (Multiple Binning Strategies) ---

    # Ensure n_bins is a list
    if isinstance(n_bins, (int, float)):
        bin_strategies = [int(n_bins)]
    else:
        bin_strategies = n_bins

    # Pre-calculate covered status to avoid re-computing inside loop
    is_covered = np.array([y_true[i] in prediction_sets[i] for i in range(len(y_true))])
    set_sizes = np.array([len(s) for s in prediction_sets])

    for n in bin_strategies:
        # A. Define Bins for this strategy
        # We use linspace, but expand edges slightly to safely catch 0.0 and 1.0
        # For n=11, edges are roughly 0.0, 0.09, 0.18... (Good for separating 0 and 0.1)
        bins = np.linspace(0, 1, n + 1)
        # Adjust outer edges to include 0.0 and 1.0 robustly
        bins[0] = -0.01
        bins[-1] = 1.01

        # B. Create DataFrame for this specific binning
        df_audit = pd.DataFrame({
            'r_star': r_star_scores,
            'covered': is_covered,
            'set_size': set_sizes
        })

        # C. Assign bins
        df_audit['bin'] = pd.cut(df_audit['r_star'], bins)

        # D. Aggregate
        bin_stats = df_audit.groupby('bin', observed=False).agg(
            coverage=('covered', 'mean'),
            avg_size=('set_size', 'mean'),
            count=('covered', 'count'),
            avg_r_star=('r_star', 'mean')
        ).reset_index()

        # E. Store results with Strategy Identifier (_N{n})
        # Bins are 1-indexed: Bin_1 = hardest (lowest r*), Bin_{n} = easiest (highest r*)
        # All n bins are always written (NaN coverage/size for empty bins, count=0)
        for i, row in bin_stats.iterrows():
            suffix = f"Bin_{i + 1}_N{n}"
            results[f'Coverage_{suffix}'] = row['coverage'] if row['count'] > 0 else float('nan')
            results[f'Size_{suffix}'] = row['avg_size'] if row['count'] > 0 else float('nan')
            results[f'Count_{suffix}'] = int(row['count'])
            results[f'r_star_Mean_{suffix}'] = row['avg_r_star'] if row['count'] > 0 else float('nan')

    # --- 4. Avg Max-Softmax (overconfidence measure) ---
    if test_pred_prob is not None:
        max_prob = test_pred_prob.max(axis=1)
        results['AvgMaxSoftmax_Overall'] = float(max_prob.mean())

        if sample_types is not None:
            easy_mask = sample_types == 1
            hard_mask = sample_types == 0
            if easy_mask.any():
                results['AvgMaxSoftmax_Easy'] = float(max_prob[easy_mask].mean())
            if hard_mask.any():
                results['AvgMaxSoftmax_Hard'] = float(max_prob[hard_mask].mean())
    return results



def evaluate_coverage_with_auditing_real(
    alpha,
    prediction_sets,
    y_true,
    X_test=None,
    n_bins=[3, 11],
    sample_types=None,
    test_pred_prob=None,
    r_star_method='knn',
    X_eval=None,
    correctness_eval=None,
    eval_auditor=None,
    k=10,
    r_hat_test=None,
):
    """
    Evaluates audited conditional coverage for REAL DATA, where r*(x) is
    approximated using an independent evaluation dataset D_eval.

    Three approximation methods:

    'precomputed' (or pass r_hat_test directly) — Use pre-computed r* estimates:
        Skips internal computation and uses the provided r_hat_test array directly.
        Useful when r* is computed externally (e.g., kNN on CNN embeddings).
        Requires: r_hat_test.

    'knn' — Nonparametric k-NN estimate:
        For each test point x, finds k nearest neighbors in X_eval and estimates
        r*(x) as the fraction of neighbors where the primary model was correct.
        Requires: X_eval, correctness_eval (= 1{primary_model.predict(X_eval)==y_eval}).

    'auditor' — Independent audit model trained on D_eval:
        Uses eval_auditor.predict_proba(X_test)[:, 1] to estimate r*(x).
        The auditor must be trained on D_eval independently of the ACP auditor.
        Requires: eval_auditor.

    Args:
        alpha:             Miscoverage level.
        prediction_sets:   List of prediction sets for test points.
        y_true:            True test labels (n_test,).
        X_test:            Test features (n_test, p). Required for 'knn' and 'auditor'.
        n_bins:            int or List[int]. Binning strategies for conditional coverage.
        sample_types:      Optional (n_test,) array: 1=easy/original, 0=hard/shifted.
        test_pred_prob:    Optional predicted probability matrix (n_test, K).
        r_star_method:     'knn', 'auditor', or 'precomputed'.
        X_eval:            D_eval features (n_eval, p). Required for 'knn'.
        correctness_eval:  Binary correctness on D_eval (n_eval,). Required for 'knn'.
        eval_auditor:      Trained independent auditor with predict_proba(). Required for 'auditor'.
        k:                 Number of nearest neighbors for 'knn'.
        r_hat_test:        Pre-computed r* estimates (n_test,). If provided, overrides r_star_method.
    """
    from sklearn.neighbors import NearestNeighbors

    # --- 1. Approximate r*(x) ---
    if r_hat_test is not None:
        r_star_scores = np.asarray(r_hat_test)

    elif r_star_method == 'knn':
        assert X_eval is not None and correctness_eval is not None, \
            "r_star_method='knn' requires X_eval and correctness_eval."
        correctness_eval = np.asarray(correctness_eval)
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nn.fit(X_eval)
        _, indices = nn.kneighbors(X_test)
        # r*(x) ≈ fraction of k neighbors where primary model was correct
        r_star_scores = correctness_eval[indices].mean(axis=1)

    elif r_star_method == 'auditor':
        assert eval_auditor is not None, \
            "r_star_method='auditor' requires eval_auditor."
        r_star_scores = eval_auditor.predict_proba(X_test)[:, 1]

    else:
        raise ValueError(
            f"Unknown r_star_method '{r_star_method}'. Choose 'knn', 'auditor', or 'precomputed'."
        )

    # --- 2. Standard metrics ---
    marginal_coverage = np.mean([y_true[i] in pred_set for i, pred_set in enumerate(prediction_sets)])
    avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])

    results = {
        'Marginal_Coverage': marginal_coverage,
        'Size': avg_size,
        'alpha': 1 - alpha,
        'r_star_method': r_star_method,
    }

    # --- 3. Binary conditional coverage (easy/hard or original/shifted) ---
    if sample_types is not None:
        original_mask = sample_types == 1
        shifted_mask  = sample_types == 0

        if np.any(original_mask):
            results['Conditional_Coverage_Original'] = np.mean(
                [y_true[i] in prediction_sets[i] for i in np.where(original_mask)[0]])
            results['Size_Original'] = np.mean(
                [len(prediction_sets[i]) for i in np.where(original_mask)[0]])

        if np.any(shifted_mask):
            results['Conditional_Coverage_Shifted'] = np.mean(
                [y_true[i] in prediction_sets[i] for i in np.where(shifted_mask)[0]])
            results['Size_Shifted'] = np.mean(
                [len(prediction_sets[i]) for i in np.where(shifted_mask)[0]])

    # --- 4. Audited conditional coverage across binning strategies ---
    if isinstance(n_bins, (int, float)):
        bin_strategies = [int(n_bins)]
    else:
        bin_strategies = list(n_bins)

    is_covered = np.array([y_true[i] in prediction_sets[i] for i in range(len(y_true))])
    set_sizes  = np.array([len(s) for s in prediction_sets])

    for n in bin_strategies:
        bins = np.linspace(0, 1, n + 1)
        bins[0]  = -0.01
        bins[-1] = 1.01

        df_audit = pd.DataFrame({
            'r_star':    r_star_scores,
            'covered':   is_covered,
            'set_size':  set_sizes,
        })
        df_audit['bin'] = pd.cut(df_audit['r_star'], bins)

        bin_stats = df_audit.groupby('bin', observed=False).agg(
            coverage=('covered',  'mean'),
            avg_size=('set_size', 'mean'),
            count=   ('covered',  'count'),
            avg_r_star=('r_star', 'mean'),
        ).reset_index()

        for i, row in bin_stats.iterrows():
            suffix = f"Bin_{i + 1}_N{n}"
            results[f'Coverage_{suffix}']    = row['coverage']   if row['count'] > 0 else float('nan')
            results[f'Size_{suffix}']        = row['avg_size']   if row['count'] > 0 else float('nan')
            results[f'Count_{suffix}']       = int(row['count'])
            results[f'r_star_Mean_{suffix}'] = row['avg_r_star'] if row['count'] > 0 else float('nan')

    # --- 5. Avg max-softmax (overconfidence measure) ---
    if test_pred_prob is not None:
        max_prob = test_pred_prob.max(axis=1)
        results['AvgMaxSoftmax_Overall'] = float(max_prob.mean())

        if sample_types is not None:
            easy_mask = sample_types == 1
            hard_mask = sample_types == 0
            if easy_mask.any():
                results['AvgMaxSoftmax_Easy'] = float(max_prob[easy_mask].mean())
            if hard_mask.any():
                results['AvgMaxSoftmax_Hard'] = float(max_prob[hard_mask].mean())

    return results


def evaluate_coverage(alpha, prediction_sets, y_true, test_pred_prob=None, sample_types=None):
    """
    Evaluate marginal and conditional coverage, overconfidence, and calibration (ECE)

    Args:
        alpha: Significance level
        prediction_sets: List of prediction sets
        y_true: True labels
        method: Method name
        test_pred_prob: Probability matrix (n, 2) for binary classification
        eta_method: Eta method name
        sample_types: Array indicating sample type (1=easy, 0=hard)
    """
    # Marginal coverage
    marginal_coverage = np.mean([y_true[i] in pred_set
                                for i, pred_set in enumerate(prediction_sets)])

    # Average prediction set size
    avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])

    results = {
        'Marginal_Coverage': marginal_coverage,
        'Size': avg_size,
        'alpha': 1 - alpha
    }

    # Conditional coverage by sample type (easy/hard)
    if sample_types is not None:
        easy_mask = sample_types == 1
        hard_mask = sample_types == 0

        if np.any(easy_mask):
            easy_coverage = np.mean([y_true[i] in prediction_sets[i]
                                    for i in range(len(y_true)) if easy_mask[i]])
            easy_size = np.mean([len(prediction_sets[i])
                                for i in range(len(y_true)) if easy_mask[i]])
            results['Conditional_Coverage_Easy'] = easy_coverage
            results['Size_Easy'] = easy_size

        if np.any(hard_mask):
            hard_coverage = np.mean([y_true[i] in prediction_sets[i]
                                    for i in range(len(y_true)) if hard_mask[i]])
            hard_size = np.mean([len(prediction_sets[i])
                                for i in range(len(y_true)) if hard_mask[i]])
            results['Conditional_Coverage_Hard'] = hard_coverage
            results['Size_Hard'] = hard_size

    # Initialize overconfidence and ECE metrics with None
    overconfidence_columns = [
        'adjusted_confidence_Mean',
        'ECE',
        'adjusted_confidence_Mean_Easy',
        'ECE_Easy',
        'adjusted_confidence_Mean_Hard',
        'ECE_Hard'
    ]

    for col in overconfidence_columns:
        results[col] = None

    # Compute overconfidence and ECE metrics if predictions are provided
    if test_pred_prob is not None:
        overconfidence_metrics = compute_overconfidence_and_ece(
            test_pred_prob, y_true, sample_types
        )
        results.update(overconfidence_metrics)

    return results



def compute_overconfidence_and_ece(test_pred_prob, y_true, sample_types=None, n_bins=10):
    """
    Compute overconfidence and Expected Calibration Error (ECE) metrics.

    Args:
        test_pred_prob: Probability matrix (n, 2) for binary classification
        y_true: True labels (n,)
        sample_types: Array indicating sample type (1=easy, 0=hard)
        n_bins: Number of bins for ECE calculation

    Returns:
        Dictionary with overconfidence and ECE metrics
    """
    # Get predicted probabilities for the predicted class
    pred_probs = np.max(test_pred_prob, axis=1)  # Max probability (confidence)

    # Overconfidence: 2 * |p - 0.5|
    adjusted_confidence = 2 * np.abs(pred_probs - 0.5)

    # All samples metrics
    metrics = {
        'adjusted_confidence_Mean': np.mean(adjusted_confidence),
    }

    # ECE for all samples
    ece_all = compute_ece(test_pred_prob, y_true, n_bins=n_bins)
    metrics['ECE'] = ece_all

    # Conditional metrics by sample type
    if sample_types is not None:
        easy_mask = sample_types == 1
        hard_mask = sample_types == 0

        # Easy samples
        if np.any(easy_mask):
            adjusted_confidence_easy = adjusted_confidence[easy_mask]
            metrics['adjusted_confidence_Mean_Easy'] = np.mean(adjusted_confidence_easy)
            ece_easy = compute_ece(test_pred_prob[easy_mask], y_true[easy_mask], n_bins=n_bins)
            metrics['ECE_Easy'] = ece_easy

        # Hard samples
        if np.any(hard_mask):
            adjusted_confidence_hard = adjusted_confidence[hard_mask]
            metrics['adjusted_confidence_Mean_Hard'] = np.mean(adjusted_confidence_hard)
            ece_hard = compute_ece(test_pred_prob[hard_mask], y_true[hard_mask], n_bins=n_bins)
            metrics['ECE_Hard'] = ece_hard

    return metrics


def compute_ece(pred_probs, y_true, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        pred_probs: Probability matrix (n, 2) for binary classification
        y_true: True labels (n,)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (float)
    """
    # Get predicted class and confidence
    pred_class = np.argmax(pred_probs, axis=1)
    confidences = np.max(pred_probs, axis=1)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = np.mean(pred_class[in_bin] == y_true[in_bin])

            # Average confidence in this bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # ECE contribution from this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


