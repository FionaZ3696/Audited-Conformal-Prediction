from scipy.stats.mstats import mquantiles
import numpy as np


def compute_nonconf_scores(y_proba, y_true, alpha = 0.1, random_state = 2025):
  n = len(y_proba)
  grey_box = ProbAccum(y_proba)
  rng = np.random.RandomState(random_state)
  epsilon = rng.uniform(low=0.0, high=1.0, size=n)
  alpha_max = grey_box.calibrate_scores(y_true, epsilon=epsilon)
  scores = alpha - alpha_max
  return scores

def calibrate_alpha(scores, alpha = 0.1):
  n_cal = len(scores)
  if n_cal == 0:
    return alpha
  level_adjusted = (1.0 - alpha)*(1.0 + 1.0/float(n_cal))
  alpha_correction = mquantiles(scores, prob=level_adjusted)
  alpha_calibrated = alpha - alpha_correction
  return alpha_calibrated

def predict_set(y_proba, alpha_calibrated, allow_empty = True, random_state = 2025):
  rng = np.random.RandomState(random_state+1)
  n = len(y_proba)
  epsilon = rng.uniform(low=0.0, high=1.0, size=n)
  grey_box = ProbAccum(y_proba)
  S_hat = grey_box.predict_sets(alpha_calibrated, epsilon=epsilon, allow_empty = allow_empty)
  return S_hat

class ProbAccum:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        if alpha>0:
            L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        else:
            L = (self.Z.shape[1]-1)*np.ones((self.Z.shape[0],)).astype(int)
        if epsilon is not None:
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        if isinstance(Y, int) == False:
          Y = list(map(int, Y))
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max


def romano_score_fn(old_model, X, y, random_state=2025):
    """
    Conformity score S(x, y) = E(x, y, u) for the Romano et al method.
    Returns a 1D numpy array of scores, one per (x_i, y_i).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)

    # Predicted probabilities for each calibration point
    y_proba = old_model.predict_proba(X)  # shape (n, K)

    # Build ProbAccum structure
    grey_box = ProbAccum(y_proba)

    # Randomization u ~ Uniform[0, 1] per calibration example
    rng = np.random.RandomState(random_state)
    epsilon = rng.uniform(low=0.0, high=1.0, size=len(X))

    # alpha_max = 1 - prob_cum + prob * epsilon
    alpha_max = grey_box.calibrate_scores(y, epsilon=epsilon)

    # E(x,y,u) = 1 - alpha_max
    scores = 1 - alpha_max  # higher score = "worse" / less conforming
    return scores

def romano_score_vector(old_model, x, eps=None):
    """
    Compute Romano-style scores S(x, y) = E(x, y, u) for all labels y at a single test point x.
    Returns a 1D array of length K with scores for each label.
    Uses one shared epsilon (u) for all labels of this x.
    """
    x = np.asarray(x)
    if x.ndim == 3:          # single image (C, H, W) → (1, C, H, W)
        x = x[np.newaxis]
    elif x.ndim == 1:        # tabular vector → (1, d)
        x = x.reshape(1, -1)

    proba = old_model.predict_proba(x)[0]  # shape (K,)
    K = proba.shape[0]

    # Sort probabilities descending
    order = np.argsort(-proba)
    prob_sort = proba[order]
    Z = prob_sort.cumsum()  # cumulative probs

    # Invert permutation to get rank of each label
    ranks = np.empty_like(order)
    ranks[order] = np.arange(K)

    # Shared epsilon for this test point
    if eps is None:
        eps = np.random.RandomState().uniform()  # or pass rng in

    # For each label y:
    #   rank = ranks[y]
    #   prob_cum = Z[rank]
    #   prob_y   = prob_sort[rank]
    #   E(x,y,u) = prob_cum - prob_y * eps
    prob_cum_by_label = Z[ranks]
    prob_by_label = prob_sort[ranks]

    scores = prob_cum_by_label - prob_by_label * eps
    return scores  # shape (K,)


def Phi_fn(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Intercept + original features
    n = X.shape[0]
    return np.concatenate([np.ones((n, 1)), X], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Real-data experiment utilities: thin model wrappers, k-NN r* approximator,
# and dataset-specific data-pool helpers (WILDS / CIFAR-10-C).
# These are imported by experiments/run_real_experiment.py.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Shared model wrappers
# =============================================================================

class EmbeddingModel:
    """
    Wraps the classifier head of a CNN to operate on (N, z_dim) embeddings.
    Exposes predict / predict_proba / get_embeddings matching the rest of the
    codebase's primary-model API.
    """
    def __init__(self, cnn_net, infer_batch_size=512):
        self.cnn              = cnn_net
        self.z_dim            = cnn_net.z_dim
        self.device           = cnn_net.device
        self.infer_batch_size = infer_batch_size

    def _run_head(self, emb, return_proba=False):
        self.cnn.eval()
        emb_t   = torch.from_numpy(emb).float() if isinstance(emb, np.ndarray) else emb.float()
        results = []
        with torch.no_grad():
            for s in range(0, len(emb_t), self.infer_batch_size):
                batch  = emb_t[s : s + self.infer_batch_size].to(self.device)
                logits = self.cnn.classifier(batch)
                if return_proba:
                    out = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                else:
                    out = logits.argmax(dim=1).cpu().numpy()
                results.append(out)
        return np.concatenate(results, axis=0)

    def predict_proba(self, emb):
        return self._run_head(emb, return_proba=True)

    def predict(self, emb):
        return self._run_head(emb, return_proba=False)

    def get_embeddings(self, emb):
        if isinstance(emb, torch.Tensor):
            return emb.cpu().numpy()
        return emb


class AuditModelWrapper:
    """
    Wraps a trained audit model that operates on embeddings so it accepts raw
    images (4D tensors) by transparently extracting CNN embeddings first.
    Used so downstream ACP code can pass raw images uniformly.
    """
    def __init__(self, trained_audit_model, cnn_net):
        self.audit_model = trained_audit_model
        self.cnn         = cnn_net

    def _to_emb(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 4:
            return self.cnn.get_embeddings(X)
        return X

    def predict(self, X):
        return self.audit_model.predict(self._to_emb(X))

    def predict_proba(self, X):
        return self.audit_model.predict_proba(self._to_emb(X))


# =============================================================================
# r_hat approximator (shared by both datasets)
# =============================================================================

def compute_r_hat_knn(old_model, test_X, eval_X, eval_y, k=50, metric='cosine'):
    """
    Approximate r*(x) for each test point using k-NN on legacy-model embeddings.

    Returns
    -------
    r_hat_test       : (n_test,)  float64
    correctness_eval : (n_eval,)  float64
    """
    assert metric in ('cosine', 'euclidean')

    print(f"Extracting embeddings  (eval={len(eval_X)}, test={len(test_X)}) ...")
    eval_emb = old_model.net.get_embeddings(eval_X)
    test_emb = old_model.net.get_embeddings(test_X)

    eval_preds       = old_model.net.predict(eval_X)
    correctness_eval = (eval_preds == eval_y).astype(np.float64)
    print(f"  Eval correctness: {correctness_eval.mean():.3f}  "
          f"({int(correctness_eval.sum())}/{len(correctness_eval)} correct)")

    print(f"Running {k}-NN  (metric={metric}) ...")
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=metric)
    nn.fit(eval_emb)
    _, indices = nn.kneighbors(test_emb)

    r_hat_test = correctness_eval[indices].mean(axis=1)
    print(f"  r_hat_test : min={r_hat_test.min():.3f}  "
          f"max={r_hat_test.max():.3f}  "
          f"mean={r_hat_test.mean():.3f}")

    return r_hat_test, correctness_eval


# =============================================================================
# Camelyon17-WILDS data helpers
# =============================================================================

def _filter_by_center(wilds_sub, center_id):
    from torch.utils.data import Subset
    hospital = wilds_sub.metadata_array[:, 0]
    idx = torch.where(hospital == center_id)[0].numpy()
    return Subset(wilds_sub, idx)


def _collect_indices(torch_dataset, indices, batch_size=256):
    from torch.utils.data import Subset, DataLoader
    sub    = Subset(torch_dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=False)
    imgs, labels = [], []
    for x, y, *_ in loader:
        imgs.append(x.numpy())
        labels.append(y.numpy())
    return (np.concatenate(imgs,   axis=0).astype(np.float32),
            np.concatenate(labels, axis=0).astype(np.int64))


def _collect_centers(torch_dataset, indices, batch_size=256):
    from torch.utils.data import Subset, DataLoader
    sub    = Subset(torch_dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=False)
    centers = []
    for x, y, meta in loader:
        centers.append(meta[:, 0].numpy())
    return np.concatenate(centers, axis=0).astype(np.int64)


def load_wilds_splits(wilds_root, transform):
    """Load Camelyon17-WILDS dataset splits."""
    from wilds import get_dataset
    print("Loading Camelyon17-WILDS ...")
    dataset      = get_dataset(dataset="camelyon17", download=True, root_dir=wilds_root)
    train_wilds  = dataset.get_subset("train",  transform=transform)
    idval_wilds  = dataset.get_subset("id_val", transform=transform)
    val_wilds    = dataset.get_subset("val",    transform=transform)
    test_wilds   = dataset.get_subset("test",   transform=transform)
    return train_wilds, idval_wilds, val_wilds, test_wilds


def load_hist(train_wilds, idval_wilds, n_hist, seed, verbose=True):
    """Draw n_hist samples from center 0 (historical population)."""
    from torch.utils.data import ConcatDataset
    hist_pool = ConcatDataset([
        _filter_by_center(train_wilds, 0),
        _filter_by_center(idval_wilds, 0),
    ])
    assert len(hist_pool) >= n_hist, \
        f"hist_pool too small: need {n_hist}, have {len(hist_pool)}"

    rng_hist = np.random.default_rng(seed)
    hist_idx = rng_hist.permutation(len(hist_pool))[:n_hist]

    if verbose:
        print(f"Loading hist (center 0, n={n_hist}) ...")
    hist_X, hist_y = _collect_indices(hist_pool, hist_idx)

    if verbose:
        centers = _collect_centers(hist_pool, hist_idx)
        assert np.all(centers == 0), "hist contains non-center-0 samples"
        print(f"  {hist_X.shape}   labels: {dict(Counter(hist_y))}")
        print(f"  Center composition: {dict(Counter(centers))}")

    return hist_X, hist_y, hist_idx


def _get_shift_pool(train_wilds, idval_wilds, val_wilds, test_wilds, shift_center):
    from torch.utils.data import ConcatDataset
    if shift_center == 1:
        return val_wilds
    elif shift_center == 2:
        return test_wilds
    elif shift_center in {3, 4}:
        return ConcatDataset([
            _filter_by_center(train_wilds, shift_center),
            _filter_by_center(idval_wilds, shift_center),
        ])
    raise ValueError(f"shift_center must be in {{1,2,3,4}}, got {shift_center}")


def load_new_test(train_wilds, idval_wilds, val_wilds, test_wilds,
                  shift_center, beta, n_new, n_test, seed,
                  hist_idx, verbose=True):
    """Build new / test pools mixing center 0 (ID) with `shift_center` (OOD)."""
    from torch.utils.data import ConcatDataset
    assert shift_center in {1, 2, 3, 4}

    n_new_hist  = int((1 - beta) * n_new);  n_new_shift  = n_new  - n_new_hist
    n_test_hist = int((1 - beta) * n_test); n_test_shift = n_test - n_test_hist

    if verbose:
        print(f"\n{'='*60}")
        print(f"load_new_test: center 0 vs center {shift_center}  |  beta={beta}")
        print(f"  new  = {n_new}  ({n_new_hist} ID + {n_new_shift} OOD)")
        print(f"  test = {n_test}  ({n_test_hist} ID + {n_test_shift} OOD)")
        print(f"{'='*60}")

    id_pool = ConcatDataset([
        _filter_by_center(train_wilds, 0),
        _filter_by_center(idval_wilds, 0),
    ])
    shift_pool = _get_shift_pool(train_wilds, idval_wilds, val_wilds, test_wilds, shift_center)

    needed_id    = n_new_hist + n_test_hist
    needed_shift = n_new_shift + n_test_shift
    assert len(id_pool) - len(hist_idx) >= needed_id, \
        (f"Not enough free ID samples after excluding hist: "
         f"need {needed_id}, have {len(id_pool) - len(hist_idx)}")
    assert len(shift_pool) >= needed_shift

    rng = np.random.default_rng(seed + 1)

    hist_idx_set = set(np.asarray(hist_idx).tolist())
    id_idx_full  = rng.permutation(len(id_pool))
    keep_mask    = np.array([i not in hist_idx_set for i in id_idx_full], dtype=bool)
    id_idx       = id_idx_full[keep_mask]

    new_id_idx   = id_idx[:n_new_hist]
    test_id_idx  = id_idx[n_new_hist : n_new_hist + n_test_hist]

    s_idx        = rng.permutation(len(shift_pool))
    new_shft_idx = s_idx[:n_new_shift]
    tst_shft_idx = s_idx[n_new_shift : n_new_shift + n_test_shift]

    assert len(hist_idx_set & set(new_id_idx.tolist()))  == 0
    assert len(hist_idx_set & set(test_id_idx.tolist())) == 0
    assert len(set(new_id_idx.tolist())   & set(test_id_idx.tolist()))  == 0
    assert len(set(new_shft_idx.tolist()) & set(tst_shft_idx.tolist())) == 0

    if verbose: print("\nLoading arrays ...")
    new_id_X,   new_id_y   = _collect_indices(id_pool,    new_id_idx)
    new_shft_X, new_shft_y = _collect_indices(shift_pool, new_shft_idx)
    tst_id_X,   tst_id_y   = _collect_indices(id_pool,    test_id_idx)
    tst_shft_X, tst_shft_y = _collect_indices(shift_pool, tst_shft_idx)

    new_X  = np.concatenate([new_id_X,  new_shft_X], axis=0)
    new_y  = np.concatenate([new_id_y,  new_shft_y], axis=0)
    sample_types = np.array([1]*n_new_hist + [0]*n_new_shift, dtype=np.int64)

    test_X = np.concatenate([tst_id_X,  tst_shft_X], axis=0)
    test_y = np.concatenate([tst_id_y,  tst_shft_y], axis=0)
    test_sample_types = np.array([1]*n_test_hist + [0]*n_test_shift, dtype=np.int64)

    perm_new  = rng.permutation(len(new_y))
    new_X, new_y, sample_types = new_X[perm_new], new_y[perm_new], sample_types[perm_new]

    perm_test = rng.permutation(len(test_y))
    test_X, test_y, test_sample_types = test_X[perm_test], test_y[perm_test], test_sample_types[perm_test]

    if verbose:
        print(f"\nnew  : {new_X.shape}  "
              f"ID={(sample_types==1).sum()} / OOD={(sample_types==0).sum()}  "
              f"labels: {dict(Counter(new_y))}")
        print(f"test : {test_X.shape}  "
              f"ID={(test_sample_types==1).sum()} / OOD={(test_sample_types==0).sum()}  "
              f"labels: {dict(Counter(test_y))}")

    return (new_X, new_y, sample_types,
            test_X, test_y, test_sample_types)


def _reconstruct_used_indices(train_wilds, idval_wilds, val_wilds, test_wilds,
                              shift_center, beta, n_hist, n_new, n_test, seed):
    """Replay RNG to recover indices consumed by hist/new/test on WILDS pools."""
    from torch.utils.data import ConcatDataset
    id_pool = ConcatDataset([
        _filter_by_center(train_wilds, 0),
        _filter_by_center(idval_wilds, 0),
    ])
    shift_pool = _get_shift_pool(train_wilds, idval_wilds, val_wilds, test_wilds, shift_center)

    rng_hist = np.random.default_rng(seed)
    hist_idx = rng_hist.permutation(len(id_pool))[:n_hist]
    hist_idx_set = set(hist_idx.tolist())

    n_new_hist  = int((1 - beta) * n_new)
    n_test_hist = int((1 - beta) * n_test)
    n_new_shift  = n_new  - n_new_hist
    n_test_shift = n_test - n_test_hist

    rng = np.random.default_rng(seed + 1)

    id_idx_full  = rng.permutation(len(id_pool))
    keep_mask    = np.array([i not in hist_idx_set for i in id_idx_full], dtype=bool)
    id_idx       = id_idx_full[keep_mask]

    new_id_idx   = id_idx[:n_new_hist]
    test_id_idx  = id_idx[n_new_hist : n_new_hist + n_test_hist]

    s_idx        = rng.permutation(len(shift_pool))
    new_shft_idx = s_idx[:n_new_shift]
    tst_shft_idx = s_idx[n_new_shift : n_new_shift + n_test_shift]

    used_id_idx   = np.concatenate([hist_idx, new_id_idx, test_id_idx])
    used_shft_idx = np.concatenate([new_shft_idx, tst_shft_idx])

    assert len(set(used_id_idx.tolist())) == len(used_id_idx), \
        "Duplicate indices detected across hist/new/test ID splits"
    assert len(set(used_shft_idx.tolist())) == len(used_shft_idx), \
        "Duplicate indices detected across new/test OOD splits"

    print(f"Reconstructed used indices -- "
          f"ID: {len(used_id_idx)} / {len(id_pool)}  "
          f"OOD: {len(used_shft_idx)} / {len(shift_pool)}")

    return used_id_idx, used_shft_idx


def load_eval(train_wilds, idval_wilds, val_wilds, test_wilds,
              shift_center, beta, n_eval,
              used_id_idx, used_shft_idx, seed, verbose=True):
    """Independent eval set on WILDS pools, no overlap with hist/new/test."""
    from torch.utils.data import ConcatDataset
    n_eval_hist  = int((1 - beta) * n_eval)
    n_eval_shift = n_eval - n_eval_hist

    if verbose:
        print(f"\n{'='*60}")
        print(f"load_eval  |  beta={beta}  n_eval={n_eval}")
        print(f"  {n_eval_hist} ID  +  {n_eval_shift} OOD")
        print(f"{'='*60}")

    id_pool = ConcatDataset([
        _filter_by_center(train_wilds, 0),
        _filter_by_center(idval_wilds, 0),
    ])
    shift_pool = _get_shift_pool(train_wilds, idval_wilds, val_wilds, test_wilds, shift_center)

    used_id_set   = set(np.asarray(used_id_idx).tolist())
    used_shft_set = set(np.asarray(used_shft_idx).tolist())

    rng = np.random.default_rng(seed + 2)

    all_id_idx   = rng.permutation(len(id_pool))
    all_shft_idx = rng.permutation(len(shift_pool))

    free_id_idx   = all_id_idx[~np.isin(all_id_idx, list(used_id_set))]
    free_shft_idx = all_shft_idx[~np.isin(all_shft_idx, list(used_shft_set))]

    assert len(free_id_idx)   >= n_eval_hist
    assert len(free_shft_idx) >= n_eval_shift

    eval_id_idx   = free_id_idx[:n_eval_hist]
    eval_shft_idx = free_shft_idx[:n_eval_shift]

    if verbose: print("\nLoading eval arrays ...")
    eval_id_X,   eval_id_y   = _collect_indices(id_pool,    eval_id_idx)
    eval_shft_X, eval_shft_y = _collect_indices(shift_pool, eval_shft_idx)

    eval_X            = np.concatenate([eval_id_X,   eval_shft_X], axis=0)
    eval_y            = np.concatenate([eval_id_y,   eval_shft_y], axis=0)
    eval_sample_types = np.array([1]*n_eval_hist + [0]*n_eval_shift, dtype=np.int64)

    perm = rng.permutation(n_eval)
    eval_X            = eval_X[perm]
    eval_y            = eval_y[perm]
    eval_sample_types = eval_sample_types[perm]

    if verbose:
        print(f"  eval_X : {eval_X.shape}  "
              f"ID={(eval_sample_types==1).sum()} / OOD={(eval_sample_types==0).sum()}")
        assert len(used_id_set   & set(eval_id_idx.tolist()))   == 0
        assert len(used_shft_set & set(eval_shft_idx.tolist())) == 0
        print("  Zero-overlap verified")

    return eval_X, eval_y, eval_sample_types


# =============================================================================
# CIFAR-10 / CIFAR-10-C data helpers
# =============================================================================

# Canonical CIFAR-10 normalization (Hendrycks 2019, RobustBench, etc.)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]


def _preprocess_cifar(X_uint8_hwc, batch_size=512):
    """Convert (N, 32, 32, 3) uint8 HWC -> (N, 3, 32, 32) float32 with CIFAR normalization."""
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    out  = np.empty((len(X_uint8_hwc), 3, 32, 32), dtype=np.float32)
    for s in range(0, len(X_uint8_hwc), batch_size):
        b   = X_uint8_hwc[s : s + batch_size]
        b_t = torch.from_numpy(b).permute(0, 3, 1, 2).float() / 255.0
        b_t = (b_t - mean) / std
        out[s : s + batch_size] = b_t.numpy()
    return out


class _CifarTrainDataset:
    """Dataset that holds raw uint8 HWC images and applies on-the-fly canonical
    augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor +
    Normalize(CIFAR mean/std). Used by `train_cifar_canonical` for legacy and
    retrain training on CIFAR-10."""
    def __init__(self, X_uint8_hwc, y, augment=True):
        from torchvision import transforms
        from torch.utils.data import Dataset  # noqa: F401  (only for issubclass docs)
        self.X = X_uint8_hwc
        self.y = y
        norm = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), norm])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), int(self.y[idx])


def train_cifar_canonical(net, X_uint8_hwc, y, n_epochs, lr, batch_size,
                          weight_decay, momentum, device, num_workers=0,
                          log_every=10):
    """Canonical Hendrycks 2019-style CIFAR training loop.

    SGD-momentum + cosine LR + RandomCrop+HFlip augmentation, on raw uint8 HWC
    images (augmentation needs PIL inputs). Used identically for the legacy
    model and the retraining benchmark.
    """
    from torch.utils.data import DataLoader
    print(f"  Canonical CIFAR training: {n_epochs} epochs  bs={batch_size}  "
          f"SGD(lr={lr}, momentum={momentum}, wd={weight_decay})  cosine LR")
    ds     = _CifarTrainDataset(X_uint8_hwc, y, augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    net.train()
    for epoch in range(n_epochs):
        running, count = 0.0, 0
        for x, yt in loader:
            x  = x.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
            logits = net(x)
            loss = criterion(logits, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item(); count += 1
        scheduler.step()
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch + 1:3d}/{n_epochs}: "
                  f"loss={running / max(1, count):.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.4f}")
    net.eval()
    return net


def _ensure_cifar10c_downloaded(cifar10c_dir):
    """Download + extract CIFAR-10-C from Zenodo if not already present."""
    import urllib.request
    import tarfile

    label_path = os.path.join(cifar10c_dir, "labels.npy")
    if os.path.exists(label_path):
        return

    parent = os.path.dirname(os.path.abspath(cifar10c_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    tar_path = os.path.join(parent, "CIFAR-10-C.tar")

    if not os.path.exists(tar_path):
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
        print(f"  Downloading CIFAR-10-C from {url} (~2.5 GB) ...")

        def _progress(blocks, blocksize, total):
            done = blocks * blocksize
            if total > 0:
                pct = min(100.0, 100.0 * done / total)
                sys.stdout.write(
                    f"\r    {done / 1e6:6.0f} / {total / 1e6:.0f} MB  ({pct:5.1f}%)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_path, reporthook=_progress)
        sys.stdout.write("\n")
    else:
        print(f"  Found existing tar at {tar_path}, skipping download")

    print(f"  Extracting {tar_path} to {parent} ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(path=parent)

    extracted_label = os.path.join(parent, "CIFAR-10-C", "labels.npy")
    if not os.path.exists(extracted_label):
        raise FileNotFoundError(
            f"After extracting {tar_path}, expected {extracted_label} "
            f"but it was not found. Inspect {parent} for the actual layout.")
    print(f"  Extracted CIFAR-10-C to {os.path.join(parent, 'CIFAR-10-C')}")


def load_cifar10_pools(data_root, cifar10c_dir, corruption, severity):
    """Load CIFAR-10 clean (ID pool) + CIFAR-10-C (OOD pool) as raw uint8 HWC.
    Both auto-download on first run if not already present."""
    import torchvision
    print(f"Loading CIFAR-10 (clean) into ID pool (download=True) ...")
    tr = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True)
    te = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
    X_id = np.concatenate([tr.data, te.data], axis=0)
    y_id = np.concatenate([np.array(tr.targets), np.array(te.targets)], axis=0).astype(np.int64)
    print(f"  ID pool : X={X_id.shape}  labels: {dict(Counter(y_id))}")

    print(f"Loading CIFAR-10-C: '{corruption}' severity {severity} (auto-download if missing) ...")
    _ensure_cifar10c_downloaded(cifar10c_dir)

    corruption_path = os.path.join(cifar10c_dir, f"{corruption}.npy")
    label_path      = os.path.join(cifar10c_dir, "labels.npy")
    if not os.path.exists(corruption_path):
        raise FileNotFoundError(
            f"Could not find {corruption_path} after download/extract.")
    X_all = np.load(corruption_path)
    y_all = np.load(label_path).astype(np.int64)
    start = (severity - 1) * 10000
    end   = severity * 10000
    X_ood = X_all[start:end]
    y_ood = y_all[start:end]
    print(f"  OOD pool: X={X_ood.shape}  labels: {dict(Counter(y_ood))}")

    return X_id, y_id, X_ood, y_ood


def _collect_indices_cifar(X_pool, y_pool, indices):
    """CIFAR analogue of `_collect_indices`: numpy slicing + preprocess."""
    X_uint8_hwc = X_pool[indices]
    y           = y_pool[indices].astype(np.int64)
    X           = _preprocess_cifar(X_uint8_hwc)
    return X, y


def load_hist_cifar(X_id, y_id, n_hist, seed, verbose=True):
    """Draw n_hist samples from the CIFAR-10 clean ID pool. Returns
    (hist_X_uint8, hist_X_norm, hist_y, hist_idx) — both raw uint8 (for
    augmented training) and CIFAR-normalized (for downstream use)."""
    assert len(X_id) >= n_hist, \
        f"ID pool too small: need {n_hist}, have {len(X_id)}"

    rng_hist = np.random.default_rng(seed)
    hist_idx = rng_hist.permutation(len(X_id))[:n_hist]

    if verbose:
        print(f"Loading hist (clean CIFAR-10, n={n_hist}) ...")
    hist_X_uint8 = X_id[hist_idx]
    hist_y       = y_id[hist_idx].astype(np.int64)
    hist_X       = _preprocess_cifar(hist_X_uint8)

    if verbose:
        print(f"  raw {hist_X_uint8.shape}  norm {hist_X.shape}  "
              f"labels: {dict(Counter(hist_y))}")

    return hist_X_uint8, hist_X, hist_y, hist_idx


def load_new_test_cifar(X_id, y_id, X_ood, y_ood, beta, n_new, n_test, seed,
                       hist_idx, verbose=True):
    """Build new / test pools mixing clean (ID) and corrupted (OOD) on CIFAR.
    Returns both raw uint8 and CIFAR-normalized arrays, indexed identically."""
    n_new_id  = int((1 - beta) * n_new);  n_new_ood  = n_new  - n_new_id
    n_test_id = int((1 - beta) * n_test); n_test_ood = n_test - n_test_id

    if verbose:
        print(f"\n{'='*60}")
        print(f"load_new_test_cifar: clean vs corrupted  |  beta={beta}")
        print(f"  new  = {n_new}  ({n_new_id} ID + {n_new_ood} OOD)")
        print(f"  test = {n_test}  ({n_test_id} ID + {n_test_ood} OOD)")
        print(f"{'='*60}")

    needed_id  = n_new_id  + n_test_id
    needed_ood = n_new_ood + n_test_ood
    assert len(X_id) - len(hist_idx) >= needed_id, \
        (f"Not enough free ID samples after excluding hist: "
         f"need {needed_id}, have {len(X_id) - len(hist_idx)}")
    assert len(X_ood) >= needed_ood

    rng = np.random.default_rng(seed + 1)

    hist_idx_set = set(np.asarray(hist_idx).tolist())
    id_idx_full  = rng.permutation(len(X_id))
    keep_mask    = np.array([i not in hist_idx_set for i in id_idx_full], dtype=bool)
    id_idx       = id_idx_full[keep_mask]

    new_id_idx   = id_idx[:n_new_id]
    test_id_idx  = id_idx[n_new_id : n_new_id + n_test_id]

    o_idx        = rng.permutation(len(X_ood))
    new_ood_idx  = o_idx[:n_new_ood]
    test_ood_idx = o_idx[n_new_ood : n_new_ood + n_test_ood]

    assert len(hist_idx_set & set(new_id_idx.tolist()))  == 0
    assert len(hist_idx_set & set(test_id_idx.tolist())) == 0
    assert len(set(new_id_idx.tolist())  & set(test_id_idx.tolist()))  == 0
    assert len(set(new_ood_idx.tolist()) & set(test_ood_idx.tolist())) == 0

    if verbose: print("\nLoading + preprocessing arrays ...")
    new_id_X,  new_id_y  = _collect_indices_cifar(X_id,  y_id,  new_id_idx)
    new_ood_X, new_ood_y = _collect_indices_cifar(X_ood, y_ood, new_ood_idx)
    tst_id_X,  tst_id_y  = _collect_indices_cifar(X_id,  y_id,  test_id_idx)
    tst_ood_X, tst_ood_y = _collect_indices_cifar(X_ood, y_ood, test_ood_idx)

    new_id_u8,  new_ood_u8 = X_id [new_id_idx],   X_ood[new_ood_idx]
    tst_id_u8,  tst_ood_u8 = X_id [test_id_idx],  X_ood[test_ood_idx]

    new_X       = np.concatenate([new_id_X,  new_ood_X],  axis=0)
    new_X_uint8 = np.concatenate([new_id_u8, new_ood_u8], axis=0)
    new_y       = np.concatenate([new_id_y,  new_ood_y],  axis=0)
    sample_types = np.array([1] * n_new_id + [0] * n_new_ood, dtype=np.int64)

    test_X       = np.concatenate([tst_id_X,  tst_ood_X],  axis=0)
    test_X_uint8 = np.concatenate([tst_id_u8, tst_ood_u8], axis=0)
    test_y       = np.concatenate([tst_id_y,  tst_ood_y],  axis=0)
    test_sample_types = np.array([1] * n_test_id + [0] * n_test_ood, dtype=np.int64)

    perm_new  = rng.permutation(len(new_y))
    new_X, new_X_uint8, new_y, sample_types = (
        new_X[perm_new], new_X_uint8[perm_new], new_y[perm_new], sample_types[perm_new])

    perm_test = rng.permutation(len(test_y))
    test_X, test_X_uint8, test_y, test_sample_types = (
        test_X[perm_test], test_X_uint8[perm_test],
        test_y[perm_test], test_sample_types[perm_test])

    if verbose:
        print(f"\nnew  : {new_X.shape}  "
              f"ID={(sample_types == 1).sum()} / OOD={(sample_types == 0).sum()}  "
              f"labels: {dict(Counter(new_y))}")
        print(f"test : {test_X.shape}  "
              f"ID={(test_sample_types == 1).sum()} / OOD={(test_sample_types == 0).sum()}  "
              f"labels: {dict(Counter(test_y))}")

    return (new_X, new_X_uint8, new_y, sample_types,
            test_X, test_X_uint8, test_y, test_sample_types)


def _reconstruct_used_indices_cifar(X_id, X_ood, beta, n_hist, n_new, n_test, seed):
    """Replay RNG to recover indices consumed by hist/new/test on CIFAR pools."""
    rng_hist = np.random.default_rng(seed)
    hist_idx = rng_hist.permutation(len(X_id))[:n_hist]
    hist_idx_set = set(hist_idx.tolist())

    n_new_id   = int((1 - beta) * n_new)
    n_test_id  = int((1 - beta) * n_test)
    n_new_ood  = n_new  - n_new_id
    n_test_ood = n_test - n_test_id

    rng = np.random.default_rng(seed + 1)

    id_idx_full = rng.permutation(len(X_id))
    keep_mask   = np.array([i not in hist_idx_set for i in id_idx_full], dtype=bool)
    id_idx      = id_idx_full[keep_mask]

    new_id_idx  = id_idx[:n_new_id]
    test_id_idx = id_idx[n_new_id : n_new_id + n_test_id]

    o_idx        = rng.permutation(len(X_ood))
    new_ood_idx  = o_idx[:n_new_ood]
    test_ood_idx = o_idx[n_new_ood : n_new_ood + n_test_ood]

    used_id_idx  = np.concatenate([hist_idx, new_id_idx, test_id_idx])
    used_ood_idx = np.concatenate([new_ood_idx, test_ood_idx])

    assert len(set(used_id_idx.tolist()))  == len(used_id_idx)
    assert len(set(used_ood_idx.tolist())) == len(used_ood_idx)

    print(f"Reconstructed used indices -- "
          f"ID: {len(used_id_idx)} / {len(X_id)}  "
          f"OOD: {len(used_ood_idx)} / {len(X_ood)}")

    return used_id_idx, used_ood_idx


def load_eval_cifar(X_id, y_id, X_ood, y_ood, beta, n_eval,
                    used_id_idx, used_ood_idx, seed, verbose=True):
    """Independent eval set on CIFAR pools, no overlap with hist/new/test."""
    n_eval_id  = int((1 - beta) * n_eval)
    n_eval_ood = n_eval - n_eval_id

    if verbose:
        print(f"\n{'='*60}")
        print(f"load_eval_cifar  |  beta={beta}  n_eval={n_eval}")
        print(f"  {n_eval_id} ID  +  {n_eval_ood} OOD")
        print(f"{'='*60}")

    used_id_set  = set(np.asarray(used_id_idx).tolist())
    used_ood_set = set(np.asarray(used_ood_idx).tolist())

    rng = np.random.default_rng(seed + 2)

    all_id_idx  = rng.permutation(len(X_id))
    all_ood_idx = rng.permutation(len(X_ood))

    free_id_idx  = all_id_idx[~np.isin(all_id_idx,  list(used_id_set))]
    free_ood_idx = all_ood_idx[~np.isin(all_ood_idx, list(used_ood_set))]

    assert len(free_id_idx)  >= n_eval_id
    assert len(free_ood_idx) >= n_eval_ood

    eval_id_idx  = free_id_idx[:n_eval_id]
    eval_ood_idx = free_ood_idx[:n_eval_ood]

    if verbose: print("\nLoading + preprocessing eval arrays ...")
    eval_id_X,  eval_id_y  = _collect_indices_cifar(X_id,  y_id,  eval_id_idx)
    eval_ood_X, eval_ood_y = _collect_indices_cifar(X_ood, y_ood, eval_ood_idx)

    eval_X            = np.concatenate([eval_id_X,  eval_ood_X], axis=0)
    eval_y            = np.concatenate([eval_id_y,  eval_ood_y], axis=0)
    eval_sample_types = np.array([1] * n_eval_id + [0] * n_eval_ood, dtype=np.int64)

    perm = rng.permutation(n_eval)
    eval_X            = eval_X[perm]
    eval_y            = eval_y[perm]
    eval_sample_types = eval_sample_types[perm]

    if verbose:
        print(f"  eval_X : {eval_X.shape}  "
              f"ID={(eval_sample_types==1).sum()} / OOD={(eval_sample_types==0).sum()}")
        assert len(used_id_set  & set(eval_id_idx.tolist()))   == 0
        assert len(used_ood_set & set(eval_ood_idx.tolist()))  == 0
        print("  Zero-overlap verified")

    return eval_X, eval_y, eval_sample_types


# =============================================================================
# Calibration helpers: temperature scaling + Platt (binary + multiclass OVR) +
# overlapping-threshold indicator matrix. Moved here from src/calibration.py.
# =============================================================================

import warnings  # noqa: F401  (preserved from original calibration module)
from scipy.special import logit, softmax
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------- OVR Platt scaling (multi-class) ----------
def fit_platt_multiclass(p_hat, y_true, eps=1e-15, C=1.0, max_iter=1000, random_state=42):
    """
    Fit multi-class Platt scaling via one-vs-rest (OVR).
    Trains one binary Platt model per class on logit(p_hat[:, k]) vs 1{y == k},
    then renormalizes at prediction time so probabilities sum to 1 across classes.
    """
    p = np.clip(np.asarray(p_hat), eps, 1 - eps)
    n, K = p.shape

    y = np.asarray(y_true).ravel()
    classes_ = np.unique(y)
    if len(classes_) != K:
        raise ValueError(f"p_hat has {K} columns but y_true has {len(classes_)} unique classes.")

    models = []
    for k, cls in enumerate(classes_):
        y_bin = (y == cls).astype(int)
        if y_bin.min() == y_bin.max():
            raise ValueError(f"Class '{cls}' not present as positive/negative in y_true; "
                             "need at least one positive and one negative for each OVR fit.")
        s_k = logit(p[:, k]).reshape(-1, 1)
        lr = LogisticRegression(
            solver="lbfgs", penalty="l2", C=C, max_iter=max_iter, random_state=random_state
        )
        lr.fit(s_k, y_bin)
        models.append(lr)

    return {"models": models, "classes_": classes_, "eps": eps}


def predict_platt_multiclass(platt_ovr, p_hat):
    """Apply fitted OVR Platt calibrators and renormalize per row."""
    eps = platt_ovr["eps"]
    p = np.clip(np.asarray(p_hat), eps, 1 - eps)
    n, K = p.shape
    if len(platt_ovr["models"]) != K:
        raise ValueError("p_hat column count doesn't match number of fitted OVR models.")

    q = np.empty((n, K), dtype=float)
    for k, lr in enumerate(platt_ovr["models"]):
        s_k = logit(p[:, k]).reshape(-1, 1)
        q[:, k] = lr.predict_proba(s_k)[:, 1]

    row_sums = q.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return q / row_sums


def fit_platt_binary(p_hat, y_true, eps=1e-7):
    p_train = np.clip(np.asarray(p_hat).ravel(), eps, 1 - eps)
    s_train = logit(p_train).reshape(-1, 1)
    s_train = np.clip(s_train, -20, 20)
    y_bin = (np.asarray(y_true).ravel() == 1).astype(int)

    platt_model = LogisticRegression(
        solver="lbfgs", penalty="l2", C=1.0, max_iter=1000, random_state=42
    )
    platt_model.fit(s_train, y_bin)
    return platt_model


def predict_platt_binary(platt_model, p_hat, eps=1e-7):
    p_test = np.clip(np.asarray(p_hat).ravel(), eps, 1 - eps)
    s_test = logit(p_test).reshape(-1, 1)
    s_test = np.clip(s_test, -20, 20)
    return platt_model.predict_proba(s_test)


def apply_temperature_scaling(p_hat, T):
    """Apply temperature scaling with a given temperature."""
    epsilon = 1e-15
    p_hat_clipped = np.clip(p_hat, epsilon, 1 - epsilon)
    logits = np.log(p_hat_clipped)
    return softmax(logits / T, axis=1)


def find_optimal_temperature(p_hat, y_true):
    """Find optimal temperature by minimizing negative log-likelihood."""
    epsilon = 1e-15
    p_hat_clipped = np.clip(p_hat, epsilon, 1 - epsilon)
    logits = np.log(p_hat_clipped)

    def nll_loss(T):
        if T <= 0:
            return np.inf
        p_scaled = softmax(logits / T, axis=1)
        p_scaled_clipped = np.clip(p_scaled, epsilon, 1 - epsilon)
        return -np.mean(np.log(p_scaled_clipped[np.arange(len(y_true)), y_true]))

    result = minimize_scalar(nll_loss, bounds=(0.1, 10), method='bounded')
    return result.x


def indicator_matrix_overlapping_thresholds(scalar_values, thresholds):
    """
    Build an indicator matrix for overlapping threshold-defined groups.
    Columns are [x < t1, ..., x < tm, x >= t1, ..., x >= tm].
    """
    x = np.asarray(scalar_values).reshape(-1, 1)
    t = np.asarray(thresholds).ravel()
    t = np.unique(t)
    lt = (x < t)
    ge = ~lt
    return np.concatenate([lt, ge], axis=1).astype(int)


# =============================================================================
# Synthetic-experiment diagnostic helpers (moved from
# experiments/run_synthetic_experiment.py).
# =============================================================================

import pandas as pd  # noqa: E402  (kept here so utils stays self-contained)
from sklearn.metrics import accuracy_score  # noqa: E402


def print_confidence_diagnostics(label, proba, test_sample_types, K):
    """Print avg max-softmax confidence and accuracy on easy vs hard test samples."""
    max_proba = proba.max(axis=1)
    easy_mask = test_sample_types == 1
    hard_mask = test_sample_types == 0

    lines = [f"\n[Confidence Diagnostics: {label}]"]
    lines.append(f"  Expected accuracy (random chance on hard): {1/K:.3f}")
    if easy_mask.any():
        lines.append(f"  Easy  — avg max-softmax: {max_proba[easy_mask].mean():.4f}  (n={easy_mask.sum()})")
    if hard_mask.any():
        lines.append(f"  Hard  — avg max-softmax: {max_proba[hard_mask].mean():.4f}  (n={hard_mask.sum()})")
    print("\n".join(lines))
    print()


def compute_binned_accuracy(y_true, y_pred, r_star_scores, n_bins_list):
    """Compute overall and oracle-r*-binned accuracy."""
    results = {'acc_overall': float(accuracy_score(y_true, y_pred))}
    for n in n_bins_list:
        bins = np.linspace(0, 1, n + 1)
        bins[0] = -0.01
        bins[-1] = 1.01
        bin_cat = pd.cut(pd.Series(r_star_scores), bins)
        correct = (y_true == y_pred).astype(int)
        for i, interval in enumerate(bin_cat.cat.categories):
            mask = (bin_cat == interval).values
            results[f'acc_Bin_{i + 1}_N{n}'] = float(correct[mask].mean()) if mask.sum() > 0 else float('nan')
    return results


def add_final_model_acc(df, final_proba, y_true, r_star_scores, n_bins_list):
    """Add final_model_acc_overall and final_model_acc_Bin_{i}_N{n} columns."""
    if final_proba is None:
        df['final_model_acc_overall'] = np.nan
        for n in n_bins_list:
            for i in range(1, n + 1):
                df[f'final_model_acc_Bin_{i}_N{n}'] = np.nan
        return df
    y_pred = np.argmax(final_proba, axis=1)
    acc_dict = compute_binned_accuracy(y_true, y_pred, r_star_scores, n_bins_list)
    df['final_model_acc_overall'] = acc_dict['acc_overall']
    for k, v in acc_dict.items():
        if k != 'acc_overall':
            df[f'final_model_{k}'] = v
    return df
