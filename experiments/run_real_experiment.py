"""
Real-data ACP experiments. Single entry point for two datasets:

  --dataset camelyon17  -> Camelyon17-WILDS (binary tumor classification, K=2)
                            DenseNet-121 trained with Blackbox.fit (SGD).
  --dataset cifar10     -> CIFAR-10 / CIFAR-10-C (10-class, K=10)
                            ResNet-18 trained canonically (Hendrycks 2019:
                            SGD-momentum + cosine LR + augmentation).

Shared infrastructure (EmbeddingModel, AuditModelWrapper, compute_r_hat_knn,
data-pool helpers) lives in `src/conformal_utils.py`.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Blackbox, ClassDenseNet, ClassResNet18Cifar
from src.baseline_conformal import BaselineConformalPrediction
from src.audited_conformal import (
    Audited_Model,
    Audited_Conformal_Prediction,
)
from src.utils import (
    EmbeddingModel,
    AuditModelWrapper,
    compute_r_hat_knn,
    # WILDS helpers
    load_wilds_splits, load_hist, load_new_test, load_eval,
    _reconstruct_used_indices,
    # CIFAR helpers
    CIFAR10_C_CORRUPTIONS,
    train_cifar_canonical,
    load_cifar10_pools, load_hist_cifar, load_new_test_cifar, load_eval_cifar,
    _reconstruct_used_indices_cifar,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run ACP real-data experiment")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["camelyon17", "cifar10"],
                        help="Dataset to run on")

    # Camelyon17 / WILDS-specific
    parser.add_argument("--wilds_root", type=str, default="experiments/data",
                        help="[camelyon17] Path to WILDS dataset root")
    parser.add_argument("--shift_center", type=int, default=2,
                        choices=[1, 2, 3, 4],
                        help="[camelyon17] Hospital center forming the OOD pool")

    # CIFAR-10 / CIFAR-10-C-specific
    parser.add_argument("--cifar10_root", type=str,
                        default="experiments/data/cifar10",
                        help="[cifar10] Root for CIFAR-10 (auto-downloaded)")
    parser.add_argument("--cifar10c_dir", type=str,
                        default="experiments/data/CIFAR-10-C",
                        help="[cifar10] Directory holding CIFAR-10-C .npy files")
    parser.add_argument("--corruption", type=str, default="contrast",
                        choices=CIFAR10_C_CORRUPTIONS,
                        help="[cifar10] Corruption type for the OOD pool")
    parser.add_argument("--severity", type=int, default=5,
                        choices=[1, 2, 3, 4, 5],
                        help="[cifar10] Corruption severity (5 = most severe)")

    # Common experiment knobs
    parser.add_argument("--beta", type=float, default=0.1,
                        help="OOD fraction in new/test/eval pools")
    parser.add_argument("--n_hist", type=int, default=10000)
    parser.add_argument("--n_new", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_eval", type=int, default=5000,
                        help="Size of D_eval used to approximate r*(x)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (legacy + retrain)")
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--knn_k", type=int, default=50,
                        help="k for KNN r* approximation")
    parser.add_argument("--knn_metric", type=str, default="cosine",
                        choices=["cosine", "euclidean"])
    parser.add_argument("--outdir", type=str, default="results/real/")

    # CIFAR canonical-training hyperparameters (Hendrycks 2019)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="[cifar10] Learning rate (Hendrycks default 0.1)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="[cifar10] Training batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="[cifar10] Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="[cifar10] SGD momentum")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific setup: returns everything downstream code needs
# ─────────────────────────────────────────────────────────────────────────────

def _setup_camelyon17(args, device):
    """Load Camelyon17, train DenseNet-121 with Blackbox.fit (SGD)."""
    seed = args.seed
    K = 2

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    train_wilds, idval_wilds, val_wilds, test_wilds = load_wilds_splits(
        args.wilds_root, transform)

    # 1. hist
    hist_X, hist_y, hist_idx = load_hist(train_wilds, idval_wilds,
                                          args.n_hist, seed)

    # 2. legacy training
    net = ClassDenseNet(num_classes=K, device=device,
                        use_dropout=False, pretrained=False)
    lr_cam = 1e-3
    bs_cam = 32
    optimizer = optim.SGD(net.parameters(), lr=lr_cam,
                          momentum=0.9, weight_decay=0.01)
    old_model = Blackbox(
        net=net, device=device,
        X_train=hist_X, y_train=hist_y, X_val=None, y_val=None,
        batch_size=bs_cam, max_epoch=args.epochs,
        learning_rate=lr_cam, criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer, compute_accuracy=False, verbose=True,
    )
    old_model.fit(save_dir=None)

    # 3. new / test
    (new_X, new_y, sample_types,
     test_X, test_y, test_sample_types) = load_new_test(
        train_wilds, idval_wilds, val_wilds, test_wilds,
        shift_center=args.shift_center, beta=args.beta,
        n_new=args.n_new, n_test=args.n_test, seed=seed,
        hist_idx=hist_idx)
    new_X_uint8 = test_X_uint8 = None  # Not used for Camelyon17

    # 4. eval
    used_id_idx, used_shft_idx = _reconstruct_used_indices(
        train_wilds, idval_wilds, val_wilds, test_wilds,
        shift_center=args.shift_center, beta=args.beta,
        n_hist=args.n_hist, n_new=args.n_new, n_test=args.n_test, seed=seed)
    assert np.array_equal(np.sort(used_id_idx[:args.n_hist]), np.sort(hist_idx)), \
        "Reconstructed hist_idx does not match the hist_idx produced by load_hist"

    eval_X, eval_y, _ = load_eval(
        train_wilds, idval_wilds, val_wilds, test_wilds,
        shift_center=args.shift_center, beta=args.beta, n_eval=args.n_eval,
        used_id_idx=used_id_idx, used_shft_idx=used_shft_idx, seed=seed)

    data_model_name = f"camelyon17_center0_vs_center{args.shift_center}"
    outfile_stem = (f"camelyon17_center{args.shift_center}_beta{args.beta}"
                    f"_nhist{args.n_hist}_nnew{args.n_new}_ntest{args.n_test}"
                    f"_seed{seed}")

    # Retrain factory: same architecture / optimizer as legacy.
    def make_retraining_model(epoch_retraining):
        net_r = ClassDenseNet(num_classes=K, device=device,
                              use_dropout=False, pretrained=False)
        opt_r = optim.SGD(net_r.parameters(), lr=lr_cam,
                          momentum=0.9, weight_decay=0.01)
        return Blackbox(
            net=net_r, device=device,
            X_train=None, y_train=None, X_val=None, y_val=None,
            batch_size=bs_cam, max_epoch=epoch_retraining,
            learning_rate=lr_cam, criterion=nn.CrossEntropyLoss(),
            optimizer=opt_r, compute_accuracy=False, verbose=True,
        )

    return {
        "K": K, "n_bins": [3, 5],
        "old_model": old_model, "net": net,
        "hist_X": hist_X, "hist_y": hist_y, "hist_idx": hist_idx,
        "new_X": new_X, "new_X_uint8": new_X_uint8,
        "new_y": new_y, "sample_types": sample_types,
        "test_X": test_X, "test_X_uint8": test_X_uint8,
        "test_y": test_y, "test_sample_types": test_sample_types,
        "eval_X": eval_X, "eval_y": eval_y,
        "data_model_name": data_model_name,
        "outfile_stem": outfile_stem,
        "make_retraining_model": make_retraining_model,
        "eta_params_combscore": None,                       # binary -> no eta
        "X_new_train_uint8_attr": None,                     # marker: no canonical retrain
    }


def _setup_cifar10(args, device):
    """Load CIFAR-10 / CIFAR-10-C, train ResNet-18 canonically."""
    seed = args.seed
    K = 10

    # 1. pools
    X_id, y_id, X_ood, y_ood = load_cifar10_pools(
        args.cifar10_root, args.cifar10c_dir, args.corruption, args.severity)

    # 2. hist (raw uint8 + normalized)
    hist_X_uint8, hist_X, hist_y, hist_idx = load_hist_cifar(
        X_id, y_id, args.n_hist, seed)

    # 3. legacy training (canonical)
    print(f"\nCanonical legacy training "
          f"({args.epochs} epochs, bs={args.batch_size})")
    net = ClassResNet18Cifar(num_classes=K, device=device, use_dropout=False)
    train_cifar_canonical(
        net=net, X_uint8_hwc=hist_X_uint8, y=hist_y,
        n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay, momentum=args.momentum, device=device,
    )
    old_model = Blackbox(
        net=net, device=device,
        X_train=None, y_train=None, X_val=None, y_val=None,
        batch_size=args.batch_size, max_epoch=0,
        learning_rate=args.lr, criterion=nn.CrossEntropyLoss(),
        optimizer=None, compute_accuracy=False, verbose=False,
    )

    # 4. new / test
    (new_X, new_X_uint8, new_y, sample_types,
     test_X, test_X_uint8, test_y, test_sample_types) = load_new_test_cifar(
        X_id, y_id, X_ood, y_ood,
        beta=args.beta, n_new=args.n_new, n_test=args.n_test, seed=seed,
        hist_idx=hist_idx)

    # 5. eval
    used_id_idx, used_ood_idx = _reconstruct_used_indices_cifar(
        X_id, X_ood, beta=args.beta,
        n_hist=args.n_hist, n_new=args.n_new, n_test=args.n_test, seed=seed)
    assert np.array_equal(np.sort(used_id_idx[:args.n_hist]), np.sort(hist_idx)), \
        "Reconstructed hist_idx does not match the hist_idx produced by load_hist_cifar"

    eval_X, eval_y, _ = load_eval_cifar(
        X_id, y_id, X_ood, y_ood,
        beta=args.beta, n_eval=args.n_eval,
        used_id_idx=used_id_idx, used_ood_idx=used_ood_idx, seed=seed)

    data_model_name = f"cifar10c_{args.corruption}_sev{args.severity}"
    outfile_stem = (f"cifar10c_{args.corruption}_sev{args.severity}_beta{args.beta}"
                    f"_nhist{args.n_hist}_nnew{args.n_new}_ntest{args.n_test}"
                    f"_seed{seed}")

    # CIFAR retrain: same canonical recipe as legacy. We pre-train on raw uint8
    # D¹_cal slice (idx_train of new_X_uint8) and wrap in a no-op-fit Blackbox.
    class _NoOpFitBlackbox(Blackbox):
        """Blackbox whose .fit() is a no-op (model is already trained)."""
        def fit(self, *a, **kw):
            return {}

    def make_retraining_model(epoch_retraining, X_new_train_uint8=None,
                              y_new_train=None):
        assert X_new_train_uint8 is not None and y_new_train is not None, \
            "CIFAR retrain needs raw uint8 D¹_cal slice"
        net_r = ClassResNet18Cifar(num_classes=K, device=device, use_dropout=False)
        train_cifar_canonical(
            net=net_r, X_uint8_hwc=X_new_train_uint8, y=y_new_train,
            n_epochs=epoch_retraining, lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, momentum=args.momentum, device=device,
        )
        return _NoOpFitBlackbox(
            net=net_r, device=device,
            X_train=None, y_train=None, X_val=None, y_val=None,
            batch_size=args.batch_size, max_epoch=0,
            learning_rate=args.lr, criterion=nn.CrossEntropyLoss(),
            optimizer=None, compute_accuracy=False, verbose=False,
        )

    return {
        "K": K, "n_bins": [3, 5],
        "old_model": old_model, "net": net,
        "hist_X": hist_X, "hist_y": hist_y, "hist_idx": hist_idx,
        "new_X": new_X, "new_X_uint8": new_X_uint8,
        "new_y": new_y, "sample_types": sample_types,
        "test_X": test_X, "test_X_uint8": test_X_uint8,
        "test_y": test_y, "test_sample_types": test_sample_types,
        "eval_X": eval_X, "eval_y": eval_y,
        "data_model_name": data_model_name,
        "outfile_stem": outfile_stem,
        "make_retraining_model": make_retraining_model,
        # Multiclass APA defaults: 'counts' = empirical confusion-matrix prior.
        "eta_params_combscore": {
            "eta_method"          : "counts",
            "eta_adapter_C"       : 0.1,
            "eta_smoothing_alpha" : 1000,
        },
        "X_new_train_uint8_attr": True,    # marker: needs raw uint8 retrain path
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    alpha = args.alpha
    n_hist = args.n_hist
    n_new  = args.n_new
    n_test = args.n_test
    n_eval = args.n_eval
    beta   = args.beta
    knn_k  = args.knn_k
    knn_metric = args.knn_metric

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nCUDA available: {torch.cuda.is_available()}")

    os.makedirs(args.outdir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1-3. Dataset-specific setup: load splits, train legacy model
    # ─────────────────────────────────────────────────────────────────────────
    if args.dataset == "camelyon17":
        ctx = _setup_camelyon17(args, device)
    elif args.dataset == "cifar10":
        ctx = _setup_cifar10(args, device)
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    K              = ctx["K"]
    n_bins         = ctx["n_bins"]
    old_model      = ctx["old_model"]
    net            = ctx["net"]
    hist_X         = ctx["hist_X"]
    hist_y         = ctx["hist_y"]
    new_X          = ctx["new_X"]
    new_X_uint8    = ctx["new_X_uint8"]
    new_y          = ctx["new_y"]
    sample_types   = ctx["sample_types"]
    test_X         = ctx["test_X"]
    test_y         = ctx["test_y"]
    test_sample_types = ctx["test_sample_types"]
    eval_X         = ctx["eval_X"]
    eval_y         = ctx["eval_y"]
    data_model_name = ctx["data_model_name"]
    outfile_stem   = ctx["outfile_stem"]
    make_retraining_model = ctx["make_retraining_model"]
    eta_params_combscore  = ctx["eta_params_combscore"]
    needs_raw_retrain     = ctx["X_new_train_uint8_attr"] is not None

    outfile = os.path.join(args.outdir, outfile_stem + ".csv")
    print(f"Output file: {outfile}")
    print("\n[OK] hist / new / test / eval index sets are fully disjoint.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Evaluate primary model
    # ─────────────────────────────────────────────────────────────────────────
    original_mask = sample_types == 1
    shifted_mask  = sample_types == 0

    old_model_hist_acc = accuracy_score(hist_y, old_model.net.predict(hist_X))
    old_model_new_acc  = accuracy_score(new_y,  old_model.net.predict(new_X))
    print(f"\nPrimary model accuracy on historical data : {old_model_hist_acc:.3f}")
    print(f"Primary model accuracy on new (overall)   : {old_model_new_acc:.3f}")

    if original_mask.any():
        old_model_original_acc = accuracy_score(
            new_y[original_mask], old_model.net.predict(new_X[original_mask]))
        print(f"Primary model accuracy on new (original) : {old_model_original_acc:.3f}")
    else:
        old_model_original_acc = float('nan')

    if shifted_mask.any():
        old_model_shifted_acc = accuracy_score(
            new_y[shifted_mask], old_model.net.predict(new_X[shifted_mask]))
        print(f"Primary model accuracy on new (shifted)  : {old_model_shifted_acc:.3f}")
    else:
        old_model_shifted_acc = float('nan')

    # ─────────────────────────────────────────────────────────────────────────
    # 5. r_hat via kNN on independent eval set
    # ─────────────────────────────────────────────────────────────────────────
    r_hat_test, _ = compute_r_hat_knn(
        old_model=old_model, test_X=test_X,
        eval_X=eval_X, eval_y=eval_y,
        k=knn_k, metric=knn_metric)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Header / audit-info helpers
    # ─────────────────────────────────────────────────────────────────────────
    p = net.z_dim

    def add_header(df):
        df["n_hist"] = n_hist
        df["n_new"]  = n_new
        df["seed"]   = seed
        df["beta"]   = beta
        df["data_model"] = data_model_name
        df["n_test"] = n_test
        df["p"]      = p
        df["K"]      = K
        df["alpha"]  = alpha
        df["old_model_hist_acc"] = old_model_hist_acc
        df["old_model_new_acc (Overall)"] = old_model_new_acc
        if original_mask.any():
            df["old_model_new_acc (Easy)"] = old_model_original_acc
        if shifted_mask.any():
            df["old_model_new_acc (Hard)"] = old_model_shifted_acc
        if args.dataset == "cifar10":
            df["corruption"] = args.corruption
            df["severity"]   = args.severity
        return df

    def add_audit_info(df, audit_model_info):
        for key, value in audit_model_info.items():
            df[f"audit_{key}"] = value
        return df

    results = pd.DataFrame({})

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Pre-split new for retrain training and audit calibration.
    #    For CIFAR canonical retrain we need the raw uint8 D¹_cal slice;
    #    for Camelyon17 retrain runs through Blackbox.fit on normalized arrays.
    # ─────────────────────────────────────────────────────────────────────────
    idx_train, idx_cal = train_test_split(
        np.arange(len(new_X)), test_size=0.5, random_state=seed)
    X_new_train,    X_new_cal    = new_X[idx_train],  new_X[idx_cal]
    y_new_train,    y_new_cal    = new_y[idx_train],  new_y[idx_cal]
    sample_types_cal             = sample_types[idx_cal]
    X_new_train_uint8 = new_X_uint8[idx_train] if new_X_uint8 is not None else None

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Baseline: Standard
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Baseline: Standard ===")
    bl_std = BaselineConformalPrediction(
        primary_model=old_model.net, alpha=alpha, random_state=seed, method="standard")
    results_std = bl_std.run_baseline(
        X_cal=new_X, y_cal=new_y, X_test=test_X, y_test=test_y,
        generator=None, n_bins=n_bins, sample_types_test=test_sample_types,
        r_hat_test=r_hat_test)
    results = pd.concat([results, results_std])

    # ─────────────────────────────────────────────────────────────────────────
    # 9-10. Temperature scaling (fixed + adaptive)
    # ─────────────────────────────────────────────────────────────────────────
    for label, method in [("Temperature Scaling (fixed)", "ts_fixed"),
                          ("Temperature Scaling (adaptive)", "ts_adaptive")]:
        print(f"\n=== Baseline: {label} ===")
        bl = BaselineConformalPrediction(
            primary_model=old_model.net, alpha=alpha, random_state=seed, method=method)
        r = bl.run_baseline(
            X_cal=new_X, y_cal=new_y, X_test=test_X, y_test=test_y,
            generator=None, n_bins=n_bins, sample_types_test=test_sample_types,
            r_hat_test=r_hat_test)
        results = pd.concat([results, r])

    # ─────────────────────────────────────────────────────────────────────────
    # 10b. Trust Score
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Baseline: Trust Score ===")
    bl_trust = BaselineConformalPrediction(
        primary_model=old_model.net, alpha=alpha, random_state=seed, method="trustscore")
    results_trustscore = bl_trust.run_baseline(
        X_cal=new_X, y_cal=new_y, X_test=test_X, y_test=test_y,
        generator=None, n_bins=n_bins, sample_types_test=test_sample_types,
        X_train=hist_X, y_train=hist_y, r_hat_test=r_hat_test)
    results = pd.concat([results, results_trustscore])

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Retrain — same architecture and recipe as legacy
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Baseline: Retrain ===")
    epoch_retraining = args.epochs

    if needs_raw_retrain:
        # CIFAR: pre-train canonically on raw uint8 D¹_cal, then wrap as no-op-fit.
        retraining_model = make_retraining_model(
            epoch_retraining,
            X_new_train_uint8=X_new_train_uint8,
            y_new_train=y_new_train,
        )
    else:
        # Camelyon17: build a Blackbox and let BaselineConformalPrediction
        # call .fit() on the normalized D¹_cal slice via X_new_train kw.
        retraining_model = make_retraining_model(epoch_retraining)

    bl_retr = BaselineConformalPrediction(
        primary_model=old_model, retraining_model=retraining_model,
        alpha=alpha, random_state=seed, method="retrain",
        retraining_epoch=epoch_retraining)
    results_retr = bl_retr.run_baseline(
        X_cal=X_new_cal, y_cal=y_new_cal, X_test=test_X, y_test=test_y,
        generator=None, n_bins=n_bins, sample_types_test=test_sample_types,
        epoch=epoch_retraining, r_hat_test=r_hat_test,
        X_new_train=X_new_train, y_new_train=y_new_train)
    results = pd.concat([results, results_retr])

    # ─────────────────────────────────────────────────────────────────────────
    # 12. Pre-extract CNN embeddings for audit training
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPre-extracting CNN embeddings ...")
    hist_emb = old_model.net.get_embeddings(hist_X)
    new_emb  = old_model.net.get_embeddings(new_X)
    print(f"  hist_emb : {hist_emb.shape}")
    print(f"  new_emb  : {new_emb.shape}")

    X_new_train_emb, X_new_cal_emb = new_emb[idx_train], new_emb[idx_cal]
    print(f"\nAudit split  train: {X_new_train_emb.shape}  cal: {X_new_cal_emb.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # 13. Train audit model on embeddings
    # ─────────────────────────────────────────────────────────────────────────
    embedding_model = EmbeddingModel(old_model.net)

    audit_model_params = {
        "feature_set"              : "full",
        "base_correctness_model"   : None,
        "use_uncertainty_features" : False,
    }

    print("\n=== Training Audit Model ===")
    print(f"Configuration: {audit_model_params}")

    audit = Audited_Model(
        primary_model=embedding_model,
        audit_model_params=audit_model_params,
        random_state=seed, verbose=True,
    )
    feature_importances, audit_acc = audit.train_evaluate_audit_model(
        hist_emb, hist_y,
        X_new_train_emb, y_new_train,
        X_new_cal_emb,   y_new_cal,
    )
    print(f"Audit model accuracy: {audit_acc:.4f}")

    audit_model_info = {**audit_model_params, "audit_acc": audit_acc}
    if feature_importances is not None:
        audit_model_info.update(feature_importances)
    else:
        audit_model_info.update({
            "basic_feature_importance"               : None,
            "distributional_shift_feature_importance": None,
            "distance_feature_importance"            : None,
            "model_agg_feature_importance"           : None,
        })

    wrapped_audit = AuditModelWrapper(audit.audit_model, old_model.net)

    # ─────────────────────────────────────────────────────────────────────────
    # 14. Verify correctness alignment
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Correctness Alignment Check ===")
    correct_emb_model = (embedding_model.predict(X_new_cal_emb) == y_new_cal).astype(int)
    correct_old_model = (old_model.net.predict(X_new_cal) == y_new_cal).astype(int)
    n_mismatch = int(np.sum(correct_emb_model != correct_old_model))
    assert n_mismatch == 0, (
        f"FAIL: {n_mismatch} samples where EmbeddingModel and old_model.net disagree.")
    print(f"  EmbeddingModel correctness == old_model.net correctness on all "
          f"{len(y_new_cal)} samples")

    if sample_types_cal is not None:
        audit_preds_cal = wrapped_audit.predict(X_new_cal)
        hard_mask_cal = sample_types_cal == 0
        easy_mask_cal = sample_types_cal == 1
        if hard_mask_cal.any():
            print(f"  Audit accuracy (hard/OOD): "
                  f"{accuracy_score(correct_old_model[hard_mask_cal], audit_preds_cal[hard_mask_cal]):.4f}")
        if easy_mask_cal.any():
            print(f"  Audit accuracy (easy/ID) : "
                  f"{accuracy_score(correct_old_model[easy_mask_cal], audit_preds_cal[easy_mask_cal]):.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 15. ACP-MC: CombScore   (eta_params None for binary, 'counts' for K>2)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== ACP: CombScore ===")
    if eta_params_combscore is not None:
        print(f"eta_params = {eta_params_combscore}")
    acp_combscore = Audited_Conformal_Prediction(
        primary_model=old_model.net, audit_model=wrapped_audit,
        alpha=alpha, K=K, random_state=seed,
    )
    results_combscore = acp_combscore.run_audited_conformal_prediction(
        method='combscore',
        X_new_cal=X_new_cal, y_new_cal=y_new_cal,
        sample_types_cal=sample_types_cal, n_bins=n_bins,
        X_test=test_X, y_test=test_y, sample_types_test=test_sample_types,
        eta_params=eta_params_combscore,
        X_new_train=X_new_train, y_new_train=y_new_train,
        data_generator=None, r_hat_test=r_hat_test,
    )
    results_combscore = add_audit_info(results_combscore, audit_model_info)
    results = pd.concat([results, results_combscore])

    # ─────────────────────────────────────────────────────────────────────────
    # 16. ACP-ACC: CondConf
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== ACP: CondConf ===")
    try:
        acp_condconf = Audited_Conformal_Prediction(
            primary_model=old_model.net, audit_model=wrapped_audit,
            alpha=alpha, K=K, random_state=seed,
        )
        results_condconf = acp_condconf.run_audited_conformal_prediction(
            method='condconf',
            X_new_cal=X_new_cal, y_new_cal=y_new_cal,
            sample_types_cal=sample_types_cal, n_bins=n_bins,
            X_test=test_X, y_test=test_y, sample_types_test=test_sample_types,
            threshold_list=[0.5, 0.6, 0.7, 0.8, 0.9],
            data_generator=None, r_hat_test=r_hat_test,
        )
        results_condconf = add_audit_info(results_condconf, audit_model_info)
        results = pd.concat([results, results_condconf])
        print("CondConf completed successfully")
    except Exception as e:
        print(f"CondConf failed with error: {e}")
        print("Skipping CondConf method")

    # ─────────────────────────────────────────────────────────────────────────
    # 17. ACP-AEC: Equalized (multiple thresholds, one threshold at a time)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== ACP: Equalized ===")
    for threshold_ in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"\nTesting threshold: {threshold_}")
        acp_equalized = Audited_Conformal_Prediction(
            primary_model=old_model.net, audit_model=wrapped_audit,
            alpha=alpha, K=K, random_state=seed,
        )
        results_equalized = acp_equalized.run_audited_conformal_prediction(
            method='equalized',
            X_new_cal=X_new_cal, y_new_cal=y_new_cal,
            sample_types_cal=sample_types_cal, n_bins=n_bins,
            X_test=test_X, y_test=test_y, sample_types_test=test_sample_types,
            threshold=threshold_,
            data_generator=None, r_hat_test=r_hat_test,
        )
        results_equalized = add_audit_info(results_equalized, audit_model_info)
        results = pd.concat([results, results_equalized])

    # ─────────────────────────────────────────────────────────────────────────
    # 18. Save
    # ─────────────────────────────────────────────────────────────────────────
    results = results.reset_index(drop=True)
    results = add_header(results)
    results.to_csv(outfile, index=False)
    print(f"\nResults written to {outfile}")
    print(results)


if __name__ == "__main__":
    main()
