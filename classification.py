"""
Classification fine-tuning script.
Adds a classification head on top of the pre-trained encoder and trains
composite activity classification (active / medium / strong).
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import Prediction_Dataset, Finetune_Collater
from metrics import AverageMeter
from model import PredictionModel
from split_utils import make_smiles_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Task Name Mapping (three-level composite classification)
# ============================================================================

TASK_LEVEL_MAP = {
    "COMPOSITE_ACTIVE": "clf1_active",
    "COMPOSITE_MEDIUM": "clf2_medium",
    "COMPOSITE_STRONG": "clf3_strong",
}
TASK_ALIAS = {
    "ACTIVE": "COMPOSITE_ACTIVE",
    "MEDIUM": "COMPOSITE_MEDIUM",
    "STRONG": "COMPOSITE_STRONG",
}
DEFAULT_TASKS = ["COMPOSITE_ACTIVE", "COMPOSITE_MEDIUM", "COMPOSITE_STRONG"]


# ============================================================================
# Utility Functions
# ============================================================================

def compute_task_auc(y_true, y_prob):
    """Compute per-task ROC-AUC, ignoring missing values (label == -1000)."""
    results = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != -1000
        if np.sum(mask) == 0:
            results.append(np.nan)
            continue
        yi = y_true[mask, i]
        pi = y_prob[mask, i]
        # AUC is undefined with only one class
        if len(np.unique(yi)) < 2:
            results.append(np.nan)
        else:
            results.append(float(roc_auc_score(yi, pi)))
    return results


def set_seed(seed):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def arch_config(arch):
    """Architecture configuration (must match the pre-trained model)."""
    return {
        "small": {"num_layers": 4, "num_heads": 4, "d_model": 128, "dff": 512},
        "medium": {"num_layers": 8, "num_heads": 8, "d_model": 256, "dff": 1024},
        "large": {"num_layers": 12, "num_heads": 12, "d_model": 576, "dff": 2304},
    }[arch]


def normalize_task_name(task):
    """Normalize abbreviated or variant task names to the standard form."""
    task_u = str(task).upper()
    if task_u in TASK_ALIAS:
        return TASK_ALIAS[task_u]
    if task_u in TASK_LEVEL_MAP:
        return task_u
    raise ValueError(f"Unknown task name: {task}")


# ============================================================================
# Data Loading & Splitting
# ============================================================================

def read_task_df(task, data_root):
    """Read a single classification task CSV.
    Returns a DataFrame with smiles and binary label columns.
    """
    task = normalize_task_name(task)
    level = TASK_LEVEL_MAP[task]
    file_path = Path(data_root) / level / f"{task}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing task file: {file_path}")

    df = pd.read_csv(file_path)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{file_path} must contain smiles and label columns")

    # Clean: drop NAs, deduplicate by SMILES
    df = df[["smiles", "label"]].dropna(subset=["smiles", "label"]) \
             .drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    df[task] = df["label"].astype(int)
    return df[["smiles", task]]


def make_seed_split(df, seed, train_ratio=0.8, val_ratio=0.1):
    """SMILES-based data split (ensures no SMILES appears in multiple partitions)."""
    smiles_list = df["smiles"].dropna().astype(str).tolist()
    return make_smiles_split(smiles_list, seed=seed,
                             train_ratio=train_ratio, val_ratio=val_ratio)


def split_task_df(df, split_obj):
    """Partition a DataFrame according to a split dictionary."""
    tr = set(split_obj.get("train", []))
    va = set(split_obj.get("valid", []))
    te = set(split_obj.get("test", []))
    train_df = df[df["smiles"].isin(tr)].copy().reset_index(drop=True)
    valid_df = df[df["smiles"].isin(va)].copy().reset_index(drop=True)
    test_df = df[df["smiles"].isin(te)].copy().reset_index(drop=True)
    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("Seed split produced an empty partition")
    return train_df, valid_df, test_df


def build_loaders(train_df, valid_df, test_df, task, batch_size, seed):
    """Build DataLoaders with deterministic worker seeds."""
    train_dataset = Prediction_Dataset(train_df, smiles_head=["smiles"],
                                       reg_heads=[], clf_heads=[task])
    valid_dataset = Prediction_Dataset(valid_df, smiles_head=["smiles"],
                                       reg_heads=[], clf_heads=[task])
    test_dataset = Prediction_Dataset(test_df, smiles_head=["smiles"],
                                      reg_heads=[], clf_heads=[task])

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    collate = Finetune_Collater(argparse.Namespace(clf_heads=[task], reg_heads=[]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, generator=generator,
                              worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, worker_init_fn=worker_init_fn)
    return train_loader, valid_loader, test_loader


# ============================================================================
# Model Loading
# ============================================================================

def load_pretrained_encoder(model, pretrain_dir, arch, pretrain_epoch):
    """Load pre-trained encoder weights.
    Naming convention: {arch}_weights_bert_encoder_weights{arch}_{epoch}.pt
    """
    pretrain_fn = f"{arch}_weights_bert_encoder_weights{arch}_{pretrain_epoch}.pt"
    pretrain_path = Path(pretrain_dir) / pretrain_fn
    if not pretrain_path.exists():
        raise FileNotFoundError(f"Missing pretrain weights: {pretrain_path}")

    checkpoint = torch.load(pretrain_path, map_location=DEVICE)
    # Strip "module." prefix if the checkpoint was saved from DDP
    if hasattr(checkpoint, "keys") and \
       any(str(k).startswith("module.") for k in checkpoint.keys()):
        checkpoint = {str(k).replace("module.", ""): v
                      for k, v in checkpoint.items()}
    model.encoder.load_state_dict(checkpoint, strict=False)
    return model


# ============================================================================
# Single Experiment
# ============================================================================

def run_single_experiment(args, task, seed, pretrain_epoch):
    """Run one complete train + evaluate experiment.
    Returns a dict with task / seed / pretrain_epoch / auc.
    """
    task = normalize_task_name(task)
    set_seed(seed)

    # ---- Data preparation ----
    df = read_task_df(task, args.data_root)
    split_obj = make_seed_split(df, seed=seed, train_ratio=0.8, val_ratio=0.1)
    train_df, valid_df, test_df = split_task_df(df, split_obj)

    current_batch_size = min(args.batch_size, max(2, len(train_df)))
    train_loader, valid_loader, test_loader = build_loaders(
        train_df, valid_df, test_df, task, current_batch_size, seed)

    # ---- Model construction ----
    conf = arch_config(args.arch)
    model = PredictionModel(
        hidden_num=512,
        finetune=args.finetune,
        vocab_size=60,
        num_layers=conf["num_layers"],
        d_model=conf["d_model"],
        dff=conf["dff"],
        num_heads=conf["num_heads"],
        dropout_rate=args.dropout_rate,
        reg_nums=0,
        clf_nums=1,              # single-task binary classification
    )
    model = load_pretrained_encoder(model, args.pretrain_dir,
                                    args.arch, pretrain_epoch)
    model = model.to(DEVICE)

    # Binary classification → BCEWithLogitsLoss (includes sigmoid internally)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")

    # ---- Early stopping setup ----
    if args.early_stop_metric == "loss":
        best_monitor = float("inf")
    else:
        best_monitor = -float("inf")
    best_model_path = None
    patience_counter = 0

    train_losses = []
    valid_losses = []

    # ---- Training loop ----
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = AverageMeter()
        for x, props in train_loader:
            y_true = props["clf"].to(DEVICE).float()
            mask = (y_true != -1000).float()       # ignore missing labels
            pred = model(x.to(DEVICE))["clf"]
            loss = (loss_func(pred, y_true) * mask).sum() / (mask.sum() + 1e-6)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss.update(float(loss.item()), x.shape[0])

        # ---- Validation ----
        model.eval()
        val_loss = AverageMeter()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for x, props in valid_loader:
                y_true = props["clf"].to(DEVICE).float()
                mask = (y_true != -1000).float()
                pred = model(x.to(DEVICE))["clf"]
                loss = (loss_func(pred, y_true) * mask).sum() / (mask.sum() + 1e-6)
                val_loss.update(float(loss.item()), x.shape[0])
                # Store sigmoid probabilities for AUC computation
                val_probs.append(torch.sigmoid(pred).cpu().numpy())
                val_labels.append(y_true.cpu().numpy())

        # Compute validation AUC
        valid_auc_results = compute_task_auc(
            np.concatenate(val_labels, axis=0),
            np.concatenate(val_probs, axis=0))
        avg_auc_valid = float(np.mean([a for a in valid_auc_results
                                       if not np.isnan(a)])) \
                        if np.any(~np.isnan(valid_auc_results)) else 0.5

        train_losses.append(epoch_loss.avg)
        valid_losses.append(val_loss.avg)

        # ---- Early stopping check ----
        if args.early_stop_metric == "loss":
            current_monitor = float(val_loss.avg)
            improved = current_monitor + 1e-12 < best_monitor
        else:
            current_monitor = float(avg_auc_valid)
            improved = current_monitor > best_monitor + 1e-12

        if improved:
            best_monitor = current_monitor
            patience_counter = 0
            best_model_path = args.model_dir / \
                f"clf_best_{task}_{args.arch}_e{pretrain_epoch}_s{seed}.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # ---- Load best model, evaluate on test set ----
    if best_model_path is None:
        best_model_path = args.model_dir / \
            f"clf_best_{task}_{args.arch}_e{pretrain_epoch}_s{seed}.pth"
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    test_probs = []
    test_labels = []
    with torch.no_grad():
        for x, props in test_loader:
            y_true = props["clf"].to(DEVICE).float()
            pred = model(x.to(DEVICE))["clf"]
            test_probs.append(torch.sigmoid(pred).cpu().numpy())
            test_labels.append(y_true.cpu().numpy())

    auc_list = compute_task_auc(np.concatenate(test_labels, axis=0),
                                np.concatenate(test_probs, axis=0))
    avg_auc = float(np.mean([a for a in auc_list if not np.isnan(a)])) \
              if np.any(~np.isnan(auc_list)) else 0.5

    # ---- Plot and save loss curve ----
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.curve_dir /
                f"clf_loss_{task}_{args.arch}_e{pretrain_epoch}_s{seed}.png")
    plt.close()

    return {
        "task": task,
        "pretrain_epoch": pretrain_epoch,
        "seed": seed,
        "auc": round(float(auc_list[0]) if len(auc_list) > 0 else np.nan, 4),
        "mean_auc": round(avg_auc, 4),
    }


# ============================================================================
# Main: sweep over seeds × tasks × pre-training epochs
# ============================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf-tasks", nargs="+", default=DEFAULT_TASKS, type=str,
                        help='List of classification tasks')
    parser.add_argument("--arch", default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--finetune", default="all",
                        choices=["partial", "all"],
                        help='Fine-tuning strategy: all = full model, partial = last 2 layers')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='AdamW learning rate')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200,
                        help='Maximum fine-tuning epochs')
    parser.add_argument("--patience", type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--early-stop-metric", type=str, default="loss",
                        choices=["auc", "loss"],
                        help='Metric monitored for early stopping')
    parser.add_argument("--seed-list", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4],
                        help='Random seeds for data splitting')
    parser.add_argument("--pretrain-start", type=int, default=85,
                        help='First pre-training epoch to sweep from')
    parser.add_argument("--pretrain-end", type=int, default=100,
                        help='Last pre-training epoch to sweep to')
    parser.add_argument("--data-root", type=str,
                        default=str(script_dir / "data" / "v2"),
                        help='Root directory for classification data')
    parser.add_argument("--pretrain-dir", type=str,
                        default=str(script_dir.parent / "weights_pubchem10M"),
                        help='Directory containing pre-trained weights')
    parser.add_argument("--output-root", type=str,
                        default=str(script_dir / "Results_old" / "Ours" / "classification"),
                        help='Root output directory')
    args = parser.parse_args()

    args.output_root = Path(args.output_root)
    args.model_dir = args.output_root / \
        f"models_bs{args.batch_size}_lr{args.lr}_ft{args.finetune}"
    args.curve_dir = args.output_root / "loss_curves"
    args.metrics_dir = args.output_root / \
        f"performance_bs{args.batch_size}_lr{args.lr}_ft{args.finetune}"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.curve_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Sweep over all task × seed × pre-training epoch combinations
    all_rows = []
    for task in args.clf_tasks:
        for seed in args.seed_list:
            for pretrain_epoch in range(args.pretrain_start, args.pretrain_end + 1):
                row = run_single_experiment(args, task, seed, pretrain_epoch)
                all_rows.append(row)
                print(f"[clf] task={row['task']} seed={seed} "
                      f"pretrain={pretrain_epoch} mean_auc={row['mean_auc']:.4f}")

    # Save aggregated results as CSV
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.metrics_dir / "classification_seeds_results.csv", index=False)


if __name__ == "__main__":
    main()
