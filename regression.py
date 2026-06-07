"""
Regression fine-tuning script.
Adds regression heads on top of the pre-trained encoder for multi-task
prediction of antioxidant activity values (TEAC, pIC50).
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import Prediction_Dataset, Finetune_Collater
from metrics import AverageMeter
from model import PredictionModel
from split_utils import make_smiles_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_REG_HEADS = ["ABTS_TEAC", "DPPH_TEAC", "DPPH_pIC50"]


# ============================================================================
# Utility Functions
# ============================================================================

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


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def read_and_merge_tasks(data_dir, reg_heads):
    """Read multiple regression task CSVs and merge them by SMILES into a wide table.
    Missing values are filled with -1000 (masked out during loss computation).
    """
    dfs = []
    for head in reg_heads:
        file_path = Path(data_dir) / f"{head}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing task file: {file_path}")
        df = pd.read_csv(file_path)
        if "smiles" not in df.columns or "value" not in df.columns:
            raise ValueError(f"{file_path} must contain smiles and value columns")
        df = df.rename(columns={"value": head})
        dfs.append(df[["smiles", head]].copy())

    # Outer join to merge all tasks
    merged_df = dfs[0]
    for idx in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[idx], on="smiles", how="outer")
    merged_df = merged_df.fillna(-1000).drop_duplicates(subset=["smiles"]) \
                         .reset_index(drop=True)
    return merged_df


def make_seed_split(merged_df, seed, train_ratio=0.8, val_ratio=0.1):
    """SMILES-based deterministic data split."""
    smiles_list = merged_df["smiles"].dropna().astype(str).tolist()
    return make_smiles_split(smiles_list, seed=seed,
                             train_ratio=train_ratio, val_ratio=val_ratio)


def split_merged_df(merged_df, split_obj):
    """Partition the merged DataFrame according to a split dictionary."""
    tr = set(split_obj.get("train", []))
    va = set(split_obj.get("valid", []))
    te = set(split_obj.get("test", []))
    train_df = merged_df[merged_df["smiles"].isin(tr)].copy()
    valid_df = merged_df[merged_df["smiles"].isin(va)].copy()
    test_df = merged_df[merged_df["smiles"].isin(te)].copy()
    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("Seed split produced an empty partition")
    return train_df, valid_df, test_df


def preprocess_regression_splits(train_df, valid_df, test_df, reg_heads):
    """Apply Z-score normalization to each regression task.
    Statistics (mean, std) are computed on the training set only and
    applied to all three partitions.
    Returns the normalized DataFrames and the per-task mean/std arrays.
    """
    means = []
    stds = []
    for head in reg_heads:
        # Compute statistics only on valid (non -1000) training values
        train_mask = train_df[head] != -1000
        valid_mask = valid_df[head] != -1000
        test_mask = test_df[head] != -1000

        train_vals = train_df.loc[train_mask, head].astype(float)
        if len(train_vals) == 0:
            means.append(0.0)
            stds.append(1.0)
            continue

        train_vals_trans = train_vals
        valid_vals_trans = valid_df.loc[valid_mask, head].astype(float)
        test_vals_trans = test_df.loc[test_mask, head].astype(float)

        mean_val = float(train_vals_trans.mean())
        std_val = float(train_vals_trans.std())
        if not np.isfinite(std_val) or std_val <= 0:
            std_val = 1.0

        # Z-score: (x - mean) / std
        train_df.loc[train_mask, head] = (train_vals_trans - mean_val) / std_val
        valid_df.loc[valid_mask, head] = (valid_vals_trans - mean_val) / std_val
        test_df.loc[test_mask, head] = (test_vals_trans - mean_val) / std_val

        means.append(mean_val)
        stds.append(std_val)

    return (train_df, valid_df, test_df,
            np.asarray(means, dtype=np.float32),
            np.asarray(stds, dtype=np.float32))


def build_loaders(train_df, valid_df, test_df, reg_heads, batch_size, seed):
    """Build DataLoaders for multi-task regression."""
    train_dataset = Prediction_Dataset(train_df, smiles_head=["smiles"],
                                       reg_heads=reg_heads, clf_heads=[])
    valid_dataset = Prediction_Dataset(valid_df, smiles_head=["smiles"],
                                       reg_heads=reg_heads, clf_heads=[])
    test_dataset = Prediction_Dataset(test_df, smiles_head=["smiles"],
                                      reg_heads=reg_heads, clf_heads=[])

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    collate = Finetune_Collater(argparse.Namespace(clf_heads=[], reg_heads=reg_heads))
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
    # Strip "module." prefix if saved from DDP
    if hasattr(checkpoint, "keys") and \
       any(str(k).startswith("module.") for k in checkpoint.keys()):
        checkpoint = {str(k).replace("module.", ""): v
                      for k, v in checkpoint.items()}
    model.encoder.load_state_dict(checkpoint, strict=False)
    return model


# ============================================================================
# Loss & Metrics
# ============================================================================

def task_mean_loss(pred, target, loss_func):
    """Multi-task loss: compute each task's loss separately, then average.
    Missing values (target == -1000) are ignored via masking.
    """
    mask = target != -1000
    raw_loss = loss_func(pred, target)          # reduction='none'
    per_task_losses = []
    for idx in range(target.shape[1]):
        task_mask = mask[:, idx]
        if task_mask.sum() > 0:
            per_task_losses.append(
                (raw_loss[:, idx] * task_mask).sum() / task_mask.sum())
    if len(per_task_losses) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(per_task_losses).mean()


def compute_task_mae_rmse(y_true, y_pred):
    """Compute per-task MAE and RMSE, ignoring missing values."""
    task_mae = []
    task_rmse = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != -1000
        if np.sum(mask) == 0:
            task_mae.append(np.nan)
            task_rmse.append(np.nan)
            continue
        yi = y_true[mask, i]
        pi = y_pred[mask, i]
        diff = pi - yi
        task_mae.append(float(np.mean(np.abs(diff))))
        task_rmse.append(float(np.sqrt(np.mean(diff ** 2))))
    return task_mae, task_rmse


# ============================================================================
# Single Experiment
# ============================================================================

def run_single_experiment(args, reg_heads, seed, pretrain_epoch):
    """Run one complete multi-task regression experiment.
    Returns two dicts: validation results and test results.
    """
    set_seed(seed)

    # ---- Data preparation ----
    merged_df = read_and_merge_tasks(args.data_dir, reg_heads)
    split_obj = make_seed_split(merged_df, seed=seed,
                                train_ratio=0.8, val_ratio=0.1)
    train_df, valid_df, test_df = split_merged_df(merged_df, split_obj)
    # Z-score normalization
    train_df, valid_df, test_df, means, stds = preprocess_regression_splits(
        train_df, valid_df, test_df, reg_heads)

    current_batch_size = min(args.batch_size, max(2, len(train_df)))
    train_loader, valid_loader, test_loader = build_loaders(
        train_df, valid_df, test_df, reg_heads, current_batch_size, seed)

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
        reg_nums=len(reg_heads),     # multi-task regression
        clf_nums=0,
    )
    model = load_pretrained_encoder(model, args.pretrain_dir,
                                    args.arch, pretrain_epoch)
    model = model.to(DEVICE)

    # MSELoss with reduction='none' to allow per-task + per-sample masking
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.MSELoss(reduction="none")

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
            y_true = props["reg"].to(DEVICE)
            pred = model(x.to(DEVICE))["reg"]
            loss = task_mean_loss(pred, y_true, loss_func)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss.update(float(loss.item()), x.shape[0])

        # ---- Validation ----
        model.eval()
        val_loss_meter = AverageMeter()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for x, props in valid_loader:
                y_true = props["reg"].to(DEVICE)
                pred = model(x.to(DEVICE))["reg"]
                loss = task_mean_loss(pred, y_true, loss_func)
                val_loss_meter.update(float(loss.item()), x.shape[0])
                val_preds.append(pred.cpu().numpy())
                val_true.append(y_true.cpu().numpy())

        train_losses.append(epoch_loss.avg)
        valid_losses.append(val_loss_meter.avg)

        # ---- Early stopping check ----
        if args.early_stop_metric == "loss":
            current_monitor = float(val_loss_meter.avg)
            improved = current_monitor + 1e-12 < best_monitor
        else:
            current_monitor = float(val_loss_meter.avg)
            improved = current_monitor + 1e-12 < best_monitor

        if improved:
            best_monitor = current_monitor
            patience_counter = 0
            best_model_path = args.model_dir / \
                f"reg_best_{args.arch}_e{pretrain_epoch}_s{seed}.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # ---- Load best model, evaluate on test set ----
    if best_model_path is None:
        best_model_path = args.model_dir / \
            f"reg_best_{args.arch}_e{pretrain_epoch}_s{seed}.pth"
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    def collect_metrics(loader):
        """Collect predictions and compute MAE/RMSE in transformed (Z-score) space."""
        preds = []
        labels = []
        with torch.no_grad():
            for x, props in loader:
                y_true = props["reg"].to(DEVICE)
                pred = model(x.to(DEVICE))["reg"]
                preds.append(pred.cpu().numpy())
                labels.append(y_true.cpu().numpy())
        y_true_all = np.concatenate(labels, axis=0)
        y_pred_all = np.concatenate(preds, axis=0)

        task_mae, task_rmse = compute_task_mae_rmse(y_true_all, y_pred_all)
        avg_mae = float(np.nanmean(task_mae)) if np.any(~np.isnan(task_mae)) else 0.0
        avg_rmse = float(np.nanmean(task_rmse)) if np.any(~np.isnan(task_rmse)) else 0.0

        return task_mae, task_rmse, avg_mae, avg_rmse

    # Test set metrics
    test_task_mae, test_task_rmse, test_avg_mae, test_avg_rmse = \
        collect_metrics(test_loader)
    # Validation set metrics
    valid_task_mae, valid_task_rmse, valid_avg_mae, valid_avg_rmse = \
        collect_metrics(valid_loader)

    # ---- Plot loss curve ----
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.curve_dir /
                f"reg_loss_{args.arch}_e{pretrain_epoch}_s{seed}.png")
    plt.close()

    # ---- Assemble validation results ----
    result_valid = {
        "pretrain_epoch": pretrain_epoch,
        "seed": seed,
        "mean_mae": round(valid_avg_mae, 4),
        "mean_rmse": round(valid_avg_rmse, 4),
    }
    # ---- Assemble test results ----
    result_test = {
        "pretrain_epoch": pretrain_epoch,
        "seed": seed,
        "mean_mae": round(test_avg_mae, 4),
        "mean_rmse": round(test_avg_rmse, 4),
    }
    # Per-task details
    for idx, head in enumerate(reg_heads):
        result_test[f"{head}_mae"] = round(float(test_task_mae[idx]), 4)
        result_valid[f"{head}_mae"] = round(float(valid_task_mae[idx]), 4)
        result_test[f"{head}_rmse"] = round(float(test_task_rmse[idx]), 4)
        result_valid[f"{head}_rmse"] = round(float(valid_task_rmse[idx]), 4)

    return result_valid, result_test


# ============================================================================
# Main: sweep over seeds × pre-training epochs
# ============================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg-heads", nargs="+", default=DEFAULT_REG_HEADS,
                        type=str, help='List of regression tasks')
    parser.add_argument("--arch", default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--finetune", default="all",
                        choices=["none", "partial", "all"],
                        help='Fine-tuning strategy')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help='AdamW learning rate')
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--early-stop-metric", type=str, default="loss",
                        choices=["loss"])
    parser.add_argument("--seed-list", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretrain-start", type=int, default=85)
    parser.add_argument("--pretrain-end", type=int, default=100)
    parser.add_argument("--data-dir", type=str,
                        default=str(script_dir / "data" / "v1"))
    parser.add_argument("--pretrain-dir", type=str,
                        default=str(script_dir.parent / "weights_pubchem10M"))
    parser.add_argument("--output-root", type=str,
                        default=str(script_dir / "Results_old" / "Ours"))
    args = parser.parse_args()

    args.output_root = Path(args.output_root)
    args.model_dir = args.output_root / \
        f"models_bs{args.batch_size}_lr{args.lr}_ft{args.finetune}" / "regression"
    args.curve_dir = args.output_root / "loss_curves" / "regression"
    args.metrics_dir = args.output_root / \
        f"performance_bs{args.batch_size}_lr{args.lr}_ft{args.finetune}"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.curve_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    all_rows_valid = []
    all_rows_test = []

    for seed in args.seed_list:
        for pretrain_epoch in range(args.pretrain_start, args.pretrain_end + 1):
            row_valid, row_test = run_single_experiment(
                args, args.reg_heads, seed, pretrain_epoch)
            all_rows_valid.append(row_valid)
            all_rows_test.append(row_test)
            print(f"[reg] seed={seed} pretrain={pretrain_epoch} "
                  f"mean_valid_mae={row_valid['mean_mae']:.4f} "
                  f"mean_test_mae={row_test['mean_mae']:.4f} "
                  f"mean_valid_rmse={row_valid['mean_rmse']:.4f} "
                  f"mean_test_rmse={row_test['mean_rmse']:.4f}")

    # Save aggregated results
    out_df_valid = pd.DataFrame(all_rows_valid)
    out_df_valid.to_csv(args.metrics_dir / "regression_seeds_results_valid.csv",
                        index=False)

    out_df_test = pd.DataFrame(all_rows_test)
    out_df_test.to_csv(args.metrics_dir / "regression_seeds_results_test.csv",
                       index=False)


if __name__ == "__main__":
    main()
