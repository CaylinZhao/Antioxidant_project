# MTL-BERT: Integrating Multi-Assay Bioactivity Data for AI-Driven Discovery of Superior Antioxidants

Official implementation of the paper **"Integrating multi-assay bioactivity data for AI-driven discovery of superior antioxidants"**.

This repository provides a Transformer-based pre-training and fine-tuning framework for predicting antioxidant properties (ABTS-TEAC, DPPH-TEAC, DPPH-pIC50) and multi-level activity labels (active/medium/strong) from SMILES strings. The model integrates bioactivity data from multiple antioxidant assays into a unified multi-task learning architecture.

## Overview

The pipeline consists of two stages:

1. **Pre-training**: A medium-sized BERT encoder (8 layers, 8 heads, d_model=256) is trained on unlabeled SMILES via masked language modeling (MLM) — randomly masking 15% of input tokens and learning to reconstruct them.
2. **Fine-tuning**: The pre-trained encoder is loaded and prediction heads are added for downstream multi-task learning — jointly predicting regression targets (ABTS-TEAC, DPPH-TEAC, DPPH-pIC50) and classification labels (active/medium/strong) across multiple antioxidant assays.

## File Structure

```
github/
├── pretrain.py           # Masked SMILES pre-training entry point
├── regression.py         # Regression fine-tuning (TEAC, pIC50)
├── classification.py     # Classification fine-tuning (composite levels)
├── model.py              # BERT Transformer model architecture
├── dataset.py            # SMILES tokenization, datasets, and collaters
├── metrics.py            # Metric tracking utilities (AverageMeter, AUC)
├── split_utils.py        # Reproducible train/val/test SMILES splitting
└── data/
    ├── v1/               # Regression data (8 CSV files, one per assay)
    └── v2/               # Classification data (3 composite levels × 6 assays + thresholds)
```

## Dependencies

- Python 3.7+
- PyTorch ≥ 1.8
- RDKit
- NumPy, Pandas, Matplotlib, scikit-learn

```bash
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install rdkit -c conda-forge
pip install numpy pandas matplotlib scikit-learn
```

## Data

### Pre-training Data

The pre-training script expects a CSV file with a SMILES column (`CAN_SMILES` by default). A sample from ChEMBL is supported:
```
--dataset_path data/chembl_select_3.csv
```

### Fine-tuning Data

- **data/v1/**: Regression tasks. Each CSV contains `smiles` and `value` columns (e.g., `ABTS_TEAC.csv`, `DPPH_pIC50.csv`).
- **data/v2/**: Classification tasks organized by composite level:
  - `clf1_active/` — binary active/inactive labels
  - `clf2_medium/` — binary medium-level labels
  - `clf3_strong/` — binary strong-level labels
  - `composite/` — the three composite task CSVs
  - `threshold_definition.csv`, `threshold_summary.csv` — composite label thresholds

## Model Architecture

Built from Transformer encoder blocks (Vaswani et al., 2017):

| Component | Description |
|-----------|-------------|
| `Encoder` | Stack of `EncoderLayer` blocks with multi-head self-attention + position-wise FFN |
| `BertModel` | `Encoder` + 2-layer MLP head (`d_model → 2*d_model → vocab_size`) for MLM |
| `PredictionModel` | Pre-trained `EncoderForPrediction` + per-task `[d_model → 512 → 1]` MLP heads |

Three model sizes (medium is the default and recommended configuration):

| Size | Layers | Heads | d_model | d_ff |
|------|--------|-------|---------|------|
| small | 4 | 4 | 128 | 512 |
| medium | 8 | 8 | 256 | 1024 |
| large | 12 | 12 | 576 | 2304 |

Key architectural details:
- Positional encoding via sine/cosine functions
- Padding mask (ignore index 0) applied in self-attention
- GELU activation in feed-forward layers
- Pre-LayerNorm with residual connections
- Dropout rate: 0.1 throughout

## SMILES Tokenization

SMILES strings are tokenized using a regex that captures atom types (e.g., `Cl`, `Br`, `Si`), brackets, charges, and bond symbols. The vocabulary has 60 tokens including special tokens:

| Token | ID | Purpose |
|-------|-----|---------|
| `<PAD>` | 0 | Padding |
| `<MASK>` | 47 | MLM masking |
| `<UNK>` | 48 | Unknown characters |
| `<GLOBAL>` | 49 | Global token prepended to each sequence |
| `<p1>`–`<p10>` | 50–59 | Prediction head tokens (prepended during fine-tuning) |

## Usage

### Step 1: Pre-training

```bash
python pretrain.py \
    --Smiles_head CAN_SMILES \
    --dataset_path data/chembl_select_3.csv \
    --arch medium \
    --batch_size 128 \
    --lr 1e-4 \
    --epochs 100 \
    --save_path weights/ \
    --accumulation_steps 1
```

This saves two checkpoints per epoch:
- `medium_weights_bert_weightsmedium_{epoch}.pt` — full `BertModel`
- `medium_weights_bert_encoder_weightsmedium_{epoch}.pt` — encoder only (used for fine-tuning)

### Step 2: Classification Fine-tuning

```bash
python classification.py \
    --clf-tasks COMPOSITE_ACTIVE COMPOSITE_MEDIUM COMPOSITE_STRONG \
    --arch medium \
    --finetune all \
    --lr 1e-4 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 \
    --seed-list 0 1 2 3 4 \
    --pretrain-start 85 \
    --pretrain-end 100 \
    --data-root data/v2 \
    --pretrain-dir ../weights_pubchem10M \
    --early-stop-metric loss \
    --output-root Results/classification
```

Fine-tuning modes (`--finetune`):
- `"all"` — train all parameters
- `"partial"` — freeze embedding layer and first `N-2` encoder layers
- `none` — freeze entire encoder (not available in this code version but supported in model)

The script runs a sweep across seeds × pre-training epochs. Results are saved as CSV with per-task AUC scores.

### Step 3: Regression Fine-tuning

```bash
python regression.py \
    --reg-heads ABTS_TEAC DPPH_TEAC DPPH_pIC50 \
    --arch medium \
    --finetune all \
    --lr 1e-3 \
    --batch-size 60 \
    --epochs 200 \
    --patience 20 \
    --seed-list 0 1 2 3 4 \
    --pretrain-start 85 \
    --pretrain-end 100 \
    --data-dir data/v1 \
    --pretrain-dir ../weights_pubchem10M \
    --early-stop-metric loss \
    --output-root Results/regression
```

Multi-task regression: the model predicts all `--reg-heads` simultaneously. Each task has its own MLP head reading from its corresponding prediction token `<p1>`, `<p2>`, etc. Metrics reported include MAE and RMSE.

## Key Parameters

### Pre-training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--arch` | medium | Model size: small/medium/large |
| `--batch_size` | 128 | Per-GPU batch size |
| `--accumulation_steps` | 1 | Gradient accumulation steps |
| `--lr` | 1e-4 | Adam learning rate |
| `--epochs` | 100 | Pre-training epochs |

### Fine-tuning
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--arch` | medium | Must match pre-trained model |
| `--finetune` | all | Transfer learning strategy |
| `--lr` | 1e-3 (reg) / 1e-4 (clf) | AdamW learning rate |
| `--patience` | 20 | Early stopping patience |
| `--pretrain-start` | 85 | First pre-train epoch to evaluate |
| `--pretrain-end` | 100 | Last pre-train epoch to evaluate |
| `--seed-list` | 0 1 2 3 4 | Random seeds for data splitting |

## Data Splitting

SMILES-based stratified splitting (`split_utils.py`): unique SMILES are sorted, shuffled with a fixed seed, then partitioned into train/valid/test (default 80/10/10). This ensures no SMILES leakage between splits.

## Loss Functions

- **Pre-training**: Cross-entropy (ignoring padding at index 0), computed only on masked positions
- **Classification**: BCEWithLogitsLoss with mask for missing labels
- **Regression**: MSELoss with per-task averaging over valid (non-missing) labels

## Citation

If you use this code in your research, please cite the corresponding paper.
