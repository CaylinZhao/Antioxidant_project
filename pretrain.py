"""
MLM pre-training script.
Trains a BERT encoder on unlabeled SMILES data to learn general-purpose
molecular representations via masked language modeling.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import BertModel
from dataset import Smiles_Bert_Dataset, Pretrain_Collater
import time
import os
from torch.utils.data import DataLoader
from metrics import AverageMeter
import argparse


# ============================================================================
# Argument Configuration
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--Smiles_head', nargs='+', default=["CAN_SMILES"], type=str,
                    help='Name of the SMILES column in the CSV')
parser.add_argument('--dataset_path', default='data/chembl_select_3.csv', type=str,
                    help='Path to pre-training data')
parser.add_argument('--save_path', default='weights/', type=str,
                    help='Directory for saving model checkpoints')
parser.add_argument('--arch', default='medium',
                    choices=['small', 'medium', 'large'], type=str,
                    help='Model size preset')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size per GPU')
parser.add_argument('--accumulation_steps', default=1, type=int,
                    help='Gradient accumulation steps (simulates larger batch size)')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Adam learning rate')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of pre-training epochs')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Architecture presets ----
small = {'name': 'small', 'num_layers': 4, 'num_heads': 4,
         'd_model': 128, 'path': 'small_weights'}
medium = {'name': 'medium', 'num_layers': 8, 'num_heads': 8,
          'd_model': 256, 'path': 'medium_weights'}
large = {'name': 'large', 'num_layers': 12, 'num_heads': 12,
         'd_model': 576, 'path': 'large_weights'}

arch_dict = {'small': small, 'medium': medium, 'large': large}
arch = arch_dict[args.arch]
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']

# Hyperparameters
dff = d_model * 4             # feed-forward hidden dimension
vocab_size = 60               # SMILES token vocabulary size
dropout_rate = 0.1

# ============================================================================
# Model & Data Initialization
# ============================================================================

model = BertModel(num_layers=num_layers, d_model=d_model, dff=dff,
                  num_heads=num_heads, vocab_size=vocab_size)
model.to(device)

# Load pre-training data
full_dataset = Smiles_Bert_Dataset(args.dataset_path, Smiles_head=args.Smiles_head)

# Train/test split: 1% for testing (capped at 10,000)
test_size = min(10000, int(0.01 * len(full_dataset)))
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=Pretrain_Collater())
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=Pretrain_Collater())

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.98))
# ignore_index=0: <PAD> positions do not contribute to loss
# reduction='none': needed for per-position weighting
loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

# Metric trackers
train_loss = AverageMeter()
train_acc = AverageMeter()
test_loss = AverageMeter()
test_acc = AverageMeter()


# ============================================================================
# Training & Evaluation Steps
# ============================================================================

def train_step(x, y, weights, batch_idx):
    """Single training step: forward → weighted loss → backward (with gradient accumulation)."""
    model.train()
    predictions = model(x)
    # Weighted average loss over masked positions
    loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum() / weights.sum()

    # Scale loss for gradient accumulation
    loss = loss / args.accumulation_steps
    loss.backward()

    # Update parameters only after enough accumulation steps
    if (batch_idx + 1) % args.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Record the de-scaled loss for logging
    train_loss.update(loss.detach().item() * args.accumulation_steps, x.shape[0])
    # Accuracy computed only on masked positions
    train_acc.update(
        ((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item()
        / weights.cpu().sum().item(),
        weights.cpu().sum().item())


def test_step(x, y, weights):
    """Single evaluation step: compute loss and accuracy on validation/test set."""
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum() / weights.sum()

        test_loss.update(loss.detach(), x.shape[0])
        test_acc.update(
            ((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item()
            / weights.cpu().sum().item(),
            weights.cpu().sum().item())


# ============================================================================
# Main Training Loop
# ============================================================================

for epoch in range(args.epochs):
    start = time.time()

    # ---- Training ----
    for (batch, (x, y, weights)) in enumerate(train_dataloader):
        train_step(x, y, weights, batch)

        # Log training metrics every 500 batches
        if batch % 500 == 0:
            print('Epoch {} Batch {} training Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.avg))
            print('traning Accuracy: {:.4f}'.format(train_acc.avg))

        # Evaluate on test set every 1000 batches
        if batch % 1000 == 0:
            model.eval()
            with torch.no_grad():
                for x, y, weights in test_dataloader:
                    test_step(x, y, weights)
            print('Test loss: {:.4f}'.format(test_loss.avg))
            print('Test Accuracy: {:.4f}'.format(test_acc.avg))

            # Reset trackers after evaluation
            test_acc.reset()
            test_loss.reset()
            train_acc.reset()
            train_loss.reset()

            # Free GPU memory fragments accumulated during evaluation
            torch.cuda.empty_cache()
            model.train()

    # ---- End of epoch ----
    print('Epoch {} is Done!'.format(epoch))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, train_loss.avg))
    print('training Accuracy: {:.4f}'.format(train_acc.avg))
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.avg))
    print('test Accuracy: {:.4f}'.format(test_acc.avg))

    # ---- Save checkpoints ----
    os.makedirs(args.save_path, exist_ok=True)
    # Full BERT model (can be used for continued training)
    torch.save(model.state_dict(),
               args.save_path + arch['path']
               + '_bert_weights{}_{}.pt'.format(arch['name'], epoch + 1))
    # Encoder only (needed for downstream fine-tuning)
    torch.save(model.encoder.state_dict(),
               args.save_path + arch['path']
               + '_bert_encoder_weights{}_{}.pt'.format(arch['name'], epoch + 1))
    print('Successfully saving checkpoint!!!')
