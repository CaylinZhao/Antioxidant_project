"""
Metric-tracking utilities for training and evaluation.
"""

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score


class AverageMeter(object):
    """Accumulates and tracks the weighted average of a metric.
    Useful for scalar metrics (loss, accuracy) that need per-batch weighting.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0        # most recently updated value
        self.avg = 0        # weighted average
        self.sum = 0        # cumulative sum
        self.count = 0      # cumulative sample count

    def update(self, val, n=1):
        """val: current batch value; n: number of samples (or weight sum) in the batch."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Records_AUC(object):
    """Accumulates predicted probabilities and labels for epoch-level AUC computation.
    Designed for classification: collects predictions across batches, then
    computes per-task ROC-AUC once at the end.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_list = []
        self.label_list = []

    def update(self, y_pred, y_true):
        """Append one batch of predicted probabilities and ground-truth labels."""
        self.pred_list.append(y_pred)
        self.label_list.append(y_true)

    def results(self):
        """Compute per-task ROC-AUC over all accumulated data.
        Uses sample_weight to ignore missing values (label == -1000).
        """
        pred = np.hstack(self.pred_list)
        label = np.concatenate(self.label_list, axis=0)

        results = []
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        for i in range(pred.shape[1]):
            results.append(roc_auc_score(
                (label[:, i] != -1000) * label[:, i],
                pred[:, i],
                sample_weight=(label[:, i] != -1000).astype('float32'),
                multi_class='ovo'))
        return results
