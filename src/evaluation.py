"""
Evaluation and Metrics Module for E-Commerce Return Intelligence.

Provides rigorous performance evaluation tailored for class-imbalanced data,
including threshold tuning, ROC/PR curves, confusion matrices, and detailed reports.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics focusing on minority class (Returned = 1).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    
    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(np.mean(y_true))
            
    report = classification_report(y_true, y_pred, target_names=['Not Returned', 'Returned'], output_dict=True, zero_division=0)

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
        'pr_auc': round(pr_auc, 4) if pr_auc is not None else None,
        'confusion_matrix': {
            'matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        },
        'classification_report': report
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Tune the classification probability threshold on validation data.
    In highly imbalanced scenarios (6.3% positive class), standard 0.5 threshold
    severely penalizes minority class recall.
    
    Returns:
        (best_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.50
    best_score = -1.0

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == 'f2':  # Gives higher weight to recall
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            score = (5 * p * r) / (4 * p + r + 1e-6)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = th

    return round(float(best_thresh), 3), round(float(best_score), 4)


def compute_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Generate ROC and Precision-Recall curve coordinate points for plotting.
    """
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

    # Downsample points for efficient JSON serialization and rendering
    step_roc = max(1, len(fpr) // 100)
    step_pr = max(1, len(prec) // 100)

    return {
        'roc_curve': {
            'fpr': [round(float(x), 4) for x in fpr[::step_roc]],
            'tpr': [round(float(x), 4) for x in tpr[::step_roc]],
            'thresholds': [round(float(x), 4) for x in roc_thresh[::step_roc]]
        },
        'pr_curve': {
            'precision': [round(float(x), 4) for x in prec[::step_pr]],
            'recall': [round(float(x), 4) for x in rec[::step_pr]],
            'thresholds': [round(float(x), 4) for x in pr_thresh[::step_pr]]
        }
    }
