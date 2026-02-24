# File: exps/evaluate.py
import logging
from pathlib import Path
import torch
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
import seaborn as sns


def compute_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)


def compute_precision(pred, target):
    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    return tp / (tp + fp + 1e-6)


def compute_recall(pred, target):
    tp = np.logical_and(pred == 1, target == 1).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    return tp / (tp + fn + 1e-6)


def compute_f1(pred, target):
    p = compute_precision(pred, target)
    r = compute_recall(pred, target)
    return 2 * p * r / (p + r + 1e-6)


def evaluate_model(model, test_loader, cfg, exp_name):
    """
    Run evaluation on the test set, computing segmentation metrics and saving results.

    Args:
      model: trained segmentation model
      test_loader: DataLoader for test set
      cfg: full configuration dict

    Returns:
      dict of averaged metrics
    """
    logger = logging.getLogger(__name__)
    
    # Set device based on config
    gpu_id = cfg['experiment'].get('gpu_id', 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # load best model weights
    ckpt_root = Path(cfg['output']['checkpoint_dir'])
    ckpt_path = ckpt_root / exp_name / f"{exp_name}_best.pth"
    if ckpt_path.exists():
        logger.info(f"Loading model weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}. Using current model weights.")
    model.to(device)
    model.eval()

    # Evaluation config
    ev = cfg['evaluation']
    metric_names = ev.get('metrics', [])

    # per class state
    stats = {
        0: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        1: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'correct': 0, 'total': 0
    }

    # Accumulate flat arrays for plot-based metrics
    all_probs  = []  # predicted probability for class=1
    all_labels = []  # ground-truth binary label

    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            imgs    = batch['image'].to(device)
            gt      = batch['label'].unsqueeze(1).cpu().numpy().astype(np.uint8)
            no_data = batch['no_data_mask'].unsqueeze(1).cpu().numpy()
            cls_gt  = batch['cls_label'].cpu().numpy()

            # Model prediction
            outputs = model(imgs)
            probs   = torch.sigmoid(outputs).cpu().numpy()    # [B,1,H,W]
            preds   = (probs > 0.5).astype(np.uint8)

            B = preds.shape[0]
            for i in range(B):
                pred_i = preds[i, 0]
                gt_i   = gt[i, 0]
                prob_i = probs[i, 0]
                valid  = ~no_data[i, 0]
                p_flat = pred_i[valid]
                g_flat = gt_i[valid]
                pr_flat = prob_i[valid]

                # Accumulate for plot-based metrics
                all_probs.append(pr_flat)
                all_labels.append(g_flat)
                
                for c in [0, 1]:
                    stats[c]['tp'] += np.logical_and(p_flat == c, g_flat == c).sum()
                    stats[c]['fp'] += np.logical_and(p_flat == c, g_flat != c).sum()
                    stats[c]['fn'] += np.logical_and(p_flat != c, g_flat == c).sum()
                    stats[c]['tn'] += np.logical_and(p_flat != c, g_flat != c).sum()

                stats['correct'] += (p_flat == g_flat).sum()
                stats['total'] += p_flat.size

    # Aggregate results
    results = {}
    def safe_div(a, b): return float(a) / (b + 1e-6)
    
    for c in [0, 1]:
        if 'precision' in metric_names:
            results[f'class{c}_precision'] = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fp'])
        if 'recall' in metric_names:
            results[f'class{c}_recall'] = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fn'])
        if 'f1' in metric_names:
            p = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fp'])
            r = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fn'])
            results[f'class{c}_f1'] = safe_div(2 * p * r, p + r)
        if 'iou' in metric_names:
            inter = stats[c]['tp']
            union = stats[c]['tp'] + stats[c]['fp'] + stats[c]['fn']
            results[f'class{c}_iou'] = safe_div(inter, union)
    
    # Macro average two classes
    if 'iou' in metric_names:
        results['res_iou_macro'] = 0.5 * (results['class0_iou'] + results['class1_iou'])
    if 'precision' in metric_names:
        results['res_precision_macro'] = 0.5 * (results['class0_precision'] + results['class1_precision'])
    if 'recall' in metric_names:
        results['res_recall_macro'] = 0.5 * (results['class0_recall'] + results['class1_recall'])
    if 'f1' in metric_names:
        results['res_f1_macro'] = 0.5 * (results['class0_f1'] + results['class1_f1'])
    if 'accuracy' in metric_names:
        results['overall_accuracy'] = safe_div(stats['correct'], stats['total'])

    # Log and return
    logger.info(f"Evaluation results:\n{results}")

    # ---- Evaluation plots ----
    results_root = Path(cfg['output']['results_dir'])
    results_dir = results_root / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation plots...")

    # Flatten collected arrays
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds  = (all_probs > 0.5).astype(np.uint8)

    # 1. ROC Curve
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(results_dir / "roc_curve.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save ROC curve: {e}")

    # 2. Precision-Recall Curve
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
        plt.figure()
        plt.plot(recall_vals, precision_vals, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curve')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(results_dir / "pr_curve.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save PR curve: {e}")

    # 3. Normalized Confusion Matrix
    try:
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Live', 'Dead'],
            yticklabels=['Live', 'Dead']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save confusion matrix: {e}")

    # 4. Metrics vs Threshold
    try:
        thresholds = np.linspace(0, 1, 101)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            p_t = (all_probs >= t).astype(np.uint8)
            tp = np.logical_and(p_t == 1, all_labels == 1).sum()
            fp = np.logical_and(p_t == 1, all_labels == 0).sum()
            fn = np.logical_and(p_t == 0, all_labels == 1).sum()
            prec = tp / (tp + fp + 1e-6)
            rec  = tp / (tp + fn + 1e-6)
            f1   = 2 * prec * rec / (prec + rec + 1e-6)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        plt.figure()
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1s, label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "threshold_metrics.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save threshold metrics plot: {e}")

    # 5. Calibration Curve (Reliability Diagram)
    try:
        _probs, _labels = all_probs, all_labels
        if len(_probs) > 500_000:  # subsample for speed
            idx = np.random.choice(len(_probs), 500_000, replace=False)
            _probs, _labels = _probs[idx], _labels[idx]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            _labels, _probs, n_bins=10, strategy='uniform'
        )
        plt.figure()
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "calibration_curve.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save calibration curve: {e}")

    return results
