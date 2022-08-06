import numpy as np
import torch
from sklearn.metrics import precision_recall_curve


def cal_metric(logits, labels):
    preds = torch.argmax(logits, dim=1)  # logits: (b, 2)
    # preds = (logits > 0).long()       # logits: (b)
    TP = ((labels == 1) & (preds == 1)).sum().item()
    FN = ((labels == 1) & (preds == 0)).sum().item()
    FP = ((labels == 0) & (preds == 1)).sum().item()
    TN = ((labels == 0) & (preds == 0)).sum().item()

    acc = P = R = f1 = P0 = 0
    if (TP + FN + FP + TN) != 0:
        acc = (TP + TN) / (TP + FN + FP + TN)
    if (TP + FP) != 0:
        P = TP / (TP + FP)
    if (TP + FN) != 0:
        R = TP / (TP + FN)
    if (P + R) != 0:
        f1 = 2 * P * R / (P + R)
    if (TN + FN) != 0:
        P0 = TN / (TN + FN)
    mP = (P + P0) / 2.0

    return acc, P, R, f1


def search_thres(label, pred):
    np.seterr(invalid="ignore")
    P, R, thres = precision_recall_curve(label, pred, pos_label=1)
    f1 = (2 * P * R) / (P + R)
    best_f1 = np.max(f1[np.isfinite(f1)])
    best_f1_index = np.argmax(f1[np.isfinite(f1)])
    return best_f1, P[best_f1_index], R[best_f1_index], thres[best_f1_index]
