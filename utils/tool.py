# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 6:07 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : tool
# @Software: PyCharm
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def accuracy_info(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    return tn, fp, fn, tp


def save_model(state, filename):
    torch.save(state, filename)


def norm(array):
    tmp = []
    for ar in array:
        a = ar[0]
        b = ar[1]
        ar[0] = a / (a + b)
        ar[1] = b / (a + b)
        tmp.append([ar[0], ar[1]])
    return np.array(tmp)


def evaluate(pred_label, true_label):
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.cpu().numpy()
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()
    if tp + fn == 0:
        black_p = 0.0
    else:
        black_p = tp / float(tp + fn)
    if tp + fp == 0:
        black_r = 0.0
    else:
        black_r = tp / float(tp + fp)
    if tn + fn == 0:
        white_p = 0.0
    else:
        white_p = tn / float(tn + fn)
    if tn + fp == 0:
        white_r = 0.0
    else:
        white_r = tn / float(tn + fp)

    acc = accuracy_score(true_label, pred_label)
    macro_f1 = f1_score(y_true=true_label, y_pred=pred_label, average="macro")
    micro_f1 = f1_score(y_true=true_label, y_pred=pred_label, average="micro")

    return acc, tn, fp, fn, tp, white_p, white_r, black_p, black_r, macro_f1, micro_f1
