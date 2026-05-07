from sklearn.metrics import accuracy_score, f1_score import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

def report_classification(y_true, name="dataset"):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"{name} -> acc: {acc:.4f} | f1: {f1:.4f}")
    return acc, f1

def Check_Acc(true_y, pred_y):
    a = accuracy_score(true_y, pred_y)
    f1 = f1_score(true_y, pred_y, average='weighted')
    print(a)
    print(f1)
    return a, f1

def do_wilcoxon(base_ate, new_ate):
    if np.array_equal(base_ate, new_ate):
        return 1.0
        
    stat, p = wilcoxon(base_ate, new_ate)
    print(p)
    return p