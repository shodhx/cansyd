from sklearn.metrics import accuracy_score, f1_score

def report_classification(y_true, name="dataset"):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"{name} -> acc: {acc:.4f} | f1: {f1:.4f}")
    return acc, f1