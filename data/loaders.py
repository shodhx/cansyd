# Append these helper functions to your data/loaders.py to cleanly support main.py
def load_cwru_all():
    # Uses the global configurations securely compiled by your classes
    from sklearn.model_selection import train_test_split
    # Mocking dummy loads matching your structure for initialization alignment
    loads_tr = np.random.choice([0, 1, 2], len(X_train_all))
    loads_te = np.random.choice([3], len(X_test_all))
    return X_train_all, y_train_all, loads_tr, X_test_all, y_test_all, loads_te