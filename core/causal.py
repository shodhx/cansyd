import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class CausalEngine:
    def __init__(self):
        self.ate_model = None
        self.ate_value = 0.0

    def fit_ate(self, test_features, y_test, load_test):
       
        norms = np.linalg.norm(test_features, axis=1)
        fault_binary = (y_test > 0).astype(int)
        
        self.ate_model = LinearRegression().fit(norms.reshape(-1, 1), fault_binary)
        self.ate_value = self.ate_model.coef_[0]
        return self.ate_value

    def get_counterfactual(self, norm, load_val, reduction=0.5):
      
        base_risk = norm * self.ate_value
        cf_risk = (norm * reduction) * self.ate_value
        return base_risk, cf_risk