# src/rule_models.py

import pandas as pd
import numpy as np

class RuleClassifier:
    def __init__(self, median_pps):
        self.median_pps = median_pps

    def predict(self, X):
        pps = pd.to_numeric(X.get('Price_per_SqFt'), errors='coerce').fillna(self.median_pps)
        bhk = pd.to_numeric(X.get('BHK'), errors='coerce').fillna(3)
        preds = ((pps <= self.median_pps) & (bhk >= 3)).astype(int)
        return preds.values

    def predict_proba(self, X):
        preds = self.predict(X)
        probs = np.vstack([1-preds, preds]).T
        return probs


class RuleRegressor:
    def __init__(self, growth_rate=0.08):
        self.growth_rate = growth_rate

    def predict(self, X):
        price = pd.to_numeric(X.get('Price_in_Lakhs'), errors='coerce').fillna(0)
        res = price * ((1+self.growth_rate)**5)
        return res.values
