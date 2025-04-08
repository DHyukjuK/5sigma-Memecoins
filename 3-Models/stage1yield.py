import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class YieldTokenDemandRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        print(f"MSE: {mean_squared_error(y_test, preds):.4f}")
        print(f"R²: {r2_score(y_test, preds):.4f}")
    
    def save_model(self, path):
        joblib.dump(self.model, path)