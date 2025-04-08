import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib

class FundingRateForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        print(f"MAE: {mean_absolute_error(y_test, preds):.6f}")
    
    def save_model(self, path):
        joblib.dump(self.model, path)