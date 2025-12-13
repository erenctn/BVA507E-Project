import pandas as pd
import numpy as np
import random # Added for randomness
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelAgent:
    def __init__(self):
        # Dynamic Hyperparameters
        
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=random.randint(90, 120), 
                max_depth=random.randint(5, 15),      
                random_state=None 
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=random.randint(90, 120),
                learning_rate=random.uniform(0.05, 0.15),
                max_depth=random.randint(3, 7),
                random_state=None
            ),
            "Histogram GB": HistGradientBoostingClassifier(
                max_iter=random.randint(90, 120),
                learning_rate=random.uniform(0.05, 0.15),
                random_state=None
            )
        }
        self.best_model_name = None
        self.best_model = None
        self.metrics = {}

    def compare_and_train(self, df):
        """
        Trains models and calculates Accuracy, Precision, Recall, F1 metrics 
        according to PDF requirements.
        """
        features = ['RSI', 'Volatility', 'Returns', 'Upper_BB', 'Lower_BB']
        X = df[features]
        y = df['Target']
        
        # Since it is a time series, we split without shuffling (Shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        results = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Metrics from PDF Page 9 - Table E:
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            # Store results in detail
            results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            }
            
        # The champion is determined by F1 Score
        self.best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
        self.best_model = self.models[self.best_model_name]
        self.metrics = results
        
        return results, self.best_model_name

    def predict_current(self, current_row):
        """Makes a prediction with current data."""
        features = ['RSI', 'Volatility', 'Returns', 'Upper_BB', 'Lower_BB']
        input_data = pd.DataFrame([current_row], columns=features)
        
        probs = self.best_model.predict_proba(input_data)[0]
        prediction = self.best_model.predict(input_data)[0] # 0 or 1
        
        # If prediction is 1 (Rise), take probability of 1, otherwise take probability of 0
        confidence = probs[1] if prediction == 1 else probs[0]
        
        return {
            "prediction_id": int(prediction), # 1: Rise, 0: Fall
            "probability": confidence,
            "used_model": self.best_model_name
        }