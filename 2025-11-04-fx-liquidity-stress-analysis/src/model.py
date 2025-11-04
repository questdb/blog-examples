# src/model.py


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

class LiquidityStressModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None

    def prepare_data(self, df, feature_cols):
        """Prepare features and labels for training"""
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df['stress_label'].values
        return X, y

    def walk_forward_validation(self, X, y, n_splits=5):
        """Perform walk-forward validation for time series data"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold+1}/{n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Compute scale_pos_weight for imbalanced classes
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            scores.append({'precision': precision, 'recall': recall, 'f1': f1})

        # Average metrics across folds
        avg_precision = np.mean([s['precision'] for s in scores])
        avg_recall = np.mean([s['recall'] for s in scores])
        avg_f1 = np.mean([s['f1'] for s in scores])
        print(f"\nAverage Performance → Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1: {avg_f1:.3f}")
        return scores

    def predict_stress(self, current_features):
        """Predict stress for a single feature row or 1D array."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        arr = np.asarray(current_features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        prob = float(self.model.predict_proba(arr)[0, 1])
        pred = bool(self.model.predict(arr)[0])
        return {
            "stress_probability": prob,
            "stress_prediction": pred,
            "alert_level": ("HIGH" if prob > 0.75 else "MEDIUM" if prob > 0.5 else "LOW"),
        }

    def save_model(self, path: str):
        """Persist model to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load_model(self, path: str):
        """Load previously saved model."""
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

    def train_final_model(self, X, y):
        """Train final model on all data"""
        print("Training final model on full dataset...")

        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X, y)
        print("Final model trained!")

        # Display feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print("\nTop 5 features by importance:")
        print(feature_importance.head())

