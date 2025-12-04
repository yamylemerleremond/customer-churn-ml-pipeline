"""
Model training pipeline for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import joblib
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """Train and evaluate churn prediction models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def load_data(self, filepath: str):
        """Load processed features."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression")
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model."""
        logger.info("Training Random Forest")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        import xgboost as xgb
        logger.info("Training XGBoost")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Evaluate model performance."""
        logger.info(f"Evaluating {model_name}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Store results
        self.results[model_name] = metrics
        
        # Print results
        print(f"\n{model_name.upper()} Results:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def compare_models(self):
        """Compare all trained models."""
        logger.info("Comparing models")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(comparison_df)
        
        # Find best model by F1 score
        best_model_name = comparison_df['f1_score'].idxmax()
        print(f"\nBest model by F1 score: {best_model_name}")
        
        return comparison_df, best_model_name
    
    def save_model(self, model_name: str, output_dir: str = 'models'):
        """Save trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_file = output_path / f"{model_name}.joblib"
        joblib.dump(self.models[model_name], model_file)
        logger.info(f"Saved {model_name} to {model_file}")
        
        # Save metrics
        metrics_file = output_path / f"{model_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.results[model_name], f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
    
    def run_training_pipeline(self, data_path: str):
        """Run complete training pipeline."""
        logger.info("Starting training pipeline")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(data_path)
        
        # Train models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        # Evaluate models
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Compare models
        comparison_df, best_model = self.compare_models()
        
        # Save all models
        for model_name in self.models.keys():
            self.save_model(model_name)
        
        logger.info("Training pipeline completed")
        
        return comparison_df, best_model


if __name__ == "__main__":
    trainer = ChurnModelTrainer()
    comparison_df, best_model = trainer.run_training_pipeline('data/telco_churn_features.csv')
    
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE - Best model: {best_model}")
    print(f"{'='*70}")