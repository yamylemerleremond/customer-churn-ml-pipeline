"""
Feature engineering for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    """Feature engineering for churn prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        logger.info("Creating new features")
        
        df = df.copy()
        
        # Tenure groups
        df['tenure_group'] = pd.cut(
            df['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
        )
        
        # Average monthly charges
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Service usage score (count of services)
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        df['service_count'] = 0
        for col in service_cols:
            if col in df.columns:
                df['service_count'] += (df[col] != 'No').astype(int)
        
        # Contract and payment interaction
        df['contract_payment'] = df['Contract'] + '_' + df['PaymentMethod']
        
        logger.info(f"Created features. New shape: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("Encoding categorical variables")
        
        df = df.copy()
        
        # Binary categorical variables (Yes/No)
        binary_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'PaperlessBilling'
        ]
        
        for col in binary_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = (df[col] == 'Yes').astype(int)
        
        # Multi-class categorical variables
        categorical_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaymentMethod', 'tenure_group', 'contract_payment'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        df[col] = df[col].apply(
                            lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                        )
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling numerical features")
        
        df = df.copy()
        
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges']
        
        # Only scale columns that exist
        cols_to_scale = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature preparation pipeline."""
        logger.info("Starting feature preparation pipeline")
        
        # Create new features
        df = self.create_features(df)
        
        # Encode categorical
        df = self.encode_categorical(df, fit=fit)
        
        # Scale numerical
        df = self.scale_features(df, fit=fit)
        
        # Drop customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        logger.info(f"Feature preparation complete. Final shape: {df.shape}")
        return df


# Test the feature engineering
if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/telco_churn_clean.csv')
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Create and apply feature engineering
    feature_engineer = ChurnFeatureEngineer()
    X_processed = feature_engineer.prepare_features(X, fit=True)
    
    print("\nProcessed features shape:", X_processed.shape)
    print("\nFeature columns:")
    print(X_processed.columns.tolist())
    print("\nFirst few rows:")
    print(X_processed.head())
    
    # Save processed features
    X_processed['Churn'] = y
    X_processed.to_csv('data/telco_churn_features.csv', index=False)
    print("\nSaved processed features to data/telco_churn_features.csv")