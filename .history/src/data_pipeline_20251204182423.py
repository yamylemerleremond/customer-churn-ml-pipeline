"""
Data pipeline for customer churn prediction.
Handles data loading, cleaning, and validation.
"""

import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnDataPipeline:
    """Pipeline for processing customer churn data."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV."""
        logger.info(f"Loading data from {self.input_path}")
        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        logger.info("Starting data cleaning")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric (it might have spaces)
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'], 
            errors='coerce'
        )
        
        # Drop rows with missing TotalCharges
        before_drop = len(df_clean)
        df_clean = df_clean.dropna(subset=['TotalCharges'])
        dropped = before_drop - len(df_clean)
        logger.info(f"Dropped {dropped} rows with missing TotalCharges")
        
        # Convert Yes/No to 1/0 for Churn
        df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality."""
        logger.info("Validating data quality")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values")
            return False
        
        # Check if Churn column exists and has correct values
        if 'Churn' not in df.columns:
            logger.error("Churn column missing")
            return False
            
        if not df['Churn'].isin([0, 1]).all():
            logger.error("Churn column has invalid values")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def save_data(self, df: pd.DataFrame):
        """Save processed data."""
        logger.info(f"Saving processed data to {self.output_path}")
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved {len(df)} rows")
    
    def run(self):
        """Run the complete pipeline."""
        logger.info("Starting data pipeline")
        
        # Load
        df = self.load_data()
        
        # Clean
        df_clean = self.clean_data(df)
        
        # Validate
        if not self.validate_data(df_clean):
            raise ValueError("Data validation failed")
        
        # Save
        self.save_data(df_clean)
        
        logger.info("Pipeline completed successfully")
        return df_clean


# Script to run the pipeline
if __name__ == "__main__":
    pipeline = ChurnDataPipeline(
        input_path="data/telco_churn_raw.csv",
        output_path="data/telco_churn_clean.csv"
    )
    
    df_processed = pipeline.run()
    print("\nProcessed data shape:", df_processed.shape)
    print("\nFirst few rows:")
    print(df_processed.head())