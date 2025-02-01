"""Data preprocessing module."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from config.config import (
    TRAIN_FILE,
    VAL_FILE,
    PROCESSED_TRAIN_FILE,
    PROCESSED_VAL_FILE,
    MISSING_THRESHOLD
)
from src.utils import calculate_age_from_dob

def load_data(train_file: str = TRAIN_FILE,
              val_file: str = VAL_FILE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and perform initial preprocessing of the data.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
    
    Returns:
        Tuple of (training_dataframe, validation_dataframe)
    """
    try:
        train_df = pd.read_csv(train_file, low_memory=False)
        val_df = pd.read_csv(val_file, low_memory=False)
        
        logging.info(f"Training Data Shape: {train_df.shape}")
        logging.info(f"Validation Data Shape: {val_df.shape}")
        
        # Save initial data stats
        log_data_stats(train_df, "Initial Training")
        log_data_stats(val_df, "Initial Validation")
        
        return train_df, val_df
    
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        logging.error(f"Training file should be at: {train_file}")
        logging.error(f"Validation file should be at: {val_file}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading data: {str(e)}")
        raise

def log_data_stats(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Log basic statistics about the dataset.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset for logging
    """
    logging.info(f"\n{dataset_name} Dataset Statistics:")
    logging.info(f"Number of rows: {len(df)}")
    logging.info(f"Number of columns: {len(df.columns)}")
    logging.info(f"Missing values: {df.isnull().sum().sum()}")
    if 'TARGET_B' in df.columns:
        logging.info(f"Positive class ratio: {df['TARGET_B'].mean():.4f}")

def identify_missing_columns(df: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> List[str]:
    """
    Identify columns with missing values above threshold.
    
    Args:
        df: Input DataFrame
        threshold: Maximum acceptable percentage of missing values
    
    Returns:
        List of column names to drop
    """
    missing_percentages = df.isnull().mean() * 100
    return missing_percentages[missing_percentages > threshold].index.tolist()

def handle_missing_values(train_df: pd.DataFrame, 
                         val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle missing values in both training and validation datasets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    
    Returns:
        Tuple of processed (training_df, validation_df)
    """
    try:
        # Identify columns to drop based on missing values
        features_to_drop = identify_missing_columns(train_df)
        logging.info(f"Dropping {len(features_to_drop)} features due to high missing values")
        
        # Drop identified features
        train_df = train_df.drop(columns=features_to_drop)
        val_df = val_df.drop(columns=features_to_drop)
        
        # Handle remaining missing values
        numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = train_df.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_columns:
            if train_df[col].isnull().any():
                median_value = train_df[col].median()
                train_df[col].fillna(median_value, inplace=True)
                val_df[col].fillna(median_value, inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if train_df[col].isnull().any():
                mode_value = train_df[col].mode()[0]
                train_df[col].fillna(mode_value, inplace=True)
                val_df[col].fillna(mode_value, inplace=True)
        
        logging.info("Missing values handled successfully")
        return train_df, val_df
    
    except Exception as e:
        logging.error(f"Error handling missing values: {str(e)}")
        raise

def process_age(train_df: pd.DataFrame, 
                val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process age-related features.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    
    Returns:
        Tuple of processed (training_df, validation_df)
    """
    try:
        # Calculate ages
        logging.info("Processing age features...")
        train_df["calculated_age"] = train_df["DOB"].apply(calculate_age_from_dob)
        val_df["calculated_age"] = val_df["DOB"].apply(calculate_age_from_dob)
        
        # Analyze age calculation accuracy (training set only)
        age_comparison = train_df[['AGE', 'calculated_age']].copy()
        age_comparison['age_difference'] = age_comparison['AGE'] - age_comparison['calculated_age']
        
        logging.info("Age calculation statistics:")
        logging.info(f"Exact matches: {(age_comparison['age_difference'] == 0).sum()}")
        logging.info(f"Mean difference: {age_comparison['age_difference'].mean():.2f}")
        
        # Fill missing ages
        median_age = train_df["AGE"].median()
        train_df["AGE"] = train_df["AGE"].fillna(train_df["calculated_age"]).fillna(median_age)
        val_df["AGE"] = val_df["AGE"].fillna(val_df["calculated_age"]).fillna(median_age)
        
        # Drop temporary column
        train_df.drop(columns=["calculated_age"], inplace=True)
        val_df.drop(columns=["calculated_age"], inplace=True)
        
        logging.info("Age processing completed")
        return train_df, val_df
    
    except Exception as e:
        logging.error(f"Error processing age features: {str(e)}")
        raise

def save_processed_data(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame) -> None:
    """
    Save processed datasets to disk.
    
    Args:
        train_df: Processed training DataFrame
        val_df: Processed validation DataFrame
    """
    try:
        train_df.to_csv(PROCESSED_TRAIN_FILE, index=False)
        val_df.to_csv(PROCESSED_VAL_FILE, index=False)
        logging.info(f"Processed data saved to:\n{PROCESSED_TRAIN_FILE}\n{PROCESSED_VAL_FILE}")
    except Exception as e:
        logging.error(f"Error saving processed data: {str(e)}")
        raise

def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute complete preprocessing pipeline.
    
    Returns:
        Tuple of processed (training_df, validation_df)
    """
    try:
        # Load data
        train_df, val_df = load_data()
        
        # Handle missing values
        train_df, val_df = handle_missing_values(train_df, val_df)
        
        # Process age features
        train_df, val_df = process_age(train_df, val_df)
        
        # Save processed data
        save_processed_data(train_df, val_df)
        
        return train_df, val_df
    
    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {str(e)}")
        raise