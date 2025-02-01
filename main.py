"""
Main execution script for the donor prediction model pipeline.
This script coordinates the entire modeling process from data loading to prediction.
"""

import os
import time
import logging
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# Import custom modules
from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_FILE,
    VAL_FILE,
    MAILING_COST_EXPENSIVE,
    MAILING_COST_CHEAP
)
from src.data_preprocessing import (
    load_data,
    handle_missing_values,
    process_age
)
from src.feature_engineering import (
    create_temporal_features,
    create_rfm_features,
    create_promotion_features,
    encode_categorical_features
)
from src.model_training import (
    prepare_training_data,
    train_classification_model,
    train_regression_model,
    save_models
)
from src.prediction import predict_for_new_data
from src.utils import print_step_time

def setup_logging() -> None:
    """Configure logging settings."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ),
            logging.StreamHandler()
        ]
    )

def validate_config() -> None:
    """Validate configuration settings and required files."""
    required_dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]
    required_files = [TRAIN_FILE, VAL_FILE]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logging.warning(f"Directory not found: {dir_path}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            raise FileNotFoundError(f"Missing required file: {file_path}")

def create_directory_structure() -> None:
    """Create necessary directories if they don't exist."""
    directories = {
        'raw_data': RAW_DATA_DIR,
        'processed_data': PROCESSED_DATA_DIR,
        'models': MODELS_DIR,
        'logs': 'logs',
        'results': 'results'
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created {name} directory: {path}")

def log_memory_usage() -> None:
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024  # in MB
    logging.info(f"Current memory usage: {mem_usage:.2f} MB")

def optimize_mailing_strategy(val_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Optimize mailing strategy based on expected donations and costs.
    
    Args:
        val_predictions: DataFrame containing prediction probabilities and amounts
    
    Returns:
        Tuple of (predictions_with_strategy, total_revenue)
    """
    # Compute net gains
    val_predictions["net_gain_expensive"] = (
        val_predictions["expected_donation"] - MAILING_COST_EXPENSIVE
    )
    val_predictions["net_gain_cheap"] = (
        val_predictions["expected_donation"] - MAILING_COST_CHEAP
    )
    
    # Determine optimal strategy
    val_predictions["mailing_strategy"] = "No Mail"
    val_predictions.loc[
        val_predictions["net_gain_expensive"] > 0, "mailing_strategy"
    ] = "Expensive Mail"
    val_predictions.loc[
        (val_predictions["net_gain_cheap"] > 0) & 
        (val_predictions["net_gain_expensive"] <= 0),
        "mailing_strategy"
    ] = "Cheap Mail"
    
    # Calculate total revenue
    total_revenue = (
        val_predictions.loc[
            val_predictions["mailing_strategy"] == "Expensive Mail",
            "net_gain_expensive"
        ].sum() +
        val_predictions.loc[
            val_predictions["mailing_strategy"] == "Cheap Mail",
            "net_gain_cheap"
        ].sum()
    )
    
    return val_predictions, total_revenue

def print_strategy_summary(predictions: pd.DataFrame) -> None:
    """Print summary statistics for each mailing strategy."""
    print("\nStrategy Summary:")
    print("=" * 50)
    
    for strategy in ["Expensive Mail", "Cheap Mail", "No Mail"]:
        strategy_data = predictions[predictions["mailing_strategy"] == strategy]
        print(f"\n{strategy}:")
        print(f"Count: {len(strategy_data)}")
        print(f"Average Expected Donation: ${strategy_data['expected_donation'].mean():.2f}")
        
        if strategy != "No Mail":
            net_gain_col = "net_gain_expensive" if strategy == "Expensive Mail" else "net_gain_cheap"
            print(f"Average Net Gain: ${strategy_data[net_gain_col].mean():.2f}")
            print(f"Total Net Gain: ${strategy_data[net_gain_col].sum():.2f}")

def save_results(val_predictions: pd.DataFrame, 
                total_revenue: float, 
                clf_metrics: Dict, 
                reg_metrics: Dict) -> None:
    """
    Save analysis results and metrics.
    
    Args:
        val_predictions: DataFrame with predictions and strategies
        total_revenue: Total expected revenue
        clf_metrics: Classification model metrics
        reg_metrics: Regression model metrics
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    predictions_path = os.path.join(results_dir, f'predictions_{timestamp}.csv')
    val_predictions.to_csv(predictions_path, index=False)
    
    # Save metrics
    metrics = {
        'classification': clf_metrics,
        'regression': reg_metrics,
        'total_revenue': total_revenue
    }
    metrics_path = os.path.join(results_dir, f'metrics_{timestamp}.json')
    pd.DataFrame(metrics).to_json(metrics_path)
    
    logging.info(f"Results saved:\n- Predictions: {predictions_path}\n- Metrics: {metrics_path}")

def main() -> None:
    """Main execution function."""
    total_start = time.time()
    
    try:
        # Setup
        setup_logging()
        create_directory_structure()
        validate_config()
        logging.info("Starting donor prediction pipeline")
        log_memory_usage()
        
        # 1. Data Loading and Preprocessing
        logging.info("Loading and preprocessing data...")
        train_df, val_df = load_data(TRAIN_FILE, VAL_FILE)
        train_df, val_df = handle_missing_values(train_df, val_df)
        train_df, val_df = process_age(train_df, val_df)
        log_memory_usage()
        
        # 2. Feature Engineering
        logging.info("Performing feature engineering...")
        train_df, val_df = create_temporal_features(train_df, val_df)
        train_df, val_df = create_rfm_features(train_df, val_df)
        train_df, val_df = create_promotion_features(train_df, val_df)
        train_df, val_df = encode_categorical_features(train_df, val_df)
        log_memory_usage()
        
        # 3. Model Training
        logging.info("Training models...")
        X_train, X_val, y_train, y_val = prepare_training_data(train_df)
        
        # Classification Model
        clf, clf_metrics = train_classification_model(X_train, y_train, X_val, y_val)
        logging.info("Classification Metrics:")
        for metric, value in clf_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Regression Model (for donors only)
        logging.info("Preparing data for regression model...")
        # Get the indices where y_train is 1 (donors)
        donors_mask = y_train == 1
        # Use loc to avoid index alignment issues
        X_donors = X_train.loc[donors_mask.index[donors_mask]]
        y_donors = train_df.loc[donors_mask.index[donors_mask], "TARGET_D"]
        
        reg, reg_metrics = train_regression_model(X_donors, y_donors)
        logging.info("Regression Metrics:")
        for metric, value in reg_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # 4. Save Models
        save_models(clf, reg)
        log_memory_usage()
        
        # 5. Make Predictions and Optimize Strategy
        logging.info("Optimizing mailing strategy...")
        val_predictions = pd.DataFrame()
        val_predictions["P_donation"] = clf.predict_proba(X_val)[:, 1]
        val_predictions["predicted_donation"] = reg.predict(X_val)
        val_predictions["expected_donation"] = (
            val_predictions["P_donation"] * val_predictions["predicted_donation"]
        )
        
        # 6. Optimize and summarize strategy
        val_predictions, total_revenue = optimize_mailing_strategy(val_predictions)
        print_strategy_summary(val_predictions)
        logging.info(f"Total Expected Net Revenue: ${total_revenue:.2f}")
        
        # 7. Save Results
        save_results(val_predictions, total_revenue, clf_metrics, reg_metrics)
        
        # Print execution time and final memory usage
        execution_time = time.time() - total_start
        logging.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
        log_memory_usage()
        print_step_time("Total Execution", total_start)
        
    except FileNotFoundError as e:
        logging.error(f"Data file error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info("Pipeline execution finished")

if __name__ == "__main__":
    main()