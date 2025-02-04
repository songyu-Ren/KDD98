"""
Main execution script for the donor prediction model pipeline.
This script coordinates the entire modeling process from data loading to prediction.
"""

import os
import time
import logging
import psutil
import pandas as pd
from datetime import datetime
from typing import Dict
import joblib

# Import custom modules
from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_FILE,
    VAL_FILE
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
from src.feature_selection import perform_feature_selection
from src.model_training import (
    prepare_training_data,
    train_classification_model,
    train_regression_model,
    save_models
)
from src.prediction import (
    DonationPredictor,
    format_strategy_report,
    save_prediction_results
)
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

def save_model_metrics(clf_metrics: Dict, reg_metrics: Dict) -> None:
    """Save model training metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        'classification': clf_metrics,
        'regression': reg_metrics
    }
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f'training_metrics_{timestamp}.json')
    pd.DataFrame(metrics).to_json(metrics_path)
    logging.info(f"Training metrics saved to: {metrics_path}")

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
        
        # 3. Feature Selection
        logging.info("Performing feature selection...")
        train_df, val_df, selected_features = perform_feature_selection(train_df, val_df)
        
        # Save selected features
        feature_dict = {
            'selected_features': selected_features
        }
        feature_path = os.path.join(MODELS_DIR, 'selected_features.pkl')
        joblib.dump(feature_dict, feature_path)
        logging.info(f"Selected features saved to: {feature_path}")

        # 4. Model Training
        logging.info("Training models...")
        X_train, X_val, y_train, y_val = prepare_training_data(train_df)
        
        # Classification Model
        clf, clf_metrics = train_classification_model(X_train, y_train, X_val, y_val)
        logging.info("Classification Metrics:")
        for metric, value in clf_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Regression Model (for donors only)
        logging.info("Preparing data for regression model...")
        donors_mask = y_train == 1
        X_donors = X_train.loc[donors_mask.index[donors_mask]]
        y_donors = train_df.loc[donors_mask.index[donors_mask], "TARGET_D"]
        
        reg, reg_metrics = train_regression_model(X_donors, y_donors)
        logging.info("Regression Metrics:")
        for metric, value in reg_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # 5. Save Models
        save_models(clf, reg)
        save_model_metrics(clf_metrics, reg_metrics)
        log_memory_usage()
        
        # 6. Make Predictions and Optimize Strategy using DonationPredictor
        logging.info("Making predictions and optimizing mailing strategy...")
        predictor = DonationPredictor()
        predictor.set_selected_features(selected_features)  # Set selected features
        predictions, metrics = predictor.make_predictions(val_df)
        
        # Log feature selection info
        logging.info(f"Number of selected features: {len(selected_features)}")
        logging.info("Selected features:")
        for feature in selected_features:
            logging.info(f"- {feature}")

        # 7. Generate and save report
        report = format_strategy_report(metrics)
        save_prediction_results(predictions, report, metrics)
        
        # Print summary
        logging.info(f"Total Expected Net Revenue: ${metrics['overall']['total_net_revenue']:.2f}")
        logging.info(f"Total Mailings: {metrics['overall']['total_mailings']}")
        
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