"""Model training and evaluation module."""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from config.config import (
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
    RF_PARAMS,
    RANDOM_STATE,
    TEST_SIZE
)

def prepare_training_data(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                         pd.Series, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        train_df: Training DataFrame containing features and targets
    
    Returns:
        Tuple containing (X_train, X_val, y_train, y_val)
    """
    # Remove target variables
    X = train_df.drop(columns=["TARGET_B", "TARGET_D"])
    y_class = train_df["TARGET_B"]
    
    return train_test_split(
        X, y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class
    )

def train_classification_model(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: pd.DataFrame,
                             y_val: pd.Series) -> Tuple[LGBMClassifier, Dict]:
    """
    Train and evaluate classification model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
    
    Returns:
        Tuple of (trained_model, performance_metrics)
    """
    try:
        # Calculate class weight
        class_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logging.info(f"Class weight calculated: {class_weight}")
        
        # Initialize classifier with parameters from config
        clf_params = RF_PARAMS.copy()
        clf_params['scale_pos_weight'] = class_weight * 1.5
        
        clf = LGBMClassifier(**clf_params)
        logging.info("Classifier initialized with parameters")
        
        # Train model with evaluation
        clf.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc', 'binary_logloss']
        )
        logging.info("Classifier training completed")
        
        # Make predictions
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        # Save the model
        joblib.dump(clf, CLASSIFIER_MODEL_PATH)
        logging.info(f"Classifier saved to {CLASSIFIER_MODEL_PATH}")
        
        # Log metrics
        for metric, value in metrics.items():
            logging.info(f"Classification {metric}: {value:.4f}")
        
        return clf, metrics
    
    except Exception as e:
        logging.error(f"Error in classification model training: {str(e)}")
        raise
    
def train_regression_model(X_donors: pd.DataFrame,
                         y_donors: pd.Series) -> Tuple[RandomForestRegressor, Dict]:
    """
    Train and evaluate regression model for donation amounts.
    
    Args:
        X_donors: Features for donors only
        y_donors: Target donation amounts
    
    Returns:
        Tuple of (trained_model, performance_metrics)
    """
    try:
        # Split donor data
        X_train_donors, X_val_donors, y_train_donors, y_val_donors = train_test_split(
            X_donors, y_donors,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        logging.info("Donor data split for regression model")
        
        # Apply log transformation to the donation amounts.
        # This converts the target using np.log1p (i.e., log(1 + value))
        y_train_donors_log = np.log1p(y_train_donors)

        # Initialize and train regressor
        reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        reg.fit(X_train_donors, y_train_donors_log)
        logging.info("Regression model training completed")
        
        # Make predictions
        y_pred_donors = reg.predict(X_val_donors)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_val_donors, y_pred_donors),
            'rmse': np.sqrt(mean_squared_error(y_val_donors, y_pred_donors)),
            'r2': r2_score(y_val_donors, y_pred_donors)
        }
        
        # Save the model
        joblib.dump(reg, REGRESSOR_MODEL_PATH)
        logging.info(f"Regressor saved to {REGRESSOR_MODEL_PATH}")
        
        # Log metrics
        for metric, value in metrics.items():
            logging.info(f"Regression {metric}: {value:.4f}")
        
        return reg, metrics
    
    except Exception as e:
        logging.error(f"Error in regression model training: {str(e)}")
        raise

def save_models(clf: LGBMClassifier, reg: RandomForestRegressor) -> None:
    """
    Save trained models to disk.
    
    Args:
        clf: Trained classifier model
        reg: Trained regressor model
    """
    try:
        joblib.dump(clf, CLASSIFIER_MODEL_PATH)
        joblib.dump(reg, REGRESSOR_MODEL_PATH)
        logging.info(
            f"Models saved successfully:\n"
            f"Classifier: {CLASSIFIER_MODEL_PATH}\n"
            f"Regressor: {REGRESSOR_MODEL_PATH}"
        )
    except Exception as e:
        logging.error(f"Error saving models: {str(e)}")
        raise

def load_models() -> Tuple[LGBMClassifier, RandomForestRegressor]:
    """
    Load trained models from disk.
    
    Returns:
        Tuple of (classifier_model, regressor_model)
    """
    try:
        clf = joblib.load(CLASSIFIER_MODEL_PATH)
        reg = joblib.load(REGRESSOR_MODEL_PATH)
        logging.info("Models loaded successfully")
        return clf, reg
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise