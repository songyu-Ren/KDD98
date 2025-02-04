"""
Feature selection module for selecting most important features using Random Forest 
and RFECV methods.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from joblib import parallel_backend
from src.utils import print_step_time

def perform_feature_selection(train_df: pd.DataFrame, 
                            val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform two-stage feature selection using Random Forest importance and RFECV.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    
    Returns:
        Tuple of (reduced_train_df, reduced_val_df)
    """
    try:
        # Get target columns that exist in each dataset
        train_targets = [col for col in ["TARGET_B", "TARGET_D"] if col in train_df.columns]
        val_targets = [col for col in ["TARGET_B", "TARGET_D"] if col in val_df.columns]
        
        # 1. Data preparation
        step_start = time.time()
        X_train = train_df.drop(columns=train_targets, errors='ignore').astype(np.float32)
        y_train = train_df["TARGET_B"] if "TARGET_B" in train_df.columns else None
        
        if y_train is None:
            raise ValueError("Training data must contain TARGET_B column")
            
        print_step_time("Data Preparation", step_start)

        # 2. Random Forest feature importance selection
        step_start = time.time()
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=False,
            warm_start=False
        )
        
        with parallel_backend('threading', n_jobs=8):
            rf.fit(X_train, y_train)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['importance_ratio'] = (
            importance_df['cumulative_importance'] / 
            importance_df['importance'].sum()
        )

        # Find features that together explain 95% of importance
        n_features_95 = len(
            importance_df[importance_df['importance_ratio'] <= 0.95]
        ) + 1
        selected_features = importance_df['feature'][:n_features_95].tolist()
        logging.info(
            f"\nFeatures selected (95% importance): {n_features_95} out of {len(X_train.columns)}"
        )
        print_step_time("Random Forest Selection", step_start)

        # Reduce features based on RF importance
        X_train_reduced = X_train[selected_features]

        # 3. RFECV selection
        step_start = time.time()
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        # Initialize RF classifier for RFECV
        rf_rfecv = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            class_weight='balanced'
        )

        # Initialize RFECV
        rfecv = RFECV(
            estimator=rf_rfecv,
            step=50,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            min_features_to_select=100
        )

        # Fit RFECV
        with parallel_backend('threading', n_jobs=8):
            rfecv = rfecv.fit(X_train_reduced, y_train)
        print_step_time("RFECV Training", step_start)

        # Get final selected features
        rfecv_selected_features = X_train_reduced.columns[rfecv.support_].tolist()
        logging.info(
            f"\nNumber of features selected by RFECV: {len(rfecv_selected_features)} "
            f"out of {X_train_reduced.shape[1]}"
        )
        logging.info(f"Optimal number of features: {rfecv.n_features_}")

        # Apply feature selection to both train and validation sets
        # Start with selected features
        train_selected = train_df[rfecv_selected_features + train_targets]
        
        # For validation set, only include features that exist
        val_features = [col for col in rfecv_selected_features if col in val_df.columns]
        val_selected = val_df[val_features + val_targets]

        # Log feature selection summary
        logging.info(f"Final number of features (including targets):")
        logging.info(f"Training set: {len(train_selected.columns)}")
        logging.info(f"Validation set: {len(val_selected.columns)}")

        # Log selected features
        logging.info("Selected features:")
        for feature in rfecv_selected_features:
            logging.info(f"- {feature}")

        return train_selected, val_selected, rfecv_selected_features

    except Exception as e:
        logging.error(f"Error in feature selection: {str(e)}")
        raise