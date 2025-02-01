"""Feature engineering module for data preparation."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def create_temporal_features(train_df: pd.DataFrame, 
                           val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create temporal features from ADATE columns.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    
    Returns:
        Tuple of (training_df, validation_df) with temporal features
    """
    try:
        # Find ADATE columns
        adate_cols = [col for col in train_df.columns if "ADATE_" in col]
        logging.info(f"Creating temporal features from {len(adate_cols)} ADATE columns")
        
        # Create new features
        new_features = {
            'days_since_last_promo': lambda df: 199812 - df[adate_cols].apply(
                pd.to_numeric, errors='coerce').max(axis=1),
            'total_promos_received': lambda df: df[adate_cols].apply(
                pd.to_numeric, errors='coerce').notna().sum(axis=1)
        }
        
        # Apply to both datasets
        for name, func in new_features.items():
            train_df[name] = func(train_df)
            val_df[name] = func(val_df)
            logging.info(f"Created feature: {name}")
        
        return train_df, val_df
        
    except Exception as e:
        logging.error(f"Error in temporal feature creation: {str(e)}")
        raise

def create_rfm_features(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create RFM (Recency, Frequency, Monetary) features.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    
    Returns:
        Tuple of (training_df, validation_df) with RFM features
    """
    try:
        logging.info("Creating RFM features")
        
        # Recency
        train_df["recency"] = 199812 - train_df["LASTGIFT"]
        val_df["recency"] = 199812 - val_df["LASTGIFT"]
        
        # Frequency
        train_df["frequency"] = train_df["NGIFTALL"]
        val_df["frequency"] = val_df["NGIFTALL"]
        
        # Monetary
        train_df["avg_donation"] = train_df["AVGGIFT"]
        val_df["avg_donation"] = val_df["AVGGIFT"]
        
        # Fill missing values
        for df in [train_df, val_df]:
            df.fillna({"avg_donation": 0}, inplace=True)
        
        logging.info("RFM features created successfully")
        return train_df, val_df
        
    except Exception as e:
        logging.error(f"Error in RFM feature creation: {str(e)}")
        raise

def create_promotion_features(train_df: pd.DataFrame, 
                            val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create promotion effectiveness features.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    
    Returns:
        Tuple of (training_df, validation_df) with promotion features
    """
    try:
        logging.info("Creating promotion effectiveness features")
        
        for df in [train_df, val_df]:
            # Calculate ratios
            df["card_promo_ratio"] = df["CARDPROM"] / (df["NUMPROM"] + 1)
            df["promo_to_gift_ratio"] = df["NUMPROM"] / (df["NGIFTALL"] + 1)
            
            # Handle missing values
            df.fillna({
                "card_promo_ratio": 0,
                "promo_to_gift_ratio": 0
            }, inplace=True)
        
        logging.info("Promotion features created successfully")
        return train_df, val_df
        
    except Exception as e:
        logging.error(f"Error in promotion feature creation: {str(e)}")
        raise

def encode_categorical_features(train_df: pd.DataFrame, 
                              val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features using appropriate encoding methods.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        
    Returns:
        Tuple of (training_df, validation_df) with encoded features
    """
    try:
        logging.info("Starting categorical feature encoding")
        
        # Handle high cardinality features first (STATE and ZIP)
        high_cardinality = ["STATE", "ZIP"]
        
        for col in high_cardinality:
            if col in train_df.columns:
                # Create a mapping dictionary from training data
                unique_values = set(train_df[col].astype(str).unique())
                # Add an 'unknown' category
                unique_values.add('UNKNOWN')
                
                # Create mapping dictionary
                value_to_int = {val: idx for idx, val in enumerate(sorted(unique_values))}
                
                # Transform training data
                train_df[col] = train_df[col].astype(str).map(value_to_int)
                
                # Transform validation data, mapping unseen values to 'UNKNOWN'
                val_df[col] = val_df[col].astype(str).map(
                    lambda x: value_to_int.get(x, value_to_int['UNKNOWN'])
                )
                
                logging.info(f"Encoded {col} column with {len(value_to_int)} unique values")
        
        # Get remaining categorical columns for one-hot encoding
        categorical_columns = train_df.select_dtypes(exclude=['int64', 'float64']).columns
        low_cardinality = [col for col in categorical_columns 
                          if col not in ["TARGET_B", "TARGET_D"]]
        
        logging.info(f"Found {len(low_cardinality)} categorical features for one-hot encoding")
        
        # Apply one-hot encoding
        if low_cardinality:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            
            # Fit and transform training data
            train_ohe = pd.DataFrame(
                ohe.fit_transform(train_df[low_cardinality]),
                columns=ohe.get_feature_names_out(low_cardinality),
                index=train_df.index
            )
            
            # Transform validation data
            val_ohe = pd.DataFrame(
                ohe.transform(val_df[low_cardinality]),
                columns=ohe.get_feature_names_out(low_cardinality),
                index=val_df.index
            )
            
            # Combine with original dataframes
            train_df = train_df.drop(columns=low_cardinality).join(train_ohe)
            val_df = val_df.drop(columns=low_cardinality).join(val_ohe)
        
        # Verify all columns are numeric
        non_numeric_cols = train_df.select_dtypes(exclude=['int64', 'float64']).columns
        if len(non_numeric_cols) > 0:
            logging.warning(f"Non-numeric columns remaining: {non_numeric_cols}")
        
        logging.info(
            f"Categorical encoding completed. New shapes - "
            f"Train: {train_df.shape}, Val: {val_df.shape}"
        )
        return train_df, val_df
        
    except Exception as e:
        logging.error(f"Error in categorical feature encoding: {str(e)}")
        raise

def verify_feature_engineering(train_df: pd.DataFrame, 
                             val_df: pd.DataFrame) -> None:
    """
    Verify the quality of engineered features.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
    """
    try:
        # Check for missing values
        train_nulls = train_df.isnull().sum().sum()
        val_nulls = val_df.isnull().sum().sum()
        
        # Check for infinite values
        train_inf = np.isinf(train_df.select_dtypes(include=np.number)).sum().sum()
        val_inf = np.isinf(val_df.select_dtypes(include=np.number)).sum().sum()
        
        if train_nulls > 0 or val_nulls > 0:
            logging.warning(f"Missing values found - Train: {train_nulls}, Val: {val_nulls}")
        
        if train_inf > 0 or val_inf > 0:
            logging.warning(f"Infinite values found - Train: {train_inf}, Val: {val_inf}")
            
        logging.info("Feature engineering verification completed")
        
    except Exception as e:
        logging.error(f"Error in feature verification: {str(e)}")
        raise