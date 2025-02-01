"""Feature engineering module for data preparation."""

import pandas as pd
import numpy as np
from typing import Tuple

def create_temporal_features(train_df: pd.DataFrame, 
                           val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create temporal features from ADATE columns."""
    # Find ADATE columns
    adate_cols = [col for col in train_df.columns if "ADATE_" in col]
    
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
    
    return train_df, val_df

def create_rfm_features(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create RFM (Recency, Frequency, Monetary) features."""
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
    
    return train_df, val_df

def create_promotion_features(train_df: pd.DataFrame, 
                            val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create promotion effectiveness features."""
    for df in [train_df, val_df]:
        df["card_promo_ratio"] = df["CARDPROM"] / (df["NUMPROM"] + 1)
        df["promo_to_gift_ratio"] = df["NUMPROM"] / (df["NGIFTALL"] + 1)
        df.fillna({
            "card_promo_ratio": 0,
            "promo_to_gift_ratio": 0
        }, inplace=True)
    
    return train_df, val_df

def encode_categorical_features(train_df: pd.DataFrame, 
                              val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features using appropriate encoding methods."""
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # High cardinality features for target encoding
    high_cardinality = ["STATE", "ZIP"]
    
    # One-hot encoding for low cardinality features
    categorical_columns = train_df.select_dtypes(exclude=['int64', 'float64']).columns
    low_cardinality = [col for col in categorical_columns 
                      if col not in high_cardinality + ["TARGET_B", "TARGET_D"]]
    
    # Apply one-hot encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
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
    
    return train_df, val_df