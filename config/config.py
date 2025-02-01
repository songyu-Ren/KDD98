# config/config.py

"""Configuration settings for the modeling pipeline."""
import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
TRAIN_FILE = os.path.join(RAW_DATA_DIR, 'cup98LRN.txt')
VAL_FILE = os.path.join(RAW_DATA_DIR, 'cup98VAL.txt')
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')
PROCESSED_VAL_FILE = os.path.join(PROCESSED_DATA_DIR, 'val_processed.csv')
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, 'classifier.pkl')
REGRESSOR_MODEL_PATH = os.path.join(MODELS_DIR, 'regressor.pkl')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Data preprocessing
MISSING_THRESHOLD = 60

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 1e-3,
    'random_state': RANDOM_STATE,
}

# Mailing strategy parameters
MAILING_COST_EXPENSIVE = 5
MAILING_COST_CHEAP = 1