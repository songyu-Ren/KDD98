# Donor Prediction Model Pipeline

## Project Overview
This project implements a machine learning pipeline for predicting donor behavior and optimizing mailing strategies. The system uses a two-stage model approach: classification to predict donation likelihood and regression to predict donation amounts.

## Project Structure
```
.
├── config/
│   └── config.py         # Configuration parameters and paths
├── data/
│   ├── models/          # Saved model files
│   ├── processed/       # Processed dataset files
│   └── raw/            # Raw input data files
├── logs/                # Log files directory
├── results/             # Model predictions and analysis results
├── src/
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── feature_selection.py     # Feature importance and selection
│   ├── model_training.py        # Model training and evaluation
│   ├── prediction.py            # Making predictions and strategy
│   └── utils.py                 # Utility functions
├── main.py              # Main execution script
└── requirements.txt     # Project dependencies
```

## Key Features
- Data preprocessing and feature engineering
- Two-stage feature selection (Random Forest + RFECV)
- Classification model for donation probability
- Regression model for donation amount
- Mailing strategy optimization
- Comprehensive logging and result tracking

## Requirements
- Python 3.11.5
- See `requirements.txt` for package dependencies

## Getting Started
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Prepare your data in the `data/raw` directory

3. Run the pipeline:
```bash
python main.py
```

## Pipeline Steps
1. Data preprocessing and cleaning
2. Feature engineering
3. Feature selection using Random Forest importance and RFECV
4. Model training (classification and regression)
5. Prediction and strategy optimization
6. Results analysis and reporting

## Output
The pipeline generates:
- Selected feature sets
- Trained models
- Prediction results
- Strategy recommendations
- Performance metrics
- Detailed logs

## Environment
- Python Version: 3.11.5
- Dependencies: See `requirements.txt` for complete list of package dependencies
- Platform: Compatible with Windows, macOS, and Linux

## Notes
- All paths and parameters can be configured in `config.py`
- Logs are stored in the `logs` directory
- Model artifacts are saved in `data/models`
- Results are saved in the `results` directory