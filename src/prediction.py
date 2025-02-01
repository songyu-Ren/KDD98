"""
Module for making predictions using trained models and implementing mailing strategies.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import joblib
import logging
from datetime import datetime
from config.config import (
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
    MAILING_COST_EXPENSIVE,
    MAILING_COST_CHEAP
)

# Add report directory to config
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

class DonationPredictor:
    """Class for making predictions and implementing mailing strategies."""
    
    def __init__(self):
        """Initialize the predictor by loading trained models."""
        try:
            self.classifier = joblib.load(CLASSIFIER_MODEL_PATH)
            self.regressor = joblib.load(REGRESSOR_MODEL_PATH)
            logging.info("Models loaded successfully")
        except FileNotFoundError as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def predict_donation_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of donation."""
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict_donation_amount(self, X: pd.DataFrame) -> np.ndarray:
        """Predict donation amount."""
        return self.regressor.predict(X)
    
    def calculate_expected_donation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected donation by combining probability and amount predictions."""
        predictions = pd.DataFrame()
        predictions['probability'] = self.predict_donation_probability(X)
        predictions['predicted_amount'] = self.predict_donation_amount(X)
        predictions['expected_donation'] = (
            predictions['probability'] * predictions['predicted_amount']
        )
        return predictions
    
    def determine_mailing_strategy(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Determine optimal mailing strategy based on expected donations and costs."""
        predictions['net_gain_expensive'] = (
            predictions['expected_donation'] - MAILING_COST_EXPENSIVE
        )
        predictions['net_gain_cheap'] = (
            predictions['expected_donation'] - MAILING_COST_CHEAP
        )
        
        predictions['mailing_strategy'] = 'No Mail'
        predictions.loc[
            predictions['net_gain_expensive'] > 0, 'mailing_strategy'
        ] = 'Expensive Mail'
        predictions.loc[
            (predictions['net_gain_cheap'] > 0) & 
            (predictions['net_gain_expensive'] <= 0),
            'mailing_strategy'
        ] = 'Cheap Mail'
        
        return predictions
    
    def calculate_strategy_metrics(self, predictions: pd.DataFrame) -> Dict:
        """Calculate metrics for each mailing strategy."""
        metrics = {}
        
        for strategy in ['Expensive Mail', 'Cheap Mail', 'No Mail']:
            strategy_data = predictions[predictions['mailing_strategy'] == strategy]
            
            metrics[strategy] = {
                'count': len(strategy_data),
                'avg_expected_donation': strategy_data['expected_donation'].mean(),
                'total_expected_donation': strategy_data['expected_donation'].sum()
            }
            
            if strategy != 'No Mail':
                net_gain_col = (
                    'net_gain_expensive' 
                    if strategy == 'Expensive Mail' 
                    else 'net_gain_cheap'
                )
                metrics[strategy].update({
                    'avg_net_gain': strategy_data[net_gain_col].mean(),
                    'total_net_gain': strategy_data[net_gain_col].sum()
                })
        
        metrics['overall'] = {
            'total_net_revenue': sum(
                metrics[strategy].get('total_net_gain', 0) 
                for strategy in ['Expensive Mail', 'Cheap Mail']
            ),
            'total_mailings': sum(
                metrics[strategy]['count'] 
                for strategy in ['Expensive Mail', 'Cheap Mail']
            )
        }
        
        return metrics
    
    def make_predictions(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Make predictions and determine mailing strategy for new data."""
        try:
            predictions = self.calculate_expected_donation(X)
            predictions = self.determine_mailing_strategy(predictions)
            metrics = self.calculate_strategy_metrics(predictions)
            return predictions, metrics
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise

def format_strategy_report(metrics: Dict) -> str:
    """Format strategy metrics into a readable report."""
    report = ["Mailing Strategy Report", "=" * 50, ""]
    
    for strategy in ['Expensive Mail', 'Cheap Mail', 'No Mail']:
        report.append(f"{strategy}:")
        report.append(f"  Count: {metrics[strategy]['count']}")
        report.append(
            f"  Average Expected Donation: "
            f"${metrics[strategy]['avg_expected_donation']:.2f}"
        )
        
        if strategy != 'No Mail':
            report.append(
                f"  Average Net Gain: "
                f"${metrics[strategy]['avg_net_gain']:.2f}"
            )
            report.append(
                f"  Total Net Gain: "
                f"${metrics[strategy]['total_net_gain']:.2f}"
            )
        report.append("")
    
    report.append("Overall Summary:")
    report.append(
        f"  Total Net Revenue: "
        f"${metrics['overall']['total_net_revenue']:.2f}"
    )
    report.append(
        f"  Total Mailings: {metrics['overall']['total_mailings']}"
    )
    
    return "\n".join(report)

def save_prediction_results(predictions: pd.DataFrame, 
                          report: str, 
                          metrics: Dict) -> Tuple[str, str]:
    """Save predictions and report to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    predictions_path = os.path.join(REPORT_DIR, f'predictions_{timestamp}.csv')
    predictions.to_csv(predictions_path, index=False)
    logging.info(f"Predictions saved to: {predictions_path}")
    
    report_path = os.path.join(REPORT_DIR, f'mailing_strategy_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
        f.write("\n\nRaw Metrics:\n")
        f.write("=" * 50 + "\n")
        for strategy, strategy_metrics in metrics.items():
            f.write(f"\n{strategy}:\n")
            for metric, value in strategy_metrics.items():
                f.write(f"  {metric}: {value}\n")
    
    logging.info(f"Report saved to: {report_path}")
    
    json_path = os.path.join(REPORT_DIR, f'metrics_{timestamp}.json')
    pd.DataFrame(metrics).to_json(json_path)
    logging.info(f"Metrics saved to: {json_path}")
    
    return predictions_path, report_path

def predict_for_new_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, str, Dict]:
    """Make predictions for new data, generate and save report."""
    predictor = DonationPredictor()
    predictions, metrics = predictor.make_predictions(data)
    report = format_strategy_report(metrics)
    
    predictions_path, report_path = save_prediction_results(predictions, report, metrics)
    
    logging.info(f"""
    Prediction results saved:
    - Predictions: {predictions_path}
    - Report: {report_path}
    """)
    
    return predictions, report, metrics