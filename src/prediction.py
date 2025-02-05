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
            self.selected_features = None  # Initialize selected_features as None
            logging.info("Models loaded successfully")
        except FileNotFoundError as e:
            logging.error(f"Error loading models: {e}")
            raise

    def set_selected_features(self, features: list) -> None:
        """Set the selected features for prediction."""
        self.selected_features = features
        logging.info(f"Set {len(features)} selected features for prediction")

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        if self.selected_features is not None:
            missing_features = set(self.selected_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            return X[self.selected_features]
        return X
    
    
    def predict_donation_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of donation."""
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict_donation_amount(self, X: pd.DataFrame) -> np.ndarray:
        """Predict donation amount with log transformation."""
        # Predict log-transformed values
        predicted_amounts_log = self.regressor.predict(X)
        # Convert back to original scale
        return np.expm1(predicted_amounts_log)
    
    def calculate_expected_donation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected donation by combining probability and amount predictions."""
        predictions = pd.DataFrame()
        predictions['probability'] = self.predict_donation_probability(X)
        predictions['predicted_amount'] = self.predict_donation_amount(X)
        predictions['expected_donation'] = (
            predictions['probability'] * predictions['predicted_amount']
        )
        return predictions
    
    def determine_mailing_strategy_improved(self, predictions: pd.DataFrame, budget: float = None) -> pd.DataFrame:
        """
        Determine an improved mailing strategy by:
        - Using a margin threshold to decide when expensive mail is justified.
        - Optionally enforcing a mailing budget by ranking customers.
        
        Parameters:
        predictions: DataFrame with expected_donation predictions.
        budget: Optional total dollar amount available for mailing.
        
        Returns:
        DataFrame with a new column 'mailing_strategy' set to:
            'Expensive Mail', 'Cheap Mail', or 'No Mail'.
        """
        # Calculate net gains for both mailing types
        predictions['net_gain_expensive'] = predictions['expected_donation'] - MAILING_COST_EXPENSIVE
        predictions['net_gain_cheap'] = predictions['expected_donation'] - MAILING_COST_CHEAP

        # Use a margin threshold to avoid marginal cases.
        margin_threshold = 5  # You can tune this threshold based on historical data
        
        # Choose preferred mailing method by comparing net gains.
        # If expensive mail provides a sufficiently higher net gain, choose it.
        predictions['preferred_strategy'] = np.where(
            predictions['net_gain_expensive'] > predictions['net_gain_cheap'] + margin_threshold,
            'Expensive Mail', 'Cheap Mail'
        )
        
        # If both mailing options have negative net gains, choose not to mail.
        max_net_gain = predictions[['net_gain_expensive', 'net_gain_cheap']].max(axis=1)
        predictions.loc[max_net_gain < margin_threshold, 'preferred_strategy'] = 'No Mail'
        
        # If a budget constraint is provided, select the best candidates until the budget is exhausted.
        if budget is not None:
            # Map mailing strategy to mailing cost
            cost_mapping = {
                'Expensive Mail': MAILING_COST_EXPENSIVE,
                'Cheap Mail': MAILING_COST_CHEAP,
                'No Mail': 0
            }
            predictions['mailing_cost'] = predictions['preferred_strategy'].map(cost_mapping)
            # Rank customers by the best net gain (you can use max(net_gain) as a proxy)
            predictions['max_net_gain'] = max_net_gain
            predictions.sort_values('max_net_gain', ascending=False, inplace=True)

            selected_indices = []
            remaining_budget = budget
            for idx, row in predictions.iterrows():
                cost = row['mailing_cost']
                # Only select if mailing incurs a cost and the budget allows it.
                if cost > 0 and remaining_budget >= cost:
                    selected_indices.append(idx)
                    remaining_budget -= cost
                else:
                    predictions.at[idx, 'preferred_strategy'] = 'No Mail'
                    predictions.at[idx, 'mailing_cost'] = 0

            # Reassign the mailing strategy based on budget-constrained selection.
            predictions['mailing_strategy'] = predictions['preferred_strategy']
        else:
            predictions['mailing_strategy'] = predictions['preferred_strategy']
        
        # Optionally, drop helper columns before returning.
        predictions.drop(columns=['preferred_strategy', 'mailing_cost', 'max_net_gain'], inplace=True, errors='ignore')
        
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