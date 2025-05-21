"""
Script to train and save the Iris species classifier model.

This script loads the Iris dataset, trains a classifier model,
evaluates its performance, and saves it to disk.
"""

import os
import argparse
from pathlib import Path

from src.data import DataLoader, DataPreprocessor
from src.model import ModelTrainer
from src.utils import Logger, PathManager, ConfigManager

def train_and_save_model(model_type="logistic_regression", 
                        model_params=None, 
                        scaling=None,
                        output_path=None,
                        test_size=0.2,
                        random_state=42):
    """
    Train and save an Iris species classifier model.
    
    Args:
        model_type (str): Type of model to train
        model_params (dict): Parameters for the model
        scaling (str): Scaling strategy (None, 'standard', or 'minmax')
        output_path (str): Path to save the trained model
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Evaluation metrics for the trained model
    """
    # Set up logging
    logger = Logger().get_logger()
    logger.info(f"Training {model_type} model with parameters: {model_params}")
    
    # Load data
    data_loader = DataLoader()
    data, target, feature_names, target_names = data_loader.load_iris_dataset()
    X_train, X_test, y_train, y_test = data_loader.split_data(
        test_size=test_size, random_state=random_state
    )
    
    # Preprocess data if needed
    if scaling:
        logger.info(f"Applying {scaling} scaling to data")
        preprocessor = DataPreprocessor(scaling_strategy=scaling)
        X_train = preprocessor.preprocess(X_train, fit=True)
        X_test = preprocessor.preprocess(X_test)
    
    # Train model
    trainer = ModelTrainer(model_type=model_type, model_params=model_params or {})
    model = trainer.train(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    logger.info(f"Model evaluation metrics: {metrics}")
    
    # Save model if output path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trainer.save_model(output_path)
        logger.info(f"Model saved to {output_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train and save an Iris species classifier model')
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                       choices=['logistic_regression', 'decision_tree', 'random_forest'],
                       help='Type of model to train')
    parser.add_argument('--scaling', type=str, choices=[None, 'standard', 'minmax'],
                       help='Scaling strategy for data preprocessing')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str,
                       help='Path to save the trained model')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    model_params = {}
    if args.config:
        config_manager = ConfigManager(args.config)
        model_config = config_manager.get_config('model')
        model_type = model_config.get('type', args.model_type)
        model_params = model_config.get('params', {})
    else:
        model_type = args.model_type
    
    # Determine output path
    output_path = args.output
    if not output_path:
        models_dir = PathManager.get_model_path()
        os.makedirs(models_dir, exist_ok=True)
        output_path = os.path.join(models_dir, f"iris_{model_type}.pkl")
    
    # Train and save model
    metrics = train_and_save_model(
        model_type=model_type,
        model_params=model_params,
        scaling=args.scaling,
        output_path=output_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Print results
    print("\nModel Training Results:")
    print(f"Model Type: {model_type}")
    print(f"Parameters: {model_params}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main()