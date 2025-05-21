"""
Model module for the GenAI-Superstream project.

This module provides functionality for training machine learning models
on the Iris dataset and making predictions.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import Logger

logger = Logger().get_logger()

class ModelTrainer:
    """
    Responsible for training the scikit-learn classifier on the Iris dataset.
    """
    
    def __init__(self, model_type="logistic_regression", model_params=None):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type (str): Type of model to train ('logistic_regression' by default)
            model_params (dict): Parameters for the model initialization
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.is_trained = False
        
    def create_model(self):
        """
        Create a new instance of the specified model type.
        
        Returns:
            object: A scikit-learn model instance
        """
        # Create the model based on the model_type
        if self.model_type == "logistic_regression":
            return LogisticRegression(**self.model_params)
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(**self.model_params)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def train(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (numpy.ndarray): Training features
            y (numpy.ndarray): Training target values
            
        Returns:
            object: The trained model
        """
        # If model is None, create a new model instance
        if self.model is None:
            self.model = self.create_model()
            
        # Log training start
        logger.info(f"Training {self.model_type} model on {X.shape[0]} samples")
        
        # Fit the model to the training data
        self.model.fit(X, y)
        
        # Set is_trained flag to True
        self.is_trained = True
        
        # Log training completion
        logger.info(f"Model training completed")
        
        return self.model
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Check if the model is trained
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Make predictions on X_test
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log evaluation results
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return metrics
        
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        # Check if the model is trained
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            object: The loaded model
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load the model
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
            
        # Set is_trained flag to True
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return self.model


class Predictor:
    """
    Provides prediction functionality using a trained model.
    """
    
    def __init__(self, model, target_names):
        """
        Initialize the Predictor.
        
        Args:
            model: Trained scikit-learn model
            target_names (list): Names of the target classes
        """
        self.model = model
        self.target_names = target_names
        
    def predict_species(self, features):
        """
        Predict the Iris species given a list of four features.
        
        Args:
            features (list): List of 4 float values [sepal_length, sepal_width, 
                             petal_length, petal_width]
                             
        Returns:
            dict: Dictionary mapping class names to probabilities
        """
        # Validate that features is a list of 4 float values
        if isinstance(features, str):
            raise ValueError("Features must be a list, not a string")
        if not isinstance(features, (list, tuple, np.ndarray)):
            raise ValueError("Features must be a list of values")
        if len(features) != 4:
            raise ValueError("Features must contain exactly 4 values")
            
        # Check each element individually - must be numeric types (not strings)
        for i, f in enumerate(features):
            if not isinstance(f, (int, float, np.number)):
                raise ValueError(f"Feature {i+1} value '{f}' must be a numeric type (not {type(f).__name__})")
            
        # Convert all to float
        features = [float(f) for f in features]
            
        # Reshape features for prediction (model expects 2D array)
        X = np.array(features).reshape(1, -1)
        
        # Get probability predictions from the model
        probabilities = self.model.predict_proba(X)[0]
        
        # Map class indices to class names and create dictionary
        # Ensure all keys are strings (JSON requirement)
        result = {str(name): float(prob) for name, prob in zip(self.target_names, probabilities)}
        
        return result
        
    def predict_class(self, features):
        """
        Predict the most likely Iris species class.
        
        Args:
            features (list): List of 4 float values
            
        Returns:
            str: Predicted class name
        """
        # Validate that features is a list of 4 float values
        if isinstance(features, str):
            raise ValueError("Features must be a list, not a string")
        if not isinstance(features, (list, tuple, np.ndarray)):
            raise ValueError("Features must be a list of values")
        if len(features) != 4:
            raise ValueError("Features must contain exactly 4 values")
            
        # Check each element individually - must be numeric types (not strings)
        for i, f in enumerate(features):
            if not isinstance(f, (int, float, np.number)):
                raise ValueError(f"Feature {i+1} value '{f}' must be a numeric type (not {type(f).__name__})")
            
        # Convert all to float
        features = [float(f) for f in features]
            
        # Reshape features for prediction
        X = np.array(features).reshape(1, -1)
        
        # Get class prediction from the model
        class_idx = self.model.predict(X)[0]
        
        # Map class index to class name
        class_name = self.target_names[class_idx]
        
        return class_name
        
    def batch_predict(self, feature_batch):
        """
        Make predictions for a batch of feature sets.
        
        Args:
            feature_batch (list): List of feature lists
            
        Returns:
            list: List of prediction dictionaries
        """
        # Validate each feature list in the batch
        if not isinstance(feature_batch, (list, tuple, np.ndarray)):
            raise ValueError("Feature batch must be a list of feature lists")
            
        results = []
        
        # Make predictions for each feature set
        for features in feature_batch:
            try:
                prediction = self.predict_species(features)
                results.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting for features {features}: {e}")
                results.append({"error": str(e)})
                
        return results


class ModelFactory:
    """
    Factory class for creating different types of models.
    """
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a new model instance of the specified type.
        
        Args:
            model_type (str): Type of model to create
            **kwargs: Additional parameters for model initialization
            
        Returns:
            object: A scikit-learn model instance
        """
        if model_type == "logistic_regression":
            return LogisticRegression(**kwargs)
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(**kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# Example usage if this module is run directly
if __name__ == "__main__":
    from src.data import DataLoader
    
    # Load data
    data_loader = DataLoader()
    data, target, feature_names, target_names = data_loader.load_iris_dataset()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    # Train model
    trainer = ModelTrainer(model_type="logistic_regression", 
                         model_params={"max_iter": 200})
    model = trainer.train(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    
    # Create predictor
    predictor = Predictor(model, target_names)
    
    # Make prediction
    features = [5.1, 3.5, 1.4, 0.2]  # Example Iris setosa features
    prediction = predictor.predict_species(features)
    print(f"Prediction: {prediction}")
    
    # Get class prediction
    class_name = predictor.predict_class(features)
    print(f"Predicted class: {class_name}")