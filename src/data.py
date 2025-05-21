"""
Data handling module for the GenAI-Superstream project.

This module provides functionality for loading and preprocessing the Iris dataset,
which is used for training and evaluating the machine learning model.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Responsible for loading the Iris dataset from scikit-learn.
    """
    
    def __init__(self):
        """
        Initialize the DataLoader.
        """
        self.dataset = None
        self.feature_names = None
        self.target_names = None
        self.data = None
        self.target = None
        
    def load_iris_dataset(self):
        """
        Load the Iris dataset from scikit-learn.
        
        Returns:
            tuple: (data, target, feature_names, target_names)
        """
        # Load the Iris dataset from scikit-learn
        self.dataset = load_iris()
        
        # Extract the data, target, feature names, and target names
        self.data = self.dataset.data
        self.target = self.dataset.target
        self.feature_names = self.dataset.feature_names
        self.target_names = self.dataset.target_names
        
        return self.data, self.target, self.feature_names, self.target_names
        
    def get_feature_names(self):
        """
        Get the names of the features in the dataset.
        
        Returns:
            list: List of feature names
        """
        if self.feature_names is None:
            self.load_iris_dataset()
        return self.feature_names
        
    def get_target_names(self):
        """
        Get the names of the target classes.
        
        Returns:
            list: List of target class names
        """
        if self.target_names is None:
            self.load_iris_dataset()
        return self.target_names
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None or self.target is None:
            self.load_iris_dataset()
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test


class DataPreprocessor:
    """
    Handles preprocessing of data before model training or prediction.
    """
    
    def __init__(self, scaling_strategy=None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            scaling_strategy (str, optional): Strategy for scaling features
                Options: 'standard', 'minmax', None
        """
        self.scaling_strategy = scaling_strategy
        self.scaler = None
        
        # Initialize scaler based on strategy
        if scaling_strategy == 'standard':
            self.scaler = StandardScaler()
        elif scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
        
    def preprocess(self, data, fit=False):
        """
        Preprocess the data according to the selected strategy.
        
        Args:
            data (numpy.ndarray): Data to preprocess
            fit (bool): Whether to fit the preprocessor on this data
            
        Returns:
            numpy.ndarray: Preprocessed data
        """
        # If no scaling strategy is selected, return the data unchanged
        if self.scaling_strategy is None or self.scaler is None:
            return data
            
        # If fit=True, fit the scaler to the data
        if fit:
            self.scaler.fit(data)
            
        # Transform the data using the selected scaler
        transformed_data = self.scaler.transform(data)
        
        return transformed_data
        
    def validate_feature_vector(self, feature_vector):
        """
        Validate that a feature vector has the correct format for the Iris dataset.
        
        Args:
            feature_vector (list): List of feature values
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check that the feature vector is a list or array
        if not isinstance(feature_vector, (list, tuple, np.ndarray)):
            return False
            
        # Check that it has exactly 4 elements (for the Iris dataset)
        if len(feature_vector) != 4:
            return False
            
        # Check that all elements are numeric (not just convertible to float)
        for x in feature_vector:
            # We want to check if the element is already a numeric type
            # not just if it can be converted to float
            if not isinstance(x, (int, float, np.number)):
                return False
            
        return True
        
    def format_features(self, features):
        """
        Format features for prediction.
        
        Args:
            features (list or str): Features as a list or comma-separated string
            
        Returns:
            list: Formatted feature list
        """
        # If features is a string, split by comma and convert to float
        if isinstance(features, str):
            try:
                features = [float(x.strip()) for x in features.split(',')]
            except ValueError:
                raise ValueError("Invalid feature format. Expected comma-separated floats.")
                
        # If features is already a list, convert elements to float
        elif isinstance(features, (list, tuple, np.ndarray)):
            try:
                features = [float(x) for x in features]
            except (ValueError, TypeError):
                raise ValueError("Invalid feature values. All values must be convertible to float.")
        else:
            raise TypeError(f"Unsupported feature type: {type(features)}")
            
        # Validate the formatted features
        if not self.validate_feature_vector(features):
            raise ValueError("Invalid feature vector. Must be a list of 4 float values.")
            
        return features


# Example usage if this module is run directly
if __name__ == "__main__":
    # Load the Iris dataset
    data_loader = DataLoader()
    data, target, feature_names, target_names = data_loader.load_iris_dataset()
    
    print(f"Dataset loaded with {data.shape[0]} samples and {data.shape[1]} features")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = data_loader.split_data(test_size=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Preprocess the data (optional)
    preprocessor = DataPreprocessor(scaling_strategy='standard')
    X_train_scaled = preprocessor.preprocess(X_train, fit=True)
    X_test_scaled = preprocessor.preprocess(X_test)
    
    print(f"Preprocessed training data shape: {X_train_scaled.shape}")
    
    # Validate a feature vector
    feature_vector = [5.1, 3.5, 1.4, 0.2]
    is_valid = preprocessor.validate_feature_vector(feature_vector)
    print(f"Feature vector {feature_vector} is valid: {is_valid}")