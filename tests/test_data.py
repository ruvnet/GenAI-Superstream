"""
Unit tests for the data module.
"""

import unittest
import numpy as np
from src.data import DataLoader, DataPreprocessor

class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
        
    def test_load_iris_dataset(self):
        """Test loading the Iris dataset."""
        data, target, feature_names, target_names = self.data_loader.load_iris_dataset()
        
        # Verify data dimensions
        self.assertEqual(data.shape[1], 4, "Data should have 4 features")
        self.assertEqual(data.shape[0], target.shape[0], "Data and target should have same number of samples")
        
        # Verify feature names
        self.assertEqual(len(feature_names), 4, "Should have 4 feature names")
        self.assertTrue(all(isinstance(name, str) for name in feature_names), 
                       "Feature names should be strings")
        
        # Verify target names
        self.assertEqual(len(target_names), 3, "Should have 3 target names")
        self.assertTrue(all(isinstance(name, str) for name in target_names), 
                       "Target names should be strings")
                       
    def test_get_feature_names(self):
        """Test getting feature names."""
        feature_names = self.data_loader.get_feature_names()
        self.assertEqual(len(feature_names), 4, "Should have 4 feature names")
        self.assertTrue(all(isinstance(name, str) for name in feature_names),
                       "Feature names should be strings")
                       
    def test_get_target_names(self):
        """Test getting target names."""
        target_names = self.data_loader.get_target_names()
        self.assertEqual(len(target_names), 3, "Should have 3 target names")
        self.assertTrue(all(isinstance(name, str) for name in target_names),
                       "Target names should be strings")
                       
    def test_split_data(self):
        """Test splitting data into train and test sets."""
        data, target, _, _ = self.data_loader.load_iris_dataset()
        X_train, X_test, y_train, y_test = self.data_loader.split_data(test_size=0.2)
        
        # Verify split proportions
        self.assertEqual(X_train.shape[0], int(data.shape[0] * 0.8), 
                        "Training set should be 80% of data")
        self.assertEqual(X_test.shape[0], int(data.shape[0] * 0.2), 
                        "Test set should be 20% of data")
        
        # Verify data shapes
        self.assertEqual(X_train.shape[1], 4, "Training data should have 4 features")
        self.assertEqual(X_test.shape[1], 4, "Test data should have 4 features")
        
        # Verify target shapes
        self.assertEqual(y_train.shape[0], X_train.shape[0],
                        "Training data and target should have same number of samples")
        self.assertEqual(y_test.shape[0], X_test.shape[0],
                        "Test data and target should have same number of samples")


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
        data, _, _, _ = self.data_loader.load_iris_dataset()
        self.data = data
        
    def test_preprocess_none(self):
        """Test preprocessing with no scaling."""
        preprocessor = DataPreprocessor(scaling_strategy=None)
        processed_data = preprocessor.preprocess(self.data)
        
        # Data should be unchanged
        np.testing.assert_array_equal(processed_data, self.data,
                                     "Data should be unchanged with no scaling")
                                     
    def test_preprocess_standard(self):
        """Test preprocessing with standard scaling."""
        preprocessor = DataPreprocessor(scaling_strategy='standard')
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Verify mean and std
        self.assertTrue(np.allclose(processed_data.mean(axis=0), 0, atol=1e-10),
                       "Standardized data should have mean close to 0")
        self.assertTrue(np.allclose(processed_data.std(axis=0), 1, atol=1e-10),
                       "Standardized data should have std close to 1")
                       
    def test_preprocess_minmax(self):
        """Test preprocessing with min-max scaling."""
        preprocessor = DataPreprocessor(scaling_strategy='minmax')
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Verify min and max
        self.assertTrue(np.allclose(processed_data.min(axis=0), 0, atol=1e-10),
                       "Min-max scaled data should have min close to 0")
        self.assertTrue(np.allclose(processed_data.max(axis=0), 1, atol=1e-10),
                       "Min-max scaled data should have max close to 1")
                       
    def test_validate_feature_vector(self):
        """Test feature vector validation."""
        preprocessor = DataPreprocessor()
        
        # Valid feature vector
        self.assertTrue(preprocessor.validate_feature_vector([5.1, 3.5, 1.4, 0.2]),
                       "Should accept valid feature vector")
        
        # Invalid feature vectors
        self.assertFalse(preprocessor.validate_feature_vector([5.1, 3.5, 1.4]),
                        "Should reject feature vector with wrong length")
        self.assertFalse(preprocessor.validate_feature_vector([5.1, 3.5, 1.4, "0.2"]),
                        "Should reject feature vector with non-numeric values")
        self.assertFalse(preprocessor.validate_feature_vector("not a list"),
                        "Should reject non-list inputs")
                        
    def test_format_features(self):
        """Test feature formatting."""
        preprocessor = DataPreprocessor()
        
        # Format list
        features = preprocessor.format_features([5.1, 3.5, 1.4, 0.2])
        self.assertEqual(features, [5.1, 3.5, 1.4, 0.2],
                        "Should return list unchanged")
        
        # Format string
        features = preprocessor.format_features("5.1, 3.5, 1.4, 0.2")
        self.assertEqual(features, [5.1, 3.5, 1.4, 0.2],
                        "Should convert string to list of floats")
        
        # Invalid inputs
        with self.assertRaises(ValueError):
            preprocessor.format_features("5.1, 3.5, 1.4")  # Too few values
        with self.assertRaises(ValueError):
            preprocessor.format_features("5.1, 3.5, 1.4, not_a_number")  # Non-numeric value


if __name__ == '__main__':
    unittest.main()