"""
Unit tests for the model module.
"""

import unittest
import os
import tempfile
import numpy as np
from src.data import DataLoader
from src.model import ModelTrainer, Predictor, ModelFactory

class TestModelTrainer(unittest.TestCase):
    """Test cases for the ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
        data, target, _, _ = self.data_loader.load_iris_dataset()
        X_train, X_test, y_train, y_test = self.data_loader.split_data(test_size=0.2)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def test_create_model_logistic_regression(self):
        """Test creating a logistic regression model."""
        trainer = ModelTrainer(model_type="logistic_regression", model_params={"max_iter": 200})
        model = trainer.create_model()
        
        # Verify model type
        self.assertEqual(model.__class__.__name__, "LogisticRegression",
                       "Should create a LogisticRegression model")
        
        # Verify model parameters
        self.assertEqual(model.max_iter, 200,
                       "Model should have the specified parameters")
                       
    def test_create_model_decision_tree(self):
        """Test creating a decision tree model."""
        trainer = ModelTrainer(model_type="decision_tree", model_params={"max_depth": 5})
        model = trainer.create_model()
        
        # Verify model type
        self.assertEqual(model.__class__.__name__, "DecisionTreeClassifier",
                       "Should create a DecisionTreeClassifier model")
        
        # Verify model parameters
        self.assertEqual(model.max_depth, 5,
                       "Model should have the specified parameters")
                       
    def test_create_model_random_forest(self):
        """Test creating a random forest model."""
        trainer = ModelTrainer(model_type="random_forest", model_params={"n_estimators": 10})
        model = trainer.create_model()
        
        # Verify model type
        self.assertEqual(model.__class__.__name__, "RandomForestClassifier",
                       "Should create a RandomForestClassifier model")
        
        # Verify model parameters
        self.assertEqual(model.n_estimators, 10,
                       "Model should have the specified parameters")
                       
    def test_create_model_invalid(self):
        """Test creating an invalid model type."""
        trainer = ModelTrainer(model_type="invalid_model_type")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            trainer.create_model()
            
    def test_train(self):
        """Test training a model."""
        trainer = ModelTrainer(model_type="logistic_regression")
        model = trainer.train(self.X_train, self.y_train)
        
        # Verify model is trained
        self.assertTrue(trainer.is_trained, "Model should be marked as trained")
        self.assertIsNotNone(model, "Should return a trained model")
        
        # Verify model performance
        score = model.score(self.X_test, self.y_test)
        self.assertGreater(score, 0.8, "Model accuracy should be greater than 0.8")
        
    def test_evaluate(self):
        """Test evaluating a trained model."""
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(self.X_train, self.y_train)
        
        metrics = trainer.evaluate(self.X_test, self.y_test)
        
        # Verify metrics format
        self.assertIn("accuracy", metrics, "Metrics should include accuracy")
        self.assertIn("precision", metrics, "Metrics should include precision")
        self.assertIn("recall", metrics, "Metrics should include recall")
        self.assertIn("f1", metrics, "Metrics should include f1 score")
        
        # Verify metrics values
        self.assertGreater(metrics["accuracy"], 0.8, "Accuracy should be greater than 0.8")
        self.assertGreater(metrics["precision"], 0.8, "Precision should be greater than 0.8")
        self.assertGreater(metrics["recall"], 0.8, "Recall should be greater than 0.8")
        self.assertGreater(metrics["f1"], 0.8, "F1 score should be greater than 0.8")
        
    def test_evaluate_untrained(self):
        """Test evaluating an untrained model."""
        trainer = ModelTrainer(model_type="logistic_regression")
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            trainer.evaluate(self.X_test, self.y_test)
            
    def test_save_load_model(self):
        """Test saving and loading a model."""
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(self.X_train, self.y_train)
        
        # Create temporary file for the model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp:
            model_path = temp.name
            
        try:
            # Save the model
            trainer.save_model(model_path)
            self.assertTrue(os.path.exists(model_path), "Model file should exist")
            
            # Create a new trainer and load the model
            new_trainer = ModelTrainer()
            loaded_model = new_trainer.load_model(model_path)
            
            # Verify loaded model is trained
            self.assertTrue(new_trainer.is_trained, "Loaded model should be marked as trained")
            self.assertIsNotNone(loaded_model, "Should return a loaded model")
            
            # Verify loaded model performance
            score = loaded_model.score(self.X_test, self.y_test)
            self.assertGreater(score, 0.8, "Loaded model accuracy should be greater than 0.8")
            
        finally:
            # Clean up temporary file
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestPredictor(unittest.TestCase):
    """Test cases for the Predictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
        data, target, feature_names, target_names = self.data_loader.load_iris_dataset()
        X_train, X_test, y_train, y_test = self.data_loader.split_data(test_size=0.2)
        
        # Train a model
        trainer = ModelTrainer(model_type="logistic_regression")
        self.model = trainer.train(X_train, y_train)
        
        # Create predictor
        self.predictor = Predictor(self.model, target_names)
        
        # Example features
        self.setosa_features = [5.1, 3.5, 1.4, 0.2]  # Typically Setosa
        self.versicolor_features = [7.0, 3.2, 4.7, 1.4]  # Typically Versicolor
        self.virginica_features = [6.3, 3.3, 6.0, 2.5]  # Typically Virginica
        
    def test_predict_species(self):
        """Test predicting species probabilities."""
        prediction = self.predictor.predict_species(self.setosa_features)
        
        # Verify prediction format
        self.assertIsInstance(prediction, dict, "Prediction should be a dictionary")
        self.assertEqual(len(prediction), 3, "Should predict probabilities for 3 classes")
        
        # Verify probabilities
        self.assertAlmostEqual(sum(prediction.values()), 1.0, places=5, 
                             msg="Probabilities should sum to 1")
        
        # Verify highest probability for the correct class
        max_class = max(prediction, key=prediction.get)
        self.assertEqual(max_class, "setosa", "Should predict setosa with highest probability")
        
    def test_predict_class(self):
        """Test predicting the most likely class."""
        # Test for each class
        setosa_class = self.predictor.predict_class(self.setosa_features)
        self.assertEqual(setosa_class, "setosa", "Should predict setosa class")
        
        versicolor_class = self.predictor.predict_class(self.versicolor_features)
        self.assertEqual(versicolor_class, "versicolor", "Should predict versicolor class")
        
        virginica_class = self.predictor.predict_class(self.virginica_features)
        self.assertEqual(virginica_class, "virginica", "Should predict virginica class")
        
    def test_batch_predict(self):
        """Test batch prediction."""
        features = [self.setosa_features, self.versicolor_features, self.virginica_features]
        predictions = self.predictor.batch_predict(features)
        
        # Verify batch predictions
        self.assertEqual(len(predictions), 3, "Should return 3 predictions")
        
        # Check individual predictions
        max_classes = [max(pred, key=pred.get) for pred in predictions]
        self.assertEqual(max_classes, ["setosa", "versicolor", "virginica"],
                       "Should predict correct classes for each example")
        
    def test_invalid_features_wrong_length(self):
        """Test prediction with wrong number of features."""
        with self.assertRaises(ValueError):
            self.predictor.predict_species([5.1, 3.5, 1.4])
            
    def test_invalid_features_non_numeric(self):
        """Test prediction with non-numeric features."""
        with self.assertRaises(ValueError):
            self.predictor.predict_species([5.1, 3.5, 1.4, "0.2"])
            
    def test_invalid_features_non_list(self):
        """Test prediction with non-list input."""
        with self.assertRaises(ValueError):
            self.predictor.predict_species("not a list")



class TestModelFactory(unittest.TestCase):
    """Test cases for the ModelFactory class."""
    
    def test_create_model_logistic_regression(self):
        """Test creating a logistic regression model."""
        model = ModelFactory.create_model("logistic_regression", max_iter=200)
        
        # Verify model type
        self.assertEqual(model.__class__.__name__, "LogisticRegression",
                       "Should create a LogisticRegression model")
        
        # Verify model parameters
        self.assertEqual(model.max_iter, 200,
                       "Model should have the specified parameters")
                       
    def test_create_model_decision_tree(self):
        """Test creating a decision tree model."""
        model = ModelFactory.create_model("decision_tree", max_depth=5)
        
        # Verify model type
        self.assertEqual(model.__class__.__name__, "DecisionTreeClassifier",
                       "Should create a DecisionTreeClassifier model")
        
        # Verify model parameters
        self.assertEqual(model.max_depth, 5,
                       "Model should have the specified parameters")
                       
    def test_create_model_random_forest(self):
        """Test creating a random forest model."""
        model = ModelFactory.create_model("random_forest", n_estimators=10)
        
        # Verify model type
        self.assertEqual(model.__class__.__name__, "RandomForestClassifier",
                       "Should create a RandomForestClassifier model")
        
        # Verify model parameters
        self.assertEqual(model.n_estimators, 10,
                       "Model should have the specified parameters")
                       
    def test_create_model_invalid(self):
        """Test creating an invalid model type."""
        with self.assertRaises(ValueError):
            ModelFactory.create_model("invalid_model_type")


if __name__ == '__main__':
    unittest.main()