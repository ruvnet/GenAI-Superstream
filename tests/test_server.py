import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

import gradio as gr
import requests
import pytest

from src.server import predict_species, create_app, create_mcp_server, MCPServer
from src.model import Predictor, ModelTrainer
from src.data import DataLoader

class TestPredictSpeciesFunction(unittest.TestCase):
    """Test the predict_species function that is exposed as an MCP tool."""
    
    def setUp(self):
        # Create a mock predictor
        self.mock_predictor = MagicMock()
        self.mock_predictor.predict_species.return_value = {
            'setosa': 0.9, 
            'versicolor': 0.05, 
            'virginica': 0.05
        }
        
        # Attach the mock predictor to the function
        predict_species._predictor = self.mock_predictor
    
    def test_predict_species_calls_predictor(self):
        """Test that predict_species calls the attached predictor."""
        features = [5.1, 3.5, 1.4, 0.2]
        result = predict_species(features)
        
        # Verify the predictor was called with the right arguments
        self.mock_predictor.predict_species.assert_called_once_with(features)
        
        # Verify result is JSON serializable and contains expected keys
        self.assertIn('setosa', result)
        self.assertIn('versicolor', result)
        self.assertIn('virginica', result)
        
        # Verify result values are all floats
        for value in result.values():
            self.assertIsInstance(value, float)
    
    def test_predict_species_handles_invalid_input(self):
        """Test that predict_species properly handles invalid inputs."""
        # Test with non-list input
        result = predict_species("not a list")
        self.assertIn('error', result)
        
        # Test with wrong length
        result = predict_species([1, 2, 3])  # only 3 values
        self.assertIn('error', result)
    
    def test_predict_species_without_predictor(self):
        """Test behavior when predictor is not attached."""
        # Remove the predictor
        if hasattr(predict_species, '_predictor'):
            delattr(predict_species, '_predictor')
            
        # Attempt prediction should fail gracefully
        result = predict_species([5.1, 3.5, 1.4, 0.2])
        self.assertIn('error', result)
        
        # Restore the predictor for other tests
        predict_species._predictor = self.mock_predictor


class TestGradioApp(unittest.TestCase):
    """Test the Gradio app creation."""
    
    def setUp(self):
        # Create a mock predictor and attach it
        self.mock_predictor = MagicMock()
        self.mock_predictor.predict_species.return_value = {
            'setosa': 0.9, 
            'versicolor': 0.05, 
            'virginica': 0.05
        }
        predict_species._predictor = self.mock_predictor
    
    def test_create_app_returns_blocks(self):
        """Test that create_app returns a Gradio Blocks instance."""
        app = create_app()
        self.assertIsInstance(app, gr.Blocks)
        
        # Ensure the app has components
        self.assertTrue(hasattr(app, 'blocks'))
        self.assertTrue(len(app.blocks) > 0)


@pytest.mark.asyncio
async def test_mcp_server_with_client():
    """Test the MCP server with a mock client."""
    # Create a mock for Gradio's launch method
    with patch('gradio.Blocks.launch') as mock_launch:
        # Configure the mock
        mock_server = MagicMock()
        mock_launch.return_value = mock_server
        
        # Create predictor
        mock_predictor = MagicMock()
        mock_predictor.predict_species.return_value = {
            'setosa': 0.9, 
            'versicolor': 0.05, 
            'virginica': 0.05
        }
        
        # Create server
        server = MCPServer(mock_predictor, port=7861)  # Use different port
        
        # Try launching
        result = server.launch(prevent_thread_lock=True)
        
        # Check launch was called
        mock_launch.assert_called()
        
        # Attach predictor should have happened
        assert hasattr(predict_species, '_predictor')
        assert predict_species._predictor is mock_predictor


class TestMCPServerCreation(unittest.TestCase):
    """Test the create_mcp_server function."""
    
    @patch('src.server.create_app')
    @patch('gradio.Blocks.launch')
    @patch('src.model.ModelTrainer')
    @patch('src.data.DataLoader')
    def test_create_mcp_server(self, mock_data_loader, mock_trainer, 
                              mock_launch, mock_create_app):
        """Test that create_mcp_server sets up everything correctly."""
        # Configure mocks
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_model = MagicMock()
        mock_trainer_instance.train.return_value = mock_model
        
        mock_data_instance = MagicMock()
        mock_data_loader.return_value = mock_data_instance
        mock_data_instance.load_iris_dataset.return_value = (
            MagicMock(), MagicMock(), MagicMock(), ['setosa', 'versicolor', 'virginica']
        )
        
        # Call function
        result = create_mcp_server(host='0.0.0.0', port=7862)  # Different port
        
        # Verify app was created and launched
        mock_create_app.assert_called_once()
        mock_app.launch.assert_called_once()
        
        # Verify predictor was attached to the prediction function
        assert hasattr(predict_species, '_predictor')


if __name__ == '__main__':
    unittest.main()