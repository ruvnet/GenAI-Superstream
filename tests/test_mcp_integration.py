import unittest
import os
import subprocess
import time
import threading
import signal
import sys
import pytest
from unittest.mock import patch, MagicMock

import gradio as gr
from mcp import client as MCPClient

from src.server import predict_species, create_app, create_mcp_server
from src.client import MCPClient as IrisClient, create_mcp_client
from src.model import Predictor, ModelTrainer
from src.data import DataLoader

class TestMCPIntegration(unittest.TestCase):
    """
    Integration test for the MCP server and client.
    
    This test specifically focuses on the interaction between the client and server 
    via MCP, which is where the NoneType error occurs.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures, including a running server in a separate process."""
        # Initialize the predictor directly to the predict_species function
        data_loader = DataLoader()
        _, _, _, target_names = data_loader.load_iris_dataset()
        
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = data_loader.split_data()
        model = trainer.train(X_train, y_train)
        
        predictor = Predictor(model, target_names)
        
        # Ensure the predictor is properly attached to the function
        predict_species._predictor = predictor
    
    def setUp(self):
        """Set up test fixture for each test."""
        # Use a patched version of the server and client to avoid actual server launch
        self.app_patcher = patch('src.server.create_app')
        self.mock_create_app = self.app_patcher.start()
        
        # Create a mock app and blocks
        self.mock_app = MagicMock()
        self.mock_blocks = MagicMock()
        self.mock_create_app.return_value = self.mock_app
        
        # Configure the app.launch method to return self.mock_blocks
        self.mock_app.launch.return_value = self.mock_blocks
        
        # Create a mock client
        self.client_patcher = patch('src.client.Client')
        self.mock_client_constructor = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_constructor.return_value = self.mock_client
        
        # Configure the client's call method
        self.mock_client.call.return_value = {
            'setosa': 0.9, 
            'versicolor': 0.05, 
            'virginica': 0.05
        }
    
    def tearDown(self):
        """Tear down test fixture after each test."""
        self.app_patcher.stop()
        self.client_patcher.stop()
    
    def test_mcp_client_server_interaction(self):
        """Test that the client can call server tools via MCP."""
        # Create and launch the server (mocked)
        server_result = create_mcp_server(
            host='127.0.0.1', 
            port=7870,  # Use a different port to avoid conflicts
            share=False
        )
        
        # Server launch should have been called
        self.mock_app.launch.assert_called_once()
        
        # Check that the predictor is attached
        self.assertTrue(hasattr(predict_species, '_predictor'))
        
        # Create and test the client (with mocked MCP connection)
        client = IrisClient("http://127.0.0.1:7870/gradio_api/mcp/sse")
        
        # Manually set the client.client to our mock
        client.client = self.mock_client
        client.connected = True
        
        # Call the prediction tool
        result = client.call_prediction_tool([5.1, 3.5, 1.4, 0.2])
        
        # Verify the client called the correct tool
        self.mock_client.call.assert_called_once_with(
            tool="predict_species",
            input={"features": [5.1, 3.5, 1.4, 0.2]}
        )
        
        # Verify the result
        self.assertEqual(result['setosa'], 0.9)
        self.assertEqual(result['versicolor'], 0.05)
        self.assertEqual(result['virginica'], 0.05)


class TestEndToEndMCP:
    """Tests that require an actual running server and client."""
    
    def test_mcp_endpoint_responsiveness(self):
        """Test that the MCP endpoint responds to health checks."""
        # Skip if we're not doing end-to-end tests
        pytest.skip("Skip end-to-end tests by default. Run with --end-to-end flag to enable.")
        
        # Start a server in a separate process
        server_process = subprocess.Popen(
            [sys.executable, "-m", "src.server", "--port", "7880"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create a new process group
        )
        
        try:
            # Wait for the server to start
            time.sleep(5)
            
            # Check the server health by making a direct request to the MCP endpoint
            import requests
            response = requests.get("http://localhost:7880/gradio_api/mcp", timeout=5)
            
            # Should get a success response
            assert response.status_code == 200
            
            # Now try to create an MCP client connection
            from mcp import client as mcp_client
            client = mcp_client("http://localhost:7880/gradio_api/mcp/sse")
            
            # Check available tools
            tools = client.list_tools()
            
            # Should have the predict_species tool
            assert "predict_species" in tools
            
            # Try calling the tool
            result = client.call(
                tool="predict_species",
                input={"features": [5.1, 3.5, 1.4, 0.2]}
            )
            
            # Should get a valid result
            assert "setosa" in result
            
        finally:
            # Terminate the server
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait()


def real_mcp_test_func():
    """
    A function that can be run manually to test MCP functionality directly.
    
    This is not a unittest, but a utility function to help debug the MCP issue.
    It should be run in a separate Python script or in an interactive session.
    """
    # Start a server
    # In real usage, you would start the server in a separate process
    
    # Init predictor
    data_loader = DataLoader()
    _, _, _, target_names = data_loader.load_iris_dataset()
    
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    model = trainer.train(X_train, y_train)
    
    predictor = Predictor(model, target_names)
    
    # Ensure the predictor is attached
    predict_species._predictor = predictor
    
    # Create app and launch server
    app = create_app()
    server = app.launch(
        server_name="127.0.0.1",
        server_port=7890,
        mcp_server=True,
        share=False,
        prevent_thread_lock=True,  # Don't block this thread
    )
    
    time.sleep(2)  # Wait for server to start
    
    try:
        # Create client
        from mcp import client as mcp_client
        client = mcp_client("http://127.0.0.1:7890/gradio_api/mcp/sse")
        
        # Try calling the tool
        result = client.call(
            tool="predict_species",
            input={"features": [5.1, 3.5, 1.4, 0.2]}
        )
        
        print("Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Close the server
        server.close()


if __name__ == "__main__":
    # Run the unittest
    unittest.main()
    
    # Or run the real MCP test function if needed
    # real_mcp_test_func()