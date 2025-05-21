"""
Standalone API server for the GenAI-Superstream project.

This module implements a FastAPI server that directly exposes 
the prediction functionality without relying on Gradio's MCP system.
"""

import os
import uvicorn
import argparse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data import DataLoader
from src.model import ModelTrainer, Predictor
from src.server import predict_species
from src.utils import Logger

# Initialize logger
logger = Logger().get_logger()

# Create FastAPI app
app = FastAPI(
    title="Iris Species Predictor API",
    description="API for predicting Iris species based on feature measurements",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input and output models
class PredictionFeatures(BaseModel):
    features: List[float]

class PredictionInputMCP(BaseModel):
    input: Dict[str, List[float]]

class PredictionInputGradio(BaseModel):
    data: List[Any]

class PredictionResult(BaseModel):
    setosa: float = 0.0
    versicolor: float = 0.0
    virginica: float = 0.0
    error: Optional[str] = None

# Initialize predictor
def initialize_predictor():
    """Initialize the model predictor for prediction."""
    try:
        # If predictor is already attached, skip initialization
        if hasattr(predict_species, "_predictor") and predict_species._predictor is not None:
            logger.info("Predictor already initialized")
            return
            
        logger.info("Initializing predictor for API server")
        data_loader = DataLoader()
        _, _, _, target_names = data_loader.load_iris_dataset()
        
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = data_loader.split_data()
        model = trainer.train(X_train, y_train)
        
        predictor = Predictor(model, target_names)
        
        # Attach predictor to the prediction function
        predict_species._predictor = predictor
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise

# API routes for different input formats
@app.post("/api/predict", response_model=PredictionResult)
async def predict_api(prediction_data: PredictionInputGradio):
    """Handle prediction via the standard API endpoint."""
    try:
        features = None
        
        # Extract features from Gradio format
        if prediction_data.data:
            if isinstance(prediction_data.data[0], str):
                # Parse comma-separated values
                features = [float(x.strip()) for x in prediction_data.data[0].split(",") if x.strip()]
            elif isinstance(prediction_data.data[0], list):
                # Use array directly
                features = prediction_data.data[0]
        
        if not features or len(features) != 4:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid features format. Expected list of 4 values, got: {features}"
            )
        
        # Call prediction function
        result = predict_species(features)
        return result
    
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        return PredictionResult(error=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return PredictionResult(error=str(e))

@app.post("/api/predict_species", response_model=PredictionResult)
async def predict_species_direct(prediction_data: PredictionFeatures):
    """Handle prediction via direct features."""
    try:
        if not prediction_data.features or len(prediction_data.features) != 4:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid features format. Expected list of 4 values, got: {prediction_data.features}"
            )
        
        # Call prediction function
        result = predict_species(prediction_data.features)
        return result
    
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        return PredictionResult(error=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return PredictionResult(error=str(e))

@app.post("/mcp/api/v1/tools/predict_species", response_model=PredictionResult)
@app.post("/gradio_api/mcp/api/v1/tools/predict_species", response_model=PredictionResult)
@app.post("/api/mcp/tools/predict_species", response_model=PredictionResult)
async def predict_mcp_tool(prediction_data: PredictionFeatures):
    """Handle prediction via MCP tool endpoints."""
    return await predict_species_direct(prediction_data)

@app.post("/mcp/call", response_model=PredictionResult)
async def mcp_call(tool_call: dict):
    """Handle direct MCP call format."""
    try:
        if tool_call.get("tool") != "predict_species":
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown tool: {tool_call.get('tool')}"
            )
        
        input_data = tool_call.get("input", {})
        features = input_data.get("features")
        
        if not features or len(features) != 4:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid features format. Expected list of 4 values, got: {features}"
            )
        
        # Call prediction function
        result = predict_species(features)
        return result
    
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        return PredictionResult(error=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return PredictionResult(error=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

def start_server(host="0.0.0.0", port=8000):
    """Start the API server."""
    # Initialize the predictor
    initialize_predictor()
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Iris Predictor API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    start_server(args.host, args.port)