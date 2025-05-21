"""
Server module for the GenAI-Superstream project.

This module implements a Gradio-based MCP server that exposes the
Iris species prediction functionality.
"""

import os
import argparse
import gradio as gr
from typing import List, Dict, Any, Union

from src.data import DataLoader
from src.model import ModelTrainer, Predictor
from src.utils import Logger, ConfigManager, InputValidator

# Initialize logger
logger = Logger().get_logger()

# ---------------------- MCP Tool Definition ----------------------

# Import the connector for reliable MCP integration
from src.mcp_connector import predict_species_proxy

# Define the predict_species tool with proper type hints for MCP
def predict_species(features: list) -> dict:
    """
    Predicts Iris species given a list of four features:
    [sepal_length, sepal_width, petal_length, petal_width].
    Returns a dict mapping class names to probabilities.
    
    Args:
        features: A list of four numeric values representing Iris measurements
            [sepal_length, sepal_width, petal_length, petal_width]
    
    Returns:
        A dictionary mapping species names to probability scores
    """
    try:
        if not isinstance(features, list) or len(features) != 4:
            raise ValueError("Features must be a list of 4 numeric values.")
        
        # Try using the attached predictor if available
        if hasattr(predict_species, "_predictor") and predict_species._predictor is not None:
            try:
                # Use direct prediction
                predictor = predict_species._predictor
                result = predictor.predict_species(features)
                
                # Ensure JSON-serializable output
                logger.info(f"MCP prediction made for features: {features}")
                return {str(k): float(v) for k, v in result.items()}
            except Exception as e:
                logger.warning(f"Direct prediction failed: {e}, falling back to API connector")
                # Fall back to API connector if direct prediction fails
                return predict_species_proxy(features)
        else:
            # Use the API connector if no predictor is attached
            logger.warning("No predictor attached, using API connector")
            return predict_species_proxy(features)
    except Exception as e:
        logger.error(f"Error in MCP tool 'predict_species': {str(e)}")
        return {"error": str(e)}

# Explicitly set function attributes for better MCP recognition
predict_species.__annotations__ = {"features": list, "return": dict}

# ---------------------- Gradio App Factory ----------------------

def create_app() -> gr.Blocks:
    """
    Builds and returns a Gradio Blocks app exposing the MCP tool.
    
    This function creates a Gradio interface and ensures the MCP tool
    is properly registered and accessible.
    """
    # Create the Gradio Blocks interface
    with gr.Blocks() as demo:
        gr.Markdown("# Iris Species Predictor MCP Demo")
        gr.Markdown("Provide 4 comma-separated values: sepal_length, sepal_width, petal_length, petal_width.")

        with gr.Row():
            inp = gr.Textbox(label="Feature Values", placeholder="e.g., 5.1, 3.5, 1.4, 0.2")
            out = gr.JSON(label="Prediction Results")

        # Example cases
        examples = [
            ["5.1, 3.5, 1.4, 0.2"],  # setosa
            ["7.0, 3.2, 4.7, 1.4"],  # versicolor
            ["6.3, 3.3, 6.0, 2.5"]   # virginica
        ]
        gr.Examples(examples=examples, inputs=inp)

        # Bind submission to MCP tool
        def wrapper(text):
            try:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 comma-separated values, got {len(parts)}")
                features = [float(p) for p in parts]
                return predict_species(features)
            except Exception as e:
                logger.error(f"Error in UI prediction: {e}")
                return {"error": str(e)}

        inp.submit(wrapper, inp, out)
        
        # Explicitly register the predict_species function as an MCP tool
        try:
            # Approach 1: Use Tools class if available
            try:
                from gradio.mcp import Tools
                logger.info("Registering predict_species with Tools API in create_app")
                
                mcp_tools = Tools()
                mcp_tools.add(
                    function=predict_species,
                    name="predict_species",
                    description="Predicts Iris species based on 4 features"
                )
                
                # Add to the app
                demo.mcp_tools = mcp_tools
                logger.info("Successfully attached MCP tools to app")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not use Tools API in create_app: {e}")
                
                # Approach 2: Try direct register_tool if available
                try:
                    from gradio import mcp
                    if hasattr(mcp, "register_tool"):
                        logger.info("Using mcp.register_tool API in create_app")
                        mcp.register_tool(
                            fn=predict_species,
                            name="predict_species",
                            description="Predicts Iris species based on 4 features"
                        )
                        logger.info("Successfully registered predict_species with direct API")
                except Exception as e:
                    logger.warning(f"Could not use register_tool API in create_app: {e}")
        except Exception as e:
            logger.error(f"Error setting up MCP tool in create_app: {e}")
            
        # Make sure the predict_species function is directly accessible from the app
        demo.predict_species = predict_species

    return demo

# ---------------------- For Backward Compatibility ----------------------

class MCPServer:
    """
    Manages the MCP server configuration and launch.
    
    This class is maintained for backwards compatibility with existing code.
    """
    
    def __init__(self, predictor, host="0.0.0.0", port=7860):
        """
        Initialize the MCPServer.
        
        Args:
            predictor: Predictor instance for making predictions
            host (str): Host address to bind the server
            port (int): Port number to bind the server
        """
        self.predictor = predictor
        self.host = host
        self.port = port
        self.blocks = None
        self.server = None
        
        # Attach predictor to the tool function
        predict_species._predictor = predictor
        
        logger.info(f"MCPServer initialized with predictor on {host}:{port}")
        
    def setup(self):
        """
        Set up the MCP server.
        
        Returns:
            gradio.Blocks: The configured Blocks interface
        """
        # Build the app
        self.blocks = create_app()
        logger.info(f"MCP server setup completed on {self.host}:{self.port}")
        return self.blocks
        
    def launch(self, **kwargs):
        """
        Launch the MCP server.
        
        Args:
            **kwargs: Additional parameters for Gradio launch
            
        Returns:
            gradio.Blocks: The launched Blocks interface
        """
        # If blocks is not set up, call setup()
        if not self.blocks:
            self.setup()
        # Set default launch parameters that are likely to be supported across most Gradio versions
        launch_params = {
            "server_name": self.host,
            "server_port": self.port,
            "share": True,           # Enable sharing to make server accessible outside localhost
            "show_error": True,      # Show detailed errors
            "favicon_path": None     # Prevent favicon issues
        }
        
        # Get the supported parameters for the current Gradio version
        try:
            import inspect
            launch_signature = inspect.signature(self.blocks.launch)
            supported_params = set(launch_signature.parameters.keys())
            
            # Advanced parameters with fallbacks
            advanced_params = {
                # Parameter name -> (value, fallback handler function or None)
                "mcp_server": (True, None),  # Enable MCP Server
                "show_api": (True, None),    # Expose the API endpoints
                "root_path": ("", None),     # Important for path resolution
                "api_name": ("/api", self._handle_api_name_fallback)  # Explicit API path
            }
            
            # Add each parameter only if it's supported
            for param_name, (param_value, fallback_handler) in advanced_params.items():
                if param_name in supported_params:
                    launch_params[param_name] = param_value
                    logger.info(f"Using supported parameter '{param_name}'")
                else:
                    logger.warning(f"Parameter '{param_name}' is not supported by this version of Gradio, skipping")
                    # Execute fallback handler if one exists
                    if fallback_handler is not None:
                        fallback_handler(param_value)
                        
        except Exception as e:
            logger.warning(f"Could not check Gradio compatibility for parameters: {e}")
        
        # Override defaults with any provided kwargs, but filter out unsupported params
        filtered_kwargs = {}
        if kwargs:
            try:
                # Only include params that are in the signature
                for key, value in kwargs.items():
                    if key in supported_params:
                        filtered_kwargs[key] = value
                    else:
                        logger.warning(f"Ignoring unsupported parameter from kwargs: '{key}'")
            except NameError:
                # If supported_params wasn't defined due to exception, just use kwargs as is
                filtered_kwargs = kwargs
                
        launch_params.update(filtered_kwargs)
        
        logger.info(f"Launching MCP server with parameters: {launch_params}")
        
        # Launch the interface
        self.server = self.blocks.launch(**launch_params)
        
        logger.info(f"MCP server launched on port {self.port}")
        
        return self.blocks
        
    def _handle_api_name_fallback(self, api_name_value):
        """
        Handle fallback for unsupported api_name parameter.
        This method implements alternative approaches when api_name is not supported.
        
        Args:
            api_name_value (str): The value that would have been used for api_name
        """
        logger.info(f"Implementing fallback for unsupported 'api_name' parameter: {api_name_value}")
        # Fallback approaches could include:
        # 1. Setting environment variables that Gradio might check
        # 2. Adding custom routes after launch
        # 3. Adapting the server configuration in other ways
        
        try:
            # Set environment variable if Gradio checks for it (some versions do)
            os.environ["GRADIO_API_NAME"] = api_name_value
        except Exception as e:
            logger.warning(f"Failed to implement api_name fallback: {e}")
    
    def shutdown(self):
        """
        Shutdown the MCP server.
        """
        if self.server:
            logger.info("Shutting down MCP server")
            self.server.close()
            self.server = None

# ---------------------- Main Server Creation Function ----------------------

def create_mcp_server(model_path=None, host="0.0.0.0", port=7860, **kwargs):
    """
    Create and launch a complete MCP server using the Iris classifier.
    
    Args:
        model_path (str, optional): Path to a saved model file
        host (str): Host address to bind the server
        port (int): Port number to bind the server
        **kwargs: Additional parameters to pass to the server launch
        
    Returns:
        gradio.Blocks: The launched Blocks interface
    """
    try:
        # Load the Iris dataset
        data_loader = DataLoader()
        data, target, feature_names, target_names = data_loader.load_iris_dataset()
        
        # Create model trainer
        trainer = ModelTrainer()
        
        # Either load an existing model or train a new one
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model = trainer.load_model(model_path)
        else:
            logger.info("Training new model")
            X_train, X_test, y_train, y_test = data_loader.split_data()
            model = trainer.train(X_train, y_train)
            
            # Log metrics for newly trained model
            metrics = trainer.evaluate(X_test, y_test)
            logger.info(f"Model metrics: {metrics}")
        
        # Create predictor
        predictor = Predictor(model, target_names)
        
        # Attach predictor to the tool function directly
        predict_species._predictor = predictor
        
        # Create app using the factory function
        app = create_app()
        
        # Set launch parameters - ensure all required parameters for MCP are set
        # Set launch parameters - ensure all required parameters for MCP are set
        # Only include parameters that are supported by the current version of Gradio
        launch_params = {
            "server_name": host,
            "server_port": port,
            "share": kwargs.get('share', True),
            "show_error": True,     # Show detailed errors
            "favicon_path": None    # Prevent favicon issues
        }
        
        # Check which parameters are supported by the current Gradio version
        import inspect
        launch_signature = inspect.signature(app.launch)
        supported_params = set(launch_signature.parameters.keys())
        
        # Define fallback handler for api_name parameter
        def _handle_api_name_fallback(api_name_value):
            """Implement fallbacks for the api_name parameter"""
            logger.info(f"Implementing fallback for unsupported 'api_name' parameter: {api_name_value}")
            
            try:
                # Set environment variable (some Gradio versions check this)
                os.environ["GRADIO_API_NAME"] = api_name_value
                
                # The custom API routes we added earlier serve as another fallback
                logger.info("Using custom API routes as fallback for api_name")
            except Exception as e:
                logger.warning(f"Failed to implement api_name fallback: {e}")
                
        # Advanced parameters with fallback strategies
        advanced_params = {
            # Parameter name -> (value, fallback function or None)
            "mcp_server": (True, None),      # Enable MCP server
            "show_api": (True, None),        # Expose the API endpoints
            "root_path": ("", None),         # Important for path resolution
            "api_name": ("/api", _handle_api_name_fallback)  # Explicit API path
        }
        
        # Add each parameter only if it's supported
        for param_name, (param_value, fallback_handler) in advanced_params.items():
            if param_name in supported_params:
                launch_params[param_name] = param_value
                logger.info(f"Using supported parameter '{param_name}'")
            else:
                logger.warning(f"Parameter '{param_name}' is not supported by this version of Gradio, skipping")
                # Execute fallback handler if one exists
                if fallback_handler is not None:
                    fallback_handler(param_value)
        # MCP-specific configuration to expose the predict_species function
        # Create a single, dedicated MCP tools registry
        mcp_tools = None
        tool_registered = False
        
        # Ensure the predict_species function has the predictor
        if not hasattr(predict_species, "_predictor"):
            logger.info("Attaching predictor to predict_species function")
            predict_species._predictor = predictor
        
        # Approach 1: Use Tools class (newer Gradio versions)
        try:
            from gradio.mcp import Tools
            logger.info("Using gradio.mcp.Tools API")
            
            # Create a single MCP tools collection
            mcp_tools = Tools()
            
            # Register predict_species function with proper input schema
            mcp_tools.add(
                function=predict_species,
                name="predict_species",
                description="Predicts Iris species based on 4 features",
                input_schema={
                    "features": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    }
                },
                output_schema={"type": "object"}
            )
            
            # Add tools to launch params
            launch_params["tools"] = mcp_tools
            tool_registered = True
            logger.info("Successfully registered predict_species with Tools API")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not use Tools API: {e}")
        
        # Approach 2: Try direct register_tool if available and Tools API failed
        if not tool_registered:
            try:
                from gradio import mcp
                if hasattr(mcp, "register_tool"):
                    logger.info("Using mcp.register_tool API")
                    mcp.register_tool(
                        fn=predict_species,
                        name="predict_species",
                        description="Predicts Iris species based on 4 features"
                    )
                    tool_registered = True
                    logger.info("Successfully registered predict_species with register_tool API")
            except Exception as e:
                logger.error(f"Could not register tool with register_tool API: {e}")
        
        # Approach 3: Set custom MCP handler for last-resort compatibility
        if not tool_registered:
            logger.info("Using custom MCP handler approach")
            # Ensure API endpoints are exposed - only if supported
            api_name_value = "/api"
            
            # Only add api_name if it's supported
            if "api_name" in supported_params:
                launch_params["api_name"] = api_name_value
                logger.info("Adding api_name parameter - it's supported by this Gradio version")
            else:
                # Call the fallback handler if api_name isn't supported
                logger.info("api_name is not supported, using fallback")
                _handle_api_name_fallback(api_name_value)
                
            # Add root_path only if supported
            if "root_path" in supported_params:
                launch_params["root_path"] = ""
            
            # Define a custom handler function that will be attached to the app
            def mcp_handler(request_data):
                if request_data.get("tool") == "predict_species":
                    features = request_data.get("input", {}).get("features", [])
                    return predict_species(features)
                return {"error": "Unknown tool"}
            
            # Attach handler to app
            app.mcp_handler = mcp_handler
            
        # Add exception handling for the MCP setup as a whole
        try:
            # Log the overall MCP tool status
            if tool_registered:
                logger.info("MCP tool registration successful")
            else:
                logger.warning("Standard MCP tool registration methods failed, falling back to manual handler")
                
            # Create a global-level route handler for direct API access as a last resort
            # Add multiple route decorators to support all the endpoints the MCP debug script attempts
            @app.routes.get("/api/mcp/tools/predict_species")
            @app.routes.post("/api/mcp/tools/predict_species")
            @app.routes.post("/gradio_api/mcp/api/v1/tools/predict_species")
            @app.routes.post("/mcp/api/v1/tools/predict_species")
            @app.routes.post("/api/predict")
            @app.routes.post("/run/predict")
            async def mcp_direct_api(request):
                try:
                    import json
                    import inspect
                    from starlette.responses import JSONResponse
                    
                    # Parse the request body if it's a POST
                    body_data = {}
                    if request.method == "POST":
                        body = await request.body()
                        if body:
                            try:
                                body_data = json.loads(body)
                            except:
                                logger.error(f"Failed to parse JSON body: {body}")
                                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
                    
                    # Extract features from different request formats
                    features = None
                    
                    # Format 1: Direct features list
                    if "features" in body_data:
                        features = body_data["features"]
                    
                    # Format 2: MCP format - input.features
                    elif "input" in body_data and isinstance(body_data["input"], dict) and "features" in body_data["input"]:
                        features = body_data["input"]["features"]
                    
                    # Format 3: Gradio format - data array
                    elif "data" in body_data:
                        if isinstance(body_data["data"], list) and len(body_data["data"]) > 0:
                            if isinstance(body_data["data"][0], str):
                                # Handle string input like "5.1, 3.5, 1.4, 0.2"
                                features = [float(x.strip()) for x in body_data["data"][0].split(",") if x.strip()]
                            elif isinstance(body_data["data"][0], list):
                                # Handle array input like [[5.1, 3.5, 1.4, 0.2]]
                                features = body_data["data"][0]
                    
                    # Format 4: GET query parameters
                    if features is None and request.method == "GET":
                        features_str = request.query_params.get("features", "")
                        if features_str:
                            features = [float(x.strip()) for x in features_str.split(",") if x.strip()]
                    
                    logger.info(f"Extracted features: {features}")
                    
                    if features is None or not isinstance(features, (list, tuple)) or len(features) != 4:
                        return JSONResponse(
                            {"error": f"Invalid features format. Expected list of 4 values, got: {features}"},
                            status_code=400
                        )
                    
                    # Ensure the predictor is attached to the function
                    if not hasattr(predict_species, "_predictor"):
                        logger.error("Predictor not attached to predict_species function")
                        return JSONResponse(
                            {"error": "Server configuration error: Predictor not attached"},
                            status_code=500
                        )
                    
                    # Call the prediction function
                    logger.info(f"Calling predict_species with features: {features}")
                    result = predict_species(features)
                    logger.info(f"Prediction result: {result}")
                    
                    # Return the result
                    return JSONResponse(result)
                except Exception as e:
                    logger.error(f"Error in direct API: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return JSONResponse({"error": str(e)}, status_code=500)
        except Exception as e:
            logger.error(f"Error setting up MCP direct handlers: {e}")
        
        # Filter kwargs to only include parameters supported by this Gradio version
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in supported_params:
                filtered_kwargs[key] = value
                logger.info(f"Using supported parameter from kwargs: '{key}'")
            else:
                logger.warning(f"Parameter '{key}' from kwargs is not supported by this version of Gradio, skipping")
                
        # Update with filtered parameters
        launch_params.update(filtered_kwargs)
        
        logger.info(f"Launching MCP server with parameters: {launch_params}")
        
        # Make sure that predict_species function is directly accessible
        # at the module level and also attached to app
        app.predict_species = predict_species
        
        # Direct FastAPI integration - this is a more reliable approach than using app.routes
        # Get the underlying FastAPI app from Gradio
        try:
            import inspect
            import fastapi
            import json
            from fastapi import FastAPI, Request
            from starlette.responses import JSONResponse
            
            # Detailed inspection and logging of Gradio app structure
            logger.info("Inspecting Gradio app structure to find FastAPI app")
            
            # Log app attributes
            app_attrs = [attr for attr in dir(app) if not attr.startswith('__')]
            logger.info(f"App attributes: {app_attrs}")
            
            # Try to find the FastAPI app
            fastapi_app = None
            possible_paths = [
                ('app',),
                ('server', 'app'),
                ('_server', 'app'),
                ('_blocks', 'app'),
                ('_blocks', '_app'),
                ('_blocks', 'server', 'app'),
            ]
            
            found_path = None
            for path in possible_paths:
                current = app
                valid_path = True
                for attr in path:
                    if hasattr(current, attr):
                        current = getattr(current, attr)
                    else:
                        valid_path = False
                        break
                
                if valid_path and isinstance(current, FastAPI):
                    fastapi_app = current
                    found_path = '.'.join(path)
                    break
            
            if fastapi_app and isinstance(fastapi_app, FastAPI):
                logger.info(f"✅ Successfully obtained FastAPI app from Gradio via path: {found_path}")
                
                # Get existing routes for debugging - safely
                existing_routes = []
                for route in fastapi_app.routes:
                    route_info = {"path": getattr(route, "path", "unknown")}
                    if hasattr(route, "methods"):
                        route_info["methods"] = route.methods
                    existing_routes.append(route_info)
                    
                logger.info(f"Existing routes: {json.dumps(existing_routes, default=str)}")
                
                # Create the prediction endpoint handler
                async def prediction_handler(request: Request):
                    try:
                        # Extract and parse the request body
                        body_data = {}
                        body = await request.body()
                        if body:
                            try:
                                import json
                                body_data = json.loads(body)
                            except:
                                logger.error(f"Failed to parse JSON: {body}")
                                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
                        
                        # Extract features from different formats
                        features = None
                        
                        # Format 1: Direct features list
                        if "features" in body_data:
                            features = body_data["features"]
                        
                        # Format 2: MCP format - input.features
                        elif "input" in body_data and isinstance(body_data["input"], dict) and "features" in body_data["input"]:
                            features = body_data["input"]["features"]
                        
                        # Format 3: Gradio format - data array
                        elif "data" in body_data:
                            if isinstance(body_data["data"], list) and len(body_data["data"]) > 0:
                                if isinstance(body_data["data"][0], str):
                                    # Handle string input like "5.1, 3.5, 1.4, 0.2"
                                    features = [float(x.strip()) for x in body_data["data"][0].split(",") if x.strip()]
                                elif isinstance(body_data["data"][0], list):
                                    # Handle array input like [[5.1, 3.5, 1.4, 0.2]]
                                    features = body_data["data"][0]
                        
                        # Log what we extracted
                        logger.info(f"FastAPI endpoint called with body: {body_data}")
                        logger.info(f"Extracted features: {features}")
                        
                        # Validate features
                        if features is None or not isinstance(features, (list, tuple)) or len(features) != 4:
                            logger.error(f"Invalid features format: {features}")
                            return JSONResponse(
                                {"error": f"Invalid features format. Expected list of 4 values, got: {features}"},
                                status_code=400
                            )
                        
                        # Ensure the predictor is attached
                        if not hasattr(predict_species, "_predictor"):
                            logger.error("Predictor not attached to predict_species function")
                            return JSONResponse(
                                {"error": "Server configuration error: Predictor not attached"},
                                status_code=500
                            )
                        
                        # Make the prediction
                        logger.info(f"Calling predict_species with features: {features}")
                        result = predict_species(features)
                        logger.info(f"Prediction result: {result}")
                        
                        return JSONResponse(result)
                    
                    except Exception as e:
                        logger.error(f"Error in FastAPI route: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        return JSONResponse({"error": str(e)}, status_code=500)
                
                # Register the route for multiple endpoints
                for endpoint in [
                    "/api/predict",
                    "/api/predict_species",
                    "/mcp/api/v1/tools/predict_species",
                    "/gradio_api/mcp/api/v1/tools/predict_species",
                    "/api/mcp/tools/predict_species",
                    "/run/predict"
                ]:
                    logger.info(f"Adding FastAPI route for {endpoint}")
                    # Add the route (supporting both GET and POST)
                    fastapi_app.add_api_route(
                        endpoint,
                        prediction_handler,
                        methods=["POST"],
                        tags=["prediction"]
                    )
                
                logger.info("Successfully added FastAPI routes")
            else:
                logger.warning("Could not access the FastAPI app from Gradio")
        
        except Exception as e:
            logger.error(f"Error setting up FastAPI routes: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Launch with modified parameters
        return app.launch(**launch_params)
        
    except Exception as e:
        logger.error(f"Error creating MCP server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

class ServerConfig:
    """
    Manages the configuration for the MCP server.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the ServerConfig.
        
        Args:
            config_file (str, optional): Path to a configuration file
        """
        self.config = self._default_config()
        if config_file:
            self.load_config(config_file)
            
    def _default_config(self):
        """
        Provide the default server configuration.
        
        Returns:
            dict: Default configuration
        """
        return {
            "host": "0.0.0.0",
            "port": 7860,
            "share": True,  # Enable sharing to make server accessible outside localhost
            "auth": None,
            "ssl_verify": True,
            "mcp_server": True
        }
        
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to the configuration file
        """
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            return
            
        try:
            # Determine file type and load
            if config_file.endswith('.json'):
                import json
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {config_file}")
                return
                
            # Update config
            self.config.update(loaded_config)
            
            # Validate
            is_valid, error = InputValidator.validate_server_config(self.config)
            if not is_valid:
                logger.error(f"Invalid server configuration: {error}")
                
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            
    def get_config(self):
        """
        Get the current configuration.
        
        Returns:
            dict: Current configuration
        """
        return self.config
        
    def update_config(self, updates):
        """
        Update the configuration with new values.
        
        Args:
            updates (dict): Dictionary of configuration updates
        """
        self.config.update(updates)
        
        # Validate
        is_valid, error = InputValidator.validate_server_config(self.config)
        if not is_valid:
            logger.error(f"Invalid server configuration after update: {error}")
        else:
            logger.info("Server configuration updated")

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch Iris MCP Server')
    parser.add_argument('--model', type=str, help='Path to saved model file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=7860, help='Port number')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--share', action='store_true', help='Create a shareable link')
    
    args = parser.parse_args()
    
    # Default kwargs for server launch
    server_kwargs = {
        'share': True  # Default to enabling share for accessibility
    }
    
    # If config file is provided, use it
    if args.config:
        config = ServerConfig(args.config)
        server_config = config.get_config()
        host = server_config.get("host", "0.0.0.0")
        port = server_config.get("port", 7860)
        
        # Update kwargs from config
        for key in ['share', 'auth', 'ssl_verify']:
            if key in server_config:
                server_kwargs[key] = server_config[key]
    else:
        host = args.host
        port = args.port
        
    # Override with command line arguments if provided
    if args.share:
        server_kwargs['share'] = True
    
    try:
        # Launch the server with all parameters
        logger.info(f"Starting MCP server with parameters: host={host}, port={port}, kwargs={server_kwargs}")
        create_mcp_server(args.model, host, port, **server_kwargs)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise