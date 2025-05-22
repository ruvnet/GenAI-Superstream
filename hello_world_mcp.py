import gradio as gr
import os

# Set environment variable to enable MCP server
os.environ["GRADIO_MCP_SERVER"] = "True"

def hello_world(name: str = "World") -> str:
    """
    A simple Hello World function that greets the provided name.
    
    Args:
        name: The name to greet, defaults to "World"
        
    Returns:
        A greeting message as a string
    """
    return f"Hello, {name}!"

# Create a Gradio Blocks app
with gr.Blocks() as demo:
    gr.Markdown("# Hello World MCP Server")
    
    # Define input and output components
    name_input = gr.Textbox(label="Enter your name", value="World")
    greeting_output = gr.Textbox(label="Greeting")
    
    # Define the function that will be called when the button is clicked
    def greet(name):
        return hello_world(name)
    
    # Create a button that submits the name and displays the greeting
    submit_button = gr.Button("Greet")
    submit_button.click(greet, inputs=name_input, outputs=greeting_output)
    
    # Also connect the input box's submit event
    name_input.submit(greet, inputs=name_input, outputs=greeting_output)

# Launch the app with MCP server enabled
# Using port 7880 which shouldn't conflict with existing servers
demo.launch(server_name="0.0.0.0", server_port=7880, mcp_server=True)

print("MCP server is available at: http://0.0.0.0:7880/gradio_api/mcp/sse")
print("The 'hello_world' tool is now registered and available through the MCP server.")