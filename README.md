# GenAI-Superstream
Agentic Engineering for Data Analysis 

## Summary

We will create a basic Model Context Protocol (MCP) server that exposes a scikit-learn classifier via Gradio’s built-in MCP capabilities. First, we’ll install the necessary Python packages and load the Iris dataset with scikit-learn. Then we’ll define a prediction function with a docstring that describes its inputs and outputs. Next, we’ll wrap that function in a Gradio Blocks app and launch it with `mcp_server=True`, which automatically exposes the function as an MCP tool. Finally, we’ll verify the MCP endpoint and demonstrate a simple Gradio client calling the tool. This end-to-end setup takes fewer than 20 lines of code and leverages Gradio’s seamless integration for both UI and tool-calling workflows ([Gradio][1], [Hugging Face][2]).

## Prerequisites

Before you begin, ensure you have:

* Python 3.10 or higher installed on your system. ([Gradio][3])
* An environment where you can install packages via `pip`. ([Gradio][4])

## Step 1: Install Dependencies

```bash
pip install scikit-learn gradio[mcp] mcp
```

This installs scikit-learn for the classifier, Gradio with MCP support, and the MCP protocol library ([Gradio][3], [Hugging Face][2]).

## Step 2: Define the Scikit-Learn Classifier

Create a file named `mcp_server.py` and add:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data and train
iris = load_iris()
clf = LogisticRegression(max_iter=200)
clf.fit(iris.data, iris.target)
```

This snippet loads the Iris dataset and trains a logistic regression model on its features and labels ([Gradio][5]).

## Step 3: Create the MCP-Exposed Prediction Function

Below your training code, define a prediction function with a descriptive docstring:

```python
def predict_species(features: list) -> dict:
    """
    Predicts the Iris species given a list of four features:
    [sepal_length, sepal_width, petal_length, petal_width].
    Returns a dict with class probabilities.
    """
    probs = clf.predict_proba([features])[0]
    classes = iris.target_names.tolist()
    return dict(zip(classes, probs))
```

Gradio will use the docstring to generate the MCP tool specification automatically ([Gradio][1]).

## Step 4: Wrap in a Gradio Blocks App and Enable MCP

Continue editing `mcp_server.py`:

```python
import gradio as gr

with gr.Blocks() as demo:
    inp = gr.Textbox(label="Comma-separated features")
    out = gr.JSON(label="Class probabilities")

    def wrapper(text):
        features = list(map(float, text.split(",")))
        return predict_species(features)

    inp.submit(wrapper, inp, out)

demo.launch(server_name="0.0.0.0", server_port=7860, mcp_server=True)
```

* We’re using Blocks for fine-grained control over input parsing and UI layout. ([Gradio][6])
* The `mcp_server=True` flag starts both the web UI and an MCP endpoint at `/gradio_api/mcp/sse`. ([Gradio][1], [Hugging Face][2])

## Step 5: Run and Test the MCP Server

Launch the server:

```bash
python mcp_server.py
```

In the console you’ll see:

```
 * Running on http://0.0.0.0:7860
 * MCP server URL: http://0.0.0.0:7860/gradio_api/mcp/sse
```

You can now point any MCP-capable client (e.g., Claude Desktop, Cursor) to that SSE endpoint ([Gradio][4], [Reddit][7]).

## Step 6: Example Gradio MCP Client

Optionally, create `mcp_client.py`:

```python
import gradio as gr
from mcp import Client

client = Client("http://localhost:7860/gradio_api/mcp/sse")

def ask_model(prompt: str):
    # Calls the MCP tool named "predict_species"
    response = client.call(
        tool="predict_species",
        input={"features": list(map(float, prompt.split(",")))}
    )
    return response

iface = gr.Interface(ask_model, gr.Textbox(label="Features"), gr.JSON(label="Prediction"))
iface.launch()
```

This Gradio interface uses the MCP Client to invoke the server’s `predict_species` tool dynamically ([Gradio][3], [MCP Servers][8]).

## Next Steps

* Secure your MCP server by adding authentication or token checks around the `predict_species` call. ([Hugging Face][2])
* Extend the example to support multiple tools (e.g., regression models, clustering tools) by defining additional functions with docstrings. ([GitHub][9])
* Deploy on Hugging Face Spaces or another cloud environment using Docker, and set `GRADIO_MCP_SERVER=True` for production builds ([Hugging Face][2]).

This setup demonstrates how quickly you can turn any Python function—including scikit-learn classifiers—into a tool that LLMs can call via MCP, all within a single Gradio app.

[1]: https://www.gradio.app/guides/building-mcp-server-with-gradio?utm_source=chatgpt.com "Building Mcp Server With Gradio"
[2]: https://huggingface.co/blog/gradio-mcp?utm_source=chatgpt.com "How to Build an MCP Server with Gradio - Hugging Face"
[3]: https://www.gradio.app/guides/building-an-mcp-client-with-gradio?utm_source=chatgpt.com "Building An Mcp Client With Gradio"
[4]: https://www.gradio.app/guides/using-docs-mcp?utm_source=chatgpt.com "Using Docs Mcp - Gradio"
[5]: https://www.gradio.app/docs?utm_source=chatgpt.com "Gradio Documentation"
[6]: https://www.gradio.app/docs/gradio/blocks?utm_source=chatgpt.com "Blocks - Gradio Docs"
[7]: https://www.reddit.com/r/mcp/comments/1kbnoev/build_an_mcp_server_in_a_few_lines_of_python_with/?utm_source=chatgpt.com "Build an MCP server in a few lines of Python with Gradio - Reddit"
[8]: https://mcpmarket.com/server/gradio-client?utm_source=chatgpt.com "Gradio Client: Tool Interaction for Language Models - MCP Market"
[9]: https://github.com/justjoehere/mcp_gradio_client?utm_source=chatgpt.com "justjoehere/mcp_gradio_client: This is a proof of concept ... - GitHub"
