# RAG Document Q&A System

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## Features

- PDF document upload and processing
- Document chunking and embedding
- Question answering using RAG
- Context-aware responses
- Interactive Streamlit interface

## Requirements

- Python 3.10+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows (Command Prompt / PowerShell)

# Windows (Git Bash) users:
# Either activate with:
#   source venv/Scripts/activate
# or run Python directly from the venv without activating:
#   ./venv/Scripts/python.exe -m pip install -r requirements.txt
#   ./venv/Scripts/python.exe -m streamlit run test2.py
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run test2.py
```

Windows (Git Bash) quick start (no activation required):

```bash
# Install dependencies using the project venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Provide your OpenAI API key (or use Streamlit secrets)
export OPENAI_API_KEY="your_api_key_here"

# Run the app via the venv's Python
./.venv/Scripts/python.exe -m streamlit run test2.py
```

One-shot setup with Streamlit secrets (recommended):

```bash
mkdir -p .streamlit && echo 'OPENAI_API_KEY = "your_api_key_here"' > .streamlit/secrets.toml && ./.venv/Scripts/python.exe -m streamlit run test2.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload a PDF document and ask questions about its content

### Run without OpenAI (Local mode)

If you see an OpenAI quota error (HTTP 429) or don't have an API key, you can run the app with a local LLM fallback:

```bash
# Install local LLM dependencies (CPU)
pip install transformers torch

# Start the app
streamlit run test2.py
```

In the sidebar, set "LLM Provider" to "Local (FLAN-T5)". The app will also auto-switch to local mode if an OpenAI quota error occurs.

## Project Structure

- `test2.py`: Main application file
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in repository)
- `vectorstore_*/`: Generated vector stores (not included in repository)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
