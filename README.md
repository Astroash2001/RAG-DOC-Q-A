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
venv\Scripts\activate  # On Windows
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

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload a PDF document and ask questions about its content

## Project Structure

- `test2.py`: Main application file
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in repository)
- `vectorstore_*/`: Generated vector stores (not included in repository)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
