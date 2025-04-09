# Necessary Langchain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

# Other modules and packages
import os
import uuid
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit interface
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System")
st.write("Upload a PDF document and ask questions about its content.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, step=50)
    num_chunks = st.slider("Number of Chunks to Retrieve", 1, 10, 4, step=1)
    
    if not OPENAI_API_KEY:
        st.error("Please set your OpenAI API key in the .env file")
        st.stop()

# Initialize LLM (Large Language Model)
try:
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.7,
        max_tokens=1000
    )
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Create embedding function
def get_embedding_function():
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            openai_api_key=OPENAI_API_KEY
        )
        return embeddings
    except Exception as e:
        st.error(f"Error creating embedding function: {str(e)}")
        return None

# Creating the vectorstore (Chroma) to organize the data
def create_vectorstore(chunks, embedding_function, vectorstore_path):
    try:
        # Generate unique IDs for each document based on content
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

        # Ensure that only unique documents are kept
        unique_ids = set()
        unique_chunks = []
        for chunk, id in zip(chunks, ids):
            if id not in unique_ids:
                unique_ids.add(id)
                unique_chunks.append(chunk)

        # Create a new Chroma database from the unique documents
        vectorstore = Chroma.from_documents(
            documents=unique_chunks,
            embedding=embedding_function,
            persist_directory=vectorstore_path
        )

        # Persist the vectorstore to disk
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main processing function
def process_document(uploaded_file, chunk_size, chunk_overlap):
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and process the PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(pages)

        # Get embedding function
        embedding_function = get_embedding_function()
        if embedding_function is None:
            return None

        # Create vectorstore path based on file name
        vectorstore_path = f"vectorstore_{uploaded_file.name.replace('.pdf', '')}"
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
        os.makedirs(vectorstore_path)

        # Create and load vectorstore
        vectorstore = create_vectorstore(chunks, embedding_function, vectorstore_path)
        if vectorstore is None:
            return None

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks}
        )

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return retriever
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following 
pieces of retrieved context to answer the question. If you don't know the 
answer, say that you don't know. Don't make up anything.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above.
"""

# Create prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Create RAG chain
def create_rag_chain(retriever):
    """Create a RAG chain for question answering."""
    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    chain = (
        {
            "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Main Streamlit interface
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        retriever = process_document(uploaded_file, chunk_size, chunk_overlap)
    
    if retriever is not None:
        rag_chain = create_rag_chain(retriever)
        
        # Question input
        question = st.text_input("Ask a question about the document:", key="question_input")
        
        if question:
            with st.spinner("Generating answer..."):
                try:
                    # Create input dictionary for the chain
                    input_dict = {"question": question}
                    # Get response from the chain
                    response = rag_chain.invoke(input_dict)
                    st.write("### Answer")
                    st.write(response)
                    
                    # Show retrieved chunks
                    with st.expander("Show Retrieved Context"):
                        relevant_chunks = retriever.get_relevant_documents(question)
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.write(f"#### Chunk {i}")
                            st.write(chunk.page_content)
                            st.write("---")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

