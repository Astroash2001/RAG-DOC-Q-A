# Necessary Langchain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma  # Fixed import
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Other modules and packages
import os  # Added missing import
import uuid
import tempfile
import streamlit as st
import pandas as pd
import shutil

# Load environment variables - Fixed the secrets access
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Fixed: use square brackets

# Initialize Streamlit interface
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System")
st.write("Upload a PDF document and ask questions about its content.")

# Check API key first
if not OPENAI_API_KEY:
    st.error("Please set your OpenAI API key in Streamlit secrets")
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, step=50)
    num_chunks = st.slider("Number of Chunks to Retrieve", 1, 10, 4, step=1)

# Initialize LLM (Large Language Model)
@st.cache_resource
def get_llm():
    try:
        return ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=1000,
            openai_api_key=OPENAI_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

llm = get_llm()
if llm is None:
    st.stop()

# Create embedding function
@st.cache_resource
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
def create_vectorstore(chunks, embedding_function):
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

        # Create a new Chroma database from the unique documents (in-memory)
        vectorstore = Chroma.from_documents(
            documents=unique_chunks,
            embedding=embedding_function,
            # Remove persist_directory to use in-memory storage
        )

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

        # Create vectorstore (in-memory)
        vectorstore = create_vectorstore(chunks, embedding_function)
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

# Create RAG chain
def create_rag_chain(retriever):
    """Create a RAG chain for question answering."""
    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}
    
    Please provide a clear and concise answer based on the context above.
    If you don't know the answer based on the context, say that you don't know.
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
        st.success("Document processed successfully!")
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
    else:
        st.error("Failed to process the document. Please try again.")