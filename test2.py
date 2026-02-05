# Necessary Langchain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Other modules and packages
import os
import uuid
import tempfile
import streamlit as st
from transformers import pipeline

# Initialize Streamlit interface
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System")
st.write("Upload a PDF document and ask questions about its content.")

"""RAG Document Q&A System (Streamlit)
This app loads an OpenAI API key from Streamlit secrets
or from the environment. It shows a clear message if no key is found.
"""

# Load environment variables - Try multiple sources
OPENAI_API_KEY = None
KEY_SOURCE = None

# Try to get API key from different sources
try:
    # First try Streamlit secrets
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    KEY_SOURCE = "secrets"
except (KeyError, FileNotFoundError):
    try:
        # Then try environment variables
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            KEY_SOURCE = "env"
    except Exception:
        pass

if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    st.error("❌ OpenAI API key not found!")
    st.info("📝 Please set your OpenAI API key in one of these ways:")
    
    with st.expander("🔧 Setup Instructions"):
        st.markdown("""
        **Option 1: Streamlit Cloud Secrets**
        1. Go to your app dashboard
        2. Click on the three dots menu (⋮)
        3. Select 'Settings'
        4. Go to 'Secrets' tab
        5. Add: `OPENAI_API_KEY = "your-api-key-here"`
        
        **Option 2: Local .streamlit/secrets.toml file**
        Create a file `.streamlit/secrets.toml` in your project with:
        ```toml
        OPENAI_API_KEY = "your-api-key-here"
        ```
        
        **Get your API key from:**
        https://platform.openai.com/api-keys
        """)
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    if OPENAI_API_KEY:
        source_label = "Streamlit secrets" if KEY_SOURCE == "secrets" else "Environment variable"
        st.success(f"✅ OpenAI API Key Configured ({source_label})")
    else:
        st.warning("Using local LLM fallback (no OpenAI key)")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, step=50)
    num_chunks = st.slider("Number of Chunks to Retrieve", 1, 10, 4, step=1)
    default_provider = "OpenAI" if OPENAI_API_KEY else "Local (FLAN-T5)"
    model_provider = st.selectbox("LLM Provider", ["OpenAI", "Local (FLAN-T5)"], index=0 if default_provider=="OpenAI" else 1)

# Initialize LLM (Large Language Model)
@st.cache_resource
def get_openai_llm():
    try:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing OpenAI LLM: {str(e)}")
        return None

@st.cache_resource
def get_local_llm():
    try:
        return pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception as e:
        st.error(f"Error initializing local LLM: {str(e)}")
        return None

llm = None
local_llm = None
if model_provider == "OpenAI":
    llm = get_openai_llm()
    if llm is None:
        st.info("Switching to local LLM fallback.")
        local_llm = get_local_llm()
        model_provider = "Local (FLAN-T5)"
else:
    local_llm = get_local_llm()
    if local_llm is None:
        st.error("Local LLM unavailable. Please install transformers/torch or provide OPENAI_API_KEY.")
        st.stop()

# Create embedding function
@st.cache_resource
def get_embedding_function():
    try:
        # Use local embeddings to avoid OpenAI quota and reduce cost
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error creating embedding function: {str(e)}")
        return None

# Creating the vectorstore (FAISS) to organize the data
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

        # Create a new FAISS database from the unique documents (in-memory)
        vectorstore = FAISS.from_documents(
            documents=unique_chunks,
            embedding=embedding_function
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
    """Create a RAG chain for question answering using OpenAI."""
    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}
    
    Please provide a clear and concise answer based on the context above.
    If you don't know the answer based on the context, say that you don't know.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
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
        rag_chain = None
        if model_provider == "OpenAI" and llm is not None:
            rag_chain = create_rag_chain(retriever)
        
        question = st.text_input("Ask a question about the document:", key="question_input")
        
        if question:
            with st.spinner("Generating answer..."):
                try:
                    if model_provider == "OpenAI" and rag_chain is not None:
                        input_dict = {"question": question}
                        response = rag_chain.invoke(input_dict)
                        st.write("### Answer (OpenAI)")
                        st.write(response)
                    else:
                        context_txt = format_docs(retriever.invoke(question))
                        prompt = (
                            "Using only the context below, answer the question.\n"
                            + "Context:\n" + context_txt + "\n\n"
                            + "Question: " + question + "\nAnswer:"
                        )
                        gen = local_llm(prompt, max_new_tokens=256)
                        response = gen[0]["generated_text"] if isinstance(gen, list) and len(gen) > 0 else ""
                        st.write("### Answer (Local LLM)")
                        st.write(response)

                    with st.expander("Show Retrieved Context"):
                        relevant_chunks = retriever.invoke(question)
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.write(f"#### Chunk {i}")
                            st.write(chunk.page_content)
                            st.write("---")
                except Exception as e:
                    err_text = str(e)
                    # If OpenAI rate limit error, try local fallback automatically
                    if "quota" in err_text.lower() or "429" in err_text or "rate limit" in err_text.lower():
                        st.info("OpenAI rate limit error. Switching to local LLM.")
                        try:
                            context_txt = format_docs(retriever.invoke(question))
                            prompt = (
                                "Using only the context below, answer the question.\n"
                                + "Context:\n" + context_txt + "\n\n"
                                + "Question: " + question + "\nAnswer:"
                            )
                            gen = local_llm(prompt, max_new_tokens=256)
                            response = gen[0]["generated_text"] if isinstance(gen, list) and len(gen) > 0 else ""
                            st.write("### Answer (Local LLM)")
                            st.write(response)

                            with st.expander("Show Retrieved Context"):
                                relevant_chunks = retriever.invoke(question)
                                for i, chunk in enumerate(relevant_chunks, 1):
                                    st.write(f"#### Chunk {i}")
                                    st.write(chunk.page_content)
                                    st.write("---")
                        except Exception as e2:
                            st.warning("Local LLM fallback failed. Showing context snippet.")
                            with st.expander("Error Details"):
                                st.code(f"OpenAI error: {err_text}\nLocal error: {str(e2)}")
                            try:
                                relevant_chunks = retriever.invoke(question)
                                st.write("### Context-Based Answer (no LLM)")
                                if len(relevant_chunks) > 0:
                                    top = relevant_chunks[0].page_content
                                    snippet = top[:800]
                                    st.write(snippet)
                                else:
                                    st.info("No relevant context found.")
                            except Exception as e3:
                                st.error(f"Fallback also failed: {str(e3)}")
    else:
        st.error("Failed to process the document. Please try again.")