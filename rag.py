import os
import glob
import pickle
from typing import List
from langchain_community.document_loaders import PyPDFLoader

# Text splitters - use langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector stores - use langchain-chroma
from langchain_chroma import Chroma

# Retrievers - moved to langchain-classic
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank

# Tools - moved to langchain-classic
from langchain_classic.tools.retriever import create_retriever_tool

# Agents - use langchain.agents (NEW in v1)
from langchain.agents import create_agent

# Memory - use langgraph.checkpoint.memory
from langgraph.checkpoint.memory import InMemorySaver

# Core messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Ollama - use langchain-ollama
from langchain_ollama import OllamaEmbeddings, ChatOllama


# --- Configuration ---
PDF_DIR = "./pdfs" 
CHROMA_PATH = "./chroma_db_hybrid_ollama"
CHUNKS_CACHE_FILE = "./cached_chunks.pkl" 
OLLAMA_BASE_URL = "http://localhost:11434" 

# IMPORTANT: Choose a tool-calling capable model for the agent (e.g., gpt-oss:120b-cloud, llama3.1, qwen2)
LLM_MODEL = "gpt-oss:120b-cloud" 
EMBEDDING_MODEL = "bge-large" 

K_RETRIEVAL = 15  
K_FINAL = 8       

# --- Utility: Get PDF Master List (For Agent Prompt Triage) ---

def get_pdf_master_list(pdf_directory: str) -> str:
    """Collects all unique PDF filenames and formats them for the prompt."""
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    file_names = sorted([os.path.basename(path) for path in pdf_files])
    
    if not file_names:
        return "No documents found."
    
    formatted_list = "\n".join([f"- {name}" for name in file_names])
    return formatted_list

# --- 1. Document Loading and Preprocessing (With Chunk Cache) ---

def load_and_chunk_pdfs(pdf_directory: str) -> list[Document]:
    """Loads all PDFs, chunks them, and applies a file-based cache for speed."""
    
    # 1. Attempt to load chunks from the cache file
    if os.path.exists(CHUNKS_CACHE_FILE):
        print(f"Loading chunks from cache file: {CHUNKS_CACHE_FILE}...")
        try:
            with open(CHUNKS_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing documents.")
            
    # --- Fallback: Load and process documents ---
    if not os.path.exists(pdf_directory) or not glob.glob(os.path.join(pdf_directory, "*.pdf")):
        return []

    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    all_documents = []

    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            documents = loader.load()
            file_name = os.path.basename(path)
            for doc in documents:
                doc.metadata["source_file"] = file_name
                if 'source' in doc.metadata:
                    doc.metadata['source_page'] = doc.metadata.pop('source')
            all_documents.extend(documents)
        except Exception:
            pass

    # Recursive Text Splitting (optimized for paragraph breaks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len,
        separators=["\n\n", "\n", " ", ""], is_separator_regex=False, 
    )
    chunks = text_splitter.split_documents(all_documents)

    # Save to cache
    try:
        with open(CHUNKS_CACHE_FILE, 'wb') as f:
            pickle.dump(chunks, f)
    except Exception:
        pass 
        
    return chunks


# --- 2. Hybrid RAG Pipeline Setup (Chroma Persistence, BM25, RRF, Reranker) ---

def setup_hybrid_retriever(chunks: list[Document]):
    """Sets up the Hybrid Retriever pipeline."""
    
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Chroma Persistence/Loading
    is_db_exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0

    if is_db_exists:
        print(f"Loading existing Chroma database from {CHROMA_PATH}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding
        )
    else:
        print(f"Creating new Chroma database at {CHROMA_PATH}...")
        if not chunks:
            raise ValueError("Cannot create new ChromaDB: No chunks available.")
            
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding,
            persist_directory=CHROMA_PATH
        )
    
    # Chroma Retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
    
    if not chunks:
        raise ValueError("Cannot create BM25 index: No chunks available.")
        
    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = K_RETRIEVAL

    # Ensemble Retriever (Performs RRF: Reciprocal Rank Fusion)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5] 
    )

    # Contextual Compression (Reranking)
    reranker = FlashrankRerank(top_n=K_FINAL)
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=ensemble_retriever
    )

    return final_retriever


# --- 3. Agent and Tool Setup (Modernized and Optimized) ---

def setup_rag_agent(retriever, pdf_list_string: str):
    """Sets up the modern LangGraph Agent with the retrieval tool and memory."""
    
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    
    # 3a. Wrap Retriever as Tool
    tool = create_retriever_tool(
        retriever,
        "document_search",
        "This tool is your primary source of knowledge. Use it to search the PDF documents for factual information, names, dates, and concepts."
    )
    tools = [tool]

    # 3b. Optimized System Prompt with Document List
    system_prompt = (
        "You are an intelligent, conversational, and helpful RAG Agent. Your knowledge is based on the following documents. "
        "Always use the 'document_search' tool to find facts. Never guess."
        "Your goal is to answer questions using facts found via the 'document_search' tool.\n\n"
        
        f"**📚 Available Documents in Knowledge Base:**\n{pdf_list_string}\n\n"
        
        "**Instructions:**\n"
        "1. **Triage:** If the user's question mentions a specific file name from the list above, include that file name in your search query to the 'document_search' tool to refine results.\n"
        "2. **Tool Use:** Always use the 'document_search' tool for factual questions. Never guess.\n"
        "3. **Refinement:** If results are vague, proactively prompt the user for clarification, referencing the document list, before trying another search.\n"
        "4. Maintain a conversational tone and cite the source file from the tool's output when answering.\n"
    )

    # 3c. Create the Agent (LangGraph/create_agent)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )

    return agent