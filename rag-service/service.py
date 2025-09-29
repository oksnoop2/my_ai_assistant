# rag-service/service.py (Rewritten for LlamaIndex)
import os
import sys
import logging
import threading
from queue import Queue
from fastapi import FastAPI, HTTPException
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LlamaIndex Imports ---
import torch
from llama_index.core import (
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Logging Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- Configuration (fail fast if required envs missing) ---
try:
    NEO4J_URI = os.environ["NEO4J_URI"]
    # These paths are the mount points inside the container
    LLM_MODEL_PATH = os.environ["LLM_MODEL_PATH"]
    # This is the name of the model, which will be downloaded to the cache volume
    EMBEDDING_MODEL_NAME = os.environ["EMBEDDING_MODEL_NAME"]
except KeyError as e:
    logging.error(f"🔥 Critical environment variable missing: {e}")
    sys.exit(1)

INPUT_DATA_DIR = "./input_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"🚀 RAG Service starting on device: {DEVICE}")


# --- Global Variables ---
# We will initialize these in the startup event
kg_index = None
indexing_queue = Queue()

# --- FastAPI App Initialization ---
app = FastAPI(title="GraphRAG Service with LlamaIndex")


# --- LlamaIndex Setup ---
@app.on_event("startup")
def configure_llama_index():
    """
    This function is called once when the FastAPI application starts.
    It sets up the global LlamaIndex components.
    """
    global kg_index
    logging.info("--- Initializing LlamaIndex Components ---")

    # 1. Set up the LLM
    # This loads the GGUF model file, configuring it to use the GPU
    llm = LlamaCPP(
        model_path=LLM_MODEL_PATH,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1}, # Offload all layers to GPU
        verbose=True,
    )

# This is the NEW, CORRECTED line
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        device=DEVICE,
        trust_remote_code=True  # <--- THIS IS THE ONLY CHANGE
    )
    # 3. Configure global LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 4. Connect to Neo4j Graph Store
    graph_store = Neo4jGraphStore(
        url=NEO4J_URI,
        username="neo4j", # Default for the user's setup
        password="neo4j", # Default for the user's setup
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # 5. Create the Knowledge Graph Index
    # This object will manage the graph and queries
    kg_index = KnowledgeGraphIndex.from_documents(
        [], # Start with no documents
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True,
    )
    logging.info("✅ LlamaIndex Components Initialized.")
    
    # --- Start File Watcher and Background Indexer ---
    os.makedirs(INPUT_DATA_DIR, exist_ok=True)
    
    indexer_thread = threading.Thread(target=background_indexer, daemon=True)
    indexer_thread.start()
    
    observer = Observer()
    observer.schedule(DocumentHandler(), INPUT_DATA_DIR, recursive=True)
    observer.start()
    
    logging.info(f"👀 Now watching for new files in '{INPUT_DATA_DIR}'...")
    
    # Seed the queue with pre-existing files from the input directory
    for root, _, files in os.walk(INPUT_DATA_DIR):
        for name in files:
            if name.endswith(('.txt', '.md')):
                path = os.path.join(root, name)
                logging.info(f"Found existing document on startup: {path}")
                indexing_queue.put(path)

# --- File Watcher and Indexing Logic ---

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md')):
            logging.info(f"New document detected: {event.src_path}")
            indexing_queue.put(event.src_path)

def background_indexer():
    """
    Monitors a queue for new file paths and indexes them into the knowledge graph.
    """
    while True:
        filepath = indexing_queue.get()
        try:
            logging.info(f"Indexing '{filepath}'...")
            # Load the new document
            reader = SimpleDirectoryReader(input_files=[filepath])
            documents = reader.load_data()
            
            # Insert the document into the knowledge graph index
            for doc in documents:
                kg_index.insert(documents=[doc])

            logging.info(f"✅ Finished indexing '{filepath}'.")
        except Exception as e:
            logging.error(f"❌ Indexing failed for '{filepath}': {e}")
        finally:
            indexing_queue.task_done()

# --- API Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "index_ready": kg_index is not None}

@app.post("/query")
def query_system(payload: dict):
    """
    Accepts a user question and queries the knowledge graph.
    """
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    if kg_index is None:
        raise HTTPException(status_code=503, detail="Index is not ready.")
        
    try:
        logging.info(f"Received query: '{question}'")
        # Create a query engine from the index
        query_engine = kg_index.as_query_engine(include_text=True, response_mode="tree_summarize")
        
        # Execute the query
        response = query_engine.query(question)
        
        logging.info(f"Generated response: '{response.response}'")
        return {"response": response.response}
        
    except Exception as e:
        logging.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
