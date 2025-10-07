# rag-service/service.py (Corrected and Complete for Phase 2)
import os
import sys
import logging
import threading
import time
import requests
from queue import Queue
from typing import Any, List, Optional, Sequence
import numpy as np
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
    Document,
    QueryBundle # <--- ADD THIS IMPORT
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from neo4j import GraphDatabase, exceptions
from llama_index.core.llms import (
    LLM, CompletionResponse, CompletionResponseGen, ChatResponse, ChatResponseGen,
    LLMMetadata, ChatMessage, MessageRole,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from pydantic import BaseModel # <--- ADD THIS IMPORT

# --- Helper functions for creating blended vectors ---
def _normalize(vec: np.ndarray) -> np.ndarray:
    """Helper to normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0: 
       return vec
    return vec / norm

def create_blended_vector(text_vector: List[float], emotion_vector: List[float], alpha: float) -> List[float]:
    """
    Creates the 'poisoned' integrated vector by blending text and emotion.
    """
    alpha = max(0.0, min(1.0, alpha))
    
    text_vec = _normalize(np.array(text_vector))
    emotion_vec_padded = np.zeros_like(text_vec)
    emotion_vec_normalized = _normalize(np.array(emotion_vector))
    emotion_vec_padded[:len(emotion_vec_normalized)] = emotion_vec_normalized
    
    blended_vec = (text_vec * alpha) + (emotion_vec_padded * (1.0 - alpha))
    
    return _normalize(blended_vec).tolist()

# --- Pydantic Models for API ---
class MemoryRequest(BaseModel):
    text: str
    emotion_vector: List[float]

class QueryRequest(BaseModel):
    input_text: str
    emotion_vector: List[float]
    alpha: float

# --- Logging and Configuration ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
try:
    NEO4J_URI = os.environ["NEO4J_URI"]
    LLM_SERVICE_URL = os.environ["LLM_SERVICE_URL"]
    EMBEDDING_SERVICE_URL = os.environ["EMBEDDING_SERVICE_URL"]
    RESOURCE_MANAGER_URL = os.environ["RESOURCE_MANAGER_URL"]
except KeyError as e:
    logging.error(f"🔥 Critical environment variable missing: {e}")
    sys.exit(1)

INPUT_DATA_DIR = "./input_data"
kg_index = None
indexing_queue = Queue()
INDEXING_COMPLETE = threading.Event()

# --- Custom LlamaIndex Classes (No Changes Here) ---
class RESTfulEmbedding(BaseEmbedding):
    def _get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "embedding"}, timeout=180).raise_for_status()
            response = requests.post(f"{EMBEDDING_SERVICE_URL}/embed-batch", json={"texts": texts}, timeout=180)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logging.error(f"❌ Failed to get embedding for batch of size {len(texts)}. Error: {e}")
            return [[0.0] * 768 for _ in texts] # Assuming embedding size is 768
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]: return self._get_embedding_batch(texts)
    def _get_query_embedding(self, query: str) -> List[float]: return self._get_embedding_batch([query])[0]
    def _get_text_embedding(self, text: str) -> List[float]: return self._get_embedding_batch([text])[0]
    async def _aget_query_embedding(self, query: str) -> List[float]: return self._get_query_embedding(query)
    async def _aget_text_embedding(self, text: str) -> List[float]: return self._get_text_embedding(text)

# In rag-service/service.py

# --- REPLACE THE ENTIRE CLASS DEFINITION WITH THIS ---
class RESTfulLLM(LLM):
    @property
    def metadata(self) -> LLMMetadata: return LLMMetadata(context_window=4096, num_output=256, model_name="local-llm-service")
    
    def _call_llm_service(self, prompt: str) -> str:
        try:
            requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "llm"}, timeout=180).raise_for_status()
            response = requests.post(f"{LLM_SERVICE_URL}/completion", json={"prompt": prompt}, timeout=300)
            response.raise_for_status()
            return response.json().get("content", "")
        except Exception as e:
            logging.error(f"❌ LLM request failed: {e}")
            return f"Error: {e}"

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse: 
        return CompletionResponse(text=self._call_llm_service(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        text = self._call_llm_service(prompt)
        def gen():
            yield CompletionResponse(text=text, delta=text)
        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        text = self._call_llm_service(prompt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        text = self._call_llm_service(prompt)
        def gen():
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=text)
        return gen()

    # --- NEW: Add async methods to satisfy the updated LLM interface ---
    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        return self.stream_complete(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        return self.stream_chat(messages, **kwargs)

# --- FastAPI App and Startup ---
app = FastAPI(title="GraphRAG Service with LlamaIndex")

@app.on_event("startup")
def configure_llama_index():
    # ... (This entire function remains unchanged) ...
    global kg_index
    logging.info("--- Initializing LlamaIndex Components ---")
    Settings.embed_batch_size = 64
    Settings.llm = RESTfulLLM()
    Settings.embed_model = RESTfulEmbedding()
    max_retries=10; wait_seconds=5
    for i in range(max_retries):
        try:
            logging.info(f"Attempting to connect to Neo4j (attempt {i+1}/{max_retries})...")
            graph_store = Neo4jGraphStore(url=NEO4J_URI, username="neo4j", password="neo4j")
            logging.info("✅ Successfully connected to Neo4j.")
            break
        except (ValueError, exceptions.ServiceUnavailable) as e:
            logging.warning(f"Neo4j not ready yet... Retrying in {wait_seconds} seconds...")
            if i == max_retries - 1: raise e
            time.sleep(wait_seconds)
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex.from_documents([], storage_context=storage_context, max_triplets_per_chunk=2, include_embeddings=True)
    logging.info("✅ LlamaIndex Components Initialized.")
    os.makedirs(INPUT_DATA_DIR, exist_ok=True)
    indexer_thread = threading.Thread(target=background_indexer, daemon=True)
    indexer_thread.start()
    observer = Observer()
    observer.schedule(DocumentHandler(), INPUT_DATA_DIR, recursive=True)
    observer.start()
    logging.info(f"👀 Now watching for new files in '{INPUT_DATA_DIR}'...")
    initial_files = [os.path.join(root, name) for root, _, files in os.walk(INPUT_DATA_DIR) for name in files if name.endswith(('.txt', '.md'))]
    if not initial_files:
        logging.info("No initial documents found. Indexing is complete.")
        INDEXING_COMPLETE.set()
    else:
        for path in initial_files:
            logging.info(f"Found existing document on startup: {path}")
            indexing_queue.put(path)

# --- Background Indexing (No Changes Here) ---
class DocumentHandler(FileSystemEventHandler):
    # ... (This class remains unchanged) ...
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md')):
            logging.info(f"New document detected: {event.src_path}")
            indexing_queue.put(event.src_path)

def background_indexer():
    # ... (This function remains unchanged) ...
    indexing_succeeded = False
    while True:
        filepath = indexing_queue.get()
        try:
            logging.info(f"Indexing '{filepath}'...")
            reader = SimpleDirectoryReader(input_files=[filepath])
            documents = reader.load_data()
            for doc in documents:
                kg_index.insert(doc)
            logging.info(f"✅ Finished indexing '{filepath}'.")
            indexing_succeeded = True
        except Exception as e:
            logging.error(f"❌ Indexing failed for '{filepath}': {e}", exc_info=True)
        finally:
            indexing_queue.task_done()
            if indexing_queue.empty() and indexing_succeeded:
                logging.info("✅ Initial document indexing complete.")
                INDEXING_COMPLETE.set()

# --- API Endpoints ---
@app.get("/health")
def health():
    if kg_index is not None and INDEXING_COMPLETE.is_set():
        return {"status": "ok", "index_ready": True}
    return {"status": "starting", "index_ready": False}

@app.post("/add_memory")
async def add_memory(payload: MemoryRequest):
    """Receives text and emotion, creates a blended vector, and stores it as a new memory."""
    if kg_index is None:
        raise HTTPException(status_code=503, detail="Index is not ready to accept new memories.")
    
    try:
        logging.info(f"📝 Received new memory to store: '{payload.text[:80]}...'")
        
        response = requests.post(f"{EMBEDDING_SERVICE_URL}/embed-batch", json={"texts": [payload.text]}, timeout=180)
        response.raise_for_status()
        text_vector = response.json()["embeddings"][0]
        
        blended_vector = create_blended_vector(text_vector, payload.emotion_vector, alpha=0.7)

        new_memory_doc = Document(
            text=payload.text,
            embedding=blended_vector,
            metadata={"emotion_vector": payload.emotion_vector}
        )
        
        kg_index.insert(document=new_memory_doc)
        logging.info("✅ New blended memory has been successfully added to the knowledge graph.")
        return {"status": "blended_memory_added"}
        
    except Exception as e:
        logging.error(f"❌ Failed to add new memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process and store the new memory.")



# --- REPLACE the /query function one last time ---
@app.post("/query")
def query_system(payload: QueryRequest):
    """
    Performs a unified search using a blended query vector, governed by a dynamic alpha.
    """
    if not payload.input_text:
        raise HTTPException(status_code=400, detail="No input_text provided.")
    if not INDEXING_COMPLETE.is_set():
        raise HTTPException(status_code=503, detail="Index is not ready.")
    
    try:
        logging.info(f"🔎 Retrieving memories for: '{payload.input_text}' with Alpha {payload.alpha:.2f}")

        response = requests.post(f"{EMBEDDING_SERVICE_URL}/embed-batch", json={"texts": [payload.input_text]}, timeout=180)
        response.raise_for_status()
        text_vector = response.json()["embeddings"][0]
        
        blended_query_vector = create_blended_vector(
            text_vector, 
            payload.emotion_vector, 
            payload.alpha
        )

        retriever = kg_index.as_retriever(
            similarity_top_k=3
        )
        
        # YOUR CORRECTED LOGIC: Use a QueryBundle to pass the custom vector
        query_bundle = QueryBundle(
            query_str=payload.input_text,
            embedding=blended_query_vector
        )
        retrieved_nodes = retriever.retrieve(query_bundle)

        raw_memories = [node.get_content() for node in retrieved_nodes]
        
        logging.info(f"⬅️  Retrieved {len(raw_memories)} memories via integrated emotional search.")
        return {"retrieved_memories": raw_memories}

    except Exception as e:
        logging.error(f"❌ Error during retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        logging.info(f"⬅️  Retrieved {len(raw_memories)} memories via integrated emotional search.")
        return {"retrieved_memories": raw_memories}

    except Exception as e:
        logging.error(f"❌ Error during retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
