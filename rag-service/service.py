# rag-service/service.py (Final Version)
import os
import sys
import logging
import threading
import time
import requests
from queue import Queue
from typing import Any, List, Optional, Sequence
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
from llama_index.core.embeddings import BaseEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from neo4j import GraphDatabase, exceptions
from llama_index.core.llms import (
    LLM, CompletionResponse, CompletionResponseGen, ChatResponse, ChatResponseGen,
    LLMMetadata, ChatMessage, MessageRole,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback

# --- Logging and Configuration (remains the same) ---
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
# --- NEW: Threading event to signal indexing completion ---
INDEXING_COMPLETE = threading.Event()

# --- Custom Embedding and LLM Classes (remain the same) ---
class RESTfulEmbedding(BaseEmbedding):
    # ... (code is correct) ...
    def _get_embedding(self, text: str) -> List[float]:
        try:
            requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "embedding"}, timeout=180).raise_for_status()
            response = requests.post(f"{EMBEDDING_SERVICE_URL}/embed", json={"text": text}, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logging.error(f"❌ Failed to get embedding for text: '{text}'. Error: {e}")
            return [0.0] * 768
    def _get_query_embedding(self, query: str) -> List[float]: return self._get_embedding(query)
    def _get_text_embedding(self, text: str) -> List[float]: return self._get_embedding(text)
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]: return [self._get_embedding(text) for text in texts]
    async def _aget_query_embedding(self, query: str) -> List[float]: return self._get_query_embedding(query)
    async def _aget_text_embedding(self, text: str) -> List[float]: return self._get_text_embedding(text)

class RESTfulLLM(LLM):
    # ... (code is correct) ...
    @property
    def metadata(self) -> LLMMetadata: return LLMMetadata(context_window=4096, num_output=256, model_name="local-llm-service")
    def _call_llm_service(self, prompt: str) -> str:
        try:
            logging.info("Requesting LLM model from Resource Manager...")
            requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "llm"}, timeout=180).raise_for_status()
            logging.info("LLM model is ready. Sending prompt...")
            response = requests.post(f"{LLM_SERVICE_URL}/completion", json={"prompt": prompt}, timeout=300)
            response.raise_for_status()
            return response.json().get("content", "")
        except Exception as e:
            logging.error(f"❌ LLM request failed: {e}")
            return f"Error: {e}"
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse: return CompletionResponse(text=self._call_llm_service(prompt))
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        text = self._call_llm_service(prompt)
        def gen(): yield CompletionResponse(text=text, delta=text)
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
        def gen(): yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=text)
        return gen()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse: return self.complete(prompt, **kwargs)
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen: return self.stream_complete(prompt, **kwargs)
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse: return self.chat(messages, **kwargs)
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen: return self.stream_chat(messages, **kwargs)

# --- FastAPI App and Startup ---
app = FastAPI(title="GraphRAG Service with LlamaIndex")

@app.on_event("startup")
def configure_llama_index():
    # ... (Neo4j connection and LlamaIndex Settings remain the same) ...
    global kg_index
    logging.info("--- Initializing LlamaIndex Components ---")
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

    # --- Start File Watcher and Background Indexer (this logic is now improved) ---
    os.makedirs(INPUT_DATA_DIR, exist_ok=True)
    
    # Start the background thread
    indexer_thread = threading.Thread(target=background_indexer, daemon=True)
    indexer_thread.start()
    
    # Schedule watching for *new* files
    observer = Observer()
    observer.schedule(DocumentHandler(), INPUT_DATA_DIR, recursive=True)
    observer.start()
    logging.info(f"👀 Now watching for new files in '{INPUT_DATA_DIR}'...")
    
    # Queue up existing files for initial processing
    initial_files = [os.path.join(root, name) for root, _, files in os.walk(INPUT_DATA_DIR) for name in files if name.endswith(('.txt', '.md'))]
    if not initial_files:
        logging.info("No initial documents found. Indexing is complete.")
        INDEXING_COMPLETE.set() # If no files, we're done
    else:
        for path in initial_files:
            logging.info(f"Found existing document on startup: {path}")
            indexing_queue.put(path)

# --- Document Handling and Background Indexer ---
class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md')):
            logging.info(f"New document detected: {event.src_path}")
            indexing_queue.put(event.src_path)

def background_indexer():
    # --- MODIFIED: The indexer now signals when it's done with the initial files ---
    while True:
        filepath = indexing_queue.get()
        try:
            logging.info(f"Indexing '{filepath}'...")
            reader = SimpleDirectoryReader(input_files=[filepath])
            documents = reader.load_data()
            for doc in documents:
                kg_index.insert(doc)
            logging.info(f"✅ Finished indexing '{filepath}'.")
        except Exception as e:
            logging.error(f"❌ Indexing failed for '{filepath}': {e}")
        finally:
            indexing_queue.task_done()
            # If the queue is empty after processing, the initial load is done.
            if indexing_queue.empty():
                logging.info("✅ Initial document indexing complete.")
                INDEXING_COMPLETE.set()

# --- MODIFIED: Health Check ---
@app.get("/health")
def health():
    # The service is only "ok" if the index is built AND the initial indexing is done.
    if kg_index is not None and INDEXING_COMPLETE.is_set():
        return {"status": "ok", "index_ready": True}
    return {"status": "starting", "index_ready": False}

# --- Query Endpoint (remains the same) ---
@app.post("/query")
def query_system(payload: dict):
    input_text = payload.get("input_text")
    if not input_text: raise HTTPException(status_code=400, detail="No input_text provided.")
    if not INDEXING_COMPLETE.is_set(): raise HTTPException(status_code=503, detail="Index is not ready.")
    try:
        logging.info(f"Received input: '{input_text}'")
        query_engine = kg_index.as_query_engine()
        response = query_engine.query(input_text)
        final_response = str(response)
        logging.info(f"⬅️  Received synthesized response: '{final_response}'")
        return {"response": final_response}
    except Exception as e:
        logging.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
