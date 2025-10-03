# rag-service/service.py (Final Version with Memory Loop)
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
    Document, # <-- NEW: Import the Document class
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from neo4j import GraphDatabase, exceptions
from llama_index.core.llms import (
    LLM, CompletionResponse, CompletionResponseGen, ChatResponse, ChatResponseGen,
    LLMMetadata, ChatMessage, MessageRole,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
# --- NEW: Import Pydantic for the new endpoint ---
from pydantic import BaseModel

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

# (The RESTfulEmbedding and RESTfulLLM classes remain unchanged)
# --- Custom Embedding Class (MODIFIED FOR BATCHING) ---
class RESTfulEmbedding(BaseEmbedding):
    def _get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            requests.post(f"{RESOURCE_MANAGER_URL}/request_model", json={"model_name": "embedding"}, timeout=180).raise_for_status()
            response = requests.post(f"{EMBEDDING_SERVICE_URL}/embed-batch", json={"texts": texts}, timeout=180)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logging.error(f"❌ Failed to get embedding for batch of size {len(texts)}. Error: {e}")
            return [[0.0] * 768 for _ in texts]
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]: return self._get_embedding_batch(texts)
    def _get_query_embedding(self, query: str) -> List[float]: return self._get_embedding_batch([query])[0]
    def _get_text_embedding(self, text: str) -> List[float]: return self._get_embedding_batch([text])[0]
    async def _aget_query_embedding(self, query: str) -> List[float]: return self._get_query_embedding(query)
    async def _aget_text_embedding(self, text: str) -> List[float]: return self._get_text_embedding(text)

# --- Custom LLM Class (remains the same) ---
class RESTfulLLM(LLM):
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
        text = self._call_llm_service(prompt); gen = (lambda: (yield CompletionResponse(text=text, delta=text)))(); return gen()
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages); text = self._call_llm_service(prompt); return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))
    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages); text = self._call_llm_service(prompt); gen = (lambda: (yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=text)))(); return gen()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse: return self.complete(prompt, **kwargs)
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen: return self.stream_complete(prompt, **kwargs)
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse: return self.chat(messages, **kwargs)
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen: return self.stream_chat(messages, **kwargs)

# --- FastAPI App and Startup ---
app = FastAPI(title="GraphRAG Service with LlamaIndex")

# (The @app.on_event("startup") function remains unchanged)
@app.on_event("startup")
def configure_llama_index():
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

# (The DocumentHandler and background_indexer functions remain unchanged)
class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md')):
            logging.info(f"New document detected: {event.src_path}")
            indexing_queue.put(event.src_path)

def background_indexer():
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

# (The /health and /query endpoints remain unchanged)
@app.get("/health")
def health():
    if kg_index is not None and INDEXING_COMPLETE.is_set():
        return {"status": "ok", "index_ready": True}
    return {"status": "starting", "index_ready": False}

@app.post("/query")
def query_system(payload: dict):
    input_text = payload.get("input_text")
    if not input_text: raise HTTPException(status_code=400, detail="No input_text provided.")
    if not INDEXING_COMPLETE.is_set(): raise HTTPException(status_code=503, detail="Index is not ready.")
    try:
        logging.info(f"Received input: '{input_text}'")
        query_engine = kg_index.as_query_engine(retriever_mode="hybrid", similarity_top_k=5)
        response = query_engine.query(input_text)
        final_response = str(response)
        logging.info(f"⬅️  Received synthesized response: '{final_response}'")
        return {"response": final_response}
    except Exception as e:
        logging.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: Endpoint for adding new memories from conversations ---
class MemoryRequest(BaseModel):
    text: str

@app.post("/add_memory")
async def add_memory(payload: MemoryRequest):
    """Receives a text snippet and adds it to the knowledge graph as a new memory."""
    if kg_index is None:
        raise HTTPException(status_code=503, detail="Index is not ready to accept new memories.")
    
    try:
        logging.info(f"📝 Received new memory to store: '{payload.text[:100]}...'")
        # We create a LlamaIndex Document object, which is the expected input for the indexer.
        new_memory_doc = Document(text=payload.text)
        kg_index.insert(new_memory_doc)
        logging.info("✅ New memory has been successfully added to the knowledge graph.")
        return {"status": "memory_added"}
    except Exception as e:
        logging.error(f"❌ Failed to add new memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process and store the new memory.")
