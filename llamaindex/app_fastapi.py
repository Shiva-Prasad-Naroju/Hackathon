from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional
import asyncio
import logging
from contextlib import asynccontextmanager

# LlamaIndex imports
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

# Other imports
from chromadb import PersistentClient
from groq import Groq
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Request/Response Models ===
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    response: str
    status: str = "success"

# === Custom GroqLLM ===
class GroqLLM(CustomLLM):
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.1
    max_tokens: int = 1024
    client: Optional[Groq] = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use environment variable or fallback to your key
        api_key = os.getenv("GROQ_API_KEY", "place the original api key here")
        self.client = Groq(api_key=api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in LLM completion: {e}")
            return CompletionResponse(text="Sorry, I encountered an error processing your request.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        yield response

# === Global State ===
class AppState:
    def __init__(self):
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.is_initialized = False

app_state = AppState()

# === Startup/Shutdown Logic ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application...")
    try:
        await initialize_query_engine()
        logger.info("Application startup complete!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down the application...")

# === Initialize Query Engine ===
async def initialize_query_engine():
    global app_state
    
    try:
        # Initialize LLM
        llm = GroqLLM()
        Settings.llm = llm
        
        # Initialize Chroma
        chroma_client = PersistentClient(path="./chroma_db")
        
        try:
            collection = chroma_client.get_collection("documents")
        except Exception as e:
            logger.error(f"Failed to get collection 'documents': {e}")
            raise HTTPException(status_code=500, detail="Database collection not found. Please run the ingestion script first.")
        
        # Set up vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=llm, 
            response_mode="compact"
        )
        
        # Create query engine
        app_state.query_engine = RetrieverQueryEngine.from_args(
            retriever=index.as_retriever(),
            response_synthesizer=response_synthesizer
        )
        
        app_state.is_initialized = True
        logger.info("Query engine initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        raise e

# === FastAPI App ===
app = FastAPI(
    title="LlamaIndex Query API",
    description="A FastAPI application using LlamaIndex with Groq LLM",
    version="1.0.0",
    lifespan=lifespan
)

# === Health Check ===
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": app_state.is_initialized
    }

# === Query Endpoint ===
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if not app_state.is_initialized or app_state.query_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Processing query: {request.query}")
        response = app_state.query_engine.query(request.query)
        
        return QueryResponse(
            response=str(response),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# === Root Endpoint ===
@app.get("/")
async def root():
    return {
        "message": "LlamaIndex Query API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

# === Run the app ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)