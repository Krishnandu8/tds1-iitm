import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "tds_virtual_ta_collection"

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file or bashrc.")
# Ensure OPENAI_BASE_URL is set
if not os.getenv("OPENAI_BASE_URL"):
    print("Warning: OPENAI_BASE_URL environment variable not set. OpenAIEmbedding/OpenAI LLM will use default base URL.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Virtual TA API",
    description="An API for a Virtual Teaching Assistant powered by LlamaIndex and OpenAI, providing answers based on course content and Discourse forum data.",
    version="0.1.0"
)

# --- ADDED CORS MIDDLEWARE HERE ---
origins = [
    "*", # Allows all origins, useful for development and submission system
    # "http://localhost",
    # "http://localhost:8000",
    # "https://your-frontend-domain.com", # Replace with your actual frontend domain if known
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)
# --- END CORS MIDDLEWARE ADDITION ---


# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    question: str
    image: str = None # Optional: base64 encoded image string if you implement multimodal capabilities later

class SourceLink(BaseModel):
    url: str
    title: str = "No Title Available" # Default title if not found in metadata


class QueryResponse(BaseModel):
    answer: str
    links: list[SourceLink]


# --- Global Variable for Query Engine ---
# This will hold our RAG query engine, initialized once on startup.
query_engine = None

# --- Helper Function to Initialize RAG Engine ---
def initialize_query_engine():
    """
    Initializes and returns the LlamaIndex query engine.
    This function will be called once when the FastAPI app starts up.
    """
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH} and collection: {COLLECTION_NAME}")
    try:
        # Initialize ChromaDB client
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Configure the embedding model (same as used for indexing)
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_base=os.getenv("OPENAI_BASE_URL")
        )

        # Configure the LLM for response generation
        llm = OpenAI(
            model="gpt-3.5-turbo", # Or "gpt-4" if preferred and available
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.1
        )

        # Load the index from the persisted storage
        print("Loading VectorStoreIndex from ChromaDB...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        print("VectorStoreIndex loaded.")

        # Create a query engine
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=5, # Retrieve top 5 most relevant chunks
            # You can add a custom prompt here if you want to refine the LLM's persona
            # e.g., system_prompt="You are a helpful Virtual TA for a Data Science course. Answer questions concisely and cite sources."
        )
    except Exception as e:
        print(f"Failed to initialize query engine: {e}")
        # Depending on your deployment, you might want to raise here to prevent startup
        # or handle gracefully if the DB might not be ready yet.
        raise

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the query engine when the FastAPI application starts.
    """
    global query_engine
    try:
        query_engine = initialize_query_engine()
        print("FastAPI app and Virtual TA engine initialized successfully.")
    except Exception as e:
        print(f"Application startup failed: {e}")
        # Depending on your environment, you might want to exit here if core component fails
        # import sys
        # sys.exit(1)


# --- API Endpoint Definition ---
# --- CHANGED ENDPOINT FROM "/api/" TO "/" ---
@app.post("/", response_model=QueryResponse)
async def query_ta(request: QueryRequest):
    """
    Endpoint to ask a question to the Virtual TA.
    """
    if not query_engine:
        raise HTTPException(status_code=503, detail="Virtual TA engine not initialized. Please try again later.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Query the RAG engine
        response = query_engine.query(question)

        # Extracting sources (for links)
        source_nodes = response.source_nodes
        links = []
        for node in source_nodes:
            metadata = node.metadata
            url = metadata.get('url')
            # Prioritize 'title' from Markdown frontmatter, fallback to 'topic_title' for Discourse
            # Also check for 'text' from the YAML schema if that's what's available
            title = metadata.get('title') or metadata.get('topic_title') or metadata.get('text') or url or "No Title Available"

            if url:
                links.append(SourceLink(url=url, title=title))

        return QueryResponse(answer=str(response), links=links)

    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your question: {e}")

# --- Root Endpoint (Optional, for basic health check) ---
@app.get("/")
async def root():
    return {"message": "Virtual TA API is running. Go to /docs for API documentation."}