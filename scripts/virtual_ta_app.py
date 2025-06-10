import os
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "tds_virtual_ta_collection"

# Ensure OPENAI_API_KEY is set.
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file or bashrc.")
# Ensure OPENAI_BASE_URL is set, especially for custom endpoints.
if not os.getenv("OPENAI_BASE_URL"):
    print("Warning: OPENAI_BASE_URL environment variable not set. OpenAIEmbedding/OpenAI LLM will use default base URL.")


def get_query_engine():
    """
    Loads the persisted ChromaDB index and sets up the RAG query engine.
    """
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH} and collection: {COLLECTION_NAME}")

    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get the existing collection
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # Configure the vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Configure the storage context to load from the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Configure the embedding model (same as used for indexing)
    # We explicitly set 'api_base' to use your custom endpoint (aipipe.org).
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_base=os.getenv("OPENAI_BASE_URL")
    )

    # Configure the LLM for response generation
    # We explicitly set 'api_base' to use your custom endpoint (aipipe.org).
    llm = OpenAI(
        model="gpt-3.5-turbo", # You can choose other models like "gpt-4" if you have access
        api_base=os.getenv("OPENAI_BASE_URL"),
        temperature=0.1 # Lower temperature for more factual, less creative answers
    )

    # Load the index from the persisted storage
    # Note: For ChromaDB, `load_index_from_storage` effectively re-initializes
    # the index pointers to the existing ChromaDB collection.
    print("Loading VectorStoreIndex from ChromaDB...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model # Ensure the same embedding model is used
    )
    print("VectorStoreIndex loaded.")

    # Create a query engine
    # Setting similarity_top_k determines how many top-k relevant chunks are retrieved
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5 # Retrieve top 5 most relevant chunks
    )
    return query_engine


if __name__ == "__main__":
    print("--- Virtual TA Application Starting ---")
    query_engine = get_query_engine()
    print("\nVirtual TA is ready! Ask your questions about the course content or Discourse forum.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting Virtual TA. Goodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue

        try:
            # Query the RAG engine
            response = query_engine.query(question)
            print("\nVirtual TA says:")
            print(str(response)) # Print the response content
        except Exception as e:
            print(f"An error occurred during query: {e}")
            print("Please ensure your API key and base URL are correct and the service is operational.")