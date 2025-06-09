import os
import json
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.file import MarkdownReader # To handle markdown files
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file.
# This ensures OPENAI_API_KEY and OPENAI_BASE_URL are available.
load_dotenv()

# --- Configuration ---
# These paths are relative to your project's root directory.
DATA_DIR = "data"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "tds_virtual_ta_collection"

# Ensure OPENAI_API_KEY is set.
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file or bashrc.")
# Ensure OPENAI_BASE_URL is set, especially for custom endpoints.
if not os.getenv("OPENAI_BASE_URL"):
    print("Warning: OPENAI_BASE_URL environment variable not set. OpenAIEmbedding will use default base URL.")

def load_documents_with_metadata(data_dir):
    """
    Loads documents from specified directories with custom metadata.
    Handles Markdown files for course content and JSON Lines for Discourse.
    """
    all_documents = []

    # 1. Load Course Content (Markdown files)
    course_content_path = os.path.join(data_dir, "course_content")
    if os.path.exists(course_content_path):
        print(f"Loading course content from: {course_content_path}")
        # Use SimpleDirectoryReader with MarkdownReader for .md files
        reader = SimpleDirectoryReader(
            input_dir=course_content_path,
            file_extractor={".md": MarkdownReader()},
            recursive=True
        )
        course_docs = reader.load_data()

        # Add 'source_type' metadata
        for doc in course_docs:
            doc.metadata["source_type"] = "course_content"
            # MarkdownReader typically parses YAML frontmatter into metadata automatically.
            # If your Markdown files have 'title' and 'original_url' in frontmatter,
            # they should appear in doc.metadata.
            all_documents.append(doc)
        print(f"Loaded {len(course_docs)} course content documents.")
    else:
        print(f"Course content directory not found: {course_content_path}. Skipping.")


    # 2. Load Discourse Posts (JSON Lines or JSON array)
    discourse_posts_path = os.path.join(data_dir, "discourse_posts")
    if os.path.exists(discourse_posts_path):
        print(f"Loading Discourse posts from: {discourse_posts_path}")
        for filename in os.listdir(discourse_posts_path):
            if filename.endswith(".jsonl") or filename.endswith(".json"):
                filepath = os.path.join(discourse_posts_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    try:
                        # Try to load as JSON Lines first (for discourse_posts_*.jsonl)
                        lines_processed = 0
                        is_jsonl = False
                        for line_num, line in enumerate(file_content.splitlines()):
                            if line.strip(): # Skip empty lines
                                try:
                                    data = json.loads(line)
                                    is_jsonl = True
                                    lines_processed += 1

                                    # --- DEBUG PRINTS FOR TAGS ---
                                    tags_value = data.get('tags')
                                    # print(f"DEBUG (Line {line_num+1}): Original tags value: {tags_value}, Type: {type(tags_value)}") # Uncomment for very verbose debug
                                    
                                    # Convert list of tags to a comma-separated string for ChromaDB compatibility
                                    processed_tags = ", ".join(tags_value) if isinstance(tags_value, list) else (tags_value if tags_value is not None else '')
                                    # print(f"DEBUG (Line {line_num+1}): Processed tags value: {processed_tags}, Type: {type(processed_tags)}") # Uncomment for very verbose debug
                                    # --- END DEBUG PRINTS ---

                                    doc = Document(
                                        text=data.get('content') or data.get('cooked', ''), # Prioritize 'content' if cleaned, fallback to 'cooked' from raw Discourse JSON
                                        metadata={
                                            "url": data.get('url', ''),
                                            "topic_title": data.get('topic_title', ''),
                                            "post_id": data.get('post_id', ''),
                                            "post_number": data.get('post_number', ''),
                                            "author": data.get('author', ''),
                                            "created_at": data.get('created_at', ''),
                                            "source_type": "discourse_post",
                                            "tags": processed_tags # Use the processed_tags variable here
                                        }
                                    )
                                    if doc.text.strip(): # Only add if text content is not empty
                                        all_documents.append(doc)
                                    else:
                                        print(f"Warning: Empty text content for Discourse post at line {line_num+1} in {filename}. Skipping.")
                                except json.JSONDecodeError:
                                    # If the first line failed as JSON, it's probably not JSONL
                                    if lines_processed == 0 and line_num == 0:
                                        is_jsonl = False
                                        break # Exit this loop and try as single JSON array
                                    else:
                                        raise # Re-raise if it's not the first line and we thought it was JSONL
                                except Exception as e:
                                    print(f"Error processing Discourse post on line {line_num+1} in {filename}: {e}")
                                    
                        # If not JSON Lines, try to load as a single JSON array (e.g., from discourse_posts.json)
                        if not is_jsonl:
                            print(f"Attempting to load {filename} as a single JSON array...")
                            json_array = json.loads(file_content)
                            if isinstance(json_array, list):
                                for item_num, data in enumerate(json_array):
                                    # --- DEBUG PRINTS FOR TAGS ---
                                    tags_value = data.get('tags')
                                    # print(f"DEBUG (Item {item_num+1}): Original tags value: {tags_value}, Type: {type(tags_value)}") # Uncomment for very verbose debug
                                    
                                    processed_tags = ", ".join(tags_value) if isinstance(tags_value, list) else (tags_value if tags_value is not None else '')
                                    # print(f"DEBUG (Item {item_num+1}): Processed tags value: {processed_tags}, Type: {type(processed_tags)}") # Uncomment for very verbose debug
                                    # --- END DEBUG PRINTS ---

                                    doc = Document(
                                        text=data.get('content') or data.get('cooked', ''),
                                        metadata={
                                            "url": data.get('url', ''),
                                            "topic_title": data.get('topic_title', ''),
                                            "post_id": data.get('post_id', ''),
                                            "post_number": data.get('post_number', ''),
                                            "author": data.get('author', ''),
                                            "created_at": data.get('created_at', ''),
                                            "source_type": "discourse_post",
                                            "tags": processed_tags # Use the processed_tags variable here
                                        }
                                    )
                                    if doc.text.strip():
                                        all_documents.append(doc)
                                    else:
                                        print(f"Warning: Empty text content for Discourse post at item {item_num+1} in {filename}. Skipping.")
                                print(f"Loaded {len([d for d in all_documents if d.metadata.get('source_type') == 'discourse_post'])} Discourse post documents from array.")
                            else:
                                print(f"Error: {filename} is a single JSON object but not an array. Skipping.")
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing {filename} as JSON or JSON Lines: {e}")
                    except Exception as e:
                        print(f"Error processing Discourse file {filename}: {e}")
    else:
        print(f"Discourse posts directory not found: {discourse_posts_path}. Skipping.")

    print(f"Loaded a total of {len(all_documents)} documents from all sources.")
    return all_documents


def create_and_persist_index(documents):
    """
    Chunks documents, generates embeddings, and stores them in ChromaDB.
    """
    if not documents:
        print("No documents to index. Exiting.")
        return

    # [ ] Chunking: Break down your scraped content into smaller chunks.
    # Adjust chunk_size and chunk_overlap based on your data and LLM context window.
    # 1024 tokens is a common chunk size, with 200 tokens of overlap for context.
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f"Split documents into {len(nodes)} nodes (chunks).")

    # [ ] Embedding Generation: Use OpenAI's text-embedding-ada-002 model.
    # We explicitly set 'api_base' to use your custom endpoint (aipipe.org).
    # LlamaIndex will automatically pick up OPENAI_API_KEY from environment variables.
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_base=os.getenv("OPENAI_BASE_URL")
    )
    print(f"Initialized OpenAI Embedding model (text-embedding-ada-002) using API base: {os.getenv('OPENAI_BASE_URL')}.")

    # [ ] Vector Database Setup: Initialize ChromaDB for persistence.
    print(f"Initializing ChromaDB client at: {CHROMA_DB_PATH}")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get or create the collection where embeddings will be stored.
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    print(f"Using ChromaDB collection: {COLLECTION_NAME}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create StorageContext which links to your vector store.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the VectorStoreIndex. This process generates embeddings for all nodes
    # and stores them in the configured ChromaDB collection.
    print("Creating VectorStoreIndex (this will generate embeddings and add to ChromaDB)...")
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context
    )
    print("VectorStoreIndex created and persisted to ChromaDB.")
    return index

if __name__ == "__main__":
    print("--- Starting data processing and indexing pipeline ---")

    # Ensure output directories for data and DB exist before processing
    os.makedirs(os.path.join(DATA_DIR, "course_content"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "discourse_posts"), exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True) # Ensure ChromaDB path exists

    documents = load_documents_with_metadata(DATA_DIR)

    if documents:
        create_and_persist_index(documents)
        print("--- Indexing complete. ChromaDB should be populated. ---")
    else:
        print("No documents found to index. Please ensure your scraping scripts ran successfully and populated the 'data/' directory.")
        print("You can check the 'data/course_content/' and 'data/discourse_posts/' directories.")