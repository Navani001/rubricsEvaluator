import os
import subprocess
import sys
import time
import json
import uuid
from typing import Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from cerebras.cloud.sdk import Cerebras
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL, HF_TOKEN, CEREBRAS_API_KEY

# Global variables for Pinecone connection
pc = None
index = None
embedding_model = None
_initialization_lock = False
_max_retries = 3
_retry_delay = 2  # seconds

def initialize_pinecone(force_reinit=False):
    """Initialize Pinecone client and embedding model with retry logic."""
    global pc, index, embedding_model, _initialization_lock
    
    if _initialization_lock:
        # Wait if another thread is initializing
        time.sleep(0.5)
        return
    
    if pc is not None and not force_reinit:
        return  # Already initialized
    
    _initialization_lock = True
    try:
        # Initialize Pinecone with retries
        for attempt in range(_max_retries):
            try:
                if PINECONE_API_KEY:
                    pc = Pinecone(api_key=PINECONE_API_KEY)
                    index = pc.Index(PINECONE_INDEX_NAME)
                    print(f"🔗 Connected to Pinecone index: {PINECONE_INDEX_NAME}")
                    break
            except Exception as e:
                if attempt < _max_retries - 1:
                    print(f"⚠️  Pinecone connection attempt {attempt + 1} failed, retrying in {_retry_delay}s...")
                    time.sleep(_retry_delay)
                else:
                    print(f"❌ Failed to connect to Pinecone after {_max_retries} attempts: {e}")
        
        # Initialize embedding model
        if embedding_model is None:
            try:
                if HF_TOKEN:
                    os.environ['HF_TOKEN'] = HF_TOKEN
                embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print(f"🤖 Loaded embedding model: {EMBEDDING_MODEL}")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
    finally:
        _initialization_lock = False

def get_index():
    """Get Pinecone index, initializing if needed."""
    global index
    if index is None:
        initialize_pinecone()
    return index

def get_embedding_model():
    """Get embedding model, initializing if needed."""
    global embedding_model
    if embedding_model is None:
        initialize_pinecone()
    return embedding_model

def store_features_in_pinecone(filename: str, img_path: str, features: dict, image_type: str):
    """Store extracted features in Pinecone database with error handling."""
    global index, embedding_model
    
    index = get_index()
    embedding_model = get_embedding_model()
    
    if index is None or embedding_model is None:
        print("⚠️ Pinecone not initialized, skipping feature storage")
        return None
    
    try:
        # Create a text representation of features for embedding
        feature_text = f"""
        Image Type: {image_type}
        Boxes: {features.get('boxes', 0)}
        Diamonds: {features.get('diamonds', 0)}
        Ellipses: {features.get('ellipses', 0)}
        Parallelograms: {features.get('parallelograms', 0)}
        Arrows: {features.get('arrows', 0)}
        Connections: {features.get('connections', 0)}
        Text: {features.get('text', '')[:500]}
        """.strip()
        
        # Generate embedding
        embedding = embedding_model.encode(feature_text).tolist()
        
        # Create metadata
        metadata = {
            "filename": filename,
            "image_path": img_path,
            "image_type": image_type,
            "boxes": int(features.get('boxes', 0)),
            "diamonds": int(features.get('diamonds', 0)),
            "ellipses": int(features.get('ellipses', 0)),
            "parallelograms": int(features.get('parallelograms', 0)),
            "arrows": int(features.get('arrows', 0)),
            "connections": int(features.get('connections', 0)),
            "text_density": float(features.get('text_density', 0)),
            "branching_depth": int(features.get('branching_depth', 0)),
            "extracted_text": features.get('text', '')[:1000],
            "pseudocode_indicators": json.dumps(features.get('pseudocode_indicators', {})),
            "shape_texts": json.dumps(features.get('shape_texts', {}))
        }
        
        # Create unique ID
        feature_id = str(uuid.uuid4())
        
        # Upsert to Pinecone with retry logic
        for attempt in range(_max_retries):
            try:
                vector = {"id": feature_id, "values": embedding, "metadata": metadata}
                index.upsert(vectors=[vector])
                print(f"✅ Stored features in Pinecone with ID: {feature_id}")
                return feature_id
            except Exception as e:
                if attempt < _max_retries - 1:
                    print(f"⚠️  Failed to store features (attempt {attempt + 1}), retrying...")
                    time.sleep(_retry_delay)
                else:
                    print(f"❌ Failed to store features after {_max_retries} attempts: {e}")
                    return None
    except Exception as e:
        print(f"❌ Error preparing features for Pinecone: {e}")
        return None

def get_existing_collection():
    global pc, index, embedding_model
    
    try:
        initialize_pinecone()
        index = get_index()
        if index is None:
            return False
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"✅ Connected to existing index with {stats.total_vector_count} documents")
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Index not found, running document processing...")
        try:
            result = subprocess.run([sys.executable, "document_pinecone.py"], timeout=600)
            if result.returncode == 0:
                initialize_pinecone(force_reinit=True)
                return True
        except Exception:
            pass
        return False

def get_available_books():
    global pc, index, embedding_model
    
    if not get_existing_collection():
        return []
    
    try:
        embedding_model = get_embedding_model()
        index = get_index()
        sample_query = embedding_model.encode("book").tolist()
        results = index.query(vector=sample_query, top_k=1000, include_metadata=True)
        
        books = set()
        for match in results.matches:
            if 'book' in match.metadata:
                books.add(match.metadata['book'])
        return list(books)
    except Exception as e:
        print(f"❌ Error getting available books: {e}")
        return []

def quizz_collection(book=None, n_results=3, question=10):
    
    global pc, index, embedding_model
    
    if not get_existing_collection():
        raise Exception("No data available in Pinecone index")
    
    embedding_model = get_embedding_model()
    index = get_index()
    query_embedding = embedding_model.encode("topics topic").tolist()
    
    query_params = {"vector": query_embedding, "top_k": n_results, "include_metadata": True}
    if book:
        query_params["filter"] = {"book": book}
        print(f"🔍 Searching in book: {book}")
    
    results = index.query(**query_params)
    print(f"📊 Found {len(results.matches)} relevant chunks")
    
    contexts = [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
    context = "\n\n".join(contexts)
    
    client = Cerebras(api_key=CEREBRAS_API_KEY)
    question_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "topic": {"type": "string"},
                        "answer": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
        },
        "additionalProperties": False
    }
    system_prompt = """You are a helpful assistant for educational books. give me ${question} quizz questions. you generate a quizz question with 4 options and also provide the correct answer. Always cite which book the information comes from when possible and also don't include based on context liked."""

    user_prompt = f"""
topics from books: {context}
Please provide a helpful quiz question with 4 options and the correct answer based on the topic above, create {question} questions based on the topic and also don't include any personal opinions or information not contained in the context and also don't include based on context liked"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-4-scout-17b-16e-instruct",
        response_format={
            "type": "json_schema", 
            "json_schema": {
                "name": "question_schema",
                "strict": True,
                "schema": question_schema
            }
        }
    )

    return {
        "message": json.loads(chat_completion.choices[0].message.content),
        "available_books": get_available_books(),
        "query_book_filter": book
    }
def query_collection(query, message=None, book=None, n_results=3):
    
    global pc, index, embedding_model
    
    if not get_existing_collection():
        raise Exception("No data available in Pinecone index")
    
    embedding_model = get_embedding_model()
    index = get_index()
    query_embedding = embedding_model.encode(query).tolist()
    
    query_params = {"vector": query_embedding, "top_k": n_results, "include_metadata": True}
    if book:
        query_params["filter"] = {"book": book}
        print(f"🔍 Searching in book: {book}")
    
    results = index.query(**query_params)
    print(f"📊 Found {len(results.matches)} relevant chunks")
    
    contexts = [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
    context = "\n\n".join(contexts)
    
    client = Cerebras(api_key=CEREBRAS_API_KEY)

    system_prompt = """You are a helpful assistant for educational books. Use the provided context to answer accurately. Always cite which book the information comes from when possible and also don't include based on context liked."""

    user_prompt = f"""Question: {query}
Context from books: {context}
Please provide a helpful answer based on the context above and also don't include any personal opinions or information not contained in the context and also don't include based on context liked"""
    if(message):
        message = [{"role": "system", "content": system_prompt}] + message + [{"role": "user", "content": user_prompt}]
    
    print("messages to send to backend: " + str(message))
    chat_completion = client.chat.completions.create(
        messages=message,
        model="llama-4-scout-17b-16e-instruct",
    )
    
    sources_info = [{
        "id": match.id,
        "book": match.metadata.get('book', 'unknown'),
        "page": match.metadata.get('page_number', 'unknown'),
        "score": match.score
    } for match in results.matches]
    
    return {
        "message": chat_completion.choices[0].message.content,
        "sources": sources_info,
        "available_books": get_available_books(),
        "query_book_filter": book
    }