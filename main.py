"""
Main FastAPI application
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

from models import PDFRequest, QuizzRequest
from routes import handle_pdf_upload
from query_pinecone import initialize_pinecone, query_collection, get_available_books, quizz_collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Initialize Pinecone on startup, cleanup on shutdown.
    """
    logger.info("🚀 Starting wemakedev API...")
    initialize_pinecone()
    yield
    logger.info("🛑 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="WeMakeDev RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files (if needed)
# IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
# app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ========================
# Health Check Routes
# ========================

@app.get('/')
async def index():
    """Root endpoint"""
    return {'message': 'WeMakeDev RAG API', 'version': '1.0.0'}


@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy'}


# ========================
# PDF Processing Routes
# ========================

@app.post('/features')
async def extract_pdf_features(file: UploadFile = File(...)):
    """
    Extract features from uploaded PDF.
    
    Args:
        file: PDF file to process
        
    Returns:
        Analysis results with extracted features
    """
    return await handle_pdf_upload(file)


# ========================
# Query Routes
# ========================

@app.post('/query')
async def query_documents(request: PDFRequest):
    """
    Query documents in Pinecone.
    
    Args:
        request: PDFRequest with query parameters
        
    Returns:
        Query results with relevant documents
    """
    try:
        result = query_collection(
            query=request.query,
            message=request.message if request.message else None,
            book=request.book,
            n_results=request.n_results or 3
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get('/books')
async def list_available_books():
    """
    Get list of available books in Pinecone.
    
    Returns:
        List of book names
    """
    try:
        books = get_available_books()
        return {"status": "success", "books": books}
    except Exception as e:
        logger.error(f"Error getting books: {str(e)}")
        return {"status": "error", "message": str(e)}


# ========================
# Quiz Routes
# ========================

@app.post('/quiz')
async def generate_quiz(request: QuizzRequest):
    """
    Generate quiz questions from documents.
    
    Args:
        request: QuizzRequest with quiz parameters
        
    Returns:
        Generated quiz questions
    """
    try:
        result = quizz_collection(
            book=request.book,
            n_results=request.n_results or 3,
            question=request.question or 10
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Quiz generation error: {str(e)}")
        return {"status": "error", "message": str(e)}


# ========================
# Application Entry Point
# ========================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
