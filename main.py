"""
Main FastAPI application
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import tempfile
import uuid

from models import PDFRequest, QuizzRequest, RubricGenRequest
from routes import handle_pdf_upload, extract_images_from_pdf, process_image
from query_pinecone import initialize_pinecone, query_collection, get_available_books, quizz_collection
from evaluation import evaluate_documents_sync, generate_rubrics_from_text

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
# Evaluation Routes
# ========================

@app.post('/evaluate')
async def evaluate_documents(
    files: list[UploadFile] = File(...),
    rubrics: str = Form(...)
):
    """
    Evaluate documents against custom rubrics.
    
    Args:
        files: PDF/DOCX files to evaluate
        rubrics: JSON string containing evaluation rubric(s)
        
    Returns:
        Evaluation results with scores and reasoning
    """
    try:
        # Parse rubrics JSON
        rubric_data = json.loads(rubrics)
        
        # Prepare uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        # Store files in uploads directory
        saved_files = []
        document_data = {}
        image_analysis_data = {}
        
        for file in files:
            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}_{Path(file.filename).name}"
            file_path = upload_dir / unique_filename
            
            # Save uploaded file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
                
            saved_files.append(file_path)
            document_data[file.filename] = str(file_path)
                
            # Extract and process images
            try:
                images = await extract_images_from_pdf(file_path, upload_dir, unique_filename)
                
                file_img_results = []
                for img_path in images:
                    res = await process_image(file.filename, len(content), img_path)
                    if res:
                        file_img_results.append(res)
                
                image_analysis_data[file.filename] = file_img_results
            except Exception as img_err:
                logger.warning(f"Image processing failed for evaluation: {img_err}")
                image_analysis_data[file.filename] = []
        
        # Run evaluation (sync function)
        evaluation_results = await asyncio.to_thread(
            evaluate_documents_sync,
            document_data,
            rubric_data,
            image_analysis_data
        )
        
        # Files are intentionally kept in 'uploads' directory
        
        return {
            "status": "success",
            "evaluations": evaluation_results,
            "saved_files": [str(p) for p in saved_files]
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid rubrics JSON: {str(e)}")
        return {"status": "error", "message": f"Invalid rubrics JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post('/generate-rubrics')
async def generate_rubrics(request: RubricGenRequest):
    """
    Generate evaluation rubrics from provided text.
    
    Args:
        request: RubricGenRequest with text content
        
    Returns:
        Generated rubrics in JSON format
    """
    try:
        result = await asyncio.to_thread(
            generate_rubrics_from_text,
            request.text,
            request.topic
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Rubric generation error: {str(e)}")
        return {"status": "error", "message": str(e)}


# ========================
# Application Entry Point
# ========================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
