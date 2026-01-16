
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
import uuid
import cv2
import pytesseract
import numpy as np
import pdfplumber
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import uvicorn
import fitz  # PyMuPDF
from query_pinecone import query_collection, get_available_books, quizz_collection

# Configure pytesseract path for Windows (update this path if needed)
pytesseract.pytesseract.tesseract_cmd  = r'c:\Program Files\Tesseract-OCR\tesseract.exe'


# Constants
UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx'}
BLUR_KERNEL = (5, 5)
CANNY_THRESHOLD1, CANNY_THRESHOLD2 = 50, 150
CONTOUR_EPSILON = 0.04
BOX_RATIO_MIN, BOX_RATIO_MAX = 0.8, 1.2
HOUGH_THRESHOLD, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP = 100, 50, 10


class PDFRequest(BaseModel):
    query: str
    book: Optional[str] = None
    message: list[dict] = []
    n_results: Optional[int] = 3
class QuizzRequest(BaseModel):
    book: Optional[str] = None
    n_results: Optional[int] = 3
    question: Optional[str] = 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting wemakedev API...")
    yield
    logger.info("🛑 Shutting down API...")

app = FastAPI(
    title="WeMakeDev RAG API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get('/')
async def index():
    return {'message': 'WeMakeDev RAG API', 'version': '1.0.0'}

@app.get('/health')
async def health_check():
    return {'status': 'healthy'}




# ---------------------------
# STEP 3: PREPROCESS IMAGE
# ---------------------------
def preprocess_image(img_path):
    """Preprocess image for feature extraction."""
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresh, 100, 200)

    # blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    # edges = cv2.Canny(blur, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    return img, gray, edges


# ---------------------------
# STEP 4: EXTRACT BASIC FEATURES
# ---------------------------
def extract_features(img_path):
    img, gray, edges = preprocess_image(img_path)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = 0
    diamonds = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.8 < ratio < 1.2:
                diamonds += 1
            else:
                boxes += 1
    
    # lines = cv2.HoughLinesP(
    #     edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    # )
    # arrows = 0 if lines is None else len(lines)
    arrowheads = detect_arrowheads(edges)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120,
        minLineLength=80,
        maxLineGap=15
    )

    line_count = 0 if lines is None else len(lines)

    arrows = min(arrowheads, line_count)

    # Extract text with error handling for missing Tesseract
    text = ""
    try:
        text = pytesseract.image_to_string(gray)
        logger.info(f"Extracted {len(text)} characters from {img_path}")
    except pytesseract.TesseractNotFoundError as e:
        logger.error(f"TesseractNotFoundError: {str(e)}")
        logger.error(f"Configured path: {pytesseract.pytesseract.pytesseract_cmd}")
        text = ""
    except Exception as e:
        logger.error(f"OCR error for {img_path}: {type(e).__name__}: {str(e)}")
        text = ""

    return {
        "boxes": boxes,
        "diamonds": diamonds,
        "arrows": arrows,
        "text": text
    }


# ---------------------------
# STEP 5: CLASSIFY IMAGE TYPE
# ---------------------------
def classify_image(features):
    if features["diamonds"] > 0 and features["connections"] > 0:
        return "flowchart"
    elif features["boxes"] > 2 and features["arrows"] > 0:
        return "architecture"
    else:
        return "example"
def detect_arrowheads(edges):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    arrowheads = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)

        # Triangle → possible arrowhead
        if len(approx) == 3:
            area = cv2.contourArea(cnt)
            if area > 50:  # filter noise
                arrowheads += 1

    return arrowheads

def detect_flow_connections(contours):
    boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            boxes.append(cv2.boundingRect(approx))

    connections = 0
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[j]

            # vertical flow
            if abs(x1 - x2) < 50 and y2 > y1:
                connections += 1

    return connections


@app.post('/features')
async def query_pdf(file: UploadFile):
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Validate file extension
        allowed_extensions = {'.pdf', '.txt', '.doc', '.docx'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")
        
        # Generate unique filename to prevent overwrites
        unique_filename = f"{uuid.uuid4()}_{Path(file.filename).name}"
        file_path = upload_dir / unique_filename
        
        # Save file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File saved: {file_path}")
        
        # Extract text and images
        text_content = ""
        images_extracted = []
        
        try:
            # Extract text with pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
            
            # Extract images with PyMuPDF in optimized way
            doc = fitz.open(file_path)
            images_dir = upload_dir / unique_filename.replace('.pdf', '_images')
            
            for page_no in range(len(doc)):
                page = doc[page_no]
                images = page.get_images(full=True)
                
                if images:
                    images_dir.mkdir(exist_ok=True)
                
                for img_no, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save to organized subdirectory
                        img_filename = f"page_{page_no}_img_{img_no}.png"
                        img_path = images_dir / img_filename
                        
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        
                        images_extracted.append(str(img_path))
                    except Exception as img_err:
                        logger.warning(f"Failed to extract image: {img_err}")
            
            doc.close()
        except Exception as pdf_err:
            logger.error(f"PDF processing error: {str(pdf_err)}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(pdf_err)}")
        # image_paths = extract_images(pdf_path)

        results = []

        for img_path in images_extracted:
            features = extract_features(img_path)
            img_type = classify_image(features)

            results.append({
                "image": img_path,
                "type": img_type,
                "features": features
            })

        
        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "saved_as": unique_filename,
            "file_size": len(contents),
            "path": str(file_path),
            "content": text_content,
            "images_extracted": len(images_extracted),
            "image_paths": images_extracted,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
