"""
API route handlers
"""
import os
import logging
from pathlib import Path
import uuid
import pdfplumber
import fitz
from fastapi import HTTPException, UploadFile

from image_processing import extract_features
from image_analysis import classify_image
from shape_analysis import build_flow_paths
from query_pinecone import store_features_in_pinecone

logger = logging.getLogger(__name__)


async def handle_pdf_upload(file: UploadFile):
    """
    Handle PDF file upload and extract features.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Create uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Validate file
        allowed_extensions = {'.pdf', '.txt', '.doc', '.docx'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")
        
        # Save file
        unique_filename = f"{uuid.uuid4()}_{Path(file.filename).name}"
        file_path = upload_dir / unique_filename
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File saved: {file_path}")
        
        # Extract images
        images_extracted = await extract_images_from_pdf(file_path, upload_dir, unique_filename)
        
        # Process images
        results = []
        for img_path in images_extracted:
            result = await process_image(file.filename, len(contents), img_path)
            if result:
                results.append(result)
        
        return {
            "status": "success",
            "file": file.filename,
            "file_size": len(contents),
            "diagrams_found": len(images_extracted),
            "analysis": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


async def extract_images_from_pdf(file_path, upload_dir, unique_filename):
    """
    Extract images from PDF using PyMuPDF.
    
    Args:
        file_path: Path to PDF file
        upload_dir: Directory to save images
        unique_filename: Base filename for organizing images
        
    Returns:
        List of image paths
    """
    images_extracted = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                _ = page.extract_text() or ""
        
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
    
    return images_extracted


async def process_image(filename, file_size, img_path):
    """
    Process single image and extract features.
    
    Args:
        filename: Original file name
        file_size: File size in bytes
        img_path: Path to image
        
    Returns:
        Dictionary with image analysis
    """
    try:
        # Import detection functions here to avoid circular imports
        from element_detection import (
            detect_arrowheads, detect_flow_connections,
            detect_branching_structure, get_shape_regions,
            extract_text_regions, find_arrow_connections
        )
        from image_analysis import detect_pseudocode_patterns
        from shape_analysis import find_text_to_shape_mapping, extract_text_from_shapes, create_element_map
        
        # Extract features
        features = extract_features(
            img_path,
            detect_arrowheads,
            detect_flow_connections,
            detect_pseudocode_patterns,
            detect_branching_structure,
            get_shape_regions,
            extract_text_regions,
            find_text_to_shape_mapping,
            extract_text_from_shapes,
            find_arrow_connections,
            create_element_map
        )
        
        img_type = classify_image(features)
        
        # Store in Pinecone
        store_features_in_pinecone(filename, img_path, features, img_type)
        
        # Build response
        element_map = features.get("element_map", {})
        flow_paths = build_flow_paths(element_map, features.get("shape_texts", {}))
        
        return {
            "file": filename,
            "file_size": file_size,
            "diagram_type": img_type,
            "image": img_path,
            "summary": {
                "boxes": features["boxes"],
                "diamonds": features["diamonds"],
                "ellipses": features["ellipses"],
                "arrows": features["arrows"],
                "depth": features["branching_depth"]
            },
            "flow_paths": flow_paths
        }
        
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        return None
