"""
Image preprocessing and feature extraction functions
"""
import cv2
import numpy as np
import pytesseract
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Image processing constants
BLUR_KERNEL = (5, 5)
CANNY_THRESHOLD1, CANNY_THRESHOLD2 = 50, 150
CONTOUR_EPSILON = 0.04
BOX_RATIO_MIN, BOX_RATIO_MAX = 0.8, 1.2
HOUGH_THRESHOLD, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP = 100, 50, 10


def preprocess_image(img_path):
    """
    Preprocess image for feature extraction.
    
    Args:
        img_path: Path to image file
        
    Returns:
        Tuple of (original_image, grayscale, edges)
    """
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresh, 100, 200)
    
    return img, gray, edges


def extract_features(img_path, detect_arrowheads, detect_flow_connections, 
                    detect_pseudocode_patterns, detect_branching_structure,
                    get_shape_regions, extract_text_regions, find_text_to_shape_mapping,
                    extract_text_from_shapes, find_arrow_connections, create_element_map):
    """
    Extract all features from an image.
    
    Args:
        img_path: Path to image
        detect_* and extract_*: Helper functions for feature detection
        
    Returns:
        Dictionary with extracted features
    """
    img, gray, edges = preprocess_image(img_path)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Count basic shapes
    boxes = 0
    diamonds = 0
    ellipses = 0
    parallelograms = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        
        if area < 100:
            continue
            
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.8 < ratio < 1.2:
                diamonds += 1
            else:
                boxes += 1
        elif len(approx) >= 8:
            ellipses += 1
        elif len(approx) == 6:
            parallelograms += 1
    
    # Detect arrows and connections
    arrowheads = detect_arrowheads(edges)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120,
        minLineLength=80,
        maxLineGap=15
    )
    line_count = 0 if lines is None else len(lines)
    arrows = min(arrowheads, line_count)

    # Extract text
    text = ""
    text_density = 0
    try:
        text = pytesseract.image_to_string(gray)
        text_density = len(text) / max(1, gray.shape[0] * gray.shape[1])
        logger.info(f"Extracted {len(text)} characters from {img_path}")
    except pytesseract.TesseractNotFoundError as e:
        logger.error(f"TesseractNotFoundError: {str(e)}")
        text = ""
    except Exception as e:
        logger.error(f"OCR error for {img_path}: {type(e).__name__}: {str(e)}")
        text = ""

    # Detect patterns and structures
    connections = detect_flow_connections(contours)
    pseudocode_indicators = detect_pseudocode_patterns(text)
    branching_depth = detect_branching_structure(contours)
    
    # Extract spatial relationships
    shapes = get_shape_regions(contours, gray)
    text_regions = extract_text_regions(gray)
    text_to_shape_map = find_text_to_shape_mapping(shapes, text_regions)
    
    # Extract text from shapes (only if full OCR didn't find much)
    shape_texts = {}
    if len(text.strip()) < 100:
        shape_texts = extract_text_from_shapes(gray, shapes)
    
    # Find connections
    arrow_connections = find_arrow_connections(edges, shapes)
    element_map = create_element_map(img_path, shapes, text_regions, 
                                    text_to_shape_map, arrow_connections, shape_texts)

    return {
        "boxes": boxes,
        "diamonds": diamonds,
        "ellipses": ellipses,
        "parallelograms": parallelograms,
        "arrows": arrows,
        "text": text,
        "text_density": text_density,
        "connections": connections,
        "pseudocode_indicators": pseudocode_indicators,
        "branching_depth": branching_depth,
        "element_map": element_map,
        "text_to_shape_mapping": text_to_shape_map,
        "arrow_connections": arrow_connections,
        "shape_texts": shape_texts
    }
