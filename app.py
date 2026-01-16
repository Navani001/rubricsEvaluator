
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
    ellipses = 0
    parallelograms = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        
        # Filter out noise
        if area < 100:
            continue
            
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.8 < ratio < 1.2:
                diamonds += 1
            else:
                boxes += 1
        elif len(approx) >= 8:  # Ellipses have more vertices
            ellipses += 1
        elif len(approx) == 6:  # Parallelograms
            parallelograms += 1
    
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
    text_density = 0
    try:
        text = pytesseract.image_to_string(gray)
        text_density = len(text) / max(1, gray.shape[0] * gray.shape[1])
        logger.info(f"Extracted {len(text)} characters from {img_path}")
    except pytesseract.TesseractNotFoundError as e:
        logger.error(f"TesseractNotFoundError: {str(e)}")
        logger.error(f"Configured path: {pytesseract.pytesseract.pytesseract_cmd}")
        text = ""
    except Exception as e:
        logger.error(f"OCR error for {img_path}: {type(e).__name__}: {str(e)}")
        text = ""

    connections = detect_flow_connections(contours)
    pseudocode_indicators = detect_pseudocode_patterns(text)
    branching_depth = detect_branching_structure(contours)
    
    # Get detailed spatial relationships
    shapes = get_shape_regions(contours, gray)
    text_regions = extract_text_regions(gray)
    text_to_shape_map = find_text_to_shape_mapping(shapes, text_regions)
    
    # Add OCR text extraction from each shape region as fallback
    shape_texts = extract_text_from_shapes(gray, shapes)
    
    arrow_connections = find_arrow_connections(edges, shapes)
    element_map = create_element_map(img_path, shapes, text_regions, text_to_shape_map, arrow_connections, shape_texts)

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


# ---------------------------
# STEP 5: CLASSIFY IMAGE TYPE
# ---------------------------
def classify_image(features):
    """
    Classify image as flowchart, algorithm, pseudocode, or architecture.
    Uses multiple indicators for better accuracy.
    """
    diamonds = features.get("diamonds", 0)
    boxes = features.get("boxes", 0)
    arrows = features.get("arrows", 0)
    ellipses = features.get("ellipses", 0)
    connections = features.get("connections", 0)
    text_density = features.get("text_density", 0)
    pseudocode_indicators = features.get("pseudocode_indicators", {})
    branching_depth = features.get("branching_depth", 0)
    
    # Calculate confidence scores
    flowchart_score = 0
    algorithm_score = 0
    pseudocode_score = 0
    
    # Flowchart detection
    if diamonds > 0:
        flowchart_score += 30  # Decision diamonds are strong indicator
    if connections > 2:
        flowchart_score += 25
    if ellipses > 0:
        flowchart_score += 15  # Terminal/start-end shapes
    if arrows > 0:
        flowchart_score += 20
    
    # Algorithm detection (structured boxes with clear flow)
    if boxes > 2 and arrows > 1:
        algorithm_score += 30
    if branching_depth > 1:
        algorithm_score += 25
    if boxes > 0 and connections > 1:
        algorithm_score += 20
    if arrows > 2:
        algorithm_score += 15
    
    # Pseudocode detection (text-heavy with code patterns)
    pseudocode_score += pseudocode_indicators.get("keyword_count", 0) * 5
    pseudocode_score += pseudocode_indicators.get("indent_count", 0) * 4
    pseudocode_score += pseudocode_indicators.get("bracket_count", 0) * 3
    
    if text_density > 0.01:
        pseudocode_score += 25
    if pseudocode_indicators.get("has_loop_keywords", False):
        pseudocode_score += 20
    if pseudocode_indicators.get("has_condition_keywords", False):
        pseudocode_score += 20
    
    # Determine classification based on highest score
    scores = {
        "flowchart": flowchart_score,
        "algorithm": algorithm_score,
        "pseudocode": pseudocode_score,
        "unknown": 0
    }
    
    max_score = max(scores.values())
    
    # If no clear winner, return the type with highest score
    if max_score == 0:
        if text_density > 0.005:
            return "pseudocode"
        elif diamonds > 0:
            return "flowchart"
        elif boxes > 1:
            return "algorithm"
        else:
            return "unknown"
    
    # Return the classification with highest score
    return max(scores, key=scores.get)


def detect_pseudocode_patterns(text):
    """
    Detect pseudocode patterns in extracted text.
    Returns indicators like keyword count, indentation, brackets, etc.
    """
    indicators = {
        "keyword_count": 0,
        "indent_count": 0,
        "bracket_count": 0,
        "has_loop_keywords": False,
        "has_condition_keywords": False,
        "has_function_keywords": False
    }
    
    if not text:
        return indicators
    
    # Common pseudocode/algorithm keywords
    loop_keywords = ["for", "while", "do", "repeat", "foreach", "loop"]
    condition_keywords = ["if", "else", "elseif", "switch", "case", "then"]
    function_keywords = ["function", "procedure", "def", "method", "return", "main"]
    
    text_lower = text.lower()
    
    # Count keywords
    for keyword in loop_keywords:
        indicators["keyword_count"] += text_lower.count(keyword)
        if keyword in text_lower:
            indicators["has_loop_keywords"] = True
    
    for keyword in condition_keywords:
        indicators["keyword_count"] += text_lower.count(keyword)
        if keyword in text_lower:
            indicators["has_condition_keywords"] = True
    
    for keyword in function_keywords:
        indicators["keyword_count"] += text_lower.count(keyword)
        if keyword in text_lower:
            indicators["has_function_keywords"] = True
    
    # Count indentation levels (lines that start with whitespace)
    lines = text.split('\n')
    for line in lines:
        if line and line[0] in [' ', '\t']:
            indicators["indent_count"] += 1
    
    # Count brackets and braces
    indicators["bracket_count"] = text.count('(') + text.count(')') + \
                                 text.count('{') + text.count('}') + \
                                 text.count('[') + text.count(']')
    
    return indicators


def detect_branching_structure(contours):
    """
    Detect the depth and complexity of branching in the diagram.
    Analyzes spatial relationships between shapes to determine if there's
    nested or multiple branching paths.
    """
    if not contours:
        return 0
    
    boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            boxes.append((x, y, w, h))
    
    if len(boxes) < 2:
        return 0
    
    # Calculate vertical depth levels
    depth_levels = set()
    for x, y, w, h in boxes:
        # Group boxes into vertical levels (within 50 pixels)
        level = y // 50
        depth_levels.add(level)
    
    # Return the number of distinct depth levels (indicates branching depth)
    return len(depth_levels) - 1


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


def get_shape_regions(contours, gray):
    """
    Extract all shapes with their bounding boxes and centroids.
    Returns list of shape dictionaries with position and type info.
    """
    shapes = []
    shape_id = 0
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = float(cv2.contourArea(cnt))
        
        # Filter out noise
        if area < 100:
            continue
        
        x, y, w, h = cv2.boundingRect(approx)
        # Convert numpy types to Python native types
        x, y, w, h = int(x), int(y), int(w), int(h)
        cx, cy = int(x + w // 2), int(y + h // 2)  # centroid
        
        # Determine shape type
        if len(approx) == 4:
            ratio = w / float(h) if h > 0 else 0
            shape_type = "diamond" if 0.8 < ratio < 1.2 else "box"
        elif len(approx) >= 8:
            shape_type = "ellipse"
        elif len(approx) == 6:
            shape_type = "parallelogram"
        else:
            shape_type = "polygon"
        
        shapes.append({
            "id": shape_id,
            "type": shape_type,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area
        })
        shape_id += 1
    
    return shapes


def extract_text_regions(gray):
    """
    Extract text regions using contour analysis and morphological operations.
    Returns list of text region bounding boxes with more sensitive detection.
    """
    text_regions = []
    
    # Apply multiple morphological operations for better text detection
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Apply closing to connect broken text characters
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1, iterations=1)
    # Apply dilation to expand text regions
    dilated = cv2.dilate(closed, kernel2, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Convert numpy types to Python native types
        x, y, w, h = int(x), int(y), int(w), int(h)
        area = w * h
        
        # Relaxed filter: detect smaller text regions and larger text blocks
        # Min area: 20 (very small text), Max area: 100000 (large text blocks)
        # Width: up to 600, Height: up to 150 (allows for multi-line text)
        if 20 < area < 100000 and 5 < w < 600 and 5 < h < 150:
            text_regions.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": int(x + w // 2),
                "cy": int(y + h // 2),
                "area": area
            })
    
    return text_regions


def find_text_to_shape_mapping(shapes, text_regions):
    """
    Map text regions to their nearest shapes with relaxed distance thresholds.
    Returns dictionary mapping shape ids to contained/nearby text regions.
    """
    text_to_shape = {}
    
    for i, text in enumerate(text_regions):
        closest_shape = None
        min_distance = float('inf')
        
        for shape in shapes:
            # Calculate distance from text centroid to shape centroid
            dist = ((text["cx"] - shape["cx"]) ** 2 + (text["cy"] - shape["cy"]) ** 2) ** 0.5
            
            # Check if text is inside shape
            text_inside = (shape["x"] <= text["cx"] <= shape["x"] + shape["w"] and
                          shape["y"] <= text["cy"] <= shape["y"] + shape["h"])
            
            # Check if text is partially overlapping with shape
            text_overlaps = not (text["x"] + text["w"] < shape["x"] or 
                               text["x"] > shape["x"] + shape["w"] or
                               text["y"] + text["h"] < shape["y"] or
                               text["y"] > shape["y"] + shape["h"])
            
            # Prefer text inside shapes, then overlapping, then closest within distance
            if text_inside:
                min_distance = 0
                closest_shape = shape["id"]
                break
            elif text_overlaps:
                min_distance = 0
                closest_shape = shape["id"]
                break
            elif dist < min_distance and dist < 200:  # Increased from 100px to 200px
                min_distance = dist
                closest_shape = shape["id"]
        
        if closest_shape is not None:
            if closest_shape not in text_to_shape:
                text_to_shape[closest_shape] = []
            text_to_shape[closest_shape].append(i)
    
    return text_to_shape


def extract_text_from_shapes(gray, shapes):
    """
    Extract text directly from each shape region using OCR.
    Returns dictionary mapping shape ids to extracted text strings.
    """
    shape_texts = {}
    
    for shape in shapes:
        x, y, w, h = shape["x"], shape["y"], shape["w"], shape["h"]
        
        # Add padding around shape to capture nearby text
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(gray.shape[1], x + w + padding)
        y_end = min(gray.shape[0], y + h + padding)
        
        # Extract region
        region = gray[y_start:y_end, x_start:x_end]
        
        # Only attempt OCR if region is large enough
        if region.shape[0] > 5 and region.shape[1] > 5:
            try:
                extracted_text = pytesseract.image_to_string(region).strip()
                if extracted_text:
                    shape_texts[shape["id"]] = extracted_text
            except Exception as e:
                logger.debug(f"Could not extract text from shape {shape['id']}: {e}")
    
    return shape_texts


def find_arrow_connections(edges, shapes):
    """
    Detect arrows and map them to the shapes they connect.
    Returns list of connection paths between shapes.
    """
    connections = []
    
    if len(shapes) < 2:
        return connections
    
    # Detect lines (arrows are lines with arrowheads)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )
    
    if lines is None:
        return connections
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Convert numpy types to Python native types
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Find which shapes these line endpoints are closest to
        start_shape = None
        end_shape = None
        min_dist_start = float('inf')
        min_dist_end = float('inf')
        
        for shape in shapes:
            # Distance from start point to shape center
            dist_start = ((x1 - shape["cx"]) ** 2 + (y1 - shape["cy"]) ** 2) ** 0.5
            # Distance from end point to shape center
            dist_end = ((x2 - shape["cx"]) ** 2 + (y2 - shape["cy"]) ** 2) ** 0.5
            
            if dist_start < min_dist_start:
                min_dist_start = dist_start
                start_shape = shape["id"]
            
            if dist_end < min_dist_end:
                min_dist_end = dist_end
                end_shape = shape["id"]
        
        # Only add connection if both endpoints are reasonably close to shapes
        if (start_shape is not None and end_shape is not None and 
            start_shape != end_shape and 
            min_dist_start < 150 and min_dist_end < 150):
            
            connections.append({
                "from_shape": start_shape,
                "to_shape": end_shape,
                "start_point": [x1, y1],
                "end_point": [x2, y2]
            })
    
    return connections


def build_flow_paths(element_map):
    """
    Build simplified flow paths showing start -> end sequences.
    Traces all connections and creates readable flow chains.
    """
    shapes = {s["shape_id"]: s for s in element_map["shapes"]}
    connections = element_map["connections"]
    
    # Find starting shapes (no incoming connections)
    starting_shapes = []
    for shape in element_map["shapes"]:
        if not shape["incoming_connections"]:
            starting_shapes.append(shape["shape_id"])
    
    # Trace paths from each starting point
    all_paths = []
    
    def trace_path(shape_id, visited=None):
        if visited is None:
            visited = set()
        
        if shape_id in visited:
            return []
        
        visited.add(shape_id)
        current_shape = shapes.get(shape_id)
        
        if not current_shape:
            return []
        
        # Build node representation
        node_text = current_shape.get("text_content", f"[{current_shape['type']}]")
        node = f"{node_text} (id:{shape_id})"
        
        outgoing = current_shape.get("outgoing_connections", [])
        
        if not outgoing:
            # End of path
            return [[node]]
        
        # Continue tracing
        all_subpaths = []
        for next_id in outgoing:
            subpaths = trace_path(next_id, visited.copy())
            for subpath in subpaths:
                all_subpaths.append([node] + subpath)
        
        return all_subpaths if all_subpaths else [[node]]
    
    # Build all paths
    for start_id in starting_shapes:
        paths = trace_path(start_id)
        for path in paths:
            all_paths.append(" → ".join(path))
    
    return all_paths


def build_simplified_response(filename, file_size, img_path, img_type, features, shape_texts):
    """
    Build a simplified response focused on flows for faster LLM evaluation.
    """
    element_map = features.get("element_map", {})
    flow_paths = build_flow_paths(element_map)
    
    # Get key shapes with text
    meaningful_shapes = []
    for shape in element_map.get("shapes", []):
        if shape.get("text_content"):
            meaningful_shapes.append({
                "id": shape["shape_id"],
                "type": shape["type"],
                "text": shape["text_content"],
                "position": shape.get("position")
            })
    
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
        "shapes_with_text": meaningful_shapes,
        "flow_paths": flow_paths,
        "all_connections": element_map.get("connections", [])
    }
    """
    Detect if text is likely OCR noise (very short, random characters, or symbols).
    """
    if not text:
        return True
    
    text = text.strip()
    
    # Very short text (single char or less) is usually noise
    if len(text) <= 1:
        return True
    
    # Text with only symbols/numbers (OCR artifacts)
    if len(text) <= 3 and not any(c.isalpha() for c in text):
        return True
    
    # Text with lots of special characters
    special_count = sum(1 for c in text if not c.isalnum() and c not in [' ', '\n', '-', '_', '(', ')', '.', ',', ':', ';', "'", '"'])
    if special_count > len(text) * 0.5:
        return True
    
    return False


def deduplicate_connections(connections):
    """
    Remove duplicate connections from the list.
    """
    seen = set()
    unique = []
    for conn in connections:
        key = (conn.get("from_shape"), conn.get("to_shape"))
        if key not in seen:
            seen.add(key)
            unique.append(conn)
    return unique


def create_element_map(img_path, shapes, text_regions, text_to_shape_map, arrow_connections, shape_texts=None):
    """
    Create a comprehensive map showing how text, shapes, and arrows are connected.
    Filters out noise and deduplicates connections for clarity.
    """
    if shape_texts is None:
        shape_texts = {}
    
    # Clean shape_texts - remove noise
    clean_shape_texts = {}
    for shape_id, text in shape_texts.items():
        text = text.strip() if isinstance(text, str) else ""
        # Skip empty, very short text, or mostly special characters
        if text and len(text) > 1:
            clean_shape_texts[shape_id] = text
    
    element_map = {
        "shapes": [],
        "connections": []
    }
    
    # Build shape information with associated text
    for shape in shapes:
        text_content = clean_shape_texts.get(shape["id"], "")
        
        shape_info = {
            "shape_id": int(shape["id"]),
            "type": shape["type"],
            "position": {
                "x": int(shape["x"]), 
                "y": int(shape["y"]), 
                "w": int(shape["w"]), 
                "h": int(shape["h"])
            },
            "centroid": {
                "cx": int(shape["cx"]), 
                "cy": int(shape["cy"])
            },
            "outgoing_connections": [],
            "incoming_connections": []
        }
        
        # Only include text_content if it exists
        if text_content:
            shape_info["text_content"] = text_content
        
        element_map["shapes"].append(shape_info)
    
    # Deduplicate and add arrow connections
    unique_connections = deduplicate_connections(arrow_connections)
    
    for conn in unique_connections:
        from_id = conn["from_shape"]
        to_id = conn["to_shape"]
        
        # Find shapes and add connection reference
        for shape in element_map["shapes"]:
            if shape["shape_id"] == from_id:
                if int(to_id) not in shape["outgoing_connections"]:
                    shape["outgoing_connections"].append(int(to_id))
            if shape["shape_id"] == to_id:
                if int(from_id) not in shape["incoming_connections"]:
                    shape["incoming_connections"].append(int(from_id))
        
        element_map["connections"].append({
            "from_shape_id": int(from_id),
            "to_shape_id": int(to_id)
        })
    
    return element_map


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
            
            # Get clean shape_texts
            clean_shape_texts = {}
            for shape_id, text in features.get("shape_texts", {}).items():
                text = text.strip() if isinstance(text, str) else ""
                if text and len(text) > 1:  # Skip empty or very short text
                    clean_shape_texts[shape_id] = text
            
            # Build simplified response
            simplified = build_simplified_response(
                file.filename, 
                len(contents),
                img_path,
                img_type,
                features,
                clean_shape_texts
            )

            results.append(simplified)

        
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
