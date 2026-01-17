"""
Text and shape analysis functions
"""
import pytesseract
import logging

logger = logging.getLogger(__name__)


def find_text_to_shape_mapping(shapes, text_regions):
    """
    Map text regions to their nearest shapes.
    
    Args:
        shapes: List of detected shapes
        text_regions: List of detected text regions
        
    Returns:
        Dictionary mapping shape IDs to text region indices
    """
    text_to_shape = {}
    
    for i, text in enumerate(text_regions):
        closest_shape = None
        min_distance = float('inf')
        
        for shape in shapes:
            dist = ((text["cx"] - shape["cx"]) ** 2 + (text["cy"] - shape["cy"]) ** 2) ** 0.5
            
            text_inside = (shape["x"] <= text["cx"] <= shape["x"] + shape["w"] and
                          shape["y"] <= text["cy"] <= shape["y"] + shape["h"])
            
            text_overlaps = not (text["x"] + text["w"] < shape["x"] or 
                               text["x"] > shape["x"] + shape["w"] or
                               text["y"] + text["h"] < shape["y"] or
                               text["y"] > shape["y"] + shape["h"])
            
            if text_inside:
                min_distance = 0
                closest_shape = shape["id"]
                break
            elif text_overlaps:
                min_distance = 0
                closest_shape = shape["id"]
                break
            elif dist < min_distance and dist < 200:
                min_distance = dist
                closest_shape = shape["id"]
        
        if closest_shape is not None:
            if closest_shape not in text_to_shape:
                text_to_shape[closest_shape] = []
            text_to_shape[closest_shape].append(i)
    
    return text_to_shape


def extract_text_from_shapes(gray, shapes, min_shape_area=500):
    """
    Extract text from significant shape regions using OCR.
    
    Args:
        gray: Grayscale image
        shapes: List of detected shapes
        min_shape_area: Minimum area to attempt OCR
        
    Returns:
        Dictionary mapping shape IDs to extracted text
    """
    shape_texts = {}
    
    for shape in shapes:
        if shape["area"] < min_shape_area:
            continue
        
        x, y, w, h = shape["x"], shape["y"], shape["w"], shape["h"]
        
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(gray.shape[1], x + w + padding)
        y_end = min(gray.shape[0], y + h + padding)
        
        region = gray[y_start:y_end, x_start:x_end]
        
        if region.shape[0] > 10 and region.shape[1] > 10:
            try:
                extracted_text = pytesseract.image_to_string(
                    region,
                    config='--psm 6 --oem 1'
                ).strip()
                
                if extracted_text and len(extracted_text) > 1:
                    shape_texts[shape["id"]] = extracted_text
            except Exception as e:
                logger.debug(f"Could not extract text from shape {shape['id']}: {e}")
    
    return shape_texts


def build_flow_paths(element_map, shape_texts=None):
    """
    Build simplified flow paths showing sequences.
    
    Args:
        element_map: Map of elements and connections
        shape_texts: Dictionary of shape text content
        
    Returns:
        List of flow path strings
    """
    if shape_texts is None:
        shape_texts = {}
    
    shapes = {s["shape_id"]: s for s in element_map["shapes"]}
    
    for shape_id, shape in shapes.items():
        if "text_content" in shape and shape["text_content"] and shape_id not in shape_texts:
            shape_texts[shape_id] = shape["text_content"]
    
    starting_shapes = []
    for shape in element_map["shapes"]:
        if not shape["incoming_connections"]:
            starting_shapes.append(shape["shape_id"])
    
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
        
        text_content = ""
        if shape_id in shape_texts:
            text_content = shape_texts[shape_id]
        elif "text_content" in current_shape and current_shape["text_content"]:
            text_content = current_shape["text_content"]
        
        if text_content:
            node_text = text_content.replace('\n', ' ').strip()
        else:
            node_text = f"[{current_shape['type']}]"
        
        node = f"{node_text} (id:{shape_id})"
        
        outgoing = current_shape.get("outgoing_connections", [])
        
        if not outgoing:
            return [[node]]
        
        all_subpaths = []
        for next_id in outgoing:
            subpaths = trace_path(next_id, visited.copy())
            for subpath in subpaths:
                all_subpaths.append([node] + subpath)
        
        return all_subpaths if all_subpaths else [[node]]
    
    for start_id in starting_shapes:
        paths = trace_path(start_id)
        for path in paths:
            all_paths.append(" → ".join(path))
    
    return all_paths


def create_element_map(img_path, shapes, text_regions, text_to_shape_map, arrow_connections, shape_texts=None):
    """
    Create comprehensive map of elements and connections.
    
    Args:
        img_path: Path to image
        shapes: List of shapes
        text_regions: List of text regions
        text_to_shape_map: Mapping of text to shapes
        arrow_connections: List of connections
        shape_texts: Dictionary of extracted text
        
    Returns:
        Comprehensive element map
    """
    if shape_texts is None:
        shape_texts = {}
    
    clean_shape_texts = {}
    for shape_id, text in shape_texts.items():
        text = text.strip() if isinstance(text, str) else ""
        if text and len(text) > 1:
            clean_shape_texts[shape_id] = text
    
    element_map = {
        "shapes": [],
        "connections": []
    }
    
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
        
        if text_content:
            shape_info["text_content"] = text_content
        
        element_map["shapes"].append(shape_info)
    
    # Deduplicate connections
    seen = set()
    unique_connections = []
    for conn in arrow_connections:
        key = (conn.get("from_shape"), conn.get("to_shape"))
        if key not in seen:
            seen.add(key)
            unique_connections.append(conn)
    
    for conn in unique_connections:
        from_id = conn["from_shape"]
        to_id = conn["to_shape"]
        
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
