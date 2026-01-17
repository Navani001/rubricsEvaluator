"""
Shape and element detection functions
"""
import cv2
import numpy as np
import pytesseract
import logging

logger = logging.getLogger(__name__)


def detect_arrowheads(edges):
    """
    Detect arrowhead shapes in edges.
    
    Args:
        edges: Edge-detected image
        
    Returns:
        Count of detected arrowheads
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    arrowheads = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)

        if len(approx) == 3:
            area = cv2.contourArea(cnt)
            if area > 50:
                arrowheads += 1

    return arrowheads


def detect_flow_connections(contours):
    """
    Detect vertical flow connections between shapes.
    
    Args:
        contours: Image contours
        
    Returns:
        Count of detected connections
    """
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

            if abs(x1 - x2) < 50 and y2 > y1:
                connections += 1

    return connections


def detect_branching_structure(contours):
    """
    Detect the depth and complexity of branching.
    
    Args:
        contours: Image contours
        
    Returns:
        Branching depth
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
    
    depth_levels = set()
    for x, y, w, h in boxes:
        level = y // 50
        depth_levels.add(level)
    
    return len(depth_levels) - 1


def get_shape_regions(contours, gray):
    """
    Extract all shapes with their bounding boxes and centroids.
    
    Args:
        contours: Image contours
        gray: Grayscale image
        
    Returns:
        List of shape dictionaries
    """
    shapes = []
    shape_id = 0
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = float(cv2.contourArea(cnt))
        
        if area < 100:
            continue
        
        x, y, w, h = cv2.boundingRect(approx)
        x, y, w, h = int(x), int(y), int(w), int(h)
        cx, cy = int(x + w // 2), int(y + h // 2)
        
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
    Extract text regions using morphological operations.
    
    Args:
        gray: Grayscale image
        
    Returns:
        List of text region dictionaries
    """
    text_regions = []
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1, iterations=1)
    dilated = cv2.dilate(closed, kernel2, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = int(x), int(y), int(w), int(h)
        area = w * h
        
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


def find_arrow_connections(edges, shapes):
    """
    Detect arrows and map them to connected shapes.
    
    Args:
        edges: Edge-detected image
        shapes: List of detected shapes
        
    Returns:
        List of connection dictionaries
    """
    connections = []
    
    if len(shapes) < 2:
        return connections
    
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
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        start_shape = None
        end_shape = None
        min_dist_start = float('inf')
        min_dist_end = float('inf')
        
        for shape in shapes:
            dist_start = ((x1 - shape["cx"]) ** 2 + (y1 - shape["cy"]) ** 2) ** 0.5
            dist_end = ((x2 - shape["cx"]) ** 2 + (y2 - shape["cy"]) ** 2) ** 0.5
            
            if dist_start < min_dist_start:
                min_dist_start = dist_start
                start_shape = shape["id"]
            
            if dist_end < min_dist_end:
                min_dist_end = dist_end
                end_shape = shape["id"]
        
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
