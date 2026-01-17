"""
Image classification and pattern detection functions
"""


def classify_image(features):
    """
    Classify image as flowchart, algorithm, pseudocode, or architecture.
    Uses multiple indicators for better accuracy.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        String classification: 'flowchart', 'algorithm', 'pseudocode', or 'unknown'
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
        flowchart_score += 30
    if connections > 2:
        flowchart_score += 25
    if ellipses > 0:
        flowchart_score += 15
    if arrows > 0:
        flowchart_score += 20
    
    # Algorithm detection
    if boxes > 2 and arrows > 1:
        algorithm_score += 30
    if branching_depth > 1:
        algorithm_score += 25
    if boxes > 0 and connections > 1:
        algorithm_score += 20
    if arrows > 2:
        algorithm_score += 15
    
    # Pseudocode detection
    pseudocode_score += pseudocode_indicators.get("keyword_count", 0) * 5
    pseudocode_score += pseudocode_indicators.get("indent_count", 0) * 4
    pseudocode_score += pseudocode_indicators.get("bracket_count", 0) * 3
    
    if text_density > 0.01:
        pseudocode_score += 25
    if pseudocode_indicators.get("has_loop_keywords", False):
        pseudocode_score += 20
    if pseudocode_indicators.get("has_condition_keywords", False):
        pseudocode_score += 20
    
    # Determine classification
    scores = {
        "flowchart": flowchart_score,
        "algorithm": algorithm_score,
        "pseudocode": pseudocode_score,
        "unknown": 0
    }
    
    max_score = max(scores.values())
    
    if max_score == 0:
        if text_density > 0.005:
            return "pseudocode"
        elif diamonds > 0:
            return "flowchart"
        elif boxes > 1:
            return "algorithm"
        else:
            return "unknown"
    
    return max(scores, key=scores.get)


def detect_pseudocode_patterns(text):
    """
    Detect pseudocode patterns in extracted text.
    
    Args:
        text: Extracted text from image
        
    Returns:
        Dictionary with pseudocode indicators
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
    
    loop_keywords = ["for", "while", "do", "repeat", "foreach", "loop"]
    condition_keywords = ["if", "else", "elseif", "switch", "case", "then"]
    function_keywords = ["function", "procedure", "def", "method", "return", "main"]
    
    text_lower = text.lower()
    
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
    
    lines = text.split('\n')
    for line in lines:
        if line and line[0] in [' ', '\t']:
            indicators["indent_count"] += 1
    
    indicators["bracket_count"] = (text.count('(') + text.count(')') + 
                                   text.count('{') + text.count('}') + 
                                   text.count('[') + text.count(']'))
    
    return indicators
