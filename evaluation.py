import json
from config import CEREBRAS_API_KEY
from cerebras.cloud.sdk import Cerebras
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_file(filepath):
    """
    Extract text from a file. Supports PDF and plain text.
    """
    if filepath.lower().endswith('.pdf'):
        try:
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            return "\n".join([p.page_content for p in pages])
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
            return ""
    else:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return ""

def evaluate_documents_sync(document_data, rubric_data, image_analysis_data=None):
    """
    Evaluate documents against rubrics using Cerebras LLM.
    
    Args:
        document_data: Dict mapping filename -> temp_filepath
        rubric_data: Dict containing evaluation rubrics
        image_analysis_data: Dict mapping filename -> list of image analysis results
        
    Returns:
        Dict containing evaluation results for each file
    """
    client = Cerebras(api_key=CEREBRAS_API_KEY)
    results = {}
    
    if image_analysis_data is None:
        image_analysis_data = {}

    for filename, filepath in document_data.items():
        text = extract_text_from_file(filepath)
        
        # Format image data if available
        image_context = ""
        if filename in image_analysis_data and image_analysis_data[filename]:
            image_context = "\n\nExtracted Visual Content (Diagrams/Flowcharts):\n"
            for i, img_data in enumerate(image_analysis_data[filename]):
                image_context += f"\n--- Diagram {i+1} ({img_data.get('diagram_type', 'unknown')}) ---\n"
                
                # Add summary counts
                summary = img_data.get('summary', {})
                if summary:
                    image_context += f"Elements: {json.dumps(summary)}\n"
                
                # Add flow paths
                flow_paths = img_data.get('flow_paths', [])
                if flow_paths:
                    image_context += "Flow Paths:\n" + "\n".join(f"- {path}" for path in flow_paths) + "\n"
                
                # Add detailed features
                details = img_data.get('detailed_features', {})
                if details:
                    if details.get('extracted_text_content'):
                        image_context += f"\nFull Diagram Text (OCR):\n{details.get('extracted_text_content')}\n"
                    
                    if details.get('pseudocode_analysis'):
                         image_context += f"Pseudocode Analysis: {json.dumps(details.get('pseudocode_analysis'))}\n"
                    
                    # Include stats
                    image_context += f"Text Density: {details.get('text_density', 0):.4f}\n"
        
        # Limit text length to avoid token limits (adjust as needed)
        # Using a reasonable context window for evaluation
        preview_text = text[:8000] 
        
        system_prompt = "You are an expert document evaluator. Evaluate the provided document text based on the given rubrics."
        
        user_prompt = f"""
Document Filename: {filename}

Document Content (Excerpt):
{preview_text}

{image_context}

Evaluation Rubrics:
{json.dumps(rubric_data, indent=2)}

Please evaluate the document according to the rubrics above.
For each rubric criteria, provide a score and a brief reasoning.
Return the output as a valid JSON object with detailed evaluations.
"""

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b", # Using a strong model for evaluation
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            try:
                results[filename] = json.loads(response_content)
            except json.JSONDecodeError:
                results[filename] = {
                    "raw_response": response_content,
                    "error": "Failed to parse JSON response from LLM"
                }
                
        except Exception as e:
            results[filename] = {
                "error": str(e),
                "status": "failed"
            }
            
    return results
