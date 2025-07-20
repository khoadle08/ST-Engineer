# resume_parser_advanced_en.py
# This version includes an advanced validation and correction layer.

import os
import json
import re
from pathlib import Path
from typing import TypedDict, List, Optional, Dict, Any

# --- Required Installations ---
# pip install google-generativeai PyPDF2 pdf2image pytesseract Pillow langgraph

import google.generativeai as genai
from PyPDF2 import PdfReader
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
# 1. Configure Google Gemini API Key
try:
    # It's recommended to use environment variables for security.
    GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini Pro model initialized successfully.")
except (ValueError, KeyError) as e:
    print(f"ERROR: Could not initialize Gemini model. Please set the GEMINI_API_KEY environment variable. Details: {e}")
    exit()

# 2. Configure Tesseract path (only needed on Windows if not in system PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 3. Define input/output directories and retry limit
INPUT_DIR = Path("resumes_pdf")
OUTPUT_FILE = Path("resume_extractions_corrected.json")
MAX_RETRIES = 2 # Set the maximum number of correction attempts

# --- LANGGRAPH STATE DEFINITION ---

class GraphState(TypedDict):
    """
    Defines the state passed between nodes in the graph.

    | Attribute      | Type                  | Description                                                                                      |
    |:---------------|:----------------------|:-------------------------------------------------------------------------------------------------|
    | `pdf_path`     | `Path`                | The file path to the current PDF being processed.                                                |
    | `raw_text`     | `Optional[str]`       | The raw text extracted from the PDF, either directly or via OCR.                                 |
    | `llm_output`   | `Optional[Dict]`      | The raw JSON data as returned by the Large Language Model.                                       |
    | `validated_data`| `Optional[Dict]`      | The cleaned and validated data after passing through our correction node.                        |
    | `error_log`    | `List[str]`           | A list of specific error messages identified during validation, used for the self-correction loop. |
    | `retry_count`  | `int`                 | A counter to track how many times the self-correction loop has run.                              |
    | `fatal_error`  | `Optional[str]`       | A critical error message that should halt the process for the current file.                      |
    """
    pdf_path: Path
    raw_text: Optional[str]
    llm_output: Optional[Dict[str, Any]]
    validated_data: Optional[Dict[str, Any]]
    error_log: List[str]
    retry_count: int
    fatal_error: Optional[str]


# --- GRAPH NODES DEFINITION ---

def extract_text_node(state: GraphState) -> Dict[str, Any]:
    """Node to extract raw text from a PDF file."""
    pdf_path = state['pdf_path']
    print(f"--- [Node: Extract Text] Processing: {pdf_path.name} ---")
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if len(text.strip()) < 200:
            print("  Short text detected, switching to OCR...")
            text = ""
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
            print("  OCR extraction successful.")
        else:
            print("  Direct text extraction successful.")
        return {"raw_text": text}
            
    except Exception as e:
        error_msg = f"Failed to extract text: {e}"
        print(f"  ERROR: {error_msg}")
        return {"raw_text": None, "fatal_error": error_msg}

def generate_extraction_prompt(raw_text: str) -> str:
    """Generates the initial prompt for extracting data."""
    return f"""
    You are an expert resume parser. Your task is to extract structured information from the resume text below.
    Your output MUST be a single, valid JSON object and nothing else. Do not add notes or markdown formatting.

    Extract the following fields:
    - "name": The full name of the candidate. If not found, return null.
    - "email": The email address. If not found, return null.
    - "phone": The phone number. If not found, return null.
    - "skills": A list of all technical and professional skills.
    - "education": A list of education items. Each item is an object with keys: "degree", "institution", and "graduation_year". If not found, return an empty list [].
    - "experience": A list of job experiences. Each item is an object with keys: "job_title", "company", "years_worked", and "description". If not found, return an empty list [].
    - "certifications": A list of certifications. If not found, return an empty list [].
    - "languages": A list of languages the candidate can speak or write. If not found, return an empty list [].

    Resume Text:
    \"\"\"
    {raw_text}
    \"\"\"
    """
def generate_correction_prompt(raw_text: str, previous_json: Dict[str, Any], errors: List[str]) -> str:
    """Generates a prompt asking the LLM to correct its previous output."""
    error_str = "\n- ".join(errors)
    return f"""
    You are an expert resume parser. Your previous attempt to parse a resume had some errors.
    Please correct your previous output based on the error list provided.
    Your output MUST be a single, valid, and corrected JSON object and nothing else.

    Here is the original resume text:
    \"\"\"
    {raw_text}
    \"\"\"

    Here was your previous, flawed JSON output:
    ```json
    {json.dumps(previous_json, indent=2)}
    ```

    Here are the errors you need to fix:
    - {error_str}

    Please provide the full, corrected JSON object.
    """

def extract_json_node(state: GraphState) -> Dict[str, Any]:
    """Node to call the LLM. It uses a different prompt for retries."""
    pdf_path = state['pdf_path']
    raw_text = state['raw_text']
    retry_count = state['retry_count']

    if retry_count > 0:
        print(f"--- [Node: Correct JSON] Attempt #{retry_count} for: {pdf_path.name} ---")
        prompt = generate_correction_prompt(raw_text, state['llm_output'], state['error_log'])
    else:
        print(f"--- [Node: Extract JSON] First attempt for: {pdf_path.name} ---")
        prompt = generate_extraction_prompt(raw_text)

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Clean the response to remove markdown wrappers for JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        data = json.loads(cleaned_response)
        print(f"  LLM call successful for {pdf_path.name}.")
        return {"llm_output": data, "retry_count": retry_count + 1}
        
    except json.JSONDecodeError:
        error_message = f"LLM did not return valid JSON for {pdf_path.name}."
        print(f"  ERROR: {error_message}")
        return {"llm_output": None, "fatal_error": error_message}
    except Exception as e:
        error_message = f"An unknown error occurred with the LLM call for {pdf_path.name}: {e}"
        print(f"  ERROR: {error_message}")
        return {"llm_output": None, "fatal_error": error_message}

def validate_data_node(state: GraphState) -> Dict[str, Any]:
    """Node to validate, clean, and log errors for the self-correction loop."""
    data = state['llm_output']
    if not data:
        return {"validated_data": None, "error_log": ["No data from LLM to validate."]}

    pdf_path = state['pdf_path']
    print(f"--- [Node: Validate Data] Analyzing data for: {pdf_path.name} ---")
    error_log = []
    
    # Create a deep copy to modify without affecting the original llm_output
    validated_data = json.loads(json.dumps(data))

    # 1. Validate Email Format
    email = validated_data.get('email')
    if email and isinstance(email, str):
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
            error_log.append(f"The email '{email}' is not a valid format. Please correct it or set it to null.")
    
    # 2. Check Data Types for List Fields
    for key in ['skills', 'education', 'experience', 'certifications', 'languages']:
        if key in validated_data and not isinstance(validated_data[key], list):
            error_log.append(f"The field '{key}' should be a list, but it is a {type(validated_data[key]).__name__}. Please correct it.")

    # 3. Perform minor sanitization (this runs every time and doesn't trigger a retry)
    if 'skills' in validated_data and isinstance(validated_data['skills'], list):
        validated_data['skills'] = [s.strip() for s in validated_data['skills'] if isinstance(s, str) and s.strip()]

    print(f"  Validation complete. Found {len(error_log)} major errors to correct.")
    return {"validated_data": validated_data, "error_log": error_log}


# --- GRAPH EDGES (LOGIC) DEFINITION ---

def should_continue_or_retry(state: GraphState) -> str:
    """Decides the next step: end the process or retry for correction."""
    print("--- [Edge: Decision] ---")
    if state.get("fatal_error"):
        print(f"  Fatal error detected: {state['fatal_error']}. Ending workflow.")
        return "end"
    
    # If there are errors in the log and we haven't exceeded the retry limit
    if state.get("error_log") and state.get("retry_count", 0) < MAX_RETRIES:
        print(f"  Errors found. Retrying... (Attempt {state['retry_count']})")
        return "retry"
        
    print("  No significant errors found or max retries reached. Ending workflow.")
    return "end"


# --- BUILD AND COMPILE THE GRAPH ---

workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("extract_text", extract_text_node)
workflow.add_node("extract_json", extract_json_node)
workflow.add_node("validate_data", validate_data_node)

# Set the entry point
workflow.set_entry_point("extract_text")

# Define the flow of the graph
workflow.add_conditional_edges(
    "extract_text",
    lambda s: "extract_json" if s.get("raw_text") else "end",
    {"extract_json": "extract_json", "end": END}
)

workflow.add_conditional_edges(
    "extract_json",
    lambda s: "validate_data" if s.get("llm_output") else "end",
    {"validate_data": "validate_data", "end": END}
)

# This is the self-correction loop
workflow.add_conditional_edges(
    "validate_data",
    should_continue_or_retry,
    {
        "retry": "extract_json", # If errors, go back to extract_json
        "end": END             # If clean, end
    }
)

# Compile the graph into a runnable application
app = workflow.compile()

# --- MAIN EXECUTION FUNCTION ---

def main():
    """Main function to run the entire resume processing pipeline."""
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"ERROR: Input directory '{INPUT_DIR}' does not exist. Please create it and add your PDF files.")
        return

    all_results = []
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}'.")
        return
        
    print(f"\nFound {len(pdf_files)} PDF files to process with Self-Correction Loop.\n")

    for pdf_path in pdf_files:
        # Initial state for each run
        initial_state = {
            "pdf_path": pdf_path,
            "retry_count": 0,
            "error_log": [],
        }
        final_state = app.invoke(initial_state)
        
        # We consider a run successful if the final data is present and the final error log is empty
        if final_state.get("validated_data") and not final_state.get("error_log"):
            print(f"SUCCESS: Processing for {pdf_path.name} completed successfully.")
            all_results.append(final_state["validated_data"])
        else:
            print(f"FAILURE: Could not produce a clean result for {pdf_path.name} after {final_state.get('retry_count')} attempts.")
            if final_state.get("error_log"):
                print("  Final uncorrected errors:", final_state.get("error_log"))
        print("="*60)

    if all_results:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"\nProcessing complete! Saved {len(all_results)} validated results to '{OUTPUT_FILE}'.")
    else:
        print("\nNo data was successfully extracted and validated.")

if __name__ == "__main__":
    main()