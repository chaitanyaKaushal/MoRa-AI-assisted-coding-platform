import os
import json
import time
import re
import logging
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "leetcode_dataset.json"
STAGE_1_OUTPUT = "leetcode_dataset_stage1.json"
STAGE_2_OUTPUT = "leetcode_dataset_stage2.json"

# We use gpt-4o or gpt-3.5-turbo. 
# 'json_object' format is crucial for strict adherence.
MODEL = "gpt-4o-mini" 

# ==========================================
# PROMPT TEMPLATES (EXACT WORDING)
# ==========================================

# STAGE 1 PROMPTS
S1_SYSTEM_PROMPT = """You are a Data Extraction Specialist and Sanitization Engine. Your task is to parse a raw LeetCode problem description into a strict JSON format."""

S1_USER_TEMPLATE = """RAW TEXT:
"{raw_problem_description}"

INSTRUCTIONS:
1. "problem_description": Extract only core logic. Remove Examples, Constraints, Hints, and Follow-ups, and Complexity Requirements (e.g., "O(n) time"). But, preserve logical guarantees even if they appear in the description. For example, "The array is 0-indexed", "'array is sorted", "tree is distinct", "The graph is a DAG" are often logical guarantees, not constraints.
2. "examples": Extract all input/output examples text blocks.
3. "constraints": Extract the full text of all constraints, including numerical limits (e.g., "1 <= n <= 100") and structural guarantees (e.g., "all elements are unique", "s consists of lowercase English letters"). If not explicitly stated, return []. Do NOT infer "small" or "large".

OUTPUT JSON (STRICT):
{{
  "problem_description": "string",
  "examples": ["string"],
  "constraints": ["string"]
}}"""

# STAGE 2 PROMPTS
S2_SYSTEM_PROMPT = """You are an expert at simulating diverse user personas in technical contexts. Your goal is to rewrite a Raw Coding Problem Statement into 4 different styles of user queries."""

S2_USER_TEMPLATE = """INPUT TEXT:
"{clean_description}"

GUIDELINES FOR VARIATIONS:
1. "layman": Zero jargon. Focus on the "story" or "scenario". Use simple words like "list", "items", "find".
2. "conversational": Natural sentences, as if recalling a question from an interview 2 hours ago. A slightly imperfect memory is okay.
3. "technical_shorthand": Ultra-concise functional specification.
4. "implementation_specific": Solution-oriented (e.g., "Python solution using hash map").

STRICT RULES:
- OUTPUT FORMAT: Return ONLY a valid JSON object with keys: "layman", "conversational", "technical_shorthand", "implementation_specific".
- FORBIDDEN: Do NOT mention specific algorithm names (e.g. "Dijkstra"), specific function names (e.g. `solve`), or class names.
- AMBIGUITY: If the raw text is vague, preserve that ambiguity.

GOAL: Describe the INPUT → OUTPUT transformation only."""

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def load_data(filepath):
    """Safely loads the dataset JSON."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # Handle cases where the file might be wrapped in a dictionary key like "dataset"
            if isinstance(data, dict) and 'dataset' in data:
                return data['dataset']
            return data
        except json.JSONDecodeError:
            return []

def save_atomic(data, filepath):
    """
    Writes data to a temp file first, then renames it. 
    This prevents file corruption if the script crashes during writing.
    """
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, filepath)

def get_completed_ids(dataset):
    """Returns a set of task_ids that are already processed."""
    return {item.get('task_id') for item in dataset}

def clean_json_str(content):
    """Removes Markdown code blocks if the LLM adds them."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
    return content.strip()

# ==========================================
# STAGE 1: EXTRACTION LOGIC
# ==========================================

def run_stage_1():
    print(f"\n--- STARTING STAGE 1: Extraction Logic ---")
    
    # 1. Load Data
    raw_dataset = load_data(INPUT_FILE)
    if not raw_dataset:
        print(f"Error: {INPUT_FILE} is missing or empty.")
        return

    # 2. Checkpoint Loading
    processed_dataset = load_data(STAGE_1_OUTPUT)
    completed_ids = get_completed_ids(processed_dataset)
    
    print(f"Total Raw Records: {len(raw_dataset)}")
    print(f"Already Processed: {len(completed_ids)}")

    # 3. Processing Loop
    for item in tqdm(raw_dataset, desc="Processing Stage 1"):
        task_id = item.get('task_id')
        
        # Skip if already done
        if task_id in completed_ids:
            continue

        raw_desc = item.get('problem_description', "")
        
        # Construct Prompt (Exact wording)
        user_prompt = S1_USER_TEMPLATE.format(raw_problem_description=raw_desc)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": S1_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # Low temp for deterministic extraction
                response_format={"type": "json_object"}
            )
            
            # Post-Processing
            content = response.choices[0].message.content
            extracted_data = json.loads(clean_json_str(content))
            
            # Update Item
            item['cleaned_problem_description'] = extracted_data.get('problem_description', "")
            item['examples'] = extracted_data.get('examples', [])
            item['constraints'] = extracted_data.get('constraints', [])

            # Append and Atomic Save
            processed_dataset.append(item)
            save_atomic(processed_dataset, STAGE_1_OUTPUT)

        except Exception as e:
            print(f"\n[ERROR] Stage 1 failed for {task_id}: {e}")
            # Continue to next item; do not crash
            continue

    print("Stage 1 Completed.")

# ==========================================
# STAGE 2: PERSONA GENERATION
# ==========================================

def run_stage_2():
    print(f"\n--- STARTING STAGE 2: Persona Generation ---")
    
    # 1. Load Data (Output of Stage 1 is Input of Stage 2)
    stage_1_data = load_data(STAGE_1_OUTPUT)
    if not stage_1_data:
        print(f"Error: {STAGE_1_OUTPUT} is missing or empty. Run Stage 1 first.")
        return

    # 2. Checkpoint Loading
    processed_dataset = load_data(STAGE_2_OUTPUT)
    completed_ids = get_completed_ids(processed_dataset)
    
    print(f"Total Stage 1 Records: {len(stage_1_data)}")
    print(f"Already Processed: {len(completed_ids)}")

    # 3. Processing Loop
    for item in tqdm(stage_1_data, desc="Processing Stage 2"):
        task_id = item.get('task_id')

        # Skip if already done
        if task_id in completed_ids:
            continue

        clean_desc = item.get('cleaned_problem_description', "")
        
        # Construct Prompt (Exact wording)
        user_prompt = S2_USER_TEMPLATE.format(clean_description=clean_desc)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": S2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7, # Higher temp for diverse personas
                response_format={"type": "json_object"}
            )
            
            # Post-Processing
            content = response.choices[0].message.content
            personas = json.loads(clean_json_str(content))

            # --- FILTERING LOGIC ---
            forbidden_keywords = ["Solution", "twoSum", "O(n)", "O(1)", "O(log n)"]
            
            # Check "layman" and "conversational" for forbidden keywords
            for style in ["layman", "conversational"]:
                if style in personas:
                    text_value = personas[style]
                    # If any keyword is found (case-insensitive)
                    if any(kw.lower() in text_value.lower() for kw in forbidden_keywords):
                        print(f"Filtered out '{style}' for {task_id} due to forbidden keywords.")
                        del personas[style]

            # Update Item
            item['vague_problem'] = personas

            # Append and Atomic Save
            processed_dataset.append(item)
            save_atomic(processed_dataset, STAGE_2_OUTPUT)

        except Exception as e:
            print(f"\n[ERROR] Stage 2 failed for {task_id}: {e}")
            continue

    print("Stage 2 Completed.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    run_stage_1()
    run_stage_2()
