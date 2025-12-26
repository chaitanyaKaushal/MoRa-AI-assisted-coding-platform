import json
import os
from openai import OpenAI
from utils import LEETCODE_CONTEXT, parse_starter_code
from execution import run_in_sandbox

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(system_prompt, user_prompt, model="gpt-4o", json_mode=False):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message.content

# --- STAGE 3: Formal Specification ---
def stage_3_formalize(raw_problem: str):
    system_prompt = f"""You are an expert Competitive Programming Problem Setter.
    EXISTING CODE CONTEXT:
    {LEETCODE_CONTEXT}
    
    TASK: Convert the human-written question into a well-structured problem specification JSON.
    CRITICAL RULES:
    1. "starter_code": Use class `Solution`.
    2. "constraints": Infer if missing.
    
    OUTPUT JSON (STRICT):
    {{
        "title": "...", "problem_description": "...", "examples": [], "constraints": "...",
        "starter_code": "...", "evaluation_type": "exact_match"
    }}
    """
    
    response = call_llm(system_prompt, raw_problem, json_mode=True)
    return json.loads(response)

# --- STAGE 4: The Oracle (Test Generator) ---
def stage_4_oracle(spec: dict):
    parsed = parse_starter_code(spec['starter_code'])
    input_signature = f"{parsed['function_name']}({', '.join([a['name'] + ': ' + a['type'] for a in parsed['args']])})"
    
    system_prompt = f"""You are The Oracle (Test Data Generator).
    CONTEXT:
    - Input Signature: {input_signature}
    - Constraints: {spec['constraints']}
    
    TASK: Write a Python script with two functions:
    1. `generate_inputs() -> List[tuple]`: Returns list of raw arguments. JSON-serializable primitives ONLY.
    2. `is_valid_input(args) -> bool`
    
    OUTPUT: Return ONLY the Python code block (no markdown).
    """
    
    code = call_llm(system_prompt, "Generate the oracle script.", json_mode=False)
    # Strip markdown if present
    code = code.replace("```python", "").replace("```", "")
    
    # Wrap in execution logic
    full_script = f"""
import random, json
{code}

inputs = generate_inputs()
valid_inputs = [i for i in inputs if is_valid_input(i)]
print(json.dumps(valid_inputs))
"""
    output = run_in_sandbox(full_script)
    return json.loads(output)

# --- STAGE 5: Reference Solution & Golden Suite ---
def stage_5_reference(spec: dict, oracle_inputs: list):
    system_prompt = f"""You are an Expert Python Developer.
    TASK: Write the Reference Solution for this problem.
    SPECIFICATION: {json.dumps(spec)}
    CONTEXT: {LEETCODE_CONTEXT}
    OUTPUT: Python code for the `Solution` class.
    """
    
    ref_code = call_llm(system_prompt, "Write the solution.", json_mode=False)
    ref_code = ref_code.replace("```python", "").replace("```", "")

    # Runner Script Logic (Simplified from your prompt for brevity)
    runner_script = f"""
import json, copy
{LEETCODE_CONTEXT}
{ref_code}

oracle_inputs = {json.dumps(oracle_inputs)}
solver = Solution()
test_suite = []

for args in oracle_inputs:
    try:
        # Assuming args is a list/tuple matches function signature
        # In a real app, complex type conversion happens here
        res = solver.solve(*copy.deepcopy(args))
        test_suite.append({{"input": args, "expected": res}})
    except Exception as e:
        pass # Skip failing cases in generation

print(json.dumps(test_suite))
"""
    output = run_in_sandbox(runner_script)
    test_suite = json.loads(output)
    return test_suite, ref_code

# --- STAGE 6: User Execution ---
def stage_6_execute_user(user_code: str, test_suite: list):
    runner_script = f"""
import json, copy, math, signal
{LEETCODE_CONTEXT}

# User Code Injection
{user_code}

test_suite = {json.dumps(test_suite)}
solver = Solution()
results = []

def timeout_handler(signum, frame): raise TimeoutError()
signal.signal(signal.SIGALRM, timeout_handler)

for case in test_suite:
    inp = case['input']
    exp = case['expected']
    try:
        signal.alarm(2) # 2s timeout
        actual = solver.solve(*copy.deepcopy(inp))
        signal.alarm(0)
        
        passed = (actual == exp)
        results.append({{"input": inp, "expected": exp, "actual": actual, "passed": passed}})
    except Exception as e:
        signal.alarm(0)
        results.append({{"input": inp, "expected": exp, "actual": str(e), "passed": False}})

print(json.dumps(results))
"""
    return json.loads(run_in_sandbox(runner_script))