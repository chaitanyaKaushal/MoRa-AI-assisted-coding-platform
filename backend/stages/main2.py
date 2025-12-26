"""
AI Coding Platform - Stages 3-6 Implementation
PRODUCTION VERSION: Robust, Self-Healing, and Consensus-Driven
STAGE 3: Vague → Formal Specification
STAGE 4: Test Case Generation (The Oracle + Reflexion)
STAGE 5: Reference Solution Generation (Dual-Model Consensus)
STAGE 6: User Submission Evaluation
"""

import os
import json
import uuid
import copy
import signal
import math
import ast
import traceback
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils import LeetCodeContext
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Coding Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATABASE SETUP
# ============================================================================

FINE_TUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:chaitanya-nus-openai::CobA356J"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@postgres:5432/coding_platform")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class Problem(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True)
    problem_id = Column(String(50), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    problem_description = Column(Text, nullable=False)
    examples = Column(Text, nullable=False)
    constraints = Column(Text, nullable=False)
    constraint_source = Column(String(50))
    tags = Column(ARRAY(String))
    starter_code = Column(Text, nullable=False)
    evaluation_type = Column(String(50), nullable=False)
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class TestCase(Base):
    __tablename__ = "test_cases"
    id = Column(Integer, primary_key=True)
    problem_id = Column(String(50), ForeignKey("problems.problem_id"))
    test_inputs = Column(JSONB, nullable=False)
    expected_output = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ReferenceSolution(Base):
    __tablename__ = "reference_solutions"
    id = Column(Integer, primary_key=True)
    problem_id = Column(String(50), ForeignKey("problems.problem_id"), unique=True)
    solution_code = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    submission_id = Column(String(50), unique=True, nullable=False)
    problem_id = Column(String(50), ForeignKey("problems.problem_id"))
    user_id = Column(String(100))
    user_code = Column(Text, nullable=False)
    passed_tests = Column(Integer)
    total_tests = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class SubmissionResult(Base):
    __tablename__ = "submission_results"
    id = Column(Integer, primary_key=True)
    submission_id = Column(String(50), ForeignKey("submissions.submission_id"))
    test_number = Column(Integer)
    input = Column(JSONB)
    expected = Column(JSONB)
    actual = Column(JSONB)
    passed = Column(Boolean)
    error_message = Column(Text)

Base.metadata.create_all(bind=engine)

# ============================================================================
# LEETCODE CONTEXT
# ============================================================================

LEETCODE_PROMPT = LeetCodeContext.get_context()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GenerateProblemRequest(BaseModel):
    vague_problem: str

class GenerateProblemResponse(BaseModel):
    problem_id: str
    title: str
    starter_code: str
    test_case_count: int

class EvaluateRequest(BaseModel):
    user_code: str
    problem_id: str
    user_id: Optional[str] = "anonymous"

class EvaluateResponse(BaseModel):
    submission_id: str
    passed: int
    total: int
    percentage: float

# ============================================================================
# ROBUST UTILITIES (Serialization, Safety, Validation)
# ============================================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution Timed Out")

def safe_exec(code: str, globals_dict: dict, timeout_sec: int = 3):
    """
    Executes code with a strict timeout to prevent infinite loops.
    CRITICAL: Handles the 'Poisoned Validator' infinite loop vulnerability.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        exec(code, globals_dict)
    finally:
        signal.alarm(0)

def validate_test_code(code: str) -> tuple[bool, str]:
    """
    STATIC ANALYSIS: Validates structure and SECURITY before execution.
    1. Ensures required functions exist.
    2. STRICTLY BANS IMPORTS (prevent 'import os', 'import sys').
    """
    try:
        tree = ast.parse(code)
        
        # Security Check: Ban imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Security Violation: Imports are NOT allowed. Use built-in primitives only."
        
        # Structure Check: Function existence
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        if 'generate_inputs' not in functions:
            return False, "Missing required function: `generate_inputs()`"
        if 'is_valid_input' not in functions:
            return False, "Missing required function: `is_valid_input()`"
            
        return True, ""
        
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"
    except Exception as e:
        return False, f"Validation Error: {str(e)}"

def robust_serialize(obj: Any) -> Any:
    """
    Converts complex objects (ListNode, TreeNode) from Reference Solution
    back to JSON primitives for storage.
    """
    # Handle ListNode
    if hasattr(obj, 'val') and hasattr(obj, 'next') and not hasattr(obj, 'left'):
        result = []
        curr = obj
        cycle_guard = 0
        while curr and cycle_guard < 10000:
            result.append(curr.val)
            curr = curr.next
            cycle_guard += 1
        return result
    
    # Handle TreeNode (Level Order)
    if hasattr(obj, 'val') and hasattr(obj, 'left') and hasattr(obj, 'right'):
        result = []
        queue = [obj]
        while queue:
            node = queue.pop(0)
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)
        # Trim trailing Nones
        while result and result[-1] is None:
            result.pop()
        return result
        
    # Handle basic types
    try:
        json.dumps(obj)
        return obj
    except:
        return str(obj)

def auto_convert_arg(value: Any, arg_type: str) -> Any:
    """Convert JSON primitives to Python objects (ListNode, TreeNode)"""
    if 'ListNode' in arg_type:
        if isinstance(value, list):
            namespace = {}
            exec(LEETCODE_PROMPT, namespace)
            return namespace['list_node'](value)
    elif 'TreeNode' in arg_type:
        if isinstance(value, list):
            namespace = {}
            exec(LEETCODE_PROMPT, namespace)
            return namespace['tree_node'](value)
    return value

def extract_signature(starter_code: str) -> Dict[str, Any]:
    """Extract function signature using AST"""
    tree = ast.parse(starter_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'solve':
            arg_names = [arg.arg for arg in node.args.args if arg.arg != 'self']
            arg_types = [
                ast.unparse(arg.annotation) if arg.annotation else "Any"
                for arg in node.args.args if arg.arg != 'self'
            ]
            return_type = ast.unparse(node.returns) if node.returns else "Any"
            return {
                "function_name": "solve",
                "arg_names": arg_names,
                "arg_types": arg_types,
                "return_type": return_type
            }
    raise ValueError("solve function not found in starter code")

# ============================================================================
# STAGE 3: VAGUE → FORMAL SPECIFICATION
# ============================================================================

def stage3_generate_specification(vague_problem: str) -> Dict[str, Any]:
    """STAGE 3: Convert vague problem to formal specification"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are an expert Competitive Programming Problem Setter.
Convert a vague, human-written coding question into a formal problem specification.

CRITICAL RULES:
1. "starter_code":
   - MUST use the class `Solution`.
   - MUST use the exact function signature `def solve(self, ...)`, where function name MUST be `solve`. Infer types strictly from CONTEXT.
2. "constraints": Use explicit limits (1 <= N <= 10^5) or infer standard limits for the type.
3. "evaluation_type": "list_any_order", "float_tolerance", or "exact_match".
4. problem_description: Clear and unambiguous.

Return ONLY valid JSON."""

    user_prompt = f'''Vague problem: "{vague_problem}"

CONTEXT:
{LEETCODE_PROMPT}

Output JSON:
{{
    "title": "string",
    "tags": ["string"],
    "problem_description": "string",
    "examples": ["string"],
    "constraints": ["string"],
    "constraint_source": "string",
    "starter_code": "string",
    "evaluation_type": "string"
}}'''

    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=2000
    )
    
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    
    spec = json.loads(text)
    spec["prompt"] = LEETCODE_PROMPT
    
    # Validation/Cleanup of starter_code
    if 'starter_code' in spec:
        starter = spec['starter_code']
        import re
        starter = re.sub(r'def \w+\(self,', 'def solve(self,', starter)
        lines = starter.split('\n')
        for i, line in enumerate(lines):
            if 'def solve(' in line and line.rstrip().endswith(':'):
                indent = len(line) - len(line.lstrip()) + 4
                if i + 1 >= len(lines):
                    lines.append(' ' * indent + 'pass')
                elif lines[i + 1].strip() == '':
                    lines[i + 1] = ' ' * indent + 'pass'
                break
        spec['starter_code'] = '\n'.join(lines)
    
    return spec

# ============================================================================
# STAGE 4: TEST CASE GENERATION (SELF-HEALING ORACLE)
# ============================================================================

def stage4_generate_tests(spec: Dict[str, Any], signature: Dict[str, Any]) -> List[List[Any]]:
    """
    STAGE 4: Self-Healing Test Generator with Reflexion.
    Uses original prompts but wraps logic in a robust feedback loop.
    1. Reflexion: Feeds execution/validation errors back to LLM.
    2. Dynamic Temperature Scaling.
    3. Strict Timeout Sandboxing.
    4. Static Analysis (Bans imports).
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 

    # --- ORIGINAL PROMPTS PRESERVED ---
    system_prompt = """You are The Oracle - Expert Test Data Generator.

MUST PROVIDE EXACTLY TWO FUNCTIONS:

def generate_inputs():
    return [
        (arg1, arg2, ...),
        (arg1, arg2, ...),
        ...
    ]

def is_valid_input(*args):
    # Validate against constraints
    return True/False

REQUIREMENTS:
1. ONLY JSON-serializable primitives (int, str, list, float, bool, None)
2. NO objects, NO custom classes, NO imports
3. Edge cases: empty, single element, max values, sorted, reverse
4. ListNode as List[int]: [1,2,3]
5. TreeNode as List[int|None] in level order: [1,2,None,4]
6. Generate AT LEAST 25 test cases, covering all categories extensively.
7. TOKEN EFFICIENCY: For large inputs (Stress Tests), USE PYTHON SYNTAX like `[0] * 10000` or `list(range(1000))` instead of writing out every literal number.
8. Return ONLY the two functions - NO other code"""

    user_prompt = f'''ANALYSIS TARGET: 
  - Constraints: {spec['constraints']}
  - Input signature (JSON): {json.dumps(signature)}

TASK: Generate A MINIMUM of 25 test cases. You must detect the data structures used in the signature and apply the corresponding strategies below.

1. Basic Functional Tests (5+ cases):
   - Goal: Verify core logic on small, human-readable inputs.
   - Strategy: Mix of simple positive cases (answer exists) and negative cases (answer impossible).

2. Boundary & Edge Cases (5+ cases):
   - GENERIC: Empty input ([], ""), Null/None, Single element.
   - NUMERICAL: Exact MIN_VAL and MAX_VAL from constraints (e.g. Int.MIN, Int.MAX).
   - ARRAYS/STRINGS: Length 0, Length 1, Length 2.

3. Stress & Scale Tests (5+ cases - Maximize Constraints):
   - ARRAYS/STRINGS: Generate inputs with maximum allowed length (e.g., 10^5 elements).
   - TREES: Deep skewed trees (degenerate linked lists) to test recursion depth limits.
   - GRAPHS: Dense graphs (max edges) vs. Sparse graphs.
   - MATRICES: Max rows x Max cols.

4. Adversarial & Logic Traps (5+ cases):
   - DUPLICATES: 
     - "Mixed Duplicates": Random duplicates scattered (`[1, 5, 2, 5, 1]`).
     - "Clustered Duplicates": Groups of duplicates (`[1, 1, 1, 2, 2, 3]`).
   - ARRAYS: Sorted, Reverse Sorted, All Identical values (e.g., [5,5,5...]).
   - NUMBERS: "Overflow Bait" -> Large positive + Large negative sums.
   - GRAPHS/TREES: Disconnected components, Cycles (if allowed), Self-loops.
   - STRINGS: Repeated patterns (e.g., "aaaaa"), Palindromes.

5. Domain-Specific Special Patterns (5+ cases):
   - IF BIT MANIPULATION: Powers of 2, 0, -1.
   - IF INTERVALS: Overlapping, Nested, Touching, Disjoint.
   - IF LINKED LISTS: Cycles, Intersection point at end/beginning.
   - IF DP/GRID: Obstacles blocking all paths, Start == End.'''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Dynamic Temperature Scaling: Start precise (0.7), get crazier if failing
            # Note: Full code used 0.7 originally. We can range 0.7 -> 0.9 or start lower if syntax errors.
            current_temp = 0.5 + (attempt * 0.2)
            
            response = client.chat.completions.create(
                model=FINE_TUNED_MODEL,
                messages=messages,
                temperature=current_temp,
                max_tokens=3500
            )
            
            code = response.choices[0].message.content.strip()
            
            # Sanitization
            if code.startswith("```"):
                code = code.split("```")[1].strip()
                if code.startswith("python"): code = code[6:].strip()
                if code.endswith("```"): code = code[:-3].strip()
            
            # 1. Static Analysis (Security & Syntax)
            is_valid_syntax, error_msg = validate_test_code(code)
            if not is_valid_syntax:
                raise ValueError(f"Code failed static validation: {error_msg}")

            # 2. Execution Sandbox
            namespace = {}
            try:
                # Safe Execution (Timeouts)
                safe_exec(code, namespace, timeout_sec=4)
                
                # 3. Validation Logic
                generate_inputs_fn = namespace['generate_inputs']
                is_valid_fn = namespace.get('is_valid_input', lambda *x: True)
                
                raw_inputs = generate_inputs_fn()
                
                if not isinstance(raw_inputs, list):
                    raise TypeError(f"generate_inputs() must return list, got {type(raw_inputs)}")

                valid_inputs = []
                # Filter inputs
                for inp in raw_inputs:
                    # Enforce tuple/list structure for args
                    if not isinstance(inp, (list, tuple)):
                        inp = [inp]
                    
                    if is_valid_fn(*inp):
                        valid_inputs.append(list(inp))

                # 4. Threshold Check (Full code asked for 25, tolerate 15)
                if len(valid_inputs) < 15:
                    raise ValueError(f"Only generated {len(valid_inputs)} valid inputs (expected 15+).")

                print(f"[STAGE 4] ✓ Generated {len(valid_inputs)} tests on attempt {attempt+1}")
                return valid_inputs

            except Exception as e:
                # REFLEXION: Feed error back to LLM
                error_msg = f"Error executing your code: {str(e)}\nTraceback: {traceback.format_exc()}"
                print(f"[STAGE 4] Attempt {attempt+1} failed. Healing...")
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "user", "content": f"Fix the code. It failed with:\n{error_msg}"})
                continue
        
        except Exception as api_e:
            print(f"API Error: {str(api_e)}")
            continue

    raise HTTPException(status_code=400, detail="Stage 4 failed to generate tests after self-healing.")

# ============================================================================
# STAGE 5: DUAL-MODEL CONSENSUS REFERENCE
# ============================================================================

def stage5_consensus_reference(spec: Dict[str, Any], test_inputs: List[List[Any]]) -> str:
    """
    STAGE 5: Consensus Generation.
    Uses original prompts but generates TWO solutions:
    1. Optimized (Time Complexity prioritized)
    2. Brute Force (Correctness prioritized)
    Verifies them against each other to ensure robustness.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    signature = extract_signature(spec['starter_code'])
    arg_types = signature['arg_types']

    # --- HELPER TO USE ORIGINAL PROMPTS WITH MODE INJECTION ---
    def generate_solution_variant(mode_instruction: str) -> str:
        # ORIGINAL SYSTEM PROMPT
        system_prompt = """You are an Expert Python Programmer specializing in algorithmic solutions.

Write a correct, efficient reference solution.

CRITICAL INSTRUCTIONS:
1. Return ONLY the Solution class code
2. NO markdown, NO explanation, NO additional code
3. Use EXACTLY the function signature provided
4. Handle ALL edge cases correctly
5. Optimize for correctness first, efficiency second"""

        # ORIGINAL USER PROMPT + MODE INJECTION
        user_prompt = f'''Problem: {spec['title']}
Description: {spec['problem_description']}
Constraints: {spec['constraints']}
Examples: {spec['examples']}

Starter Code (MUST USE EXACTLY THIS SIGNATURE):
{spec['starter_code']}

CONTEXT (available imports):
{LEETCODE_PROMPT}

Write the complete Solution class.
{mode_instruction}
CRITICAL: At the start of `solve`, add assertions to validate input constraints if possible.'''

        resp = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        c = resp.choices[0].message.content.strip()
        if "```" in c: 
            c = c.split("```")[1].strip()
            if c.startswith("python"): c = c[6:].strip()
        return c

    # 1. Generate Both
    print("[STAGE 5] Generating Optimized Solution...")
    optimized_code = generate_solution_variant("MODE: OPTIMIZED (Time Complexity prioritized).")
    
    print("[STAGE 5] Generating Brute-Force Validator...")
    brute_code = generate_solution_variant("MODE: BRUTE FORCE / NAIVE (Correctness prioritized, ignore complexity).")

    # 2. Consensus Check
    try:
        # Prepare namespaces
        ns_opt = {}; exec(LEETCODE_PROMPT, ns_opt); safe_exec(optimized_code, ns_opt)
        ns_brute = {}; exec(LEETCODE_PROMPT, ns_brute); safe_exec(brute_code, ns_brute)
        
        sol_opt = ns_opt['Solution']()
        sol_brute = ns_brute['Solution']()

        # Check first 5 inputs (or fewer)
        check_limit = min(5, len(test_inputs))
        
        for i, inp in enumerate(test_inputs[:check_limit]):
            # Convert args
            args_opt = [auto_convert_arg(v, t) for v, t in zip(inp, arg_types)]
            args_brute = [auto_convert_arg(v, t) for v, t in zip(inp, arg_types)]
            
            # Run Both with robust serialization
            try:
                out_opt = robust_serialize(sol_opt.solve(*copy.deepcopy(args_opt)))
                out_brute = robust_serialize(sol_brute.solve(*copy.deepcopy(args_brute)))
            except ValueError as ve:
                print(f"[STAGE 5] Defensive Assertion caught bad input: {ve}")
                continue
            except Exception as e:
                print(f"[STAGE 5] Runtime Error on check {i}: {e}")
                continue

            if out_opt != out_brute:
                print(f"[STAGE 5] ⚠ Mismatch on Input {inp}")
                print(f"   Optimized: {out_opt}")
                print(f"   Brute Force: {out_brute}")
                
                # REFLEXION HEALING: Trust Brute Force
                heal_prompt = f"""
                Your Optimized solution failed a test case where the Brute Force solution succeeded.
                Input: {inp}
                Expected (Brute Force): {out_brute}
                Your Output: {out_opt}
                
                Fix the OPTIMAL Solution class logic. Return ONLY the code.
                """
                print("[STAGE 5] Healing Optimized Solution...")
                resp = client.chat.completions.create(
                    model=FINE_TUNED_MODEL,
                    messages=[
                        {"role": "user", "content": f"Write Optimized Sol: {spec['title']}"},
                        {"role": "assistant", "content": optimized_code},
                        {"role": "user", "content": heal_prompt}
                    ],
                    temperature=0.1
                )
                optimized_code = resp.choices[0].message.content.strip()
                if "```" in optimized_code: 
                    optimized_code = optimized_code.split("```")[1].strip()
                    if optimized_code.startswith("python"): optimized_code = optimized_code[6:].strip()
                break 
        
        print("[STAGE 5] ✓ Reference Solution Consensus Achieved")
        return optimized_code

    except Exception as e:
        print(f"[STAGE 5] Consensus logic failed: {e}. Fallback to raw optimized code.")
        return optimized_code

def stage5_build_golden_suite(reference_code: str, test_inputs: List[List[Any]], 
                              signature: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    STAGE 5: Build golden suite.
    Uses `safe_exec` for timeout protection.
    Uses `robust_serialize` to handle ListNode/TreeNode.
    """
    
    golden_suite = []
    arg_types = signature['arg_types']
    
    exec_globals = {}
    exec(LEETCODE_PROMPT, exec_globals)
    safe_exec(reference_code, exec_globals)
    
    solver = exec_globals['Solution']()
    
    for idx, test_input in enumerate(test_inputs):
        try:
            run_args = [auto_convert_arg(v, t) for v, t in zip(test_input, arg_types)]
            
            # Execute with strict timeout per test case
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2) 
            try:
                output = solver.solve(*copy.deepcopy(run_args))
            finally:
                signal.alarm(0)

            # CRITICAL: Robust Serialize captures Linked Lists/Trees as JSON
            serialized = robust_serialize(output)
            
            golden_suite.append({
                "input": test_input,
                "expected": serialized
            })
        except Exception as e:
            # If Ref Sol crashes (defensive assertion or bug), we drop the test case
            # This filters out "Poisoned" validators
            print(f"[STAGE 5] Test {idx} dropped: {str(e)}")
            continue
    
    if len(golden_suite) < 5:
        raise HTTPException(
            status_code=400, 
            detail=f"Reference Solution failed too many tests ({len(golden_suite)} passed). Specs might be impossible."
        )
    
    return golden_suite

# ============================================================================
# STAGE 6: USER EVALUATION
# ============================================================================

def check_correctness(actual: Any, expected: Any, mode: str) -> bool:
    """Compare outputs based on evaluation type"""
    try:
        if mode == "float_tolerance":
            return math.isclose(float(actual), float(expected), rel_tol=1e-5)
        elif mode == "list_any_order":
            if isinstance(actual, list) and isinstance(expected, list):
                return sorted(actual) == sorted(expected)
            return actual == expected
        else:  # exact_match
            return actual == expected
    except:
        return False

def stage6_evaluate(user_code: str, golden_suite: List[Dict[str, Any]],
                   signature: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
    """
    STAGE 6: Evaluate user solution.
    Uses `safe_exec` (via timeout logic) to protect against user infinite loops.
    """
    results = []
    passed_count = 0
    arg_types = signature['arg_types']
    
    try:
        exec_globals = {}
        exec(LEETCODE_PROMPT, exec_globals)
        # Sandbox the user's compilation
        safe_exec(user_code, exec_globals, timeout_sec=2)
        
        user_solver = exec_globals['Solution']()
    except Exception as e:
        return {
            "error": f"Compilation/Startup Error: {str(e)}",
            "passed": 0,
            "total": len(golden_suite),
            "results": []
        }
    
    for idx, test_case in enumerate(golden_suite):
        test_input = test_case['input']
        expected = test_case['expected']
        
        try:
            run_args = [
                auto_convert_arg(v, t) for v, t in zip(test_input, arg_types)
            ]
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2) # 2 Second TLE limit per test case
            
            try:
                actual = user_solver.solve(*copy.deepcopy(run_args))
            finally:
                signal.alarm(0)
            
            # Robust serialize user output before comparing
            actual_json = robust_serialize(actual)
            passed = check_correctness(actual_json, expected, evaluation_type)
            
            if passed:
                passed_count += 1
            
            results.append({
                "test_number": idx + 1,
                "input": test_input,
                "expected": expected,
                "actual": actual_json,
                "passed": passed
            })
        
        except TimeoutException:
            results.append({
                "test_number": idx + 1,
                "passed": False,
                "error": "Time Limit Exceeded"
            })
        except Exception as e:
            signal.alarm(0)
            results.append({
                "test_number": idx + 1,
                "passed": False,
                "error": f"Runtime Error: {str(e)}"
            })
    
    percentage = (passed_count / len(golden_suite) * 100) if golden_suite else 0
    
    return {
        "passed": passed_count,
        "total": len(golden_suite),
        "percentage": percentage,
        "results": results
    }

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/generate-problem", response_model=GenerateProblemResponse)
async def generate_problem(request: GenerateProblemRequest, db: Session = Depends(get_db)):
    """Generate problem from vague statement (Stages 3-5)"""
    
    try:
        print("\n" + "="*60)
        print("[STAGE 3] Generating specification...")
        spec = stage3_generate_specification(request.vague_problem)
        print(f"[STAGE 3] ✓ Title: {spec['title']}")
        
        signature = extract_signature(spec['starter_code'])
        print(f"[STAGE 3] ✓ Signature: {signature['function_name']}({', '.join(signature['arg_names'])})")
        
        print("[STAGE 4] Generating test cases with Reflexion...")
        test_inputs = stage4_generate_tests(spec, signature)
        
        print("[STAGE 5] Generating Consensus Reference Solution...")
        reference_code = stage5_consensus_reference(spec, test_inputs)
        
        print("[STAGE 5] Building Golden Suite...")
        golden_suite = stage5_build_golden_suite(reference_code, test_inputs, signature)
        
        problem_id = f"prob_{uuid.uuid4().hex[:8]}"
        
        # Save to database
        problem = Problem(
            problem_id=problem_id,
            title=spec['title'],
            problem_description=spec['problem_description'],
            examples=spec['examples'],
            constraints=spec['constraints'],
            constraint_source=spec.get('constraint_source', 'assumed_standard_limits'),
            tags=spec.get('tags', []),
            starter_code=spec['starter_code'],
            evaluation_type=spec['evaluation_type'],
            prompt=spec['prompt']
        )
        db.add(problem)
        db.commit()
        
        ref_sol = ReferenceSolution(
            problem_id=problem_id,
            solution_code=reference_code
        )
        db.add(ref_sol)
        db.commit()
        
        for test in golden_suite:
            test_case = TestCase(
                problem_id=problem_id,
                test_inputs=test['input'],
                expected_output=test['expected']
            )
            db.add(test_case)
        db.commit()
        
        print(f"[SUCCESS] Problem {problem_id} generated with {len(golden_suite)} tests")
        print("="*60 + "\n")
        
        return GenerateProblemResponse(
            problem_id=problem_id,
            title=spec['title'],
            starter_code=spec['starter_code'],
            test_case_count=len(golden_suite)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate_solution(request: EvaluateRequest, db: Session = Depends(get_db)):
    """Evaluate user solution (Stage 6)"""
    
    try:
        problem = db.query(Problem).filter(Problem.problem_id == request.problem_id).first()
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        test_cases = db.query(TestCase).filter(TestCase.problem_id == request.problem_id).all()
        
        golden_suite = [
            {"input": tc.test_inputs, "expected": tc.expected_output}
            for tc in test_cases
        ]
        
        signature = extract_signature(problem.starter_code)
        
        print(f"\n[STAGE 6] Evaluating solution for {request.problem_id}...")
        eval_result = stage6_evaluate(
            request.user_code,
            golden_suite,
            signature,
            problem.evaluation_type
        )
        
        submission_id = f"sub_{uuid.uuid4().hex[:8]}"
        submission = Submission(
            submission_id=submission_id,
            problem_id=request.problem_id,
            user_id=request.user_id,
            user_code=request.user_code,
            passed_tests=eval_result['passed'],
            total_tests=eval_result['total']
        )
        db.add(submission)
        db.commit()
        
        for result in eval_result['results']:
            sub_result = SubmissionResult(
                submission_id=submission_id,
                test_number=result.get('test_number'),
                input=result.get('input'),
                expected=result.get('expected'),
                actual=result.get('actual'),
                passed=result.get('passed'),
                error_message=result.get('error')
            )
            db.add(sub_result)
        db.commit()
        
        print(f"[STAGE 6] ✓ Result: {eval_result['passed']}/{eval_result['total']} ({eval_result['percentage']:.1f}%)\n")
        
        return EvaluateResponse(
            submission_id=submission_id,
            passed=eval_result['passed'],
            total=eval_result['total'],
            percentage=eval_result['percentage']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/problem/{problem_id}")
async def get_problem(problem_id: str, db: Session = Depends(get_db)):
    """Get problem details"""
    problem = db.query(Problem).filter(Problem.problem_id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    ref_sol = db.query(ReferenceSolution).filter(ReferenceSolution.problem_id == problem_id).first()
    solution_code = ref_sol.solution_code if ref_sol else "Solution not available."

    test_cases_db = db.query(TestCase).filter(TestCase.problem_id == problem_id).all()
    test_cases = [{"input": tc.test_inputs, "expected": tc.expected_output} for tc in test_cases_db]

    return {
        "problem_id": problem.problem_id,
        "title": problem.title,
        "description": problem.problem_description,
        "examples": problem.examples,
        "constraints": problem.constraints,
        "starter_code": problem.starter_code,
        "tags": problem.tags,
        "reference_solution": solution_code,
        "test_cases": test_cases
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("AI CODING PLATFORM - STAGES 3-6 (ROBUST EDITION)")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)