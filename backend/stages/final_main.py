"""
AI Coding Platform - Stages 3-6 Implementation
PRODUCTION VERSION: Exclusively uses fine-tuned model
STAGE 3: Vague → Formal Specification
STAGE 4: Test Case Generation (The Oracle)
STAGE 5: Reference Solution Generation
STAGE 6: User Submission Evaluation
"""

import os
import json
import uuid
import copy
import math
import ast
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import traceback
import multiprocessing
import queue

from utils import LeetCodeContext
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

class ChallengeExample(BaseModel):
    input: str
    output: str

class ImprovementRequest(BaseModel):
    examples: List[ChallengeExample]

class ChallengeAnalysisSchema(BaseModel):
    """
    Structured audit report from the Gatekeeper.
    """
    extracted_constraints: str = Field(..., description="The specific numeric limits extracted from text (e.g. '1 <= N <= 50').")
    input_analysis: str = Field(..., description="Analysis of the user's input values against those limits.")
    violation_detected: bool = Field(..., description="True if constraints are violated.")
    decision: Literal["YES", "NO"] = Field(..., description="'NO' if violation_detected is True. 'YES' if valid input.")
    reason: str = Field(..., description="Detailed explanation for the user.")

# Validation with Reflexion Feedback.
class OracleFailure(Exception):
    """Raised when LLM-generated tests fail Oracle validation."""
    pass

class TestQualityFailure(Exception):
    """Raised when Reference Solution rejects the test suite (Circular Healing)."""
    pass

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

# TODO
# Base.metadata.create_all(bind=engine)

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
# UTILITIES
# ============================================================================

def _exec_worker(code: str, globals_dict: dict, result_queue):
    try:
        exec(code, globals_dict)
        result_queue.put(("ok", None))
    except Exception as e:
        result_queue.put(("error", str(e)))

def safe_exec(code: str, globals_dict: dict, timeout_sec: int = 6):
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_exec_worker,
        args=(code, globals_dict, result_queue),
    )
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        raise TimeoutError("Execution timed out")

    if result_queue.empty():
        raise RuntimeError("Child process exited without response")

    status, payload = result_queue.get()
    if status == "error":
        raise RuntimeError(payload)

    # IMPORTANT: exec AGAIN in parent to populate globals
    exec(code, globals_dict)

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
   - If the problem involves Linked Lists, use `ListNode` (from CONTEXT).
   - If Binary Trees, use `TreeNode` (from CONTEXT).
2. "constraint_source" & "constraints":
   Determine the source (priority: user_provided > inferred > assumed).
   
   A. "user_provided": MUST use if ONLY explicit limits appear in the text.
      - Example: "The list won't have more than 50 items" -> "list length <= 50"
      - Example: "Numbers are positive" -> "arr[i] >= 0"
      
   B. "inferred": Use if examples imply data characteristics (NOT size limits).
      - Example: Input `[-5, 10, 0]` -> "arr[i] can be negative" (Range: -10^9 to 10^9)
      - Example: Input `"hello"` -> "s consists of lowercase English letters"
      
   C. "assumed_standard_limits": Use STRICT defaults based on input type if A & B are missing.
      - Linear Structures (Arrays, Strings): Assume `1 <= N <= 10^5` (Implies O(N) or O(N log N) solution).
      - 2D Grids / Dense Graphs: Assume `1 <= N, M <= 500` (Implies O(N^2) solution).
      - Exponential / Backtracking (e.g., subsets, permutations): Assume `1 <= N <= 15` (Implies O(2^N) solution).
      - Integers: Assume standard 32-bit signed range `-2^31 <= val <= 2^31 - 1`.
3. "evaluation_type": Determine how to compare outputs. Can take one-of-three values:
    - "list_any_order": Use if the problem asks for a collection where the sequence doesn't matter (e.g., indices, sets, combinations). For example, "return indices of two numbers".
   - "float_tolerance": Use ONLY if the return type is a float (allow 1e-5 error).
   - "exact_match": Use for integers, strings, booleans, or lists where order is strictly defined (e.g., "return the list in sorted order").
4. problem_description: Should be clear, complete, and unambiguous
5. tags: Add relevant problem category tags. (e.g., ["Array", "Bit Manipulation"]).
6. "constraints": 
   - DETECT & FIX TYPOS: If the user writes "105" or "104" in a context implying size, convert to "10**5" or "10**4".
   - PYTHON SYNTAX: Constraints must use valid Python syntax for exponents (e.g., `1 <= N <= 10**5`, NOT `10^5`).

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
    if 'starter_code' in spec:
        starter = spec['starter_code']
        import re
        # Ensure function name is 'solve'
        starter = re.sub(r'def \w+\(self,', 'def solve(self,', starter)
        
        # Ensure starter code ends with pass if incomplete
        lines = starter.split('\n')
        # Find the line with def solve and add pass if needed
        for i, line in enumerate(lines):
            if 'def solve(' in line and line.rstrip().endswith(':'):
                # Calculate proper indentation for pass statement
                indent = len(line) - len(line.lstrip()) + 4
                # Insert pass on the next line if it doesn't exist
                if i + 1 >= len(lines):
                    lines.append(' ' * indent + 'pass')
                elif lines[i + 1].strip() == '':
                    lines[i + 1] = ' ' * indent + 'pass'
                break
        
        starter = '\n'.join(lines)
        spec['starter_code'] = starter
    
    return spec

# ============================================================================
# STAGE 4: TEST CASE GENERATION (THE ORACLE)
# ============================================================================

def stage4_generate_tests(spec: Dict[str, Any], signature: Dict[str, Any], feedback_history: str = "") -> tuple[List[List[Any]], Dict[str, List[Any]]]:
    """STAGE 4: Generate test cases using fine-tuned model with retries"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are a Senior QA Engineer - Expert Test Data Generator.

MUST PROVIDE EXACTLY TWO PYTHON FUNCTIONS:

def generate_inputs():
    return {
        "Basic Functional Tests": [ (arg1, arg2, ...), ... ],
        "Boundary & Edge Cases": [ (arg1, arg2, ...), ... ],
        "Stress & Scale Tests": [ (arg1, arg2, ...), ... ],
        "Adversarial & Logic Traps": [ (arg1, arg2, ...), ... ],
        "Domain-Specific Special Patterns": [ (arg1, arg2, ...), ... ],
    }

def is_valid_input(*args):
    # Validate against constraints
    # MUST return (bool, str) -> (True, "") or (False, "Reason for failure") e.g., if len(arr) > 5000: return (False, "Array too long")
    return (is_valid, reason)

REQUIREMENT 0 (CRITICAL):
    Before writing code, you MUST write a plan using ONLY single-line comments (#):
    1. Analysis of constraints (identifying coupled limits like N*M).
    2. Strategy for Edge Cases.
    3. Mathematical proof that Stress Tests fit memory limits.

    Example:
    # PLAN:
    # 1. Constraint analysis...
    # 2. Strategy...
    

OUTPUT: Provide ONLY two functions: generate_inputs() and is_valid_input().
    
REQUIREMENTS:
1. **MOST CRITICAL**: ONLY GENERATE TEST CASES THAT COMPLY WITH THE GIVEN CONSTRAINTS. (e.g., **Strict Limits:** If `1 <= length <= 100`, length 0 is INVALID. length 101 is INVALID.)
2. ONLY JSON-serializable primitives (int, str, list, float, bool, None).
3. NO objects, NO custom classes, NO imports.
4. Edge cases: empty, single element, max values, sorted, reverse.
5. **DATA STRUCTURE FORMATTING (CRITICAL)**:
  - ListNode as List[int]: [1,2,3].
  - TreeNode as List[int|None] in level order: [1,2,None,4]. MUST use LeetCode standard **Level-Order Traversal** (Flat List).
  - Linked Lists: Ensure no unintentional cycles unless testing for them.
  - Matrices: Ensure rows are consistent lengths unless "Jagged Array" is explicitly allowed/implied by the problem (e.g. "Triangle").
7. Generate AT LEAST 10 test cases, covering all categories extensively.
8. **TOKEN EFFICIENCY & COMPLEXITY (CRITICAL)**: 
   - For large inputs, USE PYTHON SYNTAX (e.g., `[0] * 10000`).
   - **DO NOT** generate O(N^2) data for N > 500. 
   - *Example:* For N=50,000, generate a sparse graph (Edges = N or 2N), NOT a dense graph.
   - For strings: **NEVER** manually type out long repeated patterns (e.g., "aaaaa..."). **ALWAYS** use Python string multiplication for stress tests. *Bad:* `["aaaaaaaaaa..."]` (Hits token limit, causes truncation error). *Good:* `["a" * 10000]` or `["/*" + "*" * 500 + "*/"]`.
9. If a parameter in the input signature is a TreeNode list, allow 'None' values in the list validation.
10. Sanity Check: Your `generate_inputs` output MUST pass your own `is_valid_input` check.
11. **COUPLED CONSTRAINT COMPLIANCE**: Ensure mathematical compatibility in stress tests (e.g., if N is max, ensure K fits).
12. **STRICT CONSTRAINT GROUNDING (CRITICAL)**: 
    - Your `is_valid_input` must derive its logic **EXCLUSIVELY** from the explicit `Constraints` text provided.
    - **ZERO ASSUMPTION POLICY**: You are FORBIDDEN from enforcing "implied" or "standard" constraints that are not written down.
      - *Example (Graph):* Do NOT enforce connectivity or acyclicity unless the text explicitly says "Tree" or "Connected".
      - *Example (Array):* Do NOT enforce distinctness or sorted order unless explicitly stated.
    - If the constraint text is silent on a property, you MUST allow ALL valid variations (e.g., both connected and disconnected graphs).
13. **CRITICAL**: For Stress & Scale Tests: BEFORE generating any stress test, you MUST:
        A. Identify all numeric constraints.
        B. Detect COUPLED limits (e.g., N vs edges, N vs K, rows × cols).
        C. Compute a FEASIBLE maximum that satisfies ALL constraints SIMULTANEOUSLY.
        EXAMPLES (MANDATORY BEHAVIOR):
        - If N <= 100000 AND edges <= 50000:
            • For N = 100000 → edges MUST be <= 50000
            • Do NOT generate edges proportional to N
            • Prefer sparse patterns: star, partial chain, random sampling
        - If matrix rows * cols <= 10^5:
            • Do NOT maximize both rows and cols simultaneously.
        FAILURE TO RESPECT COUPLED LIMITS IS A HARD ERROR.
14. Return a DICTIONARY with exactly the 5 keys listed in `generate_inputs()`.
15. **SYNTAX SAFETY (CRITICAL)**: 
   - When generating string inputs that contain code, quotes, or special characters (e.g., C++ source, SQL, nested JSON), **YOU MUST USE PYTHON TRIPLE QUOTES** (`'''` or `\"\"\"`) to enclose the string.
   - *Bad:* `["printf("hello")"]` (Causes Syntax Error).
   - *Good:* `['''printf("hello")''']` (Safe).
   - If constraints forbid specific characters (e.g., "No double quotes"), you MUST reject them.
16. **VALIDATOR LOGIC SOPHISTICATION**:
   - For problems involving **Parsing, Parentheses, or Comments**: Your `is_valid_input` **MUST USE A STATE MACHINE or STACK**.
   - **DO NOT** use naive counters (e.g., `count("/*")`) because they fail on nested content (e.g., `// /*` is a valid line comment, not an open block).
   - If the problem implies precedence (e.g., line comment beats block comment), your validator must respect that precedence.
   - For **Graph/Tree** problems:
     * Do NOT assume connectivity or acyclicity unless explicitly stated in constraints.
     * Allow disconnected components if not forbidden.
     * If input is `edges`, validate that node indices are within `0..n-1`
   - **Recursion Safety (CRITICAL)**: For Trees/Graphs, your validator MUST use **ITERATIVE** traversal (Stack/Queue). Recursive validation will crash on deep trees (RecursionError) and fail valid stress tests.
17. **Floating Point**: If inputs are floats, check constraints for precision. Do not enforce bitwise equality unless required.
18. **Graph Ambiguity**: You MUST infer the format from the variable name and problem text:
     * `edges` or `connections`: Use **Edge List** `[[u, v], [x, y]]`.
     * `graph`, `adj`, or `neighbors`: Use **Adjacency List** `[[neighbors_of_0], [neighbors_of_1]]`.
19. **ITERATIVE VALIDATION**: Your `is_valid_input` must be iterative (no recursion) to handle large inputs without StackOverflow.
20. **TRIPLE QUOTE SAFETY**: If generating strings with quotes, use 'single quotes' for the Python string to avoid breaking the script structure.
21. **NO IMPORTS ALLOWED**: Do NOT write `import sys` or `import math`. 
   - The execution environment ALREADY has standard libraries loaded (collections, itertools, math, etc.).
   - Recursion limits are ALREADY set to 5000. 
   - Writing imports will trigger a Security Violation.
22. Return ONLY the two functions - NO other code"""

    user_prompt = f'''ANALYSIS TARGET: 
  - Constraints: {spec['constraints']}
  - Input signature (JSON): {json.dumps(signature)}

TASK: Generate MINIMUM 10 VALID and MEANINGFULLY DISTINCT test cases. You must detect the data structures used in the signature and apply the corresponding strategies below.

1. Basic Functional Tests (5+ cases). Consider examples IF constraints allow:
   - Goal: Verify core logic on small, human-readable inputs.
   - Strategy: Mix of simple positive cases (answer exists) and negative cases (answer impossible).

2. Boundary & Edge Cases (5+ cases). Consider examples IF constraints allow:
   - **CRITICAL INSTRUCTION**: DON'T CONSIDER INPUTS THAT MAY BE A PART OF STRESS & SCALE TESTS.
   - GENERIC: Empty input ([], ""), Null/None, Single element.
   - NUMERICAL: Exact MIN_VAL and MAX_VAL from constraints (e.g. Int.MIN, Int.MAX).
   - ARRAYS/STRINGS: Length 0, Length 1, Length 2.
   - **Recursion**: DO NOT generate deep recursive structures that require `sys.setrecursionlimit`.

3. Stress & Scale Tests (5+ cases - Maximize Constraints). Consider examples IF constraints allow:
   - **WARNING**: Do not blindy maximize. Respect Coupled Constraints. Refer SYSTEM INSTRUCTIONS.
   - **COMPLEXITY ALERT**: Do NOT generate dense graphs/matrices for large N. Use sparse structures.
   - ARRAYS/STRINGS: Max allowed length.
   - GRAPHS: Sparse graphs (Chain, Star, Random Tree) at max Nodes.
   - ARRAYS/STRINGS: Generate inputs with maximum allowed length (e.g., 10^5 elements).
   - TREES: Deep skewed trees (degenerate linked lists) to test recursion depth limits.
   - MATRICES: Max rows x Max cols, BUT sparse matrices at large N.
   - **Recursion**: DO NOT generate deep recursive structures that require `sys.setrecursionlimit`.

4. Adversarial & Logic Traps (5+ cases). Consider examples IF constraints allow:
   - **CRITICAL**: All adversarial inputs MUST be VALID according to constraints.
   - DUPLICATES: Include duplicates ONLY IF the constraints allow them. If constraints say "Distinct", generate values that are close but not equal (e.g., [1, 2, 1] is bad, [1, 2, 3] is good). Duplicates can be of two types:
     - "Mixed Duplicates": Random duplicates scattered (`[1, 5, 2, 5, 1]`).
     - "Clustered Duplicates": Groups of duplicates (`[1, 1, 1, 2, 2, 3]`).
   - ARRAYS: Sorted, Reverse Sorted, All Identical values (e.g., [5,5,5...]).
   - NUMBERS: "Overflow Bait" -> Large positive + Large negative sums.
   - GRAPHS: Disconnected components **(if valid)**, Cycles **(if valid)**, Self-loops **(ONLY if valid)**.
   - STRINGS: Repeated patterns (e.g., "aaaaa"), Palindromes.
   - **Recursion**: DO NOT generate deep recursive structures that require `sys.setrecursionlimit`.

5. Domain-Specific Special Patterns (5+ cases). Consider examples IF constraints allow:
   - IF BIT MANIPULATION: Powers of 2, 0, -1.
   - IF INTERVALS: Overlapping, Nested, Touching, Disjoint.
   - IF LINKED LISTS: Cycles, Intersection point at end/beginning.
   - IF DP/GRID: Obstacles blocking all paths, Start == End.
   - IF STRINGS/PARSING: 
     - Nesting (e.g., "((()))", "/* /* */ */").
     - Escaping (e.g., "\\n", "\\t", "\\").
     - Incomplete markers (e.g., "/*", "(", "[[[").
     - Mixed delimiters (e.g., "/* // */", "' " '").
     - **Recursion**: DO NOT generate deep recursive structures that require `sys.setrecursionlimit`.'''
    

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # NEW: Inject feedback from previous Stage 5 failure
    if feedback_history:
        messages.append({
            "role": "user", 
            "content": f"CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:\nThe previous test set was rejected by the Reference Solution.\nReasons:\n{feedback_history}\n\nFIX: Adjust complexity, ensure constraints are respected, and remove invalid cases."
        })
        
    max_retries = 3
    MIN_EXECUTABLE = 10     # must return at least this many tests
    setup_code = "import sys; sys.setrecursionlimit(5000);"
    for attempt in range(max_retries):
        hint = ""
        try:
            print(f"Retry count: {attempt}")
            current_temp = 0.5 + (attempt * 0.2)
            response = client.chat.completions.create(
                model=FINE_TUNED_MODEL,
                messages=messages,
                temperature=current_temp,
                max_tokens=7000
            )
            
            code = response.choices[0].message.content.strip()
            print("Chaitanya response", response)

            # Sanitization
            if code.startswith("```"):
                code = code.split("```")[1].strip()
                if code.startswith("python"): code = code[6:].strip()
                if code.endswith("```"): code = code[:-3].strip()
            
            # 1. Static Analysis (Security & Syntax)
            is_valid_syntax, error_msg = validate_test_code(code)
            if not is_valid_syntax:
                raise ValueError(f"Code failed static validation: {error_msg}")

            code = f"{setup_code}\n{code}"

            # 2. Execution Sandbox
            namespace = {}
            try:
                # Safe Execution (Timeouts)
                safe_exec(code, namespace, timeout_sec=6)
                
                # 3. Validation Logic
                generate_inputs_fn = namespace['generate_inputs']
                def wrapped_validator(*args):
                    if 'is_valid_input' not in namespace:
                        return True, ""
                    res = namespace['is_valid_input'](*args)
                    if isinstance(res, bool): 
                        return res, ("Valid" if res else "Unknown Failure")
                    if isinstance(res, (tuple, list)) and len(res) == 2:
                        return res[0], res[1]
                    return False, f"Validator returned invalid format: {type(res)}"
                # is_valid_fn = namespace.get('is_valid_input', lambda *x: True)
                
                raw_data = generate_inputs_fn()
                if isinstance(raw_data, list):
                    raw_categorized = {"Basic Functional Tests": raw_data}
                elif isinstance(raw_data, dict):
                    raw_categorized = raw_data
                else:
                    raise TypeError(f"generate_inputs() must return Dict or List, got {type(raw_data)}")
                
                seen_hashes = set()
                valid_inputs = []
                failures = []
                validated_categorized = {} # To store categorization for Task 2

                # Filter inputs
                for category, inputs in raw_categorized.items():
                    if not isinstance(inputs, list): continue
                    
                    category_valid = []
                    for inp in inputs:
                        # Ensure input is a list/tuple for consistency
                        if not isinstance(inp, (list, tuple)): inp = [inp]
                        
                        try:
                            is_valid, reason = wrapped_validator(*inp)
                            
                            if is_valid:
                                # FIX: Use JSON serialization for true content-aware uniqueness
                                try:
                                    inp_hash = json.dumps(inp, sort_keys=True)
                                except TypeError:
                                    inp_hash = str(inp) # Fallback for edge cases
                                
                                if inp_hash not in seen_hashes:
                                    seen_hashes.add(inp_hash)
                                    valid_inputs.append(list(inp))
                                    category_valid.append(list(inp))
                            else:
                                failures.append(f"Invalid Input: {str(inp)[:50]}... | Reason: {reason}")
                        except Exception as e:
                            failures.append(f"Validator Crash: {str(e)}")
                            
                    if category_valid:
                        validated_categorized[category] = category_valid
                
                print("I am here 1")
                # 4. Threshold Check (Full code asked for 25, tolerate 15)
                if len(valid_inputs) < MIN_EXECUTABLE:
                    print("I am here 2")
                    failure_log = "\n".join(failures[:5])  # Show top 5 failures
                    if attempt >= 1:
                        hint = failure_log
                    error_detail = (
                        "ORACLE FAILURE: TEST GENERATION DID NOT MEET REQUIREMENTS\n\n"
                        f"SUMMARY:\n"
                        f"- Valid inputs: {len(valid_inputs)}\n"
                        f"- Required minimum: {MIN_EXECUTABLE}\n\n"
                        "OBSERVED FAILURE:\n"
                        "- The generated test set does not satisfy system requirements "
                        "(validity or structural diversity).\n\n"
                        "POTENTIAL ISSUES (NON-EXHAUSTIVE):\n"
                        "- Constraint violations detected by `is_valid_input`\n"
                        "- Structural redundancy among test cases\n"
                        "- Overly conservative generation strategy\n"
                        "- Tight or coupled constraints limiting feasible variation\n\n"
                        "MANDATORY CORRECTION:\n"
                        "1. Regenerate ALL test cases from scratch.\n"
                        "2. Ensure EVERY test passes `is_valid_input`.\n"
                        "3. Increase structural and content diversity.\n"
                        "4. Avoid repeating equivalent input patterns.\n"
                        "5. Re-evaluate feasibility under constraints.\n\n"
                        "RULE:\n"
                        "Only return test cases that are BOTH valid AND meaningfully distinct."
                    )


                    raise OracleFailure(error_detail)

                print(f"[STAGE 4] ✓ Generated {len(valid_inputs)} tests on attempt {attempt+1}")
                return valid_inputs, validated_categorized
            
            except OracleFailure as e:
                print("I am here 3")
                messages.append({"role": "assistant", "content": code})
                messages.append({
                        "role": "user",
                        "content": (
                            "Your previous code failed Oracle validation.\n\n"
                            f"{str(e)}\n\n"
                            f"{f'Hint (Top 5 failures): {hint}' if hint != '' else ''}"
                        )
                })
                continue

            except Exception as e:
                print("I am here 4", str(e))
                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your code failed during execution.\n\n"
                        f"{type(e).__name__}: {str(e)}\n"
                        "Fix the code."
                    )
                })
                continue

        
        except Exception as api_e:
            print("I am here 5")
            print(f"API Error: {str(api_e)}")
            continue

    raise HTTPException(status_code=400, detail="Stage 4 failed to generate tests after self-healing.")

# ============================================================================
# STAGE 5: REFERENCE SOLUTION (UPDATED WITH TIMEOUTS & CIRCUIT BREAKER)
# ============================================================================

def stage5_consensus_reference(spec: Dict[str, Any], 
                               test_inputs_dict: Dict[str, List[Any]], 
                               user_guidance: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
    """
    STAGE 5: Generate reference solution.
    UPDATED: Accepts 'user_guidance' (Map of InputHash -> ExpectedOutput).
    If an input exists in user_guidance, we compare against THAT, not the Brute Force model.
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    signature = extract_signature(spec['starter_code'])
    arg_types = signature['arg_types']
    user_guidance = user_guidance or {}
    
    # --- HELPER: Safe Runner with Timeout (Prevents Infinite Loops) ---
    def run_with_timeout(code_str: str, input_args: list, timeout_sec: int = 2):
        """Compiles and runs code in a separate process to enforce timeout."""
        def _worker(q, c, args, types):
            try:
                # 1. Setup Environment
                local_ns = {}
                import sys
                sys.setrecursionlimit(5000)
                exec(LEETCODE_PROMPT, local_ns)
                
                # 2. Compile
                exec(c, local_ns)
                solver = local_ns['Solution']()
                
                # 3. Convert Args
                conv_args = []
                for v, t in zip(args, types):
                    conv_args.append(auto_convert_arg(v, t))
                
                # 4. Run & Serialize
                res = solver.solve(*conv_args)
                q.put(("ok", robust_serialize(res)))
            except Exception as e:
                q.put(("error", str(e)))

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_worker, args=(q, code_str, input_args, arg_types))
        p.start()
        
        try:
            status, payload = q.get(timeout=timeout_sec)
            p.join()
            if status == "error":
                return f"ERROR: {payload}"
            return payload
        except queue.Empty:
            p.terminate()
            p.join()
            return "TIMEOUT"
    # ------------------------------------------------------------------

    def generate_solution_variant(mode_instruction: str) -> str:
        system_prompt = """You are an Expert Python Programmer specializing in algorithmic solutions.

    Write a correct, efficient reference solution.

    CRITICAL INSTRUCTIONS:
    1. Return ONLY the Solution class code
    2. NO markdown, NO explanation, NO additional code
    3. Use EXACTLY the function signature provided. Function Name MUST be `solve` and Class Name MUST be `Solution`.
    4. Handle ALL edge cases correctly
    5. Optimize for correctness first, efficiency second"""

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
    brute_code = generate_solution_variant("MODE: BRUTE FORCE (Use itertools/recursion. Ignore efficiency. Prioritize absolute correctness).")

    print("BRUTE CODE", brute_code)

    # Circuit Breaker Configuration
    MAX_HEALS = 3
    heals_performed = 0

    # METRICS
    heals_attempted = 0
    heals_successful = 0
    try:
        # Pre-validate compilation
        ns_test = {}; exec(LEETCODE_PROMPT, ns_test); safe_exec(optimized_code, ns_test)
        ns_test = {}; exec(LEETCODE_PROMPT, ns_test); safe_exec(brute_code, ns_test)

        SKIP_BRUTE_FORCE_CATEGORIES = {"Stress & Scale Tests", "Boundary & Edge Cases", "Domain-Specific Special Patterns", "Adversarial & Logic Traps"}
        
        print(f"[STAGE 5] Starting Consensus Verification...")

        for category, inputs in test_inputs_dict.items():
            print(f"   Running Category: {category} ({len(inputs)} tests)")
            should_skip_brute = category in SKIP_BRUTE_FORCE_CATEGORIES

            for i, inp in enumerate(inputs):
                # Run Optimized with Timeout
                out_opt = run_with_timeout(optimized_code, copy.deepcopy(inp), timeout_sec=2)
                
                # Check for runtime failure immediately
                if str(out_opt).startswith("ERROR") or out_opt == "TIMEOUT":
                    print(f"      [FAIL] Optimized failed on {category} input {i}: {out_opt}")
                
                # =========================================================
                # TRUTH OVERRIDE LOGIC
                # =========================================================
                # Check if this input is a User-Provided Truth
                try:
                    # Robust hashing for comparison
                    inp_hash = json.dumps(inp, sort_keys=True)
                except:
                    inp_hash = str(inp)
                
                is_user_truth = inp_hash in user_guidance
                expected_output = None
                truth_source = "Brute Force"

                if is_user_truth:
                    expected_output = user_guidance[inp_hash]
                    truth_source = "USER TRUTH (HARD OVERRIDE)"
                    # We accept the User's output blindly (assuming it's formatted correctly)
                elif should_skip_brute:
                    continue
                else:
                    # Run Brute Force
                    out_brute = run_with_timeout(brute_code, copy.deepcopy(inp), timeout_sec=4)
                    if str(out_brute).startswith("ERROR") or out_brute == "TIMEOUT":
                        continue
                    expected_output = out_brute

                # Compare Results
                if out_opt != expected_output:
                    print(f"      [MISMATCH] Category: {category} | Input: {inp}")
                    print(f"           Optimized: {out_opt}")
                    print(f"           Expected ({truth_source}): {expected_output}")
                    heals_attempted += 1
                    # --- CIRCUIT BREAKER ---
                    if heals_performed >= MAX_HEALS:
                        print(f"[STAGE 5] ⚠️ Max healing attempts ({MAX_HEALS}) reached. Stopping optimization.")
                        return optimized_code, {
                            "heals_attempted": heals_attempted,
                            "heals_successful": heals_successful,
                            "consensus_reached": False
                        }
                    
                    heals_performed += 1
                    # -----------------------

                    # ENHANCED HEALING PROMPT
                    heal_prompt = f"""
                    You provided an incorrect solution.
                    Input: {inp}
                    Your Output: {out_opt}
                    Expected Output: {expected_output}
                    
                    SOURCE OF TRUTH: {truth_source}
                    """
                    
                    if is_user_truth:
                        heal_prompt += "\nCRITICAL: The Expected Output is provided directly by the HUMAN USER. It is absolute truth. Your logic is definitely wrong. Fix it to match this case."
                    else:
                        heal_prompt += "\nThe expected output comes from a verified brute-force simulation."

                    heal_prompt += "\nReturn the complete, CORRECTED Solution class."

                    print(f"[STAGE 5] Healing Optimized Solution ({heals_performed}/{MAX_HEALS})...")
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
                    
                    # Re-verify compilation
                    try:
                        ns_opt = {}; exec(LEETCODE_PROMPT, ns_opt); safe_exec(optimized_code, ns_opt)
                        healed_out = run_with_timeout(
                            optimized_code,
                            copy.deepcopy(inp),
                            timeout_sec=2
                        )
                        if healed_out == expected_output:
                            heals_successful += 1
                    except Exception as e:
                        print(f"[STAGE 5] Healed code failed to compile: {e}")
                        return optimized_code, {
                            "heals_attempted": heals_attempted,
                            "heals_successful": heals_successful,
                            "consensus_reached": True
                        }

        print("[STAGE 5] ✓ Consensus Logic Complete")
        return optimized_code, {
            "heals_attempted": heals_attempted,
            "heals_successful": heals_successful,
            "consensus_reached": True
        }

    except Exception as e:
        print(f"[STAGE 5] Consensus logic failed: {e}. Fallback to raw optimized code.")
        traceback.print_exc()
        return optimized_code, {
            "heals_attempted": heals_attempted,
            "heals_successful": heals_successful,
            "consensus_reached": True
        }

def stage5_build_golden_suite(reference_code: str, test_inputs: List[List[Any]], 
                              signature: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    STAGE 5: Build golden suite.
    FIXED: Prevents deadlock on large outputs by serializing INSIDE the child process.
    UPDATED: Returns failure statistics for Circular Healing.
    """
    import sys
    
    golden_suite = []
    arg_types = signature['arg_types']
    
    # Pre-compile the reference solution to ensure it's valid
    exec_globals = {}
    exec(LEETCODE_PROMPT, exec_globals)
    safe_exec(reference_code, exec_globals)
    
    print(f"[STAGE 5] Calculating Golden Outputs for {len(test_inputs)} tests...")

    failures_log = []

    for idx, test_input in enumerate(test_inputs):
        try:
            # Prepare arguments in the parent
            # Note: We pass raw inputs and convert inside child to avoid pickling issues
            # with dynamically defined classes.
            
            def _run_ref(code, input_args, types, q):
                try:
                    # 1. Setup Environment inside Child
                    import sys
                    sys.setrecursionlimit(5000) # Safety for deep trees
                    
                    local_ns = {}
                    exec(LEETCODE_PROMPT, local_ns)
                    safe_exec(code, local_ns)
                    solver = local_ns['Solution']()
                    
                    # 2. Convert Args (JSON -> Objects)
                    converted_args = []
                    for v, t in zip(input_args, types):
                        if 'ListNode' in t and isinstance(v, list):
                            converted_args.append(local_ns['list_node'](v))
                        elif 'TreeNode' in t and isinstance(v, list):
                            converted_args.append(local_ns['tree_node'](v))
                        else:
                            converted_args.append(v)

                    # 3. Run Solution
                    raw_result = solver.solve(*converted_args)
                    
                    # 4. SERIALIZE INSIDE CHILD (Critical Fix)
                    json_result = robust_serialize(raw_result)
                    
                    q.put(("ok", json_result))
                except Exception as e:
                    q.put(("error", str(e)))

            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_run_ref,
                args=(reference_code, copy.deepcopy(test_input), arg_types, q),
            )
            p.start()
            
            # FIX: Use queue.get with timeout INSTEAD of p.join()
            try:
                status, payload = q.get(timeout=2.0) # 2 Second Timeout per test
                p.join() # Clean up zombie process
            except queue.Empty:
                p.terminate()
                p.join()
                print(f"[STAGE 5] Test {idx} dropped: Time Limit Exceeded (2s)")
                failures_log.append(f"Input {idx}: Time Limit Exceeded (2s)")
                continue

            if status == "error":
                print(f"[STAGE 5] Test {idx} dropped: Runtime Error: {payload}")
                failures_log.append(f"Input {idx}: Runtime Error - {str(payload)[:100]}")
                continue

            golden_suite.append({
                "input": test_input,
                "expected": payload
            })

        except Exception as e:
            print(f"[STAGE 5] Test {idx} dropped: System Error: {str(e)}")
            failures_log.append(f"Input {idx}: System Error - {str(e)}")
            continue
    
    # CIRCULAR HEALING CHECK
    # If the Reference Solution (which is verified) fails too many tests,
    # it implies the tests are invalid, "poisoned", or too complex for constraints.
    total = len(test_inputs)
    valid_count = len(golden_suite)
    
    if total > 0:
        failure_rate = 1.0 - (valid_count / total)
        if failure_rate > 0.30: # If >30% tests fail ref sol
            summary = "\n".join(failures_log[:10])
            raise TestQualityFailure(
                f"Reference Solution rejected the test suite.\n"
                f"Failure Rate: {failure_rate*100:.1f}% ({total-valid_count}/{total} failed)\n"
                f"Reasons (Sample):\n{summary}"
            )

    if valid_count < 5:
        raise HTTPException(
            status_code=400, 
            detail=f"Reference Solution failed too many tests ({valid_count} passed). Specs might be impossible."
        )
    
    print(f"[STAGE 5] ✓ Golden Suite Built ({valid_count} tests)")
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

# def stage6_evaluate(user_code: str, golden_suite: List[Dict[str, Any]],
#                    signature: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
#     """
#     STAGE 6: Evaluate user solution.
#     Uses `safe_exec` (via timeout logic) to protect against user infinite loops.
#     """
#     results = []
#     passed_count = 0
#     arg_types = signature['arg_types']
    
#     try:
#         exec_globals = {}
#         exec(LEETCODE_PROMPT, exec_globals)
#         # Sandbox the user's compilation
#         safe_exec(user_code, exec_globals, timeout_sec=6)
        
#         user_solver = exec_globals['Solution']()
#     except Exception as e:
#         return {
#             "error": f"Compilation/Startup Error: {str(e)}",
#             "passed": 0,
#             "total": len(golden_suite),
#             "results": []
#         }
    
#     for idx, test_case in enumerate(golden_suite):
#         test_input = test_case['input']
#         expected = test_case['expected']
        
#         try:
#             run_args = [
#                 auto_convert_arg(v, t) for v, t in zip(test_input, arg_types)
#             ]
            
#             def _run_user(fn, args, q):
#                 try:
#                     q.put(("ok", fn(*args)))
#                 except Exception as e:
#                     q.put(("error", str(e)))

#             q = multiprocessing.Queue()
#             p = multiprocessing.Process(
#                 target=_run_user,
#                 args=(user_solver.solve, copy.deepcopy(run_args), q),
#             )
#             p.start()
#             p.join(2)

#             if p.is_alive():
#                 p.terminate()
#                 results.append({
#                     "test_number": idx + 1,
#                     "passed": False,
#                     "error": "Time Limit Exceeded"
#                 })
#                 continue

#             if q.empty():
#                 results.append({
#                     "test_number": idx + 1,
#                     "passed": False,
#                     "error": "Runtime Error: No output"
#                 })
#                 continue

#             status, payload = q.get()
#             if status == "error":
#                 results.append({
#                     "test_number": idx + 1,
#                     "passed": False,
#                     "error": payload
#                 })
#                 continue

#             actual_json = robust_serialize(payload)
#             passed = check_correctness(actual_json, expected, evaluation_type)

#             if passed:
#                 passed_count += 1
            
#             results.append({
#                 "test_number": idx + 1,
#                 "input": test_input,
#                 "expected": expected,
#                 "actual": actual_json,
#                 "passed": passed
#             })
        
#         except Exception as e:
#             results.append({
#                 "test_number": idx + 1,
#                 "passed": False,
#                 "error": f"Runtime Error: {str(e)}"
#             })
    
#     percentage = (passed_count / len(golden_suite) * 100) if golden_suite else 0
    
#     return {
#         "passed": passed_count,
#         "total": len(golden_suite),
#         "percentage": percentage,
#         "results": results
#     }
def stage6_evaluate(user_code: str, golden_suite: List[Dict[str, Any]],
                   signature: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
    """
    STAGE 6: Evaluate user solution.
    FIXED: Compiles user code INSIDE the child process to avoid Pickling Errors
    with dynamically defined classes.
    """
    import queue
    
    results = []
    passed_count = 0
    arg_types = signature['arg_types']
    
    print(f"[STAGE 6] Evaluation started for {len(golden_suite)} tests...")
    
    # 1. Check for Syntax Errors immediately (Fast Fail)
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        return {
            "error": f"Syntax Error: {e}",
            "passed": 0, "total": len(golden_suite), "results": []
        }

    for idx, test_case in enumerate(golden_suite):
        test_input = test_case['input']
        expected = test_case['expected']
        
        try:
            # Prepare args in parent
            # Note: We pass raw data, not 'ListNode' objects, to avoid pickling issues
            
            def _run_user_sandboxed(code, input_args, types, q):
                try:
                    # A. Setup Environment (Inside Child)
                    import sys
                    sys.setrecursionlimit(5000)
                    
                    local_ns = {}
                    exec(LEETCODE_PROMPT, local_ns) # Inject TreeNode/ListNode
                    
                    # B. Compile User Code
                    try:
                        exec(code, local_ns)
                    except Exception as e:
                        q.put(("error", f"Runtime Import/Definition Error: {e}"))
                        return

                    if 'Solution' not in local_ns:
                        q.put(("error", "Class 'Solution' not found in code"))
                        return
                        
                    solver = local_ns['Solution']()
                    
                    # C. Convert Args
                    converted_args = []
                    for v, t in zip(input_args, types):
                        if 'ListNode' in t and isinstance(v, list):
                            converted_args.append(local_ns['list_node'](v))
                        elif 'TreeNode' in t and isinstance(v, list):
                            converted_args.append(local_ns['tree_node'](v))
                        else:
                            converted_args.append(v)

                    # D. Run
                    raw_result = solver.solve(*converted_args)
                    
                    # E. Serialize Output
                    json_result = robust_serialize(raw_result)
                    q.put(("ok", json_result))
                    
                except Exception as e:
                    # Catch assertions and runtime errors
                    q.put(("error", f"{type(e).__name__}: {str(e)}"))

            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_run_user_sandboxed,
                args=(user_code, copy.deepcopy(test_input), arg_types, q),
            )
            p.start()
            
            # Timeout Management
            try:
                status, payload = q.get(timeout=2.0) # 2s timeout
                p.join()
            except queue.Empty:
                p.terminate()
                p.join()
                results.append({
                    "test_number": idx + 1,
                    "passed": False,
                    "error": "Time Limit Exceeded"
                })
                continue

            if status == "error":
                results.append({
                    "test_number": idx + 1,
                    "passed": False,
                    "error": payload
                })
                continue

            # Compare Result
            passed = check_correctness(payload, expected, evaluation_type)
            if passed:
                passed_count += 1
            
            results.append({
                "test_number": idx + 1,
                "input": test_input,
                "expected": expected,
                "actual": payload,
                "passed": passed
            })
        
        except Exception as e:
            results.append({
                "test_number": idx + 1,
                "passed": False,
                "error": f"System Error: {str(e)}"
            })
    
    percentage = (passed_count / len(golden_suite) * 100) if golden_suite else 0
    
    return {
        "passed": passed_count,
        "total": len(golden_suite),
        "percentage": percentage,
        "results": results
    }

def analyze_user_challenge(problem: Problem, user_examples: List[ChallengeExample]) -> ChallengeAnalysisSchema:
    """
    The 'Gatekeeper': Audits user inputs against problem constraints.
    Refined for generic handling of Trees, Graphs, Strings, and Arrays.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 1. Format inputs (Keep this, it's good)
    examples_text = json.dumps([{"input": ex.input, "output": ex.output} for ex in user_examples], indent=2)

    # 2. Refined System Prompt
    system_prompt = """You are the 'Constraint Compliance Auditor' for an Algorithm Testing Platform.
    
    YOUR GOAL: 
    Determine if the User's "Challenge Input" is ADMISSIBLE based strictly on numerical limits, data types, and structural guarantees.
    
    ### CRITICAL DISTINCTION:
    1. **HARD CONSTRAINTS (ENFORCE THESE)**:
       - **Numerical**: `1 <= n <= 10^5`, `-10^9 <= val <= 10^9`.
       - **Structural**: "Input is a binary search tree", "Graph is a DAG", "List is sorted".
       - **Data Type**: "String contains only lowercase English letters".
       - **Uniqueness**: "All values are distinct" (only if explicitly stated).
       
    2. **PROBLEM LOGIC (IGNORE THESE)**:
       - "Find the shortest path" -> A graph with NO path is a VALID input (Output is -1).
       - "Check if tree is symmetric" -> An asymmetric tree is a VALID input (Output is False).
       - DO NOT simulate the solution. Only validate the input parameters.

    ### AUDIT PROTOCOL:
    1. **SCAN CONSTRAINTS**: Look at the explicitly provided 'Constraints Text' AND the 'Problem Description' for structural guarantees.
    2. **ANALYZE INPUT STRUCTURE**:
       - If input is **JSON/Array** for a **Tree/Graph**: Ensure the node count and values roughly align with limits.
       - If input is **String**: Check character sets (e.g., no uppercase if forbidden).
       - If input is **Linked List** (represented as array): Check length limits.
    3. **VERDICT**:
       - If a constraint is DEFINITELY violated -> Decision: **NO**.
       - If constraints are ambiguous or silent -> Decision: **YES** (Benefit of the doubt).

    ### BEHAVIOR EXAMPLES:
    - **Problem**: "Longest Substring without Repeats".
      - Constraint: `s.length <= 5000`.
      - Input: `"aaaaa"`.
      - Decision: **YES**. (Repeats are part of the logic problem, not an input violation).
    
    - **Problem**: "Number of Islands".
      - Constraint: `grid[i][j]` is '0' or '1'.
      - Input: `[["1", "2"], ["0", "1"]]`.
      - Decision: **NO**. (Value "2" violates the character set constraint).

    - **Problem**: "Validate Binary Search Tree".
      - Constraint: `root` values -10^4 to 10^4.
      - Input: `[5, 1, 4, null, null, 3, 6]` (Standard LeetCode Tree format).
      - Decision: **YES**. (Structure looks valid, values within range).
    """

    user_prompt = f"""
    ### PROBLEM DEFINITION
    Title: {problem.title}
    
    *** CONSTRAINTS SECTION ***
    {problem.constraints}
    
    *** FULL DESCRIPTION ***
    {problem.problem_description} 
    
    ### USER CHALLENGE CANDIDATE
    {examples_text}
    
    ### INSTRUCTIONS
    Perform the audit. 
    1. Extract numerical/structural constraints. 
    2. Check if the input JSON conforms to these limits. 
    3. Return the JSON verdict.
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ChallengeAnalysisSchema,
            temperature=0.0
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"[Gatekeeper Error] {e}")
        return ChallengeAnalysisSchema(
            extracted_constraints="Unknown",
            input_analysis="Analysis Failed",
            violation_detected=True,
            decision="NO", 
            reason=f"Internal Gatekeeper Error: {str(e)}"
        )

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/generate-problem")
def generate_problem(request: GenerateProblemRequest, db: Session = Depends(get_db)):
    """
    Generator endpoint that streams status updates.
    Run as standard 'def' to offload blocking OpenAI calls to a thread.
    """
    
    # Logic remains the same, but execution context changes
    def event_stream():
        metrics = {
            "stage3": {"latency": 0.0},
            "stage4": {
                "latency": 0.0,
                "retries": 0,
                "candidate_tests": 0
            },
            "stage5": {
                "latency": 0.0,
                "heals_attempted": 0,
                "heals_successful": 0,
                "consensus_reached": False
            },
            "stage5_2": {
                "latency": 0.0,
                "candidate_tests": 0,
                "golden_tests": 0,
                "discarded_tests": 0,
                "circular_healing_triggered": False
            }
        }

        try:
            # --- STAGE 1: NOTIFY START ---
            yield json.dumps({"step": 1, "message": "Formalizing Problem Specification..."}) + "\n"
            
            # This blocking call is now safe
            print("[STAGE 3] Generating specification...")
            t3 = datetime.utcnow()
            spec = stage3_generate_specification(request.vague_problem)
            metrics["stage3"]["latency"] = (datetime.utcnow() - t3).total_seconds()
            print(f"[STAGE 3] ✓ Title: {spec['title']}")
            
            signature = extract_signature(spec['starter_code'])
            
            # --- STAGE 2: NOTIFY TEST GENERATION ---
            yield json.dumps({"step": 2, "message": "Generating Test Cases (The Oracle)..."}) + "\n"
            
            test_inputs_flat = []
            test_inputs_dict = {}
            reference_code = ""
            golden_suite = []
            
            # CIRCULAR HEALING LOOP: Feedback from Ref Sol -> Test Gen
            feedback = ""
            max_circular_retries = 2
            
            for attempt in range(max_circular_retries + 1):
                try:
                    metrics["stage4"]["retries"] += 1
                    if attempt > 0:
                        yield json.dumps({"step": 2, "message": f"Refining Test Cases (Attempt {attempt+1})..."}) + "\n"
                        
                    print(f"[STAGE 4] Generating test cases (Attempt {attempt+1})...")
                    # Pass feedback if this is a retry
                    t4 = datetime.utcnow()
                    test_inputs_flat, test_inputs_dict = stage4_generate_tests(spec, signature, feedback_history=feedback)
                    metrics["stage4"]["latency"] = (datetime.utcnow() - t4).total_seconds()
                    metrics["stage4"]["candidate_tests"] = len(test_inputs_flat)
                    
                    # --- STAGE 3: NOTIFY REFERENCE SOLUTION ---
                    yield json.dumps({"step": 3, "message": "Building Reference Solution & Consensus..."}) + "\n"
                    
                    print("[STAGE 5] Generating Consensus...")
                    t5 = datetime.utcnow()
                    reference_code, stage5_meta = stage5_consensus_reference(spec, test_inputs_dict)
                    metrics["stage5"]["latency"] = (datetime.utcnow() - t5).total_seconds()
                    metrics["stage5"]["heals_attempted"] = stage5_meta["heals_attempted"]
                    metrics["stage5"]["consensus_reached"] = stage5_meta["consensus_reached"]
                    metrics["stage5"]["heals_successful"] = stage5_meta["heals_successful"]

                    
                    print("[STAGE 5] Building Golden Suite...")
                    # If this fails with TestQualityFailure, loop repeats
                    t52 = datetime.utcnow()
                    try:
                        golden_suite = stage5_build_golden_suite(reference_code, test_inputs_flat, signature)
                    except TestQualityFailure:
                        metrics["stage5_2"]["circular_healing_triggered"] = True
                        raise
                    metrics["stage5_2"]["latency"] = (datetime.utcnow() - t52).total_seconds()
                    metrics["stage5_2"]["candidate_tests"] = len(test_inputs_flat)
                    metrics["stage5_2"]["golden_tests"] = len(golden_suite)
                    metrics["stage5_2"]["discarded_tests"] = (
                        len(test_inputs_flat) - len(golden_suite)
                    )
                    
                    # If success, break loop
                    break
                    
                except TestQualityFailure as e:
                    print(f"[CIRCULAR HEALING] Stage 5 rejected Stage 4 tests: {e}")
                    feedback = str(e)
                    if attempt == max_circular_retries:
                        raise HTTPException(status_code=400, detail=f"Failed to generate valid problem after retries. {str(e)}")
                    # Loop continues...
            
            # --- DATABASE SAVING ---
            problem_id = f"prob_{uuid.uuid4().hex[:8]}"
            
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
            
            ref_sol = ReferenceSolution(problem_id=problem_id, solution_code=reference_code)
            db.add(ref_sol)
            
            for test in golden_suite:
                test_case = TestCase(
                    problem_id=problem_id,
                    test_inputs=test['input'],
                    expected_output=test['expected']
                )
                db.add(test_case)
            
            db.commit()
            
            print(f"[SUCCESS] Problem {problem_id} generated")
            
            # --- FINAL: SEND RESULT ---
            final_response = {
                "problem_id": problem_id,
                "title": spec['title'],
                "starter_code": spec['starter_code'],
                "test_case_count": len(golden_suite),
                "metrics": metrics
            }
            yield json.dumps({"final_result": final_response}) + "\n"

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            yield json.dumps({"error": str(e)}) + "\n"

    # CRITICAL: Added headers to prevent buffering
    return StreamingResponse(
        event_stream(), 
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",  # Disables buffering in Nginx/Reverse Proxies
            "Cache-Control": "no-cache", # Prevents caching
            "Connection": "keep-alive",  # Keeps socket open
        }
    )

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

@app.post("/api/improve/problem/{problem_id}")
def improve_problem(problem_id: str, request: ImprovementRequest, db: Session = Depends(get_db)):
    """
    1. Gatekeeper checks if User Examples are VALID constraints-wise.
    2. If YES: Triggers Full Healing with TRUTH INJECTION.
    """
    
    problem = db.query(Problem).filter(Problem.problem_id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    def event_stream():
        try:
            # --- STEP 1: THE GATEKEEPER ---
            yield json.dumps({"step": 1, "message": "Auditing Challenge against Constraints..."}) + "\n"
            
            audit = analyze_user_challenge(problem, request.examples)
            
            if audit.decision == "NO":
                yield json.dumps({
                    "final_result": {
                        "status": "REJECTED",
                        "decision": "NO", 
                        "violation": audit.input_analysis,
                        "reason": audit.reason
                    }
                }) + "\n"
                return

            # --- STEP 2: CHALLENGE ACCEPTED ---
            yield json.dumps({
                "step": 1, 
                "status": "ACCEPTED",
                "message": "Challenge Validated. Initiating System Healing..."
            }) + "\n"

            user_inputs_str = ", ".join([ex.input for ex in request.examples])
            feedback_context = (
                f"User submitted VALID edge cases that we missed: {user_inputs_str}. "
                f"Constraint Audit passed: {audit.extracted_constraints}. "
                "You MUST generate test cases that cover this specific logic."
            )

            # --- STEP 3: HEALING (STAGE 4: TEST GEN) ---
            yield json.dumps({"step": 2, "message": "Regenerating Test Cases (Stage 4)..."}) + "\n"
            
            spec = {
                "title": problem.title,
                "problem_description": problem.problem_description,
                "constraints": problem.constraints,
                "starter_code": problem.starter_code,
                "examples": problem.examples,
                "evaluation_type": problem.evaluation_type
            }
            # EXTRACT SIGNATURE EARLY FOR PARSING LOGIC
            signature = extract_signature(problem.starter_code)

            # Stage 4 with Explicit User Feedback
            test_inputs_flat, test_inputs_dict = stage4_generate_tests(
                spec, 
                signature, 
                feedback_history=feedback_context 
            )

            # =================================================================
            # TRUTH INJECTION (FIXED UNPACKING LOGIC)
            # =================================================================
            user_injected_count = 0
            user_category_key = "User Provided Edge Cases"
            user_ground_truth = {} 
            
            if user_category_key not in test_inputs_dict:
                test_inputs_dict[user_category_key] = []

            for ex in request.examples:
                try:
                    # 1. Parse Input
                    try:
                        parsed_input = json.loads(ex.input)
                    except:
                        parsed_input = ast.literal_eval(ex.input)
                    
                    # 2. Parse Expected Output
                    try:
                        parsed_output = json.loads(ex.output)
                    except:
                        try:
                            parsed_output = ast.literal_eval(ex.output)
                        except:
                            parsed_output = ex.output

                    # 3. CRITICAL FIX: Align Input Wrapping with Signature
                    # If function takes 1 arg (e.g. solve(nums)), we MUST wrap the input [nums]
                    # regardless of whether 'nums' is a list or not.
                    if len(signature['arg_types']) == 1:
                        final_input = [parsed_input]
                    else:
                        # For multi-arg functions, we assume parsed_input IS the list of args
                        if not isinstance(parsed_input, (list, tuple)):
                            final_input = [parsed_input]
                        else:
                            final_input = parsed_input

                    # 4. Inject into Pipeline
                    test_inputs_flat.append(final_input)
                    test_inputs_dict[user_category_key].append(final_input)
                    
                    # 5. Store Truth (Hashing input for lookup)
                    # We hash the RAW parsed_input (the value), not the wrapped one, 
                    # because Stage 5 un-wraps before looking up.
                    # Wait - Stage 5 receives the WRAPPED input in the loop.
                    # So we must hash the WRAPPED input.
                    input_hash = json.dumps(final_input, sort_keys=True)
                    user_ground_truth[input_hash] = parsed_output

                    user_injected_count += 1
                except Exception as parse_e:
                    print(f"[Improve Error] Failed to inject user input '{ex.input}': {parse_e}")
            
            print(f"[FIX] Injected {user_injected_count} user examples with Correct Argument Wrapping.")
            # =================================================================

            # --- STEP 4: HEALING (STAGE 5: REF SOL) ---
            yield json.dumps({"step": 3, "message": "Regenerating Reference Solution (Stage 5)..."}) + "\n"
            
            # Pass the user_ground_truth to Stage 5
            reference_code, _ = stage5_consensus_reference(spec, test_inputs_dict, user_guidance=user_ground_truth)
            
            # Build Golden Suite
            golden_suite = stage5_build_golden_suite(reference_code, test_inputs_flat, signature)

            # --- STEP 5: DATABASE UPDATE ---
            yield json.dumps({"step": 4, "message": "Updating Knowledge Base..."}) + "\n"

            ref_sol = db.query(ReferenceSolution).filter(ReferenceSolution.problem_id == problem_id).first()
            if ref_sol:
                ref_sol.solution_code = reference_code
                ref_sol.created_at = datetime.utcnow()

            db.query(TestCase).filter(TestCase.problem_id == problem_id).delete()
            for test in golden_suite:
                db.add(TestCase(
                    problem_id=problem_id,
                    test_inputs=test['input'],
                    expected_output=test['expected']
                ))
            
            db.commit()

            yield json.dumps({
                "final_result": {
                    "status": "HEALED",
                    "decision": "YES",
                    "reason": "Valid challenge accepted. Logic updated.",
                    "audit_log": audit.extracted_constraints,
                    "new_test_count": len(golden_suite)
                }
            }) + "\n"

        except Exception as e:
            print(f"[Improve Error] {e}")
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        event_stream(), 
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

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

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)