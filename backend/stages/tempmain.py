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
import signal
import math
import ast
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
# UTILITIES
# ============================================================================

def auto_convert_arg(value: Any, arg_type: str) -> Any:
    """Convert JSON primitives to Python objects"""
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

def auto_serialize_output(output: Any) -> Any:
    """Serialize Python objects back to JSON"""
    try:
        json.dumps(output)
        return output
    except:
        return str(output)

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

def validate_test_code(code: str) -> tuple[bool, str]:
    """Validate that test code has required functions. Returns (is_valid, error_msg)"""
    try:
        tree = ast.parse(code)
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        
        if 'generate_inputs' not in functions:
            return False, "Missing generate_inputs() function"
        if 'is_valid_input' not in functions:
            return False, "Missing is_valid_input() function"
        
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, str(e)

def stage4_generate_tests(spec: Dict[str, Any], signature: Dict[str, Any]) -> List[List[Any]]:
    """STAGE 4: Generate test cases using fine-tuned model with retries"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=FINE_TUNED_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=3500
            )
            
            code = response.choices[0].message.content.strip()
            print("Chaitanya", code)
            print("Chaitanya response", response)
            # Remove markdown wrappers
            if code.startswith("```"):
                code = code.split("```")[1].strip()
                if code.startswith("python"):
                    code = code[6:].strip()
                if code.endswith("```"):
                    code = code[:-3].strip()
            
            # Validate code structure
            is_valid, error_msg = validate_test_code(code)
            if not is_valid:
                if attempt < max_retries - 1:
                    print(f"[STAGE 4] Attempt {attempt + 1}: Invalid code - {error_msg}. Retrying...")
                    continue
                raise ValueError(f"Generated code validation failed: {error_msg}")
            
            # Execute code
            namespace = {}
            exec(code, namespace)
            
            generate_inputs_fn = namespace['generate_inputs']
            is_valid_fn = namespace.get('is_valid_input', lambda *x: True)
            
            raw_inputs = generate_inputs_fn()
            
            if not isinstance(raw_inputs, list):
                raise TypeError(f"generate_inputs() must return list, got {type(raw_inputs)}")
            
            valid_inputs = [inp for inp in raw_inputs if is_valid_fn(*inp)]
            
            if len(valid_inputs) < 5:
                if attempt < max_retries - 1:
                    print(f"[STAGE 4] Attempt {attempt + 1}: Only {len(valid_inputs)} valid tests (need 5+). Retrying...")
                    continue
                raise ValueError(f"Only {len(valid_inputs)} valid inputs (need 5+)")
            
            print(f"[STAGE 4] ✓ Generated {len(valid_inputs)} valid test cases")
            return valid_inputs[:15]
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[STAGE 4] Attempt {attempt + 1} failed: {str(e)}")
                continue
            raise HTTPException(status_code=400, detail=f"Test generation failed after {max_retries} attempts: {str(e)}")

# ============================================================================
# STAGE 5: REFERENCE SOLUTION
# ============================================================================

def stage5_5_verify_reference(reference_code: str, test_inputs: List[List[Any]], 
                              signature: Dict[str, Any]) -> bool:
    """STAGE 5.5: Sanity Check - Verify reference solution executes on first input"""
    try:
        exec_globals = {}
        exec(LEETCODE_PROMPT, exec_globals)
        exec(reference_code, exec_globals)
        
        solver = exec_globals['Solution']()
        arg_types = signature['arg_types']
        
        if not test_inputs:
            return False
        
        test_input = test_inputs[0]
        run_args = [auto_convert_arg(v, t) for v, t in zip(test_input, arg_types)]
        
        result = solver.solve(*copy.deepcopy(run_args))
        print(f"[STAGE 5.5] ✓ Reference solution verified on first test case")
        return True
    except Exception as e:
        print(f"[STAGE 5.5] ✗ Sanity Check Failed: {str(e)}")
        return False

def stage5_generate_reference(spec: Dict[str, Any]) -> str:
    """STAGE 5: Generate reference solution using fine-tuned model"""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are an Expert Python Programmer specializing in algorithmic solutions.

Write a correct, efficient reference solution.

CRITICAL INSTRUCTIONS:
1. Return ONLY the Solution class code
2. NO markdown, NO explanation, NO additional code
3. Use EXACTLY the function signature provided
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

Write the complete Solution class:'''

    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=2000
    )
    
    code = response.choices[0].message.content.strip()
    
    # Remove markdown wrappers
    if code.startswith("```"):
        code = code.split("```")[1].strip()
        if code.startswith("python"):
            code = code[6:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
    
    print(f"[STAGE 5] ✓ Generated reference solution ({len(code)} chars)")
    return code

def stage5_build_golden_suite(reference_code: str, test_inputs: List[List[Any]], 
                              signature: Dict[str, Any]) -> List[Dict[str, Any]]:
    """STAGE 5: Build golden suite by executing reference solution"""
    
    golden_suite = []
    arg_types = signature['arg_types']
    
    exec_globals = {}
    exec(LEETCODE_PROMPT, exec_globals)
    exec(reference_code, exec_globals)
    
    solver = exec_globals['Solution']()
    
    for idx, test_input in enumerate(test_inputs):
        try:
            run_args = [
                auto_convert_arg(v, t) for v, t in zip(test_input, arg_types)
            ]
            output = solver.solve(*copy.deepcopy(run_args))
            serialized = auto_serialize_output(output)
            
            golden_suite.append({
                "input": test_input,
                "expected": serialized
            })
        except Exception as e:
            print(f"[STAGE 5] Warning: Test case {idx} failed - {str(e)}")
            continue
    
    if len(golden_suite) < 5:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {len(golden_suite)} successful test executions (need 5+). Reference solution may be incorrect."
        )
    
    print(f"[STAGE 5] ✓ Built golden suite with {len(golden_suite)} test cases")
    return golden_suite

# ============================================================================
# STAGE 6: USER EVALUATION
# ============================================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Time Limit Exceeded")

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
    """STAGE 6: Evaluate user solution"""
    
    results = []
    passed_count = 0
    arg_types = signature['arg_types']
    
    try:
        exec_globals = {}
        exec(LEETCODE_PROMPT, exec_globals)
        exec(user_code, exec_globals)
        
        user_solver = exec_globals['Solution']()
    except Exception as e:
        return {
            "error": f"Compilation Error: {str(e)}",
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
            signal.alarm(2)
            
            try:
                actual = user_solver.solve(*copy.deepcopy(run_args))
            finally:
                signal.alarm(0)
            
            actual_json = auto_serialize_output(actual)
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
# FASTAPI APP
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
        print(spec)
        signature = extract_signature(spec['starter_code'])
        print(f"[STAGE 3] ✓ Signature: {signature['function_name']}({', '.join(signature['arg_names'])})")
        
        print("[STAGE 4] Generating test cases...")
        test_inputs = stage4_generate_tests(spec, signature)
        print(test_inputs)
        print("[STAGE 5] Generating reference solution...")
        reference_code = stage5_generate_reference(spec)
        print(reference_code)
        print("[STAGE 5.5] Verifying reference solution...")
        if not stage5_5_verify_reference(reference_code, test_inputs, signature):
            raise HTTPException(
                status_code=422,
                detail="Generated reference solution failed sanity check on basic input"
            )
        
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
    """Get problem details including hidden tests and reference solution"""
    
    # 1. Get Basic Info
    problem = db.query(Problem).filter(Problem.problem_id == problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    # 2. Get Reference Solution
    ref_sol = db.query(ReferenceSolution).filter(ReferenceSolution.problem_id == problem_id).first()
    solution_code = ref_sol.solution_code if ref_sol else "Solution not available."

    # 3. Get Test Cases
    test_cases_db = db.query(TestCase).filter(TestCase.problem_id == problem_id).all()
    test_cases = [
        {
            "input": tc.test_inputs,
            "expected": tc.expected_output
        }
        for tc in test_cases_db
    ]

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
    print("AI CODING PLATFORM - STAGES 3-6")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /api/generate-problem")
    print("  POST /api/evaluate")
    print("  GET /api/problem/{problem_id}")
    print("  GET /health")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)