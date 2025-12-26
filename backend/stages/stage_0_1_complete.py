"""
STAGE 0 & 1: Dataset Setup & Ground Truth Extraction
Fully implemented with real OpenAI integration
"""

import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI

# ============================================================================
# STAGE 0: DATASET SETUP & VALIDATION
# ============================================================================
OPENAI_MODEL = "gpt-4o-mini"

@dataclass
class DatasetProblem:
    """Validated problem from dataset"""
    task_id: str
    tags: List[str]
    problem_description: str
    starter_code: str

class DatasetValidator:
    """Validates LeetCode dataset contract"""
    
    REQUIRED_FIELDS = {
        'task_id': str,
        'tags': list,
        'problem_description': str,
        'starter_code': str,
    }
    
    @staticmethod
    def validate_and_parse(dataset_json: Dict) -> Tuple[List[DatasetProblem], List[str]]:
        """
        Validate dataset and return parsed problems.
        
        Returns:
            (valid_problems, error_messages)
        """
        valid_problems = []
        errors = []
        
        if 'dataset' not in dataset_json:
            errors.append("ERROR: Missing 'dataset' key in JSON")
            return [], errors
        
        problems = dataset_json['dataset']
        if not isinstance(problems, list):
            errors.append("ERROR: 'dataset' must be a list")
            return [], errors
        
        for idx, problem in enumerate(problems):
            # Validate required fields
            missing_fields = []
            for field, field_type in DatasetValidator.REQUIRED_FIELDS.items():
                if field not in problem:
                    missing_fields.append(field)
                elif not isinstance(problem[field], field_type):
                    errors.append(f"Problem {idx} ({problem.get('task_id', 'unknown')}): "
                                f"Field '{field}' has wrong type (expected {field_type.__name__})")
            
            if missing_fields:
                errors.append(f"Problem {idx}: Missing fields {missing_fields}")
                continue
            
            # Create validated problem object
            try:
                validated = DatasetProblem(
                    task_id=problem['task_id'],
                    tags=problem['tags'],
                    problem_description=problem['problem_description'],
                    starter_code=problem['starter_code']
                )
                valid_problems.append(validated)
            except Exception as e:
                errors.append(f"Problem {idx}: Parsing error: {str(e)}")
        
        return valid_problems, errors


# ============================================================================
# STAGE 1: GROUND TRUTH EXTRACTION
# ============================================================================

@dataclass
class GroundTruth:
    """Ground truth data for a problem"""
    task_id: str
    problem_description: str
    examples: List[str]
    constraints: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ExtractionPrompt:
    """Prompt templates for extraction stage"""
    
    SYSTEM = """You are a Data Extraction Specialist and Sanitization Engine.
Your task is to parse a raw LeetCode problem description into a STRICT JSON format.

CRITICAL RULES:
1. Extract ONLY core logic from problem_description. Remove Examples, Constraints, Hints, Follow-ups, and Complexity Requirements (e.g., "O(n) time").
2. BUT preserve logical guarantees even if they appear in description (e.g., "array is 0-indexed", "tree is distinct", "graph is a DAG" MUST be preserved.)
3. examples: Extract ALL input/output example blocks as complete strings
4. constraints: Extract FULL text of ALL constraints, including numerical limits (e.g., "1 <= n <= 100") and structural guarantees (e.g., "all elements are unique", "s consists of lowercase English letters").
5. If constraints are not explicitly stated, return empty list. Do NOT infer or guess.

Output ONLY valid JSON, no other text."""

    @staticmethod
    def user_prompt(raw_problem: str) -> str:
        return f"""RAW PROBLEM DESCRIPTION:
{raw_problem}

Extract and return ONLY this JSON (STRICT):
{{
    "problem_description": "string",
    "examples": ["string"],
    "constraints": ["string"] or []
}}"""

class OpenAIExtractor:
    """Extract ground truth using OpenAI API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
    
    def extract(self, problem: DatasetProblem) -> GroundTruth:
        """
        Extract ground truth from problem description using OpenAI.
        
        Args:
            problem: Validated dataset problem
            
        Returns:
            GroundTruth object with extracted data
            
        Raises:
            ValueError: If extraction fails or returns invalid JSON
        """
        
        user_msg = ExtractionPrompt.user_prompt(problem.problem_description)
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": ExtractionPrompt.SYSTEM},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            # Parse JSON
            extracted = json.loads(response_text)
            
            # Validate required fields
            if 'problem_description' not in extracted:
                raise ValueError("Missing 'problem_description' in extraction")
            if 'examples' not in extracted:
                raise ValueError("Missing 'examples' in extraction")
            if 'constraints' not in extracted:
                raise ValueError("Missing 'constraints' in extraction")
            
            # Ensure constraints is a list
            if not isinstance(extracted['constraints'], list):
                extracted['constraints'] = []
            
            # Create GroundTruth object
            return GroundTruth(
                task_id=problem.task_id,
                problem_description=extracted['problem_description'],
                examples=extracted['examples'],
                constraints=extracted['constraints']
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise ValueError(f"OpenAI extraction failed: {str(e)}")

# ============================================================================
# DATA PIPELINE: STAGE 0 -> STAGE 1
# ============================================================================

class Stage0to1Pipeline:
    """Pipeline from dataset validation to ground truth extraction"""
    
    def __init__(self, api_key: str = None):
        self.validator = DatasetValidator()
        self.extractor = OpenAIExtractor(api_key)
    
    def run(self, dataset_path: str, max_problems: int = None) -> Tuple[List[GroundTruth], Dict[str, Any]]:
        """
        Run complete pipeline from dataset to ground truth.
        
        Args:
            dataset_path: Path to LeetCode dataset JSON
            max_problems: Max problems to process (None = all)
            
        Returns:
            (ground_truths, stats)
        """
        
        print("\n" + "="*70)
        print("STAGE 0-1: DATASET VALIDATION & GROUND TRUTH EXTRACTION")
        print("="*70)
        
        # Load dataset
        print("\n[STAGE 0] Loading and validating dataset...")
        try:
            with open(dataset_path, 'r') as f:
                dataset_json = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {e}")
            return [], {"status": "failed", "error": str(e)}
        
        # Validate
        problems, errors = self.validator.validate_and_parse(dataset_json)
        
        if errors:
            print(f"\n⚠️ Validation errors:")
            for error in errors[:5]:
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more")
        
        if not problems:
            return [], {"status": "failed", "error": "No valid problems in dataset"}
        
        print(f"✓ Loaded {len(problems)} valid problems")
        
        if max_problems:
            problems = problems[:max_problems]
            print(f"✓ Processing first {max_problems} problems")
        
        # Extract ground truths
        print(f"\n[STAGE 1] Extracting ground truths ({len(problems)} problems)...")
        
        ground_truths = []
        failed = []
        
        for idx, problem in enumerate(problems):
            print(f"\n  [{idx+1}/{len(problems)}] {problem.task_id}...", end=" ", flush=True)
            
            try:
                gt = self.extractor.extract(problem)
                ground_truths.append(gt)
                print(f"✓")
                
            except ValueError as e:
                print(f"✗ {str(e)}")
                failed.append({
                    "task_id": problem.task_id,
                    "error": str(e)
                })
            except Exception as e:
                print(f"✗ Unexpected error: {str(e)}")
                failed.append({
                    "task_id": problem.task_id,
                    "error": str(e)
                })
        
        print(f"\n{'='*70}")
        print(f"✓ Extraction complete: {len(ground_truths)} successful, {len(failed)} failed")
        print(f"{'='*70}")
        
        stats = {
            "status": "success",
            "total_problems": len(problems),
            "extracted": len(ground_truths),
            "failed": len(failed),
            "failed_problems": failed
        }
        
        return ground_truths, stats

# ============================================================================
# OUTPUT & PERSISTENCE
# ============================================================================

class GroundTruthStore:
    """Store and persist ground truths"""
    
    @staticmethod
    def save(ground_truths: List[GroundTruth], output_path: str):
        """Save ground truths to JSON file"""
        data = {
            "metadata": {
                "count": len(ground_truths),
                "format_version": "1.0"
            },
            "ground_truths": [gt.to_dict() for gt in ground_truths]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Saved {len(ground_truths)} ground truths to {output_path}")
    
    @staticmethod
    def load(input_path: str) -> List[GroundTruth]:
        """Load ground truths from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        ground_truths = []
        for gt_dict in data.get('ground_truths', []):
            gt = GroundTruth(
                task_id=gt_dict['task_id'],
                problem_description=gt_dict['problem_description'],
                examples=gt_dict['examples'],
                constraints=gt_dict['constraints']
            )
            ground_truths.append(gt)
        
        return ground_truths

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Example usage
    print("\nStage 0-1: Dataset Validation & Ground Truth Extraction")
    print("Usage: python stage_0_1_complete.py <dataset.json> [max_problems]")
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        max_problems = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        # Run pipeline
        pipeline = Stage0to1Pipeline()
        ground_truths, stats = pipeline.run(dataset_path, max_problems)
        
        # Save results
        if ground_truths:
            output_path = "output_ground_truths.json"
            GroundTruthStore.save(ground_truths, output_path)
            print(f"\nStats: {json.dumps(stats, indent=2)}")
    else:
        print("\nExample:")
        print("  python stage_0_1_complete.py data/leetcode_dataset.json 10")