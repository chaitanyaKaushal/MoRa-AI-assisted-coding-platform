"""
STAGE 2: PERSONA GENERATION
Fully implemented with OpenAI integration
Input: GroundTruth from Stage 1
Output: PersonaDataset ready for fine-tuning
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from stage_0_1_complete import GroundTruth

# ============================================================================
# STAGE 2: PERSONA GENERATION
# ============================================================================

OPENAI_MODEL = "gpt-4o-mini"

@dataclass
class PersonaVariations:
    """Four persona variations for a problem"""
    task_id: str
    layman: str
    conversational: str
    technical_shorthand: str
    implementation_specific: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PersonaPrompt:
    """Prompts for persona generation"""
    
    SYSTEM = """You are an expert at simulating diverse user personas in technical contexts.
Your goal is to rewrite a raw coding problem statement into 4 different styles of user queries.

You will receive a problem description and generate 4 variations:
1. "layman": Zero technical jargon. Focus on the "story" or "scenario". Use simple words like "list", "items", "find". (e.g., "Check if a word reads the same way if you flip it backwards, like 'level' or 'racecar'.")
2. "conversational": Natural sentences, as if recalling a question from an interview 2 hours ago. Slightly imperfect memory is okay. (e.g., "I think the interviewer wanted me to write a function that tells you if a string is a palindrome. It should return true or false.")
3. "technical_shorthand": Ultra-concise functional specification. Focus: INPUT → FUNCTION → OUTPUT. (e.g., "Input: $S: string$. Output: $bool$. Return $True$ if $S[i] == S[len(S)-1-i]$ for all $i < len(S)$.")
4. "implementation_specific": Solution-oriented (e.g., "Python solution to check string symmetry by comparing a string to its reverse slice.").

STRICT RULES:
- OUTPUT FORMAT: Return ONLY valid JSON with keys: "layman", "conversational", "technical_shorthand", "implementation_specific"
- FORBIDDEN: Do NOT mention specific algorithm names (e.g., "Dijkstra", "BFS", "DFS", "greedy", "dynamic programming")
- FORBIDDEN: Do NOT mention specific function names (e.g., "solve", "Solution")
- FORBIDDEN: Do NOT mention complexity hints (e.g., "O(n)", "O(n^2)", "time complexity")
- FORBIDDEN: Do NOT mention data structure implementation details (e.g., "use hash map", "use stack", "use queue")
- FORBIDDEN: Do NOT include solution hints or specific approach hints
- AMBIGUITY: Preserve ambiguity if the original text is vague
- GOAL: Describe only INPUT → OUTPUT transformation

Return ONLY the JSON object, no other text."""

    @staticmethod
    def user_prompt(problem_description: str) -> str:
        return f"""Problem Description:
{problem_description}

Generate 4 persona variations. Return ONLY this JSON (no markdown, no extra text):
{{
    "layman": "string",
    "conversational": "string",
    "technical_shorthand": "string",
    "implementation_specific": "string"
}}"""

class ForbiddenKeywordValidator:
    """Validate personas don't contain forbidden keywords"""
    
    FORBIDDEN = [
        # Algorithms
        'dijkstra', 'bfs', 'dfs', 'breadth', 'depth', 'quicksort', 'mergesort',
        'heapsort', 'greedy', 'dynamic programming', 'dp ', 'backtracking',
        'recursion', 'memoization', 'memo',
        # Data structures
        'hash', 'hashmap', 'hash table', 'hash set', 'hashtable', 'hashset',
        'two pointer', 'sliding window', 'segment tree', 'fenwick', 'trie',
        'disjoint set', 'union find', 'max heap', 'min heap', 'priority queue',
        # Function/class names
        'def solve', 'function solve', 'solve(', ' solve ', 'class solution',
        'Solution class', 'method solve',
        # Complexity hints
        'o(n)', 'o(1)', 'o(log', 'o(n^', 'o(2^', 'o(n!', 'o(sqrt',
        'time complexity', 'space complexity', 'linear time', 'constant time',
        'logarithmic', 'exponential time',
        # Other hints
        'loop', 'for loop', 'while loop', 'if statement',
        'variable', 'counter', 'dictionary', 'map',
    ]
    
    @staticmethod
    def is_valid(text: str) -> bool:
        """Check if text is free of forbidden keywords"""
        text_lower = text.lower()
        for keyword in ForbiddenKeywordValidator.FORBIDDEN:
            if keyword in text_lower:
                return False
        return True

class CheckpointManager:
    """Manage checkpoints for resumable processing"""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """Initialize checkpoint manager"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_path(self, dataset_path: str) -> str:
        """Get checkpoint file path for a dataset"""
        basename = os.path.basename(dataset_path).replace('.json', '')
        return os.path.join(self.checkpoint_dir, f"stage2_checkpoint_{basename}.json")
    
    def load_checkpoint(self, dataset_path: str) -> Dict[str, Any]:
        """Load checkpoint if it exists"""
        if not dataset_path:
            return {}
        
        checkpoint_path = self.get_checkpoint_path(dataset_path)
        
        if not os.path.exists(checkpoint_path):
            return {}
        
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return {}
    
    def save_checkpoint(self, dataset_path: str, processed_tasks: List[str], 
                       persona_variations: List[Dict], stats: Dict):
        """Save checkpoint to resume later"""
        if not dataset_path:
            return
        
        checkpoint_path = self.get_checkpoint_path(dataset_path)
        
        checkpoint_data = {
            "dataset_path": dataset_path,
            "processed_tasks": processed_tasks,
            "persona_variations": persona_variations,
            "stats": stats
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def delete_checkpoint(self, dataset_path: str):
        """Delete checkpoint after successful completion"""
        if not dataset_path:
            return
        
        checkpoint_path = self.get_checkpoint_path(dataset_path)
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"✓ Checkpoint cleaned up")
        except Exception as e:
            print(f"Warning: Failed to delete checkpoint: {e}")

class OpenAIPersonaGenerator:
    """Generate personas using OpenAI API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, ground_truth: GroundTruth) -> PersonaVariations:
        """
        Generate persona variations from ground truth.
        
        Args:
            ground_truth: GroundTruth object from Stage 1
            
        Returns:
            PersonaVariations object with valid personas saved
            Invalid personas marked with [INVALID] prefix
            
        Raises:
            ValueError: Only if JSON parsing fails or ALL personas are invalid
        """
        
        problem_desc = ground_truth.problem_description.replace('\n', ' ').replace('\r', ' ')
        problem_desc = ' '.join(problem_desc.split())
        user_msg = PersonaPrompt.user_prompt(problem_desc)
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": PersonaPrompt.SYSTEM},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,  # Some creativity for variations
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            try:
                personas_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                # ONLY THEN replace escape sequences
                response_text = response_text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
                response_text = response_text.strip()
                personas_dict = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['layman', 'conversational', 'technical_shorthand', 'implementation_specific']
            for field in required_fields:
                if field not in personas_dict:
                    raise ValueError(f"Missing field: {field}")
            
            # Validate each persona individually and track which ones are valid
            valid_personas = {}
            invalid_reasons = {}
            
            for field in required_fields:
                is_valid = True
                reason = None
                
                # Check if empty or too short
                if not personas_dict[field] or len(personas_dict[field].strip()) < 10:
                    is_valid = False
                    reason = "too short or empty"
                # Check forbidden keywords in layman and conversational
                elif field in ['layman', 'conversational']:
                    if not ForbiddenKeywordValidator.is_valid(personas_dict[field]):
                        is_valid = False
                        reason = "contains forbidden keywords"
                
                if is_valid:
                    valid_personas[field] = personas_dict[field]
                else:
                    invalid_reasons[field] = reason
            
            # If we have at least 1 valid persona, save what we have
            if not valid_personas:
                error_details = "; ".join([f"{k}: {v}" for k, v in invalid_reasons.items()])
                raise ValueError(f"All personas failed validation: {error_details}")
            
            # Fill missing personas with a note about what failed
            fallback = next(iter(valid_personas.values())) if valid_personas else ""
            
            result_personas = {}
            for field in required_fields:
                if field in valid_personas:
                    result_personas[field] = valid_personas[field]
                else:
                    result_personas[field] = f"[INVALID: {invalid_reasons[field]}] {fallback[:100]}..."
            
            # Create PersonaVariations object with available data
            return PersonaVariations(
                task_id=ground_truth.task_id,
                layman=result_personas['layman'],
                conversational=result_personas['conversational'],
                technical_shorthand=result_personas['technical_shorthand'],
                implementation_specific=result_personas['implementation_specific']
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise ValueError(f"OpenAI persona generation failed: {str(e)}")

# ============================================================================
# DATA PIPELINE: STAGE 1 -> STAGE 2
# ============================================================================

class Stage1to2Pipeline:
    """Pipeline from ground truth to persona variations"""
    
    def __init__(self, api_key: str = None, checkpoint_dir: str = ".checkpoints"):
        self.generator = OpenAIPersonaGenerator(api_key)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    def run(self, ground_truths: List[GroundTruth], dataset_path: str = None, 
            max_problems: int = None, resume: bool = True) -> tuple:
        """
        Run complete pipeline from ground truth to personas.
        
        Args:
            ground_truths: List of GroundTruth objects from Stage 1
            dataset_path: Path to the original dataset (for checkpoint naming, optional)
            max_problems: Max problems to process
            resume: Whether to resume from checkpoint if it exists
            
        Returns:
            (persona_variations, stats)
        """
        
        print("\n" + "="*70)

        # Handle backward compatibility: if dataset_path is an int, it's actually max_problems
        if isinstance(dataset_path, int):
            max_problems = dataset_path
            dataset_path = None
        print("STAGE 2: PERSONA GENERATION")
        print("="*70)
        
        if not ground_truths:
            print("ERROR: No ground truths provided")
            return [], {"status": "failed", "error": "No input"}
        
        # Load checkpoint if resuming
        checkpoint = {}
        processed_tasks = set()
        persona_variations = []
        
        if resume and dataset_path:
            checkpoint = self.checkpoint_manager.load_checkpoint(dataset_path)
            if checkpoint:
                processed_tasks = set(checkpoint.get("processed_tasks", []))
                persona_variations = [PersonaVariations(**pv) for pv in checkpoint.get("persona_variations", [])]
                print(f"\n✓ Resuming from checkpoint: {len(processed_tasks)} already processed")
            else:
                print("\nℹ No checkpoint found, starting fresh")
        
        if max_problems:
            ground_truths = ground_truths[:max_problems]
        
        # Filter to only unprocessed tasks
        remaining_tasks = [gt for gt in ground_truths if gt.task_id not in processed_tasks]
        
        print(f"\n[STAGE 2] Generating persona variations ({len(remaining_tasks)} remaining)...")
        
        failed = []
        partial = []
        
        for idx, gt in enumerate(remaining_tasks):
            print(f"\n  [{len(processed_tasks) + idx + 1}/{len(ground_truths)}] {gt.task_id}...", end=" ", flush=True)
            
            try:
                pv = self.generator.generate(gt)
                
                # Check if any personas are marked as invalid
                has_invalid = any("[INVALID:" in getattr(pv, field) for field in 
                                  ['layman', 'conversational', 'technical_shorthand', 'implementation_specific'])
                
                persona_variations.append(pv)
                processed_tasks.add(gt.task_id)
                
                if has_invalid:
                    partial.append(gt.task_id)
                    print(f"⚠ (partial - some personas invalid)")
                else:
                    print(f"✓")
                
                # Save checkpoint after each successful task
                if dataset_path:
                    stats = self._build_stats(len(ground_truths), len(persona_variations), 
                                             len(partial), len(failed))
                    self.checkpoint_manager.save_checkpoint(
                        dataset_path, 
                        list(processed_tasks),
                        [pv.to_dict() for pv in persona_variations],
                        stats
                    )
                
            except ValueError as e:
                print(f"✗ {str(e)}")
                processed_tasks.add(gt.task_id)
                failed.append({"task_id": gt.task_id, "error": str(e)})
                
                # Save checkpoint after failure too
                if dataset_path:
                    stats = self._build_stats(len(ground_truths), len(persona_variations), 
                                             len(partial), len(failed))
                    self.checkpoint_manager.save_checkpoint(
                        dataset_path,
                        list(processed_tasks),
                        [pv.to_dict() for pv in persona_variations],
                        stats
                    )
                
            except Exception as e:
                print(f"✗ Unexpected error: {str(e)}")
                processed_tasks.add(gt.task_id)
                failed.append({"task_id": gt.task_id, "error": str(e)})
        
        # If all tasks completed successfully, delete checkpoint
        if dataset_path and len(processed_tasks) == len(ground_truths):
            self.checkpoint_manager.delete_checkpoint(dataset_path)
        
        print(f"\n{'='*70}")
        print(f"✓ Persona generation complete: {len(persona_variations)} successful, {len(failed)} failed")
        print(f"{'='*70}")
        
        stats = self._build_stats(len(ground_truths), len(persona_variations), 
                                 len(partial), len(failed))
        stats["partial_problems"] = partial
        stats["failed_problems"] = failed
        
        return persona_variations, stats
    
    def _build_stats(self, total: int, generated: int, partial: int, failed: int) -> Dict:
        """Build stats dictionary"""
        return {
            "status": "success",
            "total": total,
            "generated": generated,
            "partial": partial,
            "failed": failed
        }

# ============================================================================
# STAGE 2.1: FINE-TUNING DATASET PREPARATION
# ============================================================================

@dataclass
class FinetuningEntry:
    """Single entry for fine-tuning dataset"""
    messages: List[Dict[str, str]]
    
    def to_dict(self) -> Dict:
        return {"messages": self.messages}

class FinetuningDatasetBuilder:
    """Build fine-tuning dataset from personas and ground truths"""
    
    SYSTEM_MESSAGE = """You are an expert Competitive Programming Problem Setter.
Your task is to convert a vague human-written coding problem into a fully specified LeetCode-style problem.
Output STRICT JSON ONLY, no other text."""

    @staticmethod
    def create_entries(persona_variations: List[PersonaVariations], 
                       ground_truths_by_id: Dict[str, GroundTruth]) -> List[FinetuningEntry]:
        """
        Create fine-tuning entries from personas and ground truths.
        
        For each persona variation, create one fine-tuning entry showing:
        Input: vague problem (from persona)
        Output: formal specification (derived from ground truth)
        """
        
        entries = []
        
        for pv in persona_variations:
            gt = ground_truths_by_id.get(pv.task_id)
            if not gt:
                print(f"Warning: No ground truth for {pv.task_id}")
                continue
            
            # Build formal spec from ground truth
            formal_spec = {
                "title": pv.task_id.replace('-', ' ').title(),
                "problem_description": gt.problem_description,
                "examples": gt.examples,
                "constraints": gt.constraints
            }
            
            # Create 4 entries (one per persona)
            for persona_type in ['layman', 'conversational', 'technical_shorthand', 'implementation_specific']:
                vague_query = getattr(pv, persona_type)
                
                # Skip invalid personas
                if "[INVALID:" in vague_query:
                    continue
                
                entry = FinetuningEntry(
                    messages=[
                        {"role": "system", "content": FinetuningDatasetBuilder.SYSTEM_MESSAGE},
                        {"role": "user", "content": f"Convert this vague problem to a formal specification:\n\n{vague_query}"},
                        {"role": "assistant", "content": json.dumps(formal_spec, indent=2)}
                    ]
                )
                entries.append(entry)
        
        return entries

class FinetuningDatasetStore:
    """Save fine-tuning dataset to JSONL format"""
    
    @staticmethod
    def save_jsonl(entries: List[FinetuningEntry], output_path: str):
        """
        Save entries to JSONL format (one JSON per line).
        
        Format is compatible with OpenAI fine-tuning API.
        """
        with open(output_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + '\n')
        
        print(f"\n✓ Saved {len(entries)} fine-tuning entries to {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\nStage 2: Persona Generation")
    print("Usage: python stage_2_complete.py <ground_truths.json> [max_problems]")
    print("       Add --no-resume flag to start fresh (ignore checkpoints)")
    
    resume = True
    if "--no-resume" in sys.argv:
        resume = False
        sys.argv.remove("--no-resume")
    
    if len(sys.argv) > 1:
        gt_path = sys.argv[1]
        max_problems = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        # Load ground truths from Stage 1
        from stage_0_1_complete import GroundTruthStore
        
        print(f"\nLoading ground truths from {gt_path}...")
        ground_truths = GroundTruthStore.load(gt_path)
        print(f"Loaded {len(ground_truths)} ground truths")
        
        # Run Stage 2 pipeline
        pipeline = Stage1to2Pipeline()
        persona_variations, stats = pipeline.run(ground_truths, gt_path, max_problems, resume=resume)
        
        if persona_variations:
            # Save persona variations
            output_personas = "output_personas.json"
            personas_data = {
                "metadata": {"count": len(persona_variations), "format_version": "1.0"},
                "personas": [pv.to_dict() for pv in persona_variations]
            }
            with open(output_personas, 'w') as f:
                json.dump(personas_data, f, indent=2)
            print(f"\n✓ Saved {len(persona_variations)} persona variations to {output_personas}")
            
            # Build and save fine-tuning dataset
            gt_by_id = {gt.task_id: gt for gt in ground_truths}
            finetuning_entries = FinetuningDatasetBuilder.create_entries(persona_variations, gt_by_id)
            
            output_jsonl = "output_finetuning_dataset.jsonl"
            FinetuningDatasetStore.save_jsonl(finetuning_entries, output_jsonl)
            
            print(f"\nStats: {json.dumps(stats, indent=2)}")
    else:
        print("\nExample:")
        print("  python stage_2_complete.py output_ground_truths.json 10")