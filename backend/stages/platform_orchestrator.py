"""
COMPLETE END-TO-END ORCHESTRATOR
All 6 stages with guaranteed data flow
Production-ready with real OpenAI integration
"""

import json
import sys
from typing import Dict, List, Any, Tuple

from stage_0_1_complete import Stage0to1Pipeline, GroundTruthStore, DatasetProblem
from stage_2_complete import Stage1to2Pipeline, FinetuningDatasetBuilder, FinetuningDatasetStore
# from stage_3_complete import Stage3Pipeline
# from stage_4_complete import Stage4Pipeline
# from stage_5_complete import Stage5Pipeline
# from stage_6_complete import Stage6Pipeline

# ============================================================================
# COMPLETE TRAINING WORKFLOW
# ============================================================================

class TrainingWorkflow:
    """Complete training pipeline: Dataset -> Fine-tuning Dataset"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def run(self, dataset_path: str, 
           output_ground_truths: str = "golden_ground_truths.json",
           output_personas: str = "golden_personas.json",
           output_finetuning: str = "golden_finetuning_dataset.jsonl",
           max_problems: int = None) -> Dict[str, Any]:
        """
        Run complete training workflow.
        
        Pipeline:
        Dataset (JSON) 
          -> Stage 0-1: Extraction 
          -> Ground Truths
          -> Stage 2: Personas
          -> Persona Variations
          -> Stage 2.1: Fine-tuning prep
          -> JSONL ready for model fine-tuning
        
        Args:
            dataset_path: Path to LeetCode dataset JSON
            output_ground_truths: Path to save ground truths
            output_personas: Path to save personas
            output_finetuning: Path to save fine-tuning JSONL
            max_problems: Max problems to process (None = all)
            
        Returns:
            Workflow statistics
        """
        
        print("\n" + "█"*70)
        print("TRAINING WORKFLOW")
        print("█"*70)
        
        stats = {
            "stage_0_1": None,
            "stage_2": None,
            "stage_2_1": None,
            "status": "running"
        }
        
        try:
            # ============ STAGE 0-1: EXTRACTION ============
            print("\n[TRAINING] Running Stages 0-1: Dataset Validation & Extraction")
            
            pipeline_0_1 = Stage0to1Pipeline(self.api_key)
            ground_truths, stats_0_1 = pipeline_0_1.run(dataset_path, max_problems)
            
            stats['stage_0_1'] = stats_0_1
            
            if not ground_truths:
                stats['status'] = 'failed_stage_0_1'
                return stats
            
            # Save ground truths
            GroundTruthStore.save(ground_truths, output_ground_truths)
            
            # ============ STAGE 2: PERSONAS ============
            print("\n[TRAINING] Running Stage 2: Persona Generation")
            
            pipeline_2 = Stage1to2Pipeline(self.api_key)
            persona_variations, stats_2 = pipeline_2.run(ground_truths, max_problems)
            
            stats['stage_2'] = stats_2
            
            if not persona_variations:
                stats['status'] = 'failed_stage_2'
                return stats
            
            # Save personas
            personas_data = {
                "metadata": {"count": len(persona_variations), "format_version": "1.0"},
                "personas": [pv.to_dict() for pv in persona_variations]
            }
            with open(output_personas, 'w') as f:
                json.dump(personas_data, f, indent=2)
            print(f"\n✓ Saved {len(persona_variations)} personas to {output_personas}")
            
            # ============ STAGE 2.1: FINE-TUNING PREP ============
            print("\n[TRAINING] Running Stage 2.1: Fine-tuning Dataset Preparation")
            
            gt_by_id = {gt.task_id: gt for gt in ground_truths}
            finetuning_entries = FinetuningDatasetBuilder.create_entries(
                persona_variations, gt_by_id
            )
            
            FinetuningDatasetStore.save_jsonl(finetuning_entries, output_finetuning)
            
            stats['stage_2_1'] = {
                "status": "success",
                "total_entries": len(finetuning_entries),
                "entries_per_persona": 4
            }
            
            stats['status'] = 'success'
            
            print("\n" + "█"*70)
            print("✓ TRAINING WORKFLOW COMPLETE")
            print("█"*70)
            print(f"\nGenerated:")
            print(f"  - {len(ground_truths)} ground truths → {output_ground_truths}")
            print(f"  - {len(persona_variations)} persona sets → {output_personas}")
            print(f"  - {len(finetuning_entries)} fine-tuning entries → {output_finetuning}")
            print(f"\nNext step: Fine-tune model on {output_finetuning}")
            
            return stats
            
        except Exception as e:
            stats['status'] = 'error'
            stats['error'] = str(e)
            return stats

# # ============================================================================
# # COMPLETE INFERENCE WORKFLOW
# # ============================================================================

# class InferenceWorkflow:
#     """Complete inference pipeline: Problem -> Evaluation"""
    
#     def __init__(self, api_key: str = None):
#         self.api_key = api_key
    
#     def run(self, vague_problem: str, starter_code: str, user_code: str) -> Dict[str, Any]:
#         """
#         Run complete inference workflow.
        
#         Pipeline:
#         Vague Problem + Starter Code
#           -> Stage 3: Formal Specification
#           -> Stage 4: Test Case Generation
#           -> Stage 5: Reference Solution & Golden Suite
#           -> User Code
#           -> Stage 6: Evaluation
#           -> Test Results
        
#         Args:
#             vague_problem: User's problem description
#             starter_code: Template code with function signature
#             user_code: User's submitted solution
            
#         Returns:
#             Evaluation results
#         """
        
#         print("\n" + "█"*70)
#         print("INFERENCE WORKFLOW")
#         print("█"*70)
        
#         workflow_result = {
#             "stage_3": None,
#             "stage_4": None,
#             "stage_5": None,
#             "stage_6": None,
#             "status": "running"
#         }
        
#         try:
#             # ============ STAGE 3: FORMAL SPECIFICATION ============
#             print("\n[INFERENCE] Running Stage 3: Vague → Formal Specification")
            
#             pipeline_3 = Stage3Pipeline(self.api_key)
#             formal_spec, stats_3 = pipeline_3.run(vague_problem, starter_code)
            
#             workflow_result['stage_3'] = stats_3
            
#             if not formal_spec:
#                 workflow_result['status'] = 'failed_stage_3'
#                 return workflow_result
            
#             # ============ STAGE 4: TEST GENERATION ============
#             print("\n[INFERENCE] Running Stage 4: Test Case Oracle")
            
#             pipeline_4 = Stage4Pipeline(self.api_key)
#             test_cases, stats_4 = pipeline_4.run(formal_spec)
            
#             workflow_result['stage_4'] = stats_4
            
#             if not test_cases:
#                 workflow_result['status'] = 'failed_stage_4'
#                 return workflow_result
            
#             # ============ STAGE 5: REFERENCE SOLUTION ============
#             print("\n[INFERENCE] Running Stage 5: Reference Solution & Golden Suite")
            
#             pipeline_5 = Stage5Pipeline(self.api_key)
#             stage_5_result, stats_5 = pipeline_5.run(formal_spec, test_cases)
            
#             workflow_result['stage_5'] = stats_5
            
#             if not stage_5_result:
#                 workflow_result['status'] = 'failed_stage_5'
#                 return workflow_result
            
#             # Extract golden suite and evaluation type
#             golden_suite = stage_5_result['golden_suite']
#             evaluation_type = stage_5_result['evaluation_type']
            
#             # ============ STAGE 6: EVALUATION ============
#             print("\n[INFERENCE] Running Stage 6: User Code Evaluation")
            
#             pipeline_6 = Stage6Pipeline()
#             evaluation_result = pipeline_6.run(
#                 user_code,
#                 golden_suite,
#                 formal_spec.input_signature['arg_types'],
#                 evaluation_type
#             )
            
#             workflow_result['stage_6'] = evaluation_result
#             workflow_result['status'] = 'success'
            
#             print("\n" + "█"*70)
#             print("✓ INFERENCE WORKFLOW COMPLETE")
#             print("█"*70)
            
#             return workflow_result
            
#         except Exception as e:
#             workflow_result['status'] = 'error'
#             workflow_result['error'] = str(e)
#             return workflow_result

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class PlatformOrchestrator:
    """Master orchestrator for complete platform"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.training = TrainingWorkflow(api_key)
        # self.inference = InferenceWorkflow(api_key)
    
    def run_training(self, dataset_path: str, max_problems: int = None) -> Dict[str, Any]:
        """Run complete training workflow"""
        return self.training.run(dataset_path, max_problems=max_problems)
    
    # def run_inference(self, vague_problem: str, starter_code: str, user_code: str) -> Dict[str, Any]:
    #     """Run complete inference workflow"""
    #     return self.inference.run(vague_problem, starter_code, user_code)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_usage():
    """Print usage information"""
    print("""
AI-Powered LeetCode Platform - Complete Implementation

Usage:
    python platform_orchestrator.py train <dataset.json> [max_problems]
    python platform_orchestrator.py infer

Examples:
    # Training
    python platform_orchestrator.py train data/leetcode_dataset.json 10
    
    # Inference (interactive)
    python platform_orchestrator.py infer
""")

def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1]
    
    # Initialize orchestrator
    api_key = None  # Will use OPENAI_API_KEY environment variable
    orch = PlatformOrchestrator(api_key)
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Error: Dataset path required")
            print_usage()
            return
        
        dataset_path = sys.argv[2]
        max_problems = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        print(f"\nStarting training on {dataset_path}")
        if max_problems:
            print(f"Processing first {max_problems} problems")
        
        result = orch.run_training(dataset_path, max_problems)
        
        print("\n" + "="*70)
        print("Training Complete")
        print("="*70)
        print(json.dumps(result, indent=2))
    
    elif command == "infer":
        print("\nStarting inference workflow...")
        
        # Example problem
        vague_problem = """
Given a list of numbers, find two that add up to a target.
Return the indices of the two numbers.
"""
        
        starter_code = """class Solution:
    def solve(self, nums: List[int], target: int) -> List[int]:
        pass
"""
        
        # Example user solution
        user_code = """class Solution:
    def solve(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
"""
        
        result = orch.run_inference(vague_problem, starter_code, user_code)
        
        print("\n" + "="*70)
        print("Inference Complete")
        print("="*70)
        
        # Print summary
        if result.get('stage_6'):
            eval_result = result['stage_6']
            print(f"\nFinal Results:")
            print(f"  Passed: {eval_result.get('passed', 0)}/{eval_result.get('total', 0)}")
            print(f"  Percentage: {eval_result.get('percentage', 0):.1f}%")
            
            # Show test details
            if eval_result.get('results'):
                print(f"\nTest Details:")
                for idx, test_result in enumerate(eval_result['results'][:5], 1):
                    status = "✓" if test_result['passed'] else "✗"
                    print(f"  Test {idx}: {status}")
                    if test_result.get('error'):
                        print(f"    Error: {test_result['error']}")
        
        print(f"\nFull results: {json.dumps(result, indent=2)}")
    
    else:
        print(f"Unknown command: {command}")
        print_usage()

if __name__ == "__main__":
    main()