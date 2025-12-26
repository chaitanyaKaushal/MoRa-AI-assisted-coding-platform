#!/usr/bin/env python3
"""
Simple script to check fine-tuning job status
Usage: python check_job.py <job_id>
"""

import sys
from openai import OpenAI

def check_job_status(job_id: str):
    """Check the status of a fine-tuning job"""
    
    try:
        client = OpenAI()  # Uses OPENAI_API_KEY from environment
        
        print(f"\n{'='*70}")
        print(f"Checking job status: {job_id}")
        print(f"{'='*70}\n")
        
        # Get job status
        response = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"Job Details:")
        print(f"  Job ID: {response.id}")
        print(f"  Status: {response.status}")
        print(f"  Model: {response.model}")
        print(f"  Fine-tuned Model: {getattr(response, 'fine_tuned_model', 'N/A')}")
        print(f"  Created: {getattr(response, 'created_at', 'N/A')}")
        print(f"  Updated: {getattr(response, 'updated_at', 'N/A')}")
        
        # Show error if job failed
        if response.status == 'failed':
            error = getattr(response, 'error', None)
            if error:
                print(f"  Error: {error}")
        
        # Show checkpoints if job succeeded
        if response.status == 'succeeded':
            if hasattr(response, 'checkpoints') and response.checkpoints:
                print(f"\n  Checkpoints:")
                for cp in response.checkpoints:
                    step = getattr(cp, 'step', 'N/A')
                    model = getattr(cp, 'fine_tuned_model_checkpoint', 'N/A')
                    print(f"    - Step {step}: {model}")
        
        # Status interpretation
        print(f"\n{'─'*70}")
        if response.status == 'succeeded':
            print(f"✓ Training completed successfully!")
            model_id = getattr(response, 'fine_tuned_model', None)
            if model_id:
                print(f"\n  Your fine-tuned model is ready:")
                print(f"  {model_id}")
                print(f"\n  Use this model ID for inference!")
        elif response.status == 'running':
            print(f"⏳ Job is currently running...")
            print(f"  Check again in 1-5 minutes")
        elif response.status == 'queued':
            print(f"⏳ Job is queued...")
            print(f"  Will start training soon")
        elif response.status == 'validating_files':
            print(f"⏳ Validating training/validation files...")
            print(f"  This may take 1-5 minutes")
        elif response.status == 'failed':
            print(f"✗ Job failed!")
            error = getattr(response, 'error', 'Unknown error')
            print(f"  Error: {error}")
        elif response.status == 'cancelled':
            print(f"⊗ Job was cancelled")
        
        print(f"{'─'*70}\n")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_job.py <job_id>")
        print("Example: python check_job.py ftjob-abc123xyz456def")
        sys.exit(1)
    
    job_id = sys.argv[1]
    check_job_status(job_id)