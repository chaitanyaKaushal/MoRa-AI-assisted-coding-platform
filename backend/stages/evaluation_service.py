# evaluation_service.py

import json
import time
import traceback
import csv
from collections import defaultdict
from pathlib import Path

# === IMPORT YOUR ACTUAL PIPELINE ===
from main import (
    stage3_generate_specification,
    stage4_generate_tests,
    stage5_consensus_reference,
    stage5_build_golden_suite,
    extract_signature,
    OracleFailure,
    TestQualityFailure
)

DATASET = "finetuning_dataset_validation.jsonl"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ============================
# GLOBAL METRICS
# ============================

metrics = {
    "total_inputs": 0,
    "successful_problems": 0,

    "stage3_failures": 0,
    "stage4_failures": 0,
    "stage5_failures": 0,

    "circular_healing_count": 0,
    "consensus_success": 0,
    "consensus_fail": 0,

    "total_candidate_tests": 0,
    "total_golden_tests": 0,

    "healing_attempts": [],
    "persona_success": defaultdict(int),
    "persona_total": defaultdict(int),
}

latency_rows = []
problem_rows = []

# ============================
# HELPER
# ============================

def extract_persona(record):
    text = record["messages"][1]["content"].lower()
    if "remember" in text or "interviewer" in text:
        return "conversational"
    if "calculate" in text and "return" in text:
        return "technical"
    if "i think" in text:
        return "layman"
    return "implementation"

# ============================
# MAIN LOOP
# ============================

with open(DATASET) as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        metrics["total_inputs"] += 1
        record = json.loads(line)
        vague_problem = record["messages"][1]["content"]
        persona = extract_persona(record)
        metrics["persona_total"][persona] += 1

        problem_id = f"eval_{metrics['total_inputs']}"

        try:
            t0 = time.time()

            # ---------- STAGE 3 ----------
            t3 = time.time()
            spec = stage3_generate_specification(vague_problem)
            stage3_time = time.time() - t3

            signature = extract_signature(spec["starter_code"])

            # ---------- STAGE 4 ----------
            t4 = time.time()
            test_inputs_flat, test_inputs_dict = stage4_generate_tests(
                spec, signature
            )
            stage4_time = time.time() - t4

            # ---------- STAGE 5 (CONSENSUS) ----------
            t5 = time.time()
            reference_code = stage5_consensus_reference(
                spec, test_inputs_dict
            )
            stage5_consensus_time = time.time() - t5

            # ---------- STAGE 5.2 (GOLDEN) ----------
            try:
                golden_suite = stage5_build_golden_suite(
                    reference_code,
                    test_inputs_flat,
                    signature
                )
            except TestQualityFailure:
                metrics["circular_healing_count"] += 1
                raise

            stage5_total_time = time.time() - t5
            total_time = time.time() - t0

            # ---------- METRICS ----------
            metrics["successful_problems"] += 1
            metrics["persona_success"][persona] += 1
            metrics["consensus_success"] += 1

            metrics["total_candidate_tests"] += len(test_inputs_flat)
            metrics["total_golden_tests"] += len(golden_suite)

            latency_rows.append({
                "problem_id": problem_id,
                "stage3": stage3_time,
                "stage4": stage4_time,
                "stage5": stage5_total_time,
                "total": total_time
            })

            problem_rows.append({
                "problem_id": problem_id,
                "persona": persona,
                "candidate_tests": len(test_inputs_flat),
                "golden_tests": len(golden_suite),
                "success": True
            })

        except OracleFailure:
            metrics["stage4_failures"] += 1
            metrics["consensus_fail"] += 1

        except TestQualityFailure:
            metrics["stage5_failures"] += 1
            metrics["consensus_fail"] += 1

        except Exception:
            metrics["stage3_failures"] += 1
            metrics["consensus_fail"] += 1
            traceback.print_exc()

# ============================
# DERIVED METRICS
# ============================

summary = {
    "PSR": metrics["successful_problems"] / metrics["total_inputs"],
    "TSR": (
        metrics["total_golden_tests"] /
        max(metrics["total_candidate_tests"], 1)
    ),
    "CHF": metrics["circular_healing_count"] / metrics["total_inputs"],
    "CCR": metrics["consensus_success"] /
           max(metrics["consensus_success"] + metrics["consensus_fail"], 1),
    "persona_breakdown": {
        p: metrics["persona_success"][p] / metrics["persona_total"][p]
        for p in metrics["persona_total"]
    },
    "raw": metrics
}

# ============================
# WRITE OUTPUTS
# ============================

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(OUT_DIR / "per_problem.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=problem_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(problem_rows)

with open(OUT_DIR / "latency.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=latency_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(latency_rows)

print("Evaluation complete. Results saved to /results")
