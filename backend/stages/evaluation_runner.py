import json
import time
import random
import requests
import subprocess
from collections import defaultdict

DATASET_PATH = "finetuning_dataset_validation.jsonl"
IMAGE_NAME = "stages-api"

MAX_EXAMPLES = 100

CONNECT_TIMEOUT = 240
READ_TIMEOUT = 240
EVAL_DEADLINE = 240
SERVER_START_TIMEOUT = 240


# -------------------------
# Docker helpers
# -------------------------

def start_container():
    port = random.randint(10000, 60000)
    result = subprocess.run(
        [
            "docker", "run",
            "-d",
            "--network", "mora-net",
            "-p", f"{port}:8000",
            "-e", "DATABASE_URL=postgresql://admin:password@postgres:5432/coding_platform",
            IMAGE_NAME
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip(), port


def kill_container(container_id):
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def wait_for_server(port, timeout=SERVER_START_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:8000/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def dump_logs(container_id):
    print("\n=== CONTAINER LOGS ===")
    subprocess.run(["docker", "logs", container_id])


# -------------------------
# Evaluation logic
# -------------------------

def run_evaluation():
    per_problem_metrics = []

    with open(DATASET_PATH) as f:
        for idx, line in enumerate(f):
            if idx >= MAX_EXAMPLES:
                break

            print(f"\n=== Evaluating problem {idx + 1} ===")

            container_id, port = start_container()
            api_url = f"http://127.0.0.1:8000/api/generate-problem"

            if not wait_for_server(port):
                dump_logs(container_id)
                kill_container(container_id)
                per_problem_metrics.append({
                    "success": 0,
                    "error": "Server failed to start"
                })
                continue

            record = json.loads(line)
            vague_problem = record["messages"][1]["content"]

            req_start = time.time()
            stream_start = req_start
            metrics = None
            success = 0

            try:
                resp = requests.post(
                    api_url,
                    json={"vague_problem": vague_problem},
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
                )

                try:
                    for raw in resp.iter_lines():
                        if time.time() - stream_start > EVAL_DEADLINE:
                            kill_container(container_id)
                            raise TimeoutError("Evaluation deadline exceeded")

                        if not raw:
                            continue
                        # print("[RAW STREAM]", raw)
                        event = json.loads(raw)

                        if "error" in event:
                            metrics = {
                                "success": 0,
                                "error": event["error"]
                            }
                            break

                        if "final_result" in event:
                            metrics = event["final_result"]["metrics"]
                            success = 1
                            break

                except requests.exceptions.ReadTimeout:
                    metrics = {
                        "success": 0,
                        "error": "Read timeout"
                    }

                finally:
                    resp.close()

            except Exception as e:
                metrics = {
                    "success": 0,
                    "error": str(e)
                }

            elapsed = time.time() - req_start

            if metrics is None:
                metrics = {
                    "success": 0,
                    "error": "No response"
                }

            metrics["success"] = success
            metrics["end_to_end_latency"] = elapsed
            per_problem_metrics.append(metrics)

            kill_container(container_id)
            time.sleep(1)

    return per_problem_metrics


# -------------------------
# Aggregation
# -------------------------

def aggregate(metrics_list):
    agg = defaultdict(float)
    count = len(metrics_list)

    for m in metrics_list:
        agg["psr"] += m["success"]

        if m["success"] == 0:
            continue

        agg["candidate_tests"] += m["stage5_2"]["candidate_tests"]
        agg["golden_tests"] += m["stage5_2"]["golden_tests"]

        agg["ccr"] += 1 if m["stage5"]["consensus_reached"] else 0

        agg["heals_attempted"] += m["stage5"]["heals_attempted"]
        agg["heals_successful"] += m["stage5"]["heals_successful"]

        agg["chf"] += 1 if m["stage5_2"]["circular_healing_triggered"] else 0

        agg["lat_stage3"] += m["stage3"]["latency"]
        agg["lat_stage4"] += m["stage4"]["latency"]
        agg["lat_stage5"] += m["stage5"]["latency"]
        agg["lat_stage5_2"] += m["stage5_2"]["latency"]
        agg["lat_e2e"] += m["end_to_end_latency"]

    successful = max(1, agg["psr"])

    return {
        "PSR": agg["psr"] / count,
        "TSR": agg["golden_tests"] / max(1, agg["candidate_tests"]),
        "CHF": agg["chf"] / successful,
        "CCR": agg["ccr"] / successful,
        "HY": agg["heals_successful"] / max(1, agg["heals_attempted"]),
        "Latency (Stage 3)": agg["lat_stage3"] / successful,
        "Latency (Stage 4)": agg["lat_stage4"] / successful,
        "Latency (Stage 5)": agg["lat_stage5"] / successful,
        "Latency (Stage 5.2)": agg["lat_stage5_2"] / successful,
        "Latency (End-to-End)": agg["lat_e2e"] / successful,
    }


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    metrics = run_evaluation()
    results = aggregate(metrics)

    print("\n===== FINAL AGGREGATE METRICS =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
