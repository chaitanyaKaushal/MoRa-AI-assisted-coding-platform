import json
import os

STAGE_1_OUTPUT = "leetcode_dataset_stage1.json"
STAGE_2_OUTPUT = "leetcode_dataset_stage2.json"
GROUND_TRUTHS = "stages/ground_truths.json"

def load_data(filepath):
    """Safely loads the dataset JSON."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # Handle cases where the file might be wrapped in a dictionary key like "dataset"
            if isinstance(data, dict) and 'dataset' in data:
                return data['dataset']
            return data
        except json.JSONDecodeError:
            return []

if __name__ == "__main__":
    stage1 = load_data(STAGE_1_OUTPUT)
    print(len(stage1))
    stage2 = load_data(STAGE_2_OUTPUT)
    print(len(stage2))
    ground_truths = load_data(GROUND_TRUTHS)
    print(len(ground_truths['ground_truths']))
