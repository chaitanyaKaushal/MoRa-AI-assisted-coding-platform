# import json
# import tiktoken

# # Choose the model you plan to fine-tune
# MODEL_NAME = "gpt-4.1-mini-2025-04-14"
# enc = tiktoken.encoding_for_model(MODEL_NAME)

# def count_tokens_in_jsonl(path):
#     total_tokens = 0

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)
#             for msg in obj["messages"]:
#                 total_tokens += len(enc.encode(msg["content"]))

#     return total_tokens

# tokens = count_tokens_in_jsonl("finetuning_dataset.jsonl")
# print("Total tokens:", tokens)

import json

file_path = 'data/leetcode_dataset.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Ensure the 'dataset' key exists
    if 'dataset' in data:
        # Extract question_ids from the list of dictionaries
        # Using a set comprehension automatically handles uniqueness
        unique_ids_set = {item.get('question_id') for item in data['dataset'] if 'question_id' in item}
        
        # Convert the set back to a list (optional, for sorting or indexing)
        unique_ids_list = sorted(list(unique_ids_set))
        
        print(f"Successfully extracted {len(unique_ids_list)} unique question IDs.")
        print(unique_ids_list)
        
    else:
        print("Error: The key 'dataset' was not found in the JSON file.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from '{file_path}'.")