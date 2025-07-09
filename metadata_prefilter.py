import json
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
def prefilter():
    dataset = load_dataset("nvidia/OpenMathReasoning", split="cot")
    pass_conditions = lambda data: data["problem_type"] == "has_answer_extracted" and data["expected_answer"].isdigit() and data["pass_rate_72b_tir"] != "n/a" and float(data["pass_rate_72b_tir"]) > 0.3 and float(data["pass_rate_72b_tir"]) < 0.7

    # Filter the dataset based on pass conditions
    filtered_dataset = dataset.filter(pass_conditions)

    # Convert to list of dictionaries for JSON serialization
    filtered_data = [item for item in filtered_dataset]

    # Save to JSON file
    with open("metas/filtered_openmathreasoning.json", "w") as fp:
        json.dump(filtered_data, fp, indent=2)

    print(f"Filtered dataset saved with {len(filtered_data)} samples")