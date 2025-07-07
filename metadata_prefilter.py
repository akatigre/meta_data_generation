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


def meta_upload_hf():
    import json, pandas as pd, pyarrow as pa
    from datasets import Dataset
    from huggingface_hub import HfApi

    JSON_PATH      = "metas/openmathreasoning_meta.json"                 # your source file
    PARQUET_TRAIN_PATH   = "metas/openmathreasoning_meta_train.parquet"
    PARQUET_TEST_PATH   = "metas/openmathreasoning_meta_test.parquet"
    HF_REPO_ID     = "yjyjyj98/openmathreasoning-meta"   # change this!
    PRIVATE_REPO   = False                           # or False for public

    # ────────────────────────── 1. load ──────────────────────────
    with open(JSON_PATH) as f:
        raw = json.load(f)          # raw is dict[id] -> dict[fields]

    if isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            value['pid'] = key
            records.append(value)
    else:                            # already list-like
        records = raw

    # ────────────────── 2. pandas dataframe ──────────────────────
    df = pd.DataFrame.from_records(records)
    if {"summary", "meta"}.issubset(df.columns):
        # expand each nested dict into columns
        summary_flat = df.pop("summary").apply(pd.Series)
        meta_flat    = df.pop("meta").apply(pd.Series)

        # merge everything back together
        df = pd.concat([df, summary_flat, meta_flat], axis=1)

        # optional: put pid first for readability
        if "pid" in df.columns:
            cols = ["pid"] + [c for c in df.columns if c != "pid"]
            df = df[cols]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_parquet(PARQUET_TRAIN_PATH, index=False)
    print(f"✅  saved {PARQUET_TRAIN_PATH} with {len(train_df)} samples")
    
    val_df.to_parquet(PARQUET_TEST_PATH, index=False)
    print(f"✅  saved {PARQUET_TEST_PATH} with {len(val_df)} samples")

    # ────────────────── 4. push to HF hub ─────────────────────────
    ds = Dataset(pa.Table.from_pandas(train_df, preserve_index=False), split="train")
    ds_val = Dataset(pa.Table.from_pandas(val_df, preserve_index=False), split="test")
    
    api = HfApi()
    if not api.repo_exists(HF_REPO_ID, repo_type="dataset"):
        api.create_repo(HF_REPO_ID,
                        repo_type="dataset",
                        private=PRIVATE_REPO,
                        exist_ok=True)

    ds.push_to_hub(HF_REPO_ID, private=PRIVATE_REPO)
    ds_val.push_to_hub(HF_REPO_ID, private=PRIVATE_REPO)
    print(f"🚀  uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
    
if __name__ == "__main__":
    meta_upload_hf()