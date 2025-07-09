from __future__ import annotations
import asyncio, json, os, re, enum
import itertools
from typing import Any
import dotenv, pydantic, openai
from tqdm.asyncio import tqdm_asyncio  # async-aware progress bar

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ──────────────────────────── Pydantic Schemas ─────────────────────────────
# class Solvable(str, enum.Enum):
#     Yes = "Yes"
#     No = "No"
#     Uncertain = "Uncertain"

class Difficulty(str, enum.Enum):
    MiddleSchoolEasy = "MiddleSchoolEasy"
    MiddleSchoolMedium = "MiddleSchoolMedium"
    MiddleSchoolHard = "MiddleSchoolHard"
    HighSchoolEasy = "HighSchoolEasy"
    HighSchoolMedium = "HighSchoolMedium"
    HighSchoolHard = "HighSchoolHard"
    UndergraduateEasy = "UndergraduateEasy"
    UndergraduateMedium = "UndergraduateMedium"
    UndergraduateHard = "UndergraduateHard"

class MetaCognition(pydantic.BaseModel):
    math_notion : list[str]
    difficulty : Difficulty
    length : int

class SolutionSummary(pydantic.BaseModel):
    steps: list[str]
    final_answer: str

class MetaCognitionCoT(pydantic.BaseModel):
    summary: str
    reasoning: str
    
# ─────────────────────────── Prompt builders ───────────────────────────────
def meta_message(question: str, summary: str):
    return [
        {
            "role": "system",
            "content": """
            You are a helpful expert in mathematics. You are given a problem and a solution.
            
            - math_notion: a list of strings that describe the mathematical notions used in the problem. Do not include problem specific information such as equations, variable names, entities, etc.
            - difficulty: Evaluate the problem's overall difficulty based on the math_notion. Estimate the minimum level of mathematical knowledge required to understand and solve the problem.
            - length: Estimate the minimum required length of the solution in terms of the number of logic steps.
            """
        },
        {"role": "user", "content": f"Problem: {question}\nSolution: {summary}"},
    ]

def solution_summary(solution: str):
    return [
        {
            "role": "system",
            "content": """
            You are a helpful expert in mathematics. You are given a solution to a problem.
            
            - steps: a list of strings that describe the steps in the solution
            - final_answer: a string that describes the final answer
            """
        },
        {"role": "user", "content": solution},
    ]
    
# ──────────────────────────── Core async helpers ───────────────────────────
async def meta_generation(client: openai.AsyncOpenAI, question: str, summary: str) -> MetaCognition:
    try:
        rsp = await client.responses.parse(
            model       = MODEL,
            input       = meta_message(question, summary),
            text_format = MetaCognition,
            )
        return rsp.output_parsed
    except Exception as e:
        print(f"Error in meta_generation: {e}")
        return None


async def solution_summary_generation(client: openai.AsyncOpenAI, solution: str, question: str) -> SolutionSummary:
    try:
        rsp = await client.responses.parse(
            model       = MODEL,
            input       = solution_summary(solution),
            text_format = SolutionSummary,
        )
        return rsp.output_parsed
    except Exception as e:
        print(f"Error in solution_summary_generation: {e}")
        return None

async def generate_meta_cot(client: openai.AsyncOpenAI, question: str, solution: str, terminologies: list[str], difficulty: str, length: str):
    """Convert ASCII maths to non-ASCII LaTeX."""
    try:
        rsp = await client.responses.parse(
            model    = MODEL,
            input = [
                {"role": "system",
                 "content": (
                     "You are provided with problem, solution, core mathematical terms, problem difficulty and solution length."
                     "Provide a high level problem solution by removing all detailed entity names, equations, variable names and leave the core mathematical logic in english."
                     f"Analyze the problem using all provided mathematical terms, then determine the problem difficulty and the length of the solution."
                     
                 )},
                {"role": "user", "content": f"Problem: {question} \n Solution: {solution} \n Core Mathematical Terms: {terminologies} \n Problem Difficulty: {difficulty} \n Solution Length: {length}"},
            ],
            text_format = MetaCognitionCoT,
        )
        return rsp.output_parsed
    except Exception as e:
        print("Error in generate_meta_cot:", e)
        return None

    
# ───────────────────────────── Row-level task ──────────────────────────────
async def process_row_meta(
    client: openai.AsyncOpenAI,
    sem:    asyncio.Semaphore,
    pid:    str,
    data: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Returns (pid, result-dict).  Concurrency is gated by the semaphore."""
    async with sem:
        question = data["problem"]
        solution = data["generated_solution"]
        
        summary_task = asyncio.create_task(solution_summary_generation(client, solution, question))
        summary_output = await summary_task
        if summary_output is None:
            return pid, None
        summary = " ".join(summary_output.model_dump()["steps"])
        meta_task = asyncio.create_task(meta_generation(client, question, summary))
        meta_output = await meta_task
        if meta_output is None:
            return pid, None
        data.update({"summary": summary_output.model_dump(), "meta": meta_output.model_dump()})
        return pid, data
    
# ──────────────────────────────── main async ───────────────────────────────
async def async_main_meta_generation() -> None:
    client   = openai.AsyncOpenAI(api_key=api_key)
    sem      = asyncio.Semaphore(MAX_CONCURRENCY)
    
    meta_save_path = f"metas/openmathreasoning_meta.json"
    if os.path.exists(meta_save_path):
        with open(meta_save_path, "r") as fp:
            meta_data = json.load(fp)
    else:
        meta_data = {}

    with open("metas/filtered_openmathreasoning.json", "r") as fp:
        dataset = json.load(fp)
    # ---------- task creation ----------
    valid_pids = [pid for pid in dataset.keys() if int(pid) < 50000 and pid not in meta_data.keys()]
    tasks = [
        asyncio.create_task(process_row_meta(client, sem, pid, dataset[pid]))
        for pid in valid_pids
    ]
        
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        pid, meta = await coro
        if meta is None:
            continue
        meta_data[pid] = meta
        with open(meta_save_path.replace(".json", "_test.json"), "w") as fp:
            json.dump(meta_data, fp, indent=4, ensure_ascii=False)

async def async_main_summary_nonascii():
    client = openai.AsyncOpenAI(api_key=api_key)
    sem    = asyncio.Semaphore(MAX_CONCURRENCY)

    path   = "metas/openmathreasoning_meta.json"
    with open(path, "r") as fp:
        meta_data = json.load(fp)

    async def worker(pid: str, meta_dict: dict):
        # turn dict → plain text
        summarized_solution = meta_dict["meta"]["solution"]
        question = meta_dict["problem"]
        terminologies = meta_dict["meta"]["math_notion"]
        difficulty = meta_dict["meta"]["difficulty"]
        length = meta_dict["meta"]["length"]
        if length < 5:
            length = "Short"
        elif length > 9:
            length = "Long"
        else:
            length = "Medium"
        async with sem:
            meta_cot = await generate_meta_cot(client, question, summarized_solution, terminologies, difficulty, length)
        if meta_cot:
            meta_dict["meta"]["reasoning"] = meta_cot.model_dump()["reasoning"]
            meta_dict["meta"]["summary"] = meta_cot.model_dump()["summary"]
        return pid

    # first_ten = itertools.islice(meta_data.items(), 10)

    tasks = [
        asyncio.create_task(worker(pid, meta))
        for pid, meta in meta_data.items()
    ]

    for done in asyncio.as_completed(tasks):
        pid = await done
        print("✓ processed", pid)

    out_path = path.replace(".json", "_cot.json")
    with open(out_path, "w") as fp:
        json.dump(meta_data, fp, indent=2, ensure_ascii=False)
    
    
def meta_upload_hf():
    import json, pandas as pd, pyarrow as pa
    from datasets import Dataset
    from huggingface_hub import HfApi
    from sklearn.model_selection import train_test_split
    
    JSON_PATH      = "metas/openmathreasoning_meta_cot.json"
    HF_REPO_ID     = "yjyjyj98/omr-meta-cot"   # change this!
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
    if {"meta"}.issubset(df.columns):
        # expand each nested dict into columns
        # summary_flat = df.pop("summary").apply(pd.Series)
        meta_flat    = df.pop("meta").apply(pd.Series)
        meta_flat["math_notion"] = meta_flat["math_notion"].apply(lambda x: ", ".join(x))
        # merge everything back together
        df = pd.concat([df, meta_flat], axis=1)

        # optional: put pid first for readability
        if "pid" in df.columns:
            cols = ["pid"] + [c for c in df.columns if c != "pid"]
            df = df[cols]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

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
    
# ───────────────────────────────── Entrypoint ──────────────────────────────
if __name__ == "__main__":
    MODEL = "gpt-4.1"          # snapshot that supports structured output
    MAX_CONCURRENCY = 10                  # tweak for your own rate-limit comfort
    # asyncio.run(async_main_meta_generation())
    # asyncio.run(async_main_summary_nonascii())
    meta_upload_hf()