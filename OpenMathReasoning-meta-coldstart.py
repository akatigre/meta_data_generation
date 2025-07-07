from __future__ import annotations
import asyncio, json, os, re, enum
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
            json.dump(meta_data, fp, indent=4)

# ───────────────────────────────── Entrypoint ──────────────────────────────
if __name__ == "__main__":
    MODEL = "o4-mini"          # snapshot that supports structured output
    MAX_CONCURRENCY = 10                  # tweak for your own rate-limit comfort
    asyncio.run(async_main_meta_generation())