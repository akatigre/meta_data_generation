from __future__ import annotations
import asyncio, json, os, re, enum
from typing import Any
import dotenv, pydantic, openai
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio  # async-aware progress bar

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ──────────────────────────── Pydantic Schemas ─────────────────────────────
class MathReasoning(pydantic.BaseModel):
    steps: list[str]
    final_answer: str

class MathType(str, enum.Enum):
    Algebra       = "Algebra"
    Geometry      = "Geometry"
    Calculus      = "Calculus"
    Statistics    = "Statistics"
    NumberTheory  = "Number Theory"
    Others        = "Others"

class DapoClassification(pydantic.BaseModel):
    category : MathType
    glossary : list[str]
    solvable : bool

class DapoSubquestion(pydantic.BaseModel):
    questions : list[str]
    answers: list[int]

# ─────────────────────────── Prompt builders ───────────────────────────────
def classify_messages(question: str):
    return [
        {
            "role": "user",
            "content": (
                "Generate a JSON that classifies the given question into the mathematical "
                "field among algebra, geometry, calculus, statistics, number theory, and others. "
                "Extract the important key mathematical terms that appear in the question. "
                "Determine if you have sufficient knowledge to solve this question. "
                "Output True if you can solve it, False otherwise.\n\n"
                f"Question: {question}"
            ),
        }
    ]

def reasoning_messages(question: str):
    return [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": question},
    ]

def subquestion_messages(solution: str):
    return [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Design subquestions that yields integer valued output.",
        },
        {
            "role": "user",
            "content": f"""
            From the step-by-step solution, I want to extract subquestions that yields integer valued output.
            The subquestions are in sequential order to guide the solution trajectory to the final output.
            For example, if the solution trajectory consists of Q1, Q2, Q3, Q4 (final output),
            Then the answer to Q1 directly leads to Q2, and Q2 to Q3, and Q3 to final output.
            Design each sub-question to integer valued and all premises are given from the reasoning trajectory upto the question point.

            Solution: {solution}
            """,
        }
    ]

# ──────────────────────────── Core async helpers ───────────────────────────
async def classify_problem(client: openai.AsyncOpenAI, question: str) -> DapoSubquestion:
    """Return a parsed DapoSubquestion object."""
    rsp = await client.responses.parse(
        model       = MODEL,
        input       = classify_messages(question),
        text_format = DapoClassification,
    )
    return rsp.output_parsed  # already a pydantic object

async def reason_problem(client: openai.AsyncOpenAI, question: str) -> MathReasoning:
    """Return a parsed MathReasoning object."""
    try:
        rsp = await client.responses.parse(
            model       = MODEL,
            input       = reasoning_messages(question),
            text_format = MathReasoning,
        )
        return rsp.output_parsed
    except Exception as e:
        print(f"Error: {e}")
        return None

async def subquestion_problem(client: openai.AsyncOpenAI, solution: str) -> DapoSubquestion:
    rsp = await client.responses.parse(
        model       = MODEL,
        input       = subquestion_messages(solution),
        text_format = DapoSubquestion,
    )
    return rsp.output_parsed


def strip_final_answer(raw: str) -> str:
    """Remove \\boxed{…} and inline LaTeX from the answer string."""
    ans = re.sub(r'\\boxed\{([^}]+)\}', r'\1', raw)
    ans = re.sub(r'\$([^$]+)\$', r'\1', ans)
    return ans.strip()

# ───────────────────────────── Row-level task ──────────────────────────────
async def process_row(
    client: openai.AsyncOpenAI,
    sem:    asyncio.Semaphore,
    data_row: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Returns (pid, result-dict).  Concurrency is gated by the semaphore."""
    async with sem:
        pid       = data_row["extra_info"]["index"]
        question  = data_row["prompt"]
        gt_answer = data_row["solution"]
        
        # run the two structured requests concurrently
        cls_task   = asyncio.create_task(classify_problem(client, question))
        math_task  = asyncio.create_task(reason_problem(client, question))
        subquestion_task = asyncio.create_task(subquestion_problem(client, question))
        cls_obj, reasoning = await asyncio.gather(cls_task, math_task)

        cleaned_ans = strip_final_answer(reasoning.final_answer)

        result = {
            "classification": cls_obj.model_dump(),
            "reasoning":      reasoning.model_dump(),
            "gt_answer":      gt_answer,
            "correct":        cleaned_ans == gt_answer,
        }
        return pid, result

async def process_row_subquestion(
    client: openai.AsyncOpenAI,
    sem:    asyncio.Semaphore,
    pid,
    trajectory_row: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Returns (pid, result-dict).  Concurrency is gated by the semaphore."""
    async with sem:
        solution_trajectory = "\n".join(trajectory_row["reasoning"]["steps"])
        subquestion_task = asyncio.create_task(subquestion_problem(client, solution_trajectory))
        subquestion = await subquestion_task
        result = {
            "subquestions": subquestion.model_dump(),
        }
        return pid, result
    
# ──────────────────────────────── main async ───────────────────────────────
async def async_main_answer_trajectory() -> None:
    client   = openai.AsyncOpenAI(api_key=api_key)
    dataset  = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")["train"]
    sem      = asyncio.Semaphore(MAX_CONCURRENCY)
    out: dict[int, Any] = {}
    with open(SAVE_PATH, "r") as fp:
        out = json.load(fp)
        
    tasks = [process_row(client, sem, row) for row in dataset if row["extra_info"]["index"] not in out]

    # NOTE: ordinary `for`, not `async for`
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        pid, res = await fut              # <- await each finished coroutine
        out[pid] = res

        with open(SAVE_PATH, "w") as fp:  # incremental save
            json.dump(out, fp, indent=2)

    print(f"Saved {len(out)} results to {SAVE_PATH}")


async def async_main_subquestion_generation() -> None:
    client   = openai.AsyncOpenAI(api_key=api_key)
    sem      = asyncio.Semaphore(MAX_CONCURRENCY)
    out: dict[int, Any] = {}
    
    trajectory_path = f"dapo_subquestion_structured_{MODEL}.json"

    with open(trajectory_path, "r") as fp:
        traj_data = json.load(fp)
    
    # keep only correct trajectories; convert keys to int
    valid_pids = [pid for pid, row in traj_data.items() if row.get("correct")]

    # DEBUG LIMIT ─ remove when ready
    # DEBUG_LIMIT = 10
    # valid_pids = valid_pids[:DEBUG_LIMIT]

    # ---------- task creation ----------
    tasks = [
        asyncio.create_task(
            process_row_subquestion(client, sem, pid, traj_data[str(pid)])
        )
        for pid in valid_pids          # however you selected these pids
    ]

    # synchronous iteration, await inside
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        pid, res = await fut
        traj_data[pid]["subquestions"] = res
        with open(trajectory_path, "w") as fp:
            json.dump(traj_data, fp, indent=2)
        
    print(f"Saved {len(out)} results to {trajectory_path}")

# ───────────────────────────────── Entrypoint ──────────────────────────────
if __name__ == "__main__":

    MODEL = "o4-mini"          # snapshot that supports structured output
    SAVE_PATH = f"dapo_subquestion_structured_{MODEL}.json"
    MAX_CONCURRENCY = 10                  # tweak for your own rate-limit comfort
    # asyncio.run(async_main_answer_trajectory())
    asyncio.run(async_main_subquestion_generation())
