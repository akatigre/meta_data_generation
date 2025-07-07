from __future__ import annotations
import json, os, re
from typing import Any
from collections import defaultdict
from ast import literal_eval

import dotenv, pydantic, openai
from datasets import load_dataset
from tqdm import tqdm
import enum

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


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

class DapoSubquestion(pydantic.BaseModel):
    category : MathType
    glossary : list[str]
    solvable : bool

# ──────────────────────────── Prompt helpers ────────────────────────────────
def classify_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Generate a JSON that classifies the given question into the mathematical "
                "field among algebra, geometry, calculus, statistics, number theory, and others. "
                "Extract the important key mathematical terms that appear in the question"
                "Determine if you have sufficient knowledge to solve this question. Output True if you can solve it, False otherwise.\n\n"
                f"Question: {question}"
            ),
        }
    ]

def reasoning_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful math tutor.  Guide the user through the solution step by step.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

MODEL = "o4-mini"          # snapshot that supports structured output

# ───────────────────────────── Main loop (sync) ─────────────────────────────
def main() -> None:
    client = openai.OpenAI(api_key=api_key)
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")["train"]
    out: dict[int, Any] = {}
    save_path = "dapo_subquestion_structured.json"

    for row in tqdm(dataset, total=len(dataset)):
        pid        = row["extra_info"]["index"]
        question   = row["prompt"]
        gt_answer  = row["solution"]

        # 1. classification (structured)
        cls_resp = client.responses.parse(
            model      = MODEL,
            input      = classify_messages(question),
            text_format = DapoSubquestion,
        )
        cls_obj: DapoSubquestion = cls_resp.output_parsed

        # 2. step-by-step reasoning (structured)
        math_resp = client.responses.parse(
            model      = MODEL,
            input      = reasoning_messages(question),
            text_format= MathReasoning,
        )
        reasoning: MathReasoning = math_resp.output_parsed

        # strip any latex / \boxed{…} decorations from final_answer
        cleaned_ans = re.sub(r'\\boxed\{([^}]+)\}', r'\1', reasoning.final_answer)
        cleaned_ans = re.sub(r'\$([^$]+)\$',   r'\1', cleaned_ans).strip()

        out[pid] = {
            "classification": cls_obj.model_dump(),
            "reasoning": reasoning.model_dump(),
            "gt_answer": gt_answer,
            "correct": cleaned_ans == gt_answer,
        }

        # incremental save
        with open(save_path, "w") as fp:
            json.dump(out, fp, indent=2)

if __name__ == "__main__":
    main()
