
# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
import re
import asyncio
import enum
import json
import os
from ast import literal_eval
from typing import Any
from datasets import load_dataset
import openai
import pydantic
from collections import defaultdict
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class StepByStepAnswer(pydantic.BaseModel):
    step_by_step: list[str]
    answer: str

class MathType(str, enum.Enum):
    Algebra = "Algebra"
    Geometry = "Geometry"
    Calculus = "Calculus"
    Statistics = "Statistics"
    NumberTheory = "Number Theory"
    Others = "Others"


class DapoSubquestion(pydantic.BaseModel):
    category: MathType
    glossary: list[str]
    solvable: bool


def build_prompts(question: str):
    PARAMS: dict[dict[str, Any]] = {
        # "dapo_classify": {
        #         "messages": [
        #             {
        #                 "role": "user",
        #                 "content": """
        #                 Generate a JSON that classifies the given question into mathematical field among algebra, geometry, calculus, statistics, number theory, and others. 
        #                 Find two key mathematical glossary (terminology or theorem) that appear in the question or must be used to solve the problem.
        #                 Determine if you have sufficient knowledge to solve this question. Output True if you can solve it, False otherwise.
        #                 Question: {question}
        #                 """.format(question=question),
        #             }
        #         ],
        #         "response_format": {
        #             "type": "json_schema",
        #             "json_schema": {
        #                 "name": "dapo-subquestion",
        #                 "schema": DapoSubquestion.model_json_schema(),
        #             },
        #         },
        #     },
        
        # "choice": {
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": """
        #             Determine if you have sufficient knowledge to solve this question. First think what knowledge is required to solve this question and then determine if you can solve this question correctly.
        #             Question: {question}
        #             """.format(question=question),
        #         }
        #     ],
        #     "extra_body": {
        #         "guided_choice": ["yes", "no", "unanswerable"]
        #     }
        # },
        "reasoning_answer": {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Provide the concise overview of the solution to the following math problem. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. \n\n {question}"  
                    }
                ],
            }
    }

    return PARAMS


async def cli():
    save_path = "dapo_subquestion_thinking.json" if ENABLE_THINKING else "dapo_subquestion_no_thinking.json"
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    model = (await client.models.list()).data[0].id
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")['train'] 
    total_result = {}
    for data_idx, data in tqdm(enumerate(dataset)):
        question = data['prompt']
        gt_answer = data['solution']
        pid = data['extra_info']['index']
        PARAMS = build_prompts(question)
        
        results = await asyncio.gather(
            *[
                client.chat.completions.create(
                    model=model,
                    stream=False,
                    # max_tokens=4096,
                    n=4,
                    temperature=0.6 if ENABLE_THINKING else 0.7,
                    top_p=0.95 if ENABLE_THINKING else 0.8,
                    extra_body={
                        "top_k": 20, 
                        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
                    },
                    **PARAMS[name],
                )
                for name in PARAMS
            ]
        )
        result = defaultdict(list)
        for name, response in zip(PARAMS, results):
            for choice in response.choices:
                message = choice.message
                if message.content is None:
                    continue
                # Extract answer using regex
                if name == "reasoning_answer":
                    answer_match = re.search(r'Answer:\s*([^\n]+)', message.content)
                    # Remove LaTeX formatting from the answer
                    
                    if answer_match:
                        extracted_answer = answer_match.group(1).strip()
                        extracted_answer = extracted_answer.strip('"\'')
                        extracted_answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', extracted_answer)
                        extracted_answer = re.sub(r'\$([^$]+)\$', r'\1', extracted_answer)
                    else:
                        extracted_answer = ""
                    result[name].append({
                        "response": message.content,
                        "answer": extracted_answer,
                        "gt_answer": gt_answer,
                        "correct": extracted_answer == gt_answer
                    })
                
                elif name == "dapo_classify":
                    content = message.content.replace("true", "True").replace("false", "False")
                    
                    try:
                        content = literal_eval(content) 
                    except:
                        content = content
                    result[name].append(content)
                
        total_result[pid] = result

        with open(save_path, "w") as f:
            json.dump(total_result, f, indent=4)
        

def main():
    asyncio.run(cli())

if __name__ == "__main__":
    ENABLE_THINKING = True
    main()
    
    unsloth/OpenMathReasoning-mini