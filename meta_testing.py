#! Ask the model to determine how long the solution should be for the given problem. Then test the length of the actual generation and correctness.
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# Load the model and tokenizer
# ckpt_name = "Qwen/Qwen2.5-Math-7B"
ckpt_name = "/data/yoonjeon_workspace/meta_sft_mix_reason_v3/checkpoint-1500"
tokenizer = AutoTokenizer.from_pretrained(ckpt_name)
model = AutoModelForCausalLM.from_pretrained(
    ckpt_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer.padding_side = "left"

SYSTEM_MSG = (
    "You are a math metadata assistant."
    f"Given a math problem, respond with a JSON object between <meta> and </meta>."
    "The JSON object should contain three keys: math_notion (array of strings), problem_difficulty (string), and solution_length (integer)."
    "Do **not** add anything outside the JSON."
)

data = load_dataset("yjyjyj98/openmathreasoning-meta", split="test")
# with open("./meta_chat_template.jinja", "r") as f:  
    # chat_template = f.read()
# tokenizer.chat_template = chat_template

for item in data:
    problem, gt_answer, gt_length, gt_difficulty, gt_math_notion = item["problem"], item["expected_answer"], item["length"], item["difficulty"], item["math_notion"]
    with torch.no_grad():
        prompts = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Given a math problem, respond with a JSON object between <meta> and </meta>. The JSON object should contain three keys: math_notion (array of strings), problem_difficulty (string), and solution_length (integer). Do **not** add anything outside the JSON. " + problem + " /meta",
                },
            ],
            [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. "
                },
                {
                    "role": "user",
                    "content": "Please reason step by step and output the final answer in \\boxed{}. Do not provide meta information. " + problem + " /no_meta",
                }
            ]
        ]
        texts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=3072)
        generated_ids = [
            output_ids for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        meta_response, answer_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        breakpoint()