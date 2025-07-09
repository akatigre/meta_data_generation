import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ckpt_name = "Qwen/Qwen2.5-Math-1.5B" # Base
# ckpt_name = "Qwen/Qwen2.5-Math-1.5B-Instruct" # Instruction Following
version_name, step = "meta_sft_mix_reason_v4", 1500
ckpt_name = f"/data/yoonjeon_workspace/{version_name}/checkpoint-{step}"

tokenizer = AutoTokenizer.from_pretrained(ckpt_name)
model = AutoModelForCausalLM.from_pretrained(
    ckpt_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer.padding_side = "left"

SYSTEM_MSG_META = (
    "You are a math metadata assistant."
    f"Given a math problem, respond with a JSON object between <meta> and </meta>."
    "The JSON object should contain three keys: math_notion (array of strings), problem_difficulty (string), and solution_length (integer)."
    "Do **not** add anything outside the JSON."
)
SYSTEM_MSG_NO_META = (
    "You are a math metadata assistant."
    "Do not provide meta information and directly think step by step to answer the problem."
)

data = load_dataset("yjyjyj98/openmathreasoning-meta", split="test")
log_response = {}
for idx, item in tqdm(enumerate(data)):
    problem, gt_answer, gt_length, gt_difficulty, gt_math_notion = item["problem"], item["expected_answer"], item["length"], item["difficulty"], item["math_notion"]
    with torch.no_grad():
        prompts = [
            [
                {
                    "role": "system",
                    "content": SYSTEM_MSG_META
                },
                {
                    "role": "user",
                    "content": "Given a math problem, provide the meta information between <meta> and </meta>. " + problem + " /meta",
                },
            ],
            [
                {
                    "role": "system", 
                    "content": SYSTEM_MSG_NO_META
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
        
        log_response[idx] = {
            "problem": problem,
            "meta_response": meta_response,
            "answer_response": answer_response
        }
        if idx == 10:
            break
with open(f"{version_name}_{step}.json", "w") as f:
    json.dump(log_response, f)