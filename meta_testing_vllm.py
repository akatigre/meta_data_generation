"""
python run_offline_vllm.py
"""
import json, textwrap
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams      # pip install vllm>=0.4.0

# ───────────────────────────── 1 · model paths ──────────────────────────────
version_name, step = "meta_sft_mix_reason_v4", 1990
ckpt_path = f"/data/yoonjeon_workspace/{version_name}/checkpoint-{step}"

# ───────────────────────────── 2 · tokenizer (for template) ──────────────────
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

# ───────────────────────────── 3 · vLLM engine  ─────────────────────────────
llm = LLM(
    model               = ckpt_path,
    dtype               = "float16",
    trust_remote_code   = True,
    tensor_parallel_size= 2,           # >1 if you want multi-GPU tensor-parallel
)

# ───────────────────────────── 4 · static prompt pieces ─────────────────────
SYSTEM_MSG_META = textwrap.dedent("""
    You are a math metadata assistant.
    Given a math problem, respond with a JSON object between <meta> and </meta>.
    The JSON object should contain three keys: math_notion (array of strings),
    problem_difficulty (string), and solution_length (integer).
    Do **not** add anything outside the JSON.
""").strip()

SYSTEM_MSG_NO_META = textwrap.dedent("""
    You are a math metadata assistant.
    Do not provide meta information and directly think step by step to answer the problem.
""").strip()

sampler = SamplingParams(
    max_tokens            = 3072,
    temperature           = 0.0,
    top_p                 = 1.0,
    stop                  = None,      # rely on </meta> / <|im_end|>
)

# ───────────────────────────── 5 · load dataset ─────────────────────────────
data = load_dataset("yjyjyj98/openmathreasoning-meta", split="test")

log_response = {}
for idx, item in tqdm(enumerate(data), total=len(data)):
    problem = item["problem"]

    # build two chat-style messages
    prompts = [
        [
            {"role": "system", "content": SYSTEM_MSG_META},
            {"role": "user",
             "content": (
                 "Given a math problem, provide the meta information between "
                 "<meta> and </meta>. " + problem + " /meta"
             )},
        ],
        [
            {"role": "system", "content": SYSTEM_MSG_NO_META},
            {"role": "user",
             "content": (
                 "Please reason step by step and output the final answer in "
                 "\\boxed{}. Do not provide meta information. " + problem + " /no_meta"
             )},
        ],
    ]

    prompt_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    # ───────── 6 · vLLM inference (batch of two prompts) ──────────
    outputs = llm.generate(prompt_texts, sampler)

    # outputs is a list[RequestOutput]; each has .outputs[0].text
    meta_response   = outputs[0].outputs[0].text
    answer_response = outputs[1].outputs[0].text

    log_response[idx] = {
        "problem":         problem,
        "meta_response":   meta_response,
        "answer_response": answer_response,
    }

    if idx == 10:               # quick smoke-test like original script
        break

# ───────────────────────────── 7 · dump log ─────────────────────────────────
out_file = f"{version_name}_{step}_vllm.json"
with open(out_file, "w") as f:
    json.dump(log_response, f, indent=2, ensure_ascii=False)

print("saved →", out_file)
