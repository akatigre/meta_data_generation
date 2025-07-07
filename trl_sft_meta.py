import json
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datasets import Dataset, load_dataset
import pandas as pd
import textwrap

def main(MODEL_NAME, MAX_SEQ_LENGTH):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # attn_implementation="sdpa",
        use_cache=False,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=True
    )

    META_START = "<meta>" # Acts as <think>
    META_END   = "</meta>"   # Acts as </think>

    if META_START not in tokenizer.special_tokens_map.get("additional_special_tokens", []):
        tokenizer.add_special_tokens({"additional_special_tokens": [META_START, META_END]})
        model.resize_token_embeddings(len(tokenizer))
        
    SYSTEM_MSG = (
        "You are a math metadata assistant. "
        f"Given a math problem, respond with a JSON object between {META_START} and {META_END}."
        "The JSON object should contain three keys: math_notion (array of strings), problem_difficulty (string), and solution_length (integer)."
        "Do **not** add anything outside the JSON."
    )
    chat_template = \
            "{%- if messages[0]['role'] == 'system' %}"\
                "{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"\
            "{%- else %}"\
                "{{- '<|im_start|>system\n' + '{system_prompt}' + '<|im_end|>\n' }}"\
            "{%- endif %}"\
            "{%- for message in messages %}"\
                "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first)%}"\
                    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}"\
                "{%- elif message.role == 'assistant' %}"\
                    "{{- '<|im_start|>' + message.role }}"\
                    "{%- if message.content %}"\
                        "{{- '\n' + message.content }}"\
                    "{%- endif %}"\
                    "{{- '<|im_end|>\n' }}"\
                "{%- elif message.role == 'assistant_meta' %}"\
                    "{{- '<|im_start|>assistant\n<meta>' }}"\
                    "{%- if message.content %}"\
                        "{{- '\n' + message.content + '</meta>' + '\n' }}"\
                    "{%- endif %}"\
                    "{{- '<|im_end|>\n' }}"\
                "{%- endif %}"\
            "{%- endfor %}"\
            "{%- if add_generation_prompt %}"\
                "{{- '<|im_start|>assistant\n<meta>' }}"\
            "{%- endif %}"

    # ─────────────── 3. build chat messages for each training row ────────────────
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{SYSTEM_MSG}'")
    tokenizer.chat_template = chat_template
    
    # --------------- 4. Load Dataset and process it ---------------
    
    ds_train = load_dataset("yjyjyj98/openmathreasoning-meta", split = "train")
    ds_test = load_dataset("yjyjyj98/openmathreasoning-meta", split = "test")
    
    cols_to_keep = ["problem", "math_notion", "difficulty", "length"]
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in cols_to_keep])
    ds_test = ds_test.remove_columns([c for c in ds_test.column_names if c not in cols_to_keep])
    
    def to_messages(example):
        meta_dict = {
            "math_notion":       example["math_notion"],
            "problem_difficulty": example["difficulty"],
            "solution_length":    int(example["length"]),
        }
        example["messages"] = [
            {"role": "system",         "content": SYSTEM_MSG},
            {"role": "user",           "content": example["problem"]},
            {"role": "assistant_meta", "content": json.dumps(meta_dict,
                                                            ensure_ascii=False)},
        ]
        return example

    ds_train = ds_train.map(to_messages, num_proc=1)
    ds_test = ds_test.map(to_messages, num_proc=1)

    # 3 ────────────────────── turn messages → prompt string
    def formatting_func(example):
        prompt = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,             # we want the raw text
            add_generation_prompt=False # assistant turn already present
        )
        return {"text": prompt}

    ds_train = ds_train.map(formatting_func, num_proc=1)
    ds_test = ds_test.map(formatting_func, num_proc=1)
    
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c != "text"])
    ds_test = ds_test.remove_columns([c for c in ds_test.column_names if c != "text"])
    
    trainer = SFTTrainer(
        model = model,
        train_dataset = ds_train,
        eval_dataset = ds_test,
        args = SFTConfig(
            per_device_train_batch_size = 64,
            gradient_accumulation_steps = 1,
            dataset_text_field = "text",
            packing = False,
            warmup_steps = 5,
            num_train_epochs = 5,
            output_dir = "/data/yoonjeon_workspace/meta_sft",
            learning_rate = 5e-5,
            logging_steps = 5,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            report_to = "wandb",
        ),
    )
    trainer.train()
    
    #! Check
    _ = model.generate(
        **tokenizer(ds_test[0]["text"], return_tensors = "pt").to("cuda"),
        temperature = 0.8,
        max_new_tokens = MAX_SEQ_LENGTH,
        streamer = TextStreamer(tokenizer, skip_prompt = False),
    )
    
if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
    MAX_SEQ_LENGTH = 3072 # Can increase for longer reasoning traces
    main(MODEL_NAME, MAX_SEQ_LENGTH)
    # test("meta_sft/checkpoint-500")