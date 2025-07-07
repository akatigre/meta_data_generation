import re
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
 
    # --------------- 4. Load Dataset and process it ---------------
    
    ds_train = load_dataset("yjyjyj98/openmathreasoning-meta", split = "train")
    # ds_test = load_dataset("yjyjyj98/openmathreasoning-meta", split = "test")
    
    cols_to_keep = ["problem", "generated_solution", "math_notion", "difficulty", "length", "steps", "expected_answer"]
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in cols_to_keep])
    # ds_test = ds_test.remove_columns([c for c in ds_test.column_names if c not in cols_to_keep])
    
    def to_conversations(examples: dict[list]):
        """
        \n<|im_start|>assistant\n<think>\n\n</think>\n\n
        """
        SYSTEM_PROMPT = """
        You are given a problem.
        When required, provide the meta information between <meta> and </meta>, else <meta>\n\n</meta>.
        Then, think step by step and provide your solution inside \\boxed{}.
        """
        
        meta_conversations = []
        reasoning_conversations = []
        for problem, solution, notion, diff, length, steps, answer in zip(examples["problem"], examples["generated_solution"], examples["math_notion"], examples["difficulty"], examples["length"], examples["steps"], examples["expected_answer"]):
            meta_dict = json.dumps(
                {
                "math_notion":       notion,
                "problem_difficulty": diff,
                "solution_length":    length,
            }, ensure_ascii=False)
            # Extract the thinking part from the solution
            
            think_match = re.search(r'<think>(.*?)</think>', solution, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
            else:
                thinking_content = ""
                break
            meta_conversations.append(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": "Given a math problem, respond with a JSON object between <meta> and </meta>. The JSON object should contain three keys: math_notion (array of strings), problem_difficulty (string), and solution_length (integer). Think step by step and output the final answer in \\boxed{}. " + problem + " /meta",
                    },
                    {
                        "role": "assistant",
                        "content": f"{META_START}\n{meta_dict}\n{META_END} \n\n {" ".join(steps)}. Answer: \\boxed{answer}",
                    },
                ]
            )
            reasoning_conversations.append([
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": "Please reason step by step and output the final answer in \\boxed{}. " + problem + "/no_meta",
                },
                {
                    "role": "assistant",
                    "content": f"{META_START}\n\n{META_END}\n {thinking_content}. Answer: \\boxed{answer}",
                },
            ]
            )
        return {"meta_conversations": meta_conversations, "reasoning_conversations": reasoning_conversations}

    ds_train = ds_train.map(to_conversations, batched=True)
    meta_conversations = tokenizer.apply_chat_template(
            ds_train['meta_conversations'],
            tokenize=False,             # we want the raw text
            add_generation_prompt=False # assistant turn already present
        )
    reasoning_conversations = tokenizer.apply_chat_template(
            ds_train['reasoning_conversations'],
            tokenize=False,             # we want the raw text
            add_generation_prompt=False # assistant turn already present
        )
    
    ds_train = pd.concat([pd.Series(meta_conversations), pd.Series(reasoning_conversations)])
    ds_train.name = "text" # if trainig in chat mode
    ds_train = Dataset.from_pandas(pd.DataFrame(ds_train))
    ds_train = ds_train.shuffle(seed=42)
    trainer = SFTTrainer(
        model = model,
        train_dataset = ds_train,
        # eval_dataset = ds_test,
        args = SFTConfig(
            # assistant_only_loss=True, # Only train on the assistant messages
            per_device_train_batch_size = 32,
            gradient_accumulation_steps = 1,
            dataset_text_field = "text",
            packing = True,
            warmup_steps = 5,
            num_train_epochs = 5,
            output_dir = "/data/yoonjeon_workspace/meta_sft_mix_reason_v2",
            learning_rate = 5e-5,
            logging_steps = 5,
            optim = "adamw_torch_fused",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            report_to = "wandb",
        ),
    )
    trainer.train()

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
    MAX_SEQ_LENGTH = 3072 # Can increase for longer reasoning traces
    main(MODEL_NAME, MAX_SEQ_LENGTH)
    # test("meta_sft/checkpoint-500")