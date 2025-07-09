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
    
    ds_train = load_dataset("yjyjyj98/omr-meta-cot", split = "train")
    # ds_test = load_dataset("yjyjyj98/openmathreasoning-meta", split = "test")
    
    cols_to_keep = ["problem", "solution", "reasoning", "math_notion", "difficulty", "length", "expected_answer"]
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in cols_to_keep])
    # ds_test = ds_test.remove_columns([c for c in ds_test.column_names if c not in cols_to_keep])
    
    def to_conversations(examples: dict[list]):
        """
        \n<|im_start|>assistant\n<think>\n\n</think>\n\n
        """
        BASE_SYSTEM = textwrap.dedent("""\
            You are a helpful math assistant.
            Given a math problem you must:

            - If the user requests "/meta": think step-by-step and return a JSON object between
                <meta> and </meta> with three keys:
                math_notion (list[str]),
                problem_difficulty (str),
                solution_length (int).

            - If the user requests "/no_meta": skip the JSON and leave an empty
                <meta></meta> block.
            """.strip())
                
        meta_conversations = []
        reasoning_conversations = []
        for problem, notion, diff, length, solution, meta_cot, answer in zip(examples["problem"], examples["math_notion"], examples["difficulty"], examples["length"], examples["solution"], examples["reasoning"], examples["expected_answer"]):
            meta_dict = json.dumps(
                {
                "math_notion":        notion,
                "problem_difficulty": diff,
                "solution_length":    length,
            }, ensure_ascii=False)
            
            meta_conversations.append(
                [
                    {
                        "role": "system",
                        "content": BASE_SYSTEM
                    },
                    {
                        "role": "user",
                        "content": "Please reason step by step and return meta information.\nProblem: " + problem + " /meta",
                    },
                    {
                        "role": "assistant",
                        "content": f"{META_START}\n{meta_cot}\n{meta_dict}\n{META_END}\n The meta information is {meta_dict}.",
                    },
                ]
            )
            reasoning_conversations.append([
                {
                    "role": "system",
                    "content": BASE_SYSTEM
                },
                {
                    "role": "user",
                    "content": "Please reason step by step and output the final answer in \\boxed{}. " + problem + "/no_meta",
                },
                {
                    "role": "assistant",
                    "content": f"{META_START}{META_END}\n {solution}\nAnswer: \\boxed{{{answer}}}", # change thinking_content into summary (shorter)
                },
            ]
            )
        return {"meta_conversations": meta_conversations, "reasoning_conversations": reasoning_conversations}

    ds_train = ds_train.map(to_conversations, batched=True)
    all_conv = pd.concat(
        [pd.Series(ds_train["meta_conversations"]),
        pd.Series(ds_train["reasoning_conversations"])],
        ignore_index=True,
    ).to_frame(name="conversation")
    
    all_conv["text"] = all_conv["conversation"].apply(
        lambda conv: tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False)
    )
    print("Coverted to text")

    tokenised = tokenizer(
        all_conv["text"].tolist(),
        add_special_tokens=False,     # template already added them
        padding=False,                # no pad → true lengths
        return_attention_mask=False,
    )

    all_conv["N"] = [len(ids) for ids in tokenised["input_ids"]]
    print("Calculated token count")
    
    ds_train = all_conv.drop(columns="conversation")
    ds_train = ds_train.loc[ds_train["N"] <= MAX_SEQ_LENGTH * 0.8].copy()
    
    ds_train = Dataset.from_pandas(ds_train.reset_index(drop=True))
    ds_train = ds_train.shuffle(seed=42)
    trainer = SFTTrainer(
        model = model,
        train_dataset = ds_train,
        # eval_dataset = ds_test,
        args = SFTConfig(
            # assistant_only_loss=True, # Only train on the assistant messages
            max_length = MAX_SEQ_LENGTH,
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            dataset_text_field = "text",
            packing = False,
            warmup_steps = 5,
            num_train_epochs = 5,
            output_dir = "/data/yoonjeon_workspace/meta_sft_mix_reason_v6",
            learning_rate = 3e-5,
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
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