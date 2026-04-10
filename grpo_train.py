import logging
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("WANDB_DISABLED", "true")

try:
    from math_answer_extractor import (
        answer_verifier,
        extract_final_answer,
        normalize_text,
        split_into_parts,
    )
except ModuleNotFoundError:
    from time_scaling_techniques.math_answer_extractor import (
        answer_verifier,
        extract_final_answer,
        normalize_text,
        split_into_parts,
    )
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from trl import GRPOConfig, GRPOTrainer


logger = logging.getLogger(__name__)


def grad_function(prediction, gtruth):
    results = [False]
    if prediction and gtruth:
        prediction = normalize_text(prediction)
        gtruth = normalize_text(gtruth)
        prediction = split_into_parts(prediction)
        gtruth = split_into_parts(gtruth)
        if prediction and gtruth and (len(gtruth) == len(prediction)):
            results = [answer_verifier(x, y) for x, y in zip(prediction, gtruth)]
    if all(results):
        return True
    return False


def correctness_reward(
    completions: list[str],
    answer: list[str] | None = None,
    ground_truth: list[str] | None = None,
    **kwargs,
) -> list[float]:
    truths = answer if answer is not None else ground_truth
    if truths is None:
        raise ValueError(
            "correctness_reward expected an `answer` or `ground_truth` field in the batch."
        )

    rewards = []
    for completion, truth in zip(completions, truths):
        completion = extract_final_answer(completion)
        grad = grad_function(completion, truth)
        rewards.append(float(grad))
    return rewards


def format_reward(completions, **kargs):
    import re

    pattern = r"<think>.+?</think>\s*"  # <answer>.+?</answer>"
    return [0.5 if re.search(pattern, c, re.DOTALL) else 0.0 for c in completions]


def format_prompt(prompt, tokenizer):
    system_prompt = (
        "You are a helpful math assistant.\n"  # personality
        "When solving the problem, first write your reasoning inside <think> and </think> tags.\n"  # format
        "Then write the final result on a newline as:\n"  # rules ...
        "\\boxed{ANSWER}\n\n"
    )
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )


def calculate_eval_acc(model, tokenizer, eval_dataset, max_new_tokens=2048):
    model.eval()
    correct = 0
    total = 0
    for ed in eval_dataset:
        right_answer = ed["answer"]
        problem = ed["formated_prompt"]
        tokens = tokenizer.encode(problem, return_tensors="pt").to(model.device)
        with torch.no_grad():
            response = model.generate(tokens, max_new_tokens=max_new_tokens)
        prediction = tokenizer.decode(response.tolist())
        correct += correctness_reward(prediction, [right_answer])[0]
        total += 1
    return correct / total


# integrate it with trl as callback
class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % args.eval_steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            acc = calculate_eval_acc(
                model, self.tokenizer, self.eval_dataset, args.max_completion_length
            )
            kwargs["trainer"].log({"eval_acc": acc, "step": state.global_step})


if __name__ == "__main__":
    logger.info("Load Model/Tokenizer...")
    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Load Dataset...")
    dataset_id = "HuggingFaceH4/MATH-500"
    dataset = load_dataset(dataset_id, split="test[:100]")
    # format dataset
    logger.info("Format Dataset...")
    dataset = dataset.map(
        lambda x: {"formated_prompt": format_prompt(x, tokenizer)},
        input_columns=["problem"],
    )
    logger.info("Split Dataset...")
    train_test_dataset = dataset.train_test_split(0.1)
    logger.info("Modify/Rename Dataset Columns...")
    train_test_dataset = train_test_dataset.select_columns(
        [
            "formated_prompt",
            "answer",
        ]
    ).rename_columns({"formated_prompt": "prompt"})
    logger.info("Prepare Trainer Configs...")
    grpo_config = GRPOConfig(
        output_dir="qwen3_grpo_0.6b",
        num_generations=4,
        max_completion_length=2048,
        temperature=0.8,
        beta=0,  # no KL
        epsilon=10,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        steps_per_generation=2,
        use_vllm=True,  # use vllm as engine for rollout generation
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        vllm_importance_sampling_correction=True,
        bf16=True,
        eval_steps=25,
        logging_steps=10,
        report_to="trackio",
        run_name="grpo_qwen3-0.6b_lr1e-6_beta0",
    )
    logger.info("Prepare Trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=grpo_config,
        callbacks=[EvalCallback(tokenizer, train_test_dataset["test"])],
        train_dataset=train_test_dataset["train"],
    )
    logger.info("Train...")
    trainer.train()
