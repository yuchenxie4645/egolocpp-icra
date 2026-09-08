#!/usr/bin/env python3
"""Train Qwen3.5-9B grid adapters with multimodal GRPO.

The contact and separation adapters are trained independently. A run can start
either from its existing SFT LoRA or from a fresh LoRA on the vanilla base for
the no-SFT + GRPO ablation. Existing SFT checkpoints are never modified.

Examples:
    # One-step contact smoke test
    CUDA_VISIBLE_DEVICES=1 python train_grpo.py \
        --tasks contact --init_from sft --max_steps 1 --limit 8

    # Full SFT + GRPO and no-SFT + GRPO ablations
    CUDA_VISIBLE_DEVICES=1 python train_grpo.py --init_from sft
    CUDA_VISIBLE_DEVICES=1 python train_grpo.py --init_from base
"""

import argparse
import gc
import os
import shutil
from pathlib import Path

# Keep the same hardware constraint as train_qlora.py.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)
from trl import GRPOConfig, GRPOTrainer

from grid_reward import (
    exact_cell_metric,
    gaussian_cell_reward,
    parse_success_metric,
)


TASKS = ("contact", "separation")
MODEL_ID = "Qwen/Qwen3.5-9B"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_ROOTS = {
    "contact": Path("/home/data_labeling/data/v3-contact_grpo_dataset"),
    "separation": Path("/home/data_labeling/data/v3-separation_grpo_dataset"),
}
DEFAULT_SFT_ADAPTER_ROOT = SCRIPT_DIR / "v3-sft"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "v3-grpo"

# Conservative single-RTX-5090 starting point.
EPOCHS = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 32
TEMPERATURE = 1.2
TOP_P = 0.95
SEED = 42
NUM_WORKERS = 0
ATTN_IMPL = "sdpa"
VISION_ATTN_IMPL = "sdpa"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"


def validate_dataset(dataset, task):
    required = {
        "image",
        "prompt",
        "task",
        "gt_cell",
        "cell_rewards",
        "invalid_reward",
    }
    missing = required - set(dataset.column_names)
    if missing:
        raise RuntimeError(
            f"{task} dataset is missing columns: {sorted(missing)}"
        )
    if len(dataset) == 0:
        raise RuntimeError(f"{task} training dataset is empty")

    # Validate all cheap scalar/reward invariants without decoding every image.
    for index, row in enumerate(dataset):
        if row["task"] != task:
            raise RuntimeError(
                f"{task} row {index} declares task={row['task']!r}"
            )
        gt_cell = int(row["gt_cell"])
        rewards = row["cell_rewards"]
        if not 1 <= gt_cell <= 9:
            raise RuntimeError(
                f"{task} row {index} has invalid gt_cell={gt_cell}"
            )
        if len(rewards) != 9:
            raise RuntimeError(
                f"{task} row {index} has {len(rewards)} rewards"
            )
        if float(rewards[gt_cell - 1]) != 1.0:
            raise RuntimeError(
                f"{task} row {index} does not give GT reward 1"
            )


def make_conversational_prompts(dataset):
    """Wrap raw prompt text in the conversational VLM structure TRL expects."""

    def format_row(row):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["prompt"]},
                    ],
                }
            ]
        }

    return dataset.map(
        format_row, desc="Building conversational multimodal prompts"
    )


def load_task_dataset(root, task, limit=0):
    if not root.is_dir():
        raise FileNotFoundError(f"{task} dataset not found: {root}")
    dataset_dict = load_dataset("imagefolder", data_dir=str(root))
    if "train" not in dataset_dict or "validation" not in dataset_dict:
        raise RuntimeError(
            f"{root} must expose train and validation ImageFolder splits"
        )
    train_dataset = dataset_dict["train"]
    validation_dataset = dataset_dict["validation"]
    validate_dataset(train_dataset, task)
    validate_dataset(validation_dataset, task)

    if limit:
        train_dataset = train_dataset.select(
            range(min(limit, len(train_dataset)))
        )
    train_dataset = make_conversational_prompts(train_dataset)
    validation_dataset = make_conversational_prompts(validation_dataset)
    print(
        f"[data] {task}: {len(train_dataset)} train / "
        f"{len(validation_dataset)} validation rows",
        flush=True,
    )
    return train_dataset, validation_dataset


def build_model(model_id, adapter_dir, args):
    if args.init_from == "sft" and not adapter_dir.is_dir():
        raise FileNotFoundError(f"SFT adapter not found: {adapter_dir}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    attention = {
        "text_config": args.attn_impl,
        "vision_config": args.vision_attn_impl,
    }
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation=attention,
        device_map={"": 0},
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    base.enable_input_require_grads()

    if args.init_from == "sft":
        # Continue the existing SFT LoRA directly; do not stack an adapter.
        model = PeftModel.from_pretrained(
            base,
            str(adapter_dir),
            is_trainable=True,
        )
    else:
        # Fresh LoRA begins behaviorally identical to the vanilla base because
        # PEFT initializes the LoRA output projection to zero.
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def train_task(task, data_root, args):
    sft_adapter_dir = args.sft_adapter_root / f"{task}_adapter"
    run_tag = f"{args.init_from}_grpo"
    final_adapter_dir = args.output_root / f"{task}_{run_tag}_adapter"
    run_dir = args.output_root / f"{task}_{run_tag}_run"

    if final_adapter_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output adapter already exists: {final_adapter_dir}. "
            "Pass --overwrite to replace it."
        )
    if args.overwrite:
        for path in (final_adapter_dir, run_dir):
            if path.exists():
                shutil.rmtree(path)

    start_description = (
        f"[grpo] SFT start={sft_adapter_dir}\n"
        if args.init_from == "sft"
        else "[grpo] SFT start=none (fresh LoRA on vanilla base)\n"
    )
    print(
        f"\n{'=' * 72}\n"
        f"[grpo] task={task}\n"
        f"[grpo] dataset={data_root}\n"
        f"[grpo] initialization={args.init_from}\n"
        f"{start_description}"
        f"{'=' * 72}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    train_dataset, _ = load_task_dataset(data_root, task, args.limit)
    model = build_model(args.model_id, sft_adapter_dir, args)

    max_steps = args.max_steps if args.max_steps > 0 else -1
    config = GRPOConfig(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        chat_template_kwargs={"enable_thinking": False},
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        reward_weights=[1.0, 0.0, 0.0],
        remove_unused_columns=False,
        logging_steps=1 if max_steps > 0 else 10,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=args.num_generations,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            gaussian_cell_reward,
            parse_success_metric,
            exact_cell_metric,
        ],
        args=config,
        train_dataset=train_dataset,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))
    print(f"[grpo] saved {task} adapter -> {final_adapter_dir}", flush=True)

    del trainer, model, processor, train_dataset
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--init_from",
        choices=("sft", "base"),
        default="sft",
        help=(
            "sft continues the existing adapter; base creates a fresh LoRA "
            "for the no-SFT + GRPO ablation."
        ),
    )
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument(
        "--contact_root",
        type=Path,
        default=DEFAULT_DATA_ROOTS["contact"],
    )
    parser.add_argument(
        "--separation_root",
        type=Path,
        default=DEFAULT_DATA_ROOTS["separation"],
    )
    parser.add_argument(
        "--sft_adapter_root",
        type=Path,
        default=DEFAULT_SFT_ADAPTER_ROOT,
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument(
        "--num_generations", type=int, default=NUM_GENERATIONS
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=MAX_COMPLETION_LENGTH,
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--attn_impl", default=ATTN_IMPL)
    parser.add_argument(
        "--vision_attn_impl", default=VISION_ATTN_IMPL
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If positive, override epochs (use 1 for a smoke test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If positive, use only the first N training rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing GRPO run/output directories.",
    )
    args = parser.parse_args()

    effective_batch = args.batch_size * args.grad_accum
    if effective_batch % args.num_generations != 0:
        parser.error(
            "batch_size * grad_accum must be divisible by num_generations "
            f"({effective_batch} is not divisible by {args.num_generations})"
        )
    if args.limit and args.limit < effective_batch:
        parser.error(
            f"--limit must be at least the effective batch ({effective_batch})"
        )
    return args


def main():
    args = parse_args()
    args.sft_adapter_root = args.sft_adapter_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    roots = {
        "contact": args.contact_root.resolve(),
        "separation": args.separation_root.resolve(),
    }
    for task in args.tasks:
        train_task(task, roots[task], args)

    print("\n[grpo] all requested tasks complete.", flush=True)


if __name__ == "__main__":
    main()
