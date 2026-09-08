#!/usr/bin/env python3
"""QLoRA fine-tune Qwen/Qwen3.5-9B into TWO independent PEFT adapters.

One adapter is trained on the *contact* grid dataset and a second, fully
independent adapter on the *separation* grid dataset. The base model is loaded
fresh (4-bit NF4) for each task so the adapters never share weights, then each
LoRA adapter is saved to ``<output_dir>/<task>_adapter``.

Design (matches the approved plan):
  * Base: ``Qwen/Qwen3.5-9B`` unified VLM (``Qwen3_5ForConditionalGeneration``
    + ``AutoProcessor`` -> ``Qwen3VLProcessor``).
  * 4-bit QLoRA (bitsandbytes NF4, double-quant, bf16 compute) + gradient
    checkpointing + ``paged_adamw_8bit``. Fits a single 32 GB card in bf16.
  * Completion-only loss: only the trailing ``{"points": [N]}`` answer tokens
    contribute to the loss; the (long, shared) prompt + image tokens are masked
    to -100. Non-thinking mode (``enable_thinking=False``).
  * If present, ``validation/metadata.jsonl`` is used as the held-out
    validation set and ``train/metadata.jsonl`` is treated as training-only. Otherwise a
    grouped train/val split is made keyed on ``(video_source, video_index)`` so
    all 9 offset variants of one video stay on the same side. A
    ``<task>_val.jsonl`` is written for ``infer_eval.py``.

Attention backend
  Qwen3.5 is a hybrid model. This 5090 (Blackwell sm_120) cannot run
  FlashAttention-2, so the text LM defaults to ``flash_attention_4`` and the
  vision tower to ``sdpa`` (the FA4 kernel does not support the vision tower's
  ragged attention). If FlashAttention-4 is not working in your environment,
  pass ``--attn_impl sdpa`` (and optionally ``--vision_attn_impl sdpa``) to run
  everything on PyTorch SDPA, which is fully supported on Blackwell.

Run on GPU 1 only (per hardware constraint); CUDA_VISIBLE_DEVICES defaults to
"1" below but respects an externally-set value.

Example:
    CUDA_VISIBLE_DEVICES=1 python train_qlora.py \
        --output_dir /home/EgoLoc/training/adapters
    # SDPA-only fallback if FA4 is unavailable:
    CUDA_VISIBLE_DEVICES=1 python train_qlora.py --attn_impl sdpa
"""

import argparse
import gc
import json
import os
import random
import shutil

# Hard constraint: only GPU 1 is available on this machine. Set before torch
# is imported so the process only ever sees that device (respects an existing
# externally-provided value, e.g. when launched by a scheduler).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

TASKS = ("contact", "separation")
END_OF_TURN = "<|im_end|>\n"

# --------------------------------------------------------------------------- #
# Hardcoded config (was CLI args; pinned to the approved baseline)
# --------------------------------------------------------------------------- #
CONTACT_ROOT = "/home/data_labeling/data/v3-contact_dataset"
SEPARATION_ROOT = "/home/data_labeling/data/v3-separation_dataset"
MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3-sft")

# Training
EPOCHS = 5.0
SEED = 42
VAL_FRAC = 0.1
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 1e-4
NUM_WORKERS = 4
GRADIENT_CHECKPOINTING = True

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = "all-linear"

# Attention backends
# NOTE: flash_attention_4 (cute DSL) fails to JIT-compile on this stack with
# Qwen3.5's GQA layout — see MLIRError in flash_attn/cute/pack_gqa.py:139
# ("unable to compute crd2idx with '!cute.layout<\"(?):(1)\">' ...").
# Fall back to PyTorch SDPA, which is fully supported on Blackwell sm_120.
ATTN_IMPL = "sdpa"
VISION_ATTN_IMPL = "sdpa"


# --------------------------------------------------------------------------- #
# Data loading + grouped split
# --------------------------------------------------------------------------- #
def load_task_rows(metadata_path, task):
    """Read a metadata JSONL into a list of rows.

    Each row keeps the absolute image path, the prompt, the JSON-only answer
    ``{"points": [cell_number]}`` and the metadata needed for the grouped split
    and for downstream evaluation.
    """
    metadata_dir = os.path.dirname(metadata_path)
    rows = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            md = r["metadata"]
            cell = int(md["cell_number"])
            rows.append({
                "image": os.path.join(metadata_dir, r["file_name"]),
                "prompt": r["prompt"],
                "answer": json.dumps({"points": [cell]}),
                "task": task,
                "video_source": md["video_source"],
                "video_index": int(md["video_index"]),
                "gt_frame": int(md["gt_frame"]),
                "cell_number": cell,
                "frame_indices": md["frame_indices"],
            })
    if not rows:
        raise RuntimeError(f"No rows found in {metadata_path}")
    return rows


def find_train_metadata(root):
    """Prefer the Hugging Face ImageFolder split layout, then legacy layout."""
    candidates = [
        os.path.join(root, "train", "metadata.jsonl"),
        os.path.join(root, "metadata.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No training metadata found under {root}")


def find_eval_metadata(root):
    """Return explicit generated eval metadata if present."""
    candidates = [
        os.path.join(root, "validation", "metadata.jsonl"),
        os.path.join(root, "eval", "metadata.jsonl"),
        os.path.join(root, "eval_metadata.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def grouped_split(rows, val_frac, seed):
    """Split *rows* into (train, val) keeping each ``(source, index)`` group
    entirely on one side to prevent leakage across the 9 offset variants."""
    groups = {}
    for r in rows:
        groups.setdefault((r["video_source"], r["video_index"]), []).append(r)
    keys = sorted(groups.keys())
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac)))
    val_keys = set(keys[:n_val])
    train_rows, val_rows = [], []
    for key, group_rows in groups.items():
        (val_rows if key in val_keys else train_rows).extend(group_rows)
    return train_rows, val_rows


# --------------------------------------------------------------------------- #
# Collator: builds completion-only-masked multimodal batches
# --------------------------------------------------------------------------- #
class GridCollator:
    """Render each chat example and mask everything except the JSON answer.

    The prompt (user turn + image placeholder tokens + the empty ``<think>``
    scaffold produced by ``add_generation_prompt`` in non-thinking mode) is
    processed *with* the image so image tokens are expanded correctly, then the
    answer tokens (``{"points": [N]}<|im_end|>``) are appended. Labels are the
    input ids with every prompt / image / pad token set to -100.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        self.pad_id = pad_id

    def _encode(self, example):
        image = Image.open(example["image"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": example["prompt"]},
        ]}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt = self.processor(text=[prompt_text], images=[image],
                                return_tensors="pt")
        prompt_ids = prompt["input_ids"][0]
        answer_ids = self.tokenizer(
            example["answer"] + END_OF_TURN, add_special_tokens=False,
            return_tensors="pt")["input_ids"][0]

        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        labels = input_ids.clone()
        labels[:prompt_ids.shape[0]] = -100

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": prompt["pixel_values"],
            "image_grid_thw": prompt["image_grid_thw"],
        }
        if "mm_token_type_ids" in prompt:
            mm = prompt["mm_token_type_ids"][0]
            item["mm_token_type_ids"] = torch.cat(
                [mm, torch.zeros(answer_ids.shape[0], dtype=mm.dtype)], dim=0)
        return item

    def _pad(self, seq, length, value):
        out = seq.new_full((length,), value)
        out[:seq.shape[0]] = seq
        return out

    def __call__(self, examples):
        items = [self._encode(ex) for ex in examples]
        max_len = max(it["input_ids"].shape[0] for it in items)
        batch = {
            "input_ids": torch.stack(
                [self._pad(it["input_ids"], max_len, self.pad_id) for it in items]),
            "attention_mask": torch.stack(
                [self._pad(it["attention_mask"], max_len, 0) for it in items]),
            "labels": torch.stack(
                [self._pad(it["labels"], max_len, -100) for it in items]),
            "pixel_values": torch.cat([it["pixel_values"] for it in items], dim=0),
            "image_grid_thw": torch.cat(
                [it["image_grid_thw"] for it in items], dim=0),
        }
        if "mm_token_type_ids" in items[0]:
            batch["mm_token_type_ids"] = torch.stack(
                [self._pad(it["mm_token_type_ids"], max_len, 0) for it in items])
        return batch


# --------------------------------------------------------------------------- #
# Train one adapter
# --------------------------------------------------------------------------- #
def build_model(args):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    attn_implementation = {
        "text_config": args.attn_impl,
        "vision_config": args.vision_attn_impl,
    }
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def train_one_task(task, root, args, processor):
    print(f"\n{'=' * 70}\n[train] task={task} root={root}\n{'=' * 70}", flush=True)
    train_meta_path = find_train_metadata(root)
    eval_meta_path = find_eval_metadata(root)
    rows = load_task_rows(train_meta_path, task)
    if eval_meta_path is None:
        random.Random(args.seed).shuffle(rows)
        train_rows, val_rows = grouped_split(rows, args.val_frac, args.seed)
        split_desc = "grouped by video"
    else:
        train_rows = rows
        val_rows = load_task_rows(eval_meta_path, task)
        split_desc = "generated holdout"
    random.Random(args.seed).shuffle(train_rows)
    print(f"[train] {task}: {len(rows)} rows -> {len(train_rows)} train / "
          f"{len(val_rows)} val ({split_desc})", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    val_path = os.path.join(args.output_dir, f"{task}_val.jsonl")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[train] wrote {val_path}", flush=True)

    train_ds = Dataset.from_list([
        {"image": r["image"], "prompt": r["prompt"], "answer": r["answer"]}
        for r in train_rows
    ])

    # Build the base fresh per task. get_peft_model injects LoRA layers in
    # place, so reusing one base across tasks would train the second adapter
    # on top of the first one's weights.
    model = build_model(args)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=os.path.join(args.output_dir, f"{task}_run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        data_collator=GridCollator(processor),
        processing_class=processor,
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, f"{task}_adapter")
    trainer.save_model(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"[train] saved {task} adapter -> {adapter_dir}", flush=True)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Config -> args shim (keeps the rest of the code reading `args.*`)
# --------------------------------------------------------------------------- #
class _Args:
    """Lightweight stand-in for the old argparse Namespace."""

    def __init__(self):
        self.contact_root = CONTACT_ROOT
        self.separation_root = SEPARATION_ROOT
        self.model_id = MODEL_ID
        self.output_dir = OUTPUT_DIR
        self.tasks = list(TASKS)
        self.epochs = EPOCHS
        self.seed = SEED
        self.val_frac = VAL_FRAC
        self.batch_size = BATCH_SIZE
        self.grad_accum = GRAD_ACCUM
        self.lr = LR
        self.lora_r = LORA_R
        self.lora_alpha = LORA_ALPHA
        self.lora_dropout = LORA_DROPOUT
        self.target_modules = (
            [m.strip() for m in TARGET_MODULES.split(",")]
            if "," in TARGET_MODULES
            else TARGET_MODULES
        )
        self.attn_impl = ATTN_IMPL
        self.vision_attn_impl = VISION_ATTN_IMPL
        self.num_workers = NUM_WORKERS
        self.gradient_checkpointing = GRADIENT_CHECKPOINTING


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--contact_root", default=CONTACT_ROOT)
    parser.add_argument("--separation_root", default=SEPARATION_ROOT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--attn_impl", default=ATTN_IMPL)
    parser.add_argument("--vision_attn_impl", default=VISION_ATTN_IMPL)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing adapter/run directories.",
    )
    parsed = parser.parse_args()

    args = _Args()
    for key, value in vars(parsed).items():
        setattr(args, key, value)
    return args


def main():
    args = parse_args()
    print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[env] attn text={args.attn_impl} vision={args.vision_attn_impl}",
          flush=True)

    roots = {"contact": args.contact_root, "separation": args.separation_root}
    for task in args.tasks:
        adapter_dir = os.path.join(args.output_dir, f"{task}_adapter")
        if os.path.exists(adapter_dir):
            if not args.overwrite:
                raise FileExistsError(
                    f"Adapter already exists: {adapter_dir}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(adapter_dir)
            run_dir = os.path.join(args.output_dir, f"{task}_run")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

    # One shared processor (same tokenizer / image processor for both tasks).
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    for task in args.tasks:
        train_one_task(task, roots[task], args, processor)

    print("\n[train] all requested adapters complete.", flush=True)


if __name__ == "__main__":
    main()
