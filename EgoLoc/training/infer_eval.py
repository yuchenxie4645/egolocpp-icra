#!/usr/bin/env python3
"""Evaluate the two Qwen3.5-9B grid adapters (contact + separation).

Loads the base model once (4-bit NF4) and attaches BOTH LoRA adapters without
merging; the correct adapter is selected per example via ``set_adapter(task)``.
For every row in each ``<task>_val.jsonl`` (written by ``train_qlora.py``) it
renders the prompt in non-thinking mode, greedily generates a short answer,
parses the trailing ``{"points": [N]}`` JSON, and reports:

  * JSON parse-success rate
  * exact cell accuracy (pred_cell == cell_number)
  * frame MAE via ``frame_indices[pred_cell - 1]`` vs ``gt_frame``
    (off-by-one cell => off-by-one frame), comparable to EgoLoc's frame MAE
  * mean sigma-1 Gaussian cell reward used by GRPO

broken down by task and by video_source.

Attention backend: same story as training — text LM defaults to
``flash_attention_4`` and the vision tower to ``sdpa`` on this Blackwell 5090.
Pass ``--attn_impl sdpa`` to run entirely on PyTorch SDPA.

Run on GPU 1 only:
    CUDA_VISIBLE_DEVICES=1 python infer_eval.py \
        --output_dir /home/EgoLoc/training/adapters
"""

import argparse
import functools
import json
import os
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from grid_reward import gaussian_reward_for_cell, parse_pred_cell

TASKS = ("contact", "separation")
COMPUTE_DTYPE = torch.bfloat16


def load_val_rows(output_dir, task, dataset_root=None):
    """Load explicit dataset validation rows or the legacy saved val JSONL."""
    if dataset_root:
        path = os.path.join(dataset_root, "validation", "metadata.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        metadata_dir = os.path.dirname(path)
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                md = raw.get("metadata", raw)
                gt_cell = md.get("gt_cell", md.get("cell_number"))
                if gt_cell is None:
                    raise KeyError(f"Missing gt_cell/cell_number in {path}")
                rows.append({
                    "image": os.path.join(metadata_dir, raw["file_name"]),
                    "prompt": raw["prompt"],
                    "task": task,
                    "video_source": md["video_source"],
                    "video_index": int(md["video_index"]),
                    "gt_frame": int(md["gt_frame"]),
                    "cell_number": int(gt_cell),
                    "frame_indices": [
                        int(value) for value in md["frame_indices"]
                    ],
                })
        return rows

    path = os.path.join(output_dir, f"{task}_val.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def install_legacy_rope_default():
    """Restore ``ROPE_INIT_FUNCTIONS['default']`` for 4.x-era remote code.

    transformers 5.x moved unscaled RoPE onto each rotary class as
    ``compute_default_rope_parameters`` and deleted the registry's
    ``default`` entry. Molmo2 ships 4.x remote code that still builds
    ``Molmo2RotaryEmbedding(rope_type="default")`` and indexes the
    registry, so it dies with ``KeyError: 'default'``. Native models never
    look this key up, so re-adding it cannot affect them.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default_rope(config=None, device=None, seq_len=None,
                      layer_type=None):
        params = getattr(config, "rope_parameters", None)
        base = params.get("rope_theta") if isinstance(params, dict) else None
        if base is None:
            base = getattr(config, "rope_theta", 10000.0)
        head_dim = (getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope


def install_legacy_processor_kwargs():
    """Let 4.x-era processors forward custom kwargs through ProcessorMixin.

    transformers 5.x restricts ``ProcessorMixin.__init__`` to the declared
    attributes and raises ``TypeError`` on anything else. Molmo2Processor
    passes flags such as ``image_use_col_tokens`` that it later reads off
    ``self``, which 4.x set implicitly. Strip them, run the strict
    ``__init__``, then attach them.
    """
    from transformers.processing_utils import ProcessorMixin

    if getattr(ProcessorMixin.__init__, "_legacy_kwargs_shim", False):
        return

    original_init = ProcessorMixin.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        allowed = set(self.get_attributes())
        allowed.update(("chat_template", "audio_tokenizer"))
        extras = {key: kwargs.pop(key)
                  for key in list(kwargs) if key not in allowed}
        original_init(self, *args, **kwargs)
        for key, value in extras.items():
            setattr(self, key, value)

    patched_init._legacy_kwargs_shim = True
    ProcessorMixin.__init__ = patched_init


def build_inputs(processor, prompt, image_path, enable_thinking=False):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
    return processor(text=[text], images=[image], return_tensors="pt")


@torch.no_grad()
def main():
    args = parse_args()
    print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"cuda_available={torch.cuda.is_available()}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    if args.vision_attn_impl == "skip":
        # Molmo2 and friends have no nested vision_config, so the
        # per-subconfig dict form is rejected.
        attn_implementation = args.attn_impl
    else:
        attn_implementation = {
            "text_config": args.attn_impl,
            "vision_config": args.vision_attn_impl,
        }
    print(f"[env] attn text={args.attn_impl} vision={args.vision_attn_impl} "
          f"enable_thinking={args.enable_thinking}", flush=True)

    install_legacy_rope_default()
    install_legacy_processor_kwargs()
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        dtype=COMPUTE_DTYPE,
        attn_implementation=attn_implementation,
        device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )

    if args.base_only:
        # Evaluate the quantized base model directly, without loading LoRA.
        model = base
        attached = list(args.tasks)
        print("[eval] mode=base-only (no LoRA adapters loaded)", flush=True)
    else:
        # Attach both adapters (no merge); route per example with set_adapter().
        attached = []
        model = None
        for task in args.tasks:
            adapter_dir = os.path.join(
                args.output_dir, f"{task}{args.adapter_suffix}"
            )
            if not os.path.isdir(adapter_dir):
                print(f"[eval] skip {task}: no adapter at {adapter_dir}", flush=True)
                continue
            if model is None:
                model = PeftModel.from_pretrained(base, adapter_dir, adapter_name=task)
            else:
                model.load_adapter(adapter_dir, adapter_name=task)
            attached.append(task)
        if model is None:
            raise RuntimeError(f"No adapters found under {args.output_dir}")
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code)
    processor.tokenizer.padding_side = "left"

    # stats[(task, source)] = dict of counters
    stats = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0,
                                 "correct": 0, "abs_frame_err": 0.0,
                                 "mae_n": 0, "gaussian_reward": 0.0})

    for task in attached:
        dataset_root = (
            args.contact_root if task == "contact"
            else args.separation_root
        )
        rows = load_val_rows(args.output_dir, task, dataset_root)
        if not rows:
            print(f"[eval] skip {task}: no {task}_val.jsonl", flush=True)
            continue
        if args.limit:
            rows = rows[:args.limit]
        if not args.base_only:
            model.set_adapter(task)
        print(f"\n[eval] task={task}: {len(rows)} val rows", flush=True)

        for i, r in enumerate(rows):
            inputs = build_inputs(processor, r["prompt"], r["image"],
                                  args.enable_thinking).to(model.device)
            # Gemma-4 gates its own pixel cast on patch_dense.weight.dtype,
            # which NF4 turns into uint8, so fp32 pixels would reach bf16
            # vision norms. Align them here instead.
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    COMPUTE_DTYPE)
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(gen, skip_special_tokens=True)
            pred = parse_pred_cell(text)

            frame_indices = r["frame_indices"]
            gt_frame = int(r["gt_frame"])
            gt_cell = int(r["cell_number"])
            key = (task, r.get("video_source", "?"))
            s = stats[key]
            s["n"] += 1
            s["gaussian_reward"] += gaussian_reward_for_cell(
                pred,
                gt_cell,
                num_cells=len(frame_indices),
                sigma=args.reward_sigma,
            )
            if pred is not None:
                s["parsed"] += 1
                if 1 <= pred <= len(frame_indices):
                    s["valid"] += 1
                    if pred == gt_cell:
                        s["correct"] += 1
                    pred_frame = int(frame_indices[pred - 1])
                    s["abs_frame_err"] += abs(pred_frame - gt_frame)
                    s["mae_n"] += 1
            if args.verbose:
                print(f"  [{task} {i}] gt_cell={gt_cell} pred={pred} "
                      f"raw={text!r}", flush=True)

    print_report(stats)


def print_report(stats):
    print("\n" + "=" * 78)
    print(f"{'task/source':<28}{'n':>5}{'parse%':>9}{'cell_acc%':>11}"
          f"{'frameMAE':>11}{'avgReward':>11}")
    print("-" * 78)

    def line(label, s):
        n = s["n"] or 1
        parse_pct = 100.0 * s["parsed"] / n
        acc_pct = 100.0 * s["correct"] / n
        mae = (s["abs_frame_err"] / s["mae_n"]) if s["mae_n"] else float("nan")
        avg_reward = s["gaussian_reward"] / n
        print(f"{label:<28}{s['n']:>5}{parse_pct:>8.1f}%{acc_pct:>10.1f}%"
              f"{mae:>11.3f}{avg_reward:>11.4f}")

    per_task = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0,
                                    "correct": 0, "abs_frame_err": 0.0,
                                    "mae_n": 0, "gaussian_reward": 0.0})
    for (task, source), s in sorted(stats.items()):
        line(f"{task}/{source}", s)
        agg = per_task[task]
        for k in agg:
            agg[k] += s[k]
    print("-" * 78)
    for task, s in sorted(per_task.items()):
        line(f"{task} (all)", s)
    print("=" * 78, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    p.add_argument("--output_dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "adapters"),
                   help="Dir containing <task>_adapter/ and <task>_val.jsonl.")
    p.add_argument(
        "--adapter_suffix",
        default="_adapter",
        help=(
            "Adapter directory suffix. train_grpo.py produces "
            "_sft_grpo_adapter and _base_grpo_adapter."
        ),
    )
    p.add_argument(
        "--contact_root",
        default=None,
        help=(
            "Optional dataset root whose validation/metadata.jsonl replaces "
            "the saved contact_val.jsonl."
        ),
    )
    p.add_argument(
        "--separation_root",
        default=None,
        help=(
            "Optional dataset root whose validation/metadata.jsonl replaces "
            "the saved separation_val.jsonl."
        ),
    )
    p.add_argument("--reward_sigma", type=float, default=1.0)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--attn_impl", default="flash_attention_4",
                   help="Text LM attention (flash_attention_4 | sdpa | eager).")
    p.add_argument("--vision_attn_impl", default="sdpa",
                   help="Vision tower attention, or 'skip' when the config "
                        "has no nested vision_config (e.g. Molmo2).")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Needed for model types transformers does not "
                        "register natively (e.g. Molmo2).")
    p.add_argument("--enable_thinking", action="store_true",
                   help="Pass enable_thinking=True to the chat template. "
                        "Off by default: Qwen3.5 and Gemma-4 both suppress "
                        "reasoning when this is False.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, evaluate only the first N rows per task.")
    p.add_argument("--base_only", action="store_true",
                   help="Evaluate the 4-bit base model without loading LoRA.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
