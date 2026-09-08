"""Persistent, offline NF4 Transformers backend for the OccluBench ablation.

Heavy ML imports intentionally happen in :class:`TransformersBackend` rather
than at module import time. This keeps the CPU-only test suite lightweight.
"""

import functools
import json
import os
import threading

# These must be set before importing transformers or huggingface_hub.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_ID = "Qwen/Qwen3.5-9B"
CONTACT_ADAPTER = "/home/EgoLoc/training/v3-grpo/contact_sft_grpo_adapter"
SEPARATION_ADAPTER = "/home/EgoLoc/training/v3-grpo/separation_sft_grpo_adapter"


def install_legacy_rope_default():
    """Compatibility shim copied from training/infer_eval.py (2026-09-08)."""
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default_rope(config=None, device=None, seq_len=None, layer_type=None):
        params = getattr(config, "rope_parameters", None)
        base = params.get("rope_theta") if isinstance(params, dict) else None
        if base is None:
            base = getattr(config, "rope_theta", 10000.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64)
                .to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope


def install_legacy_processor_kwargs():
    """Compatibility shim copied from training/infer_eval.py (2026-09-08)."""
    from transformers.processing_utils import ProcessorMixin

    if getattr(ProcessorMixin.__init__, "_legacy_kwargs_shim", False):
        return

    original_init = ProcessorMixin.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        allowed = set(self.get_attributes())
        allowed.update(("chat_template", "audio_tokenizer"))
        extras = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key not in allowed
        }
        original_init(self, *args, **kwargs)
        for key, value in extras.items():
            setattr(self, key, value)

    patched_init._legacy_kwargs_shim = True
    ProcessorMixin.__init__ = patched_init


def parse_point(raw_text):
    """Parse the first value from the model's trailing ``points`` JSON."""
    if not isinstance(raw_text, str):
        return -1
    compact = raw_text.strip().replace(" ", "").replace("\n", "")
    marker = '{"points":'
    try:
        start = compact.index(marker)
        fragment = compact[start : compact.index("}", start) + 1]
        points = json.loads(fragment, strict=False).get("points", [])
    except (ValueError, TypeError, json.JSONDecodeError):
        return -1
    return points[0] if points else -1


class TransformersBackend:
    """One warm Qwen3.5 base with two unmerged PEFT adapters."""

    name = "transformers-nf4"

    def __init__(
        self,
        model_id=MODEL_ID,
        contact_adapter=CONTACT_ADAPTER,
        separation_adapter=SEPARATION_ADAPTER,
    ):
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TransformersBackend requires CUDA; run with CUDA_VISIBLE_DEVICES=1"
            )

        self._torch = torch
        self._lock = threading.RLock()
        self.model_id = model_id
        self.adapters = {
            "contact": contact_adapter,
            "separation": separation_adapter,
        }
        for task, path in self.adapters.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Missing {task} adapter: {path}")

        self.load_count = 0
        self.request_count = 0
        self.adapter_switch_count = 0
        self.current_task = None
        self.seed_strategy = "torch.Generator"
        self.decode_config = {
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "max_new_tokens": 200,
            # Transformers has no frequency/presence penalty parameters.
            # A value of zero means disabled, represented by no corresponding
            # logits processor and repetition_penalty=1.0.
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "enable_thinking": False,
        }
        self.vram_config = {
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_map": {"": 0},
            "quantization": "bitsandbytes-nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "model_dtype": "bfloat16",
            "text_attention": "sdpa",
            "vision_attention": "sdpa",
        }

        install_legacy_rope_default()
        install_legacy_processor_kwargs()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb,
            dtype=torch.bfloat16,
            attn_implementation={
                "text_config": "sdpa",
                "vision_config": "sdpa",
            },
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            base,
            contact_adapter,
            adapter_name="contact",
            is_trainable=False,
        )
        model.load_adapter(
            separation_adapter,
            adapter_name="separation",
            is_trainable=False,
        )
        # load_adapter does not intentionally change the active adapter, but
        # make the initial state explicit. Runtime switches are counted below.
        model.set_adapter("contact")
        model.eval()

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.current_task = "contact"
        self.load_count = 1
        self.load_provenance = {
            "model_id": model_id,
            "adapters": dict(self.adapters),
            "adapter_names": ["contact", "separation"],
            "adapters_merged": False,
            "held_out_dataset": "OccluBench",
            "held_out": True,
            "loader": "training/infer_eval.py-compatible",
            "offline": True,
        }

    def _activate(self, task):
        if task not in self.adapters:
            raise ValueError(
                f"Unknown task {task!r}; expected 'contact' or 'separation'"
            )
        if task != self.current_task:
            self.model.set_adapter(task)
            self.current_task = task
            self.adapter_switch_count += 1

    def _build_inputs(self, image_bgr, prompt_message):
        import cv2
        from PIL import Image

        if image_bgr is None or getattr(image_bgr, "ndim", 0) != 3:
            raise ValueError("image_bgr must be a HxWx3 BGR image")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        # Keep Trial-2-final's image-request content order: text, then image.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_message},
                    {"type": "image"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self._torch.bfloat16
            )
        return inputs

    def _seed_globally(self, seed):
        if seed is None:
            return
        self._torch.manual_seed(int(seed))
        self._torch.cuda.manual_seed_all(int(seed))

    def _run_generate(self, inputs, seed):
        kwargs = {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "repetition_penalty": 1.0,
        }
        if self.seed_strategy == "torch.Generator" and seed is not None:
            generator = self._torch.Generator(device=self.model.device)
            generator.manual_seed(int(seed))
            try:
                return self.model.generate(
                    **inputs, **kwargs, generator=generator
                )
            except (TypeError, ValueError) as exc:
                text = str(exc).lower()
                if "generator" not in text and "model_kwargs" not in text:
                    raise
                # Some remote GenerationMixin implementations reject the
                # generator keyword before decoding. Sequential execution is
                # locked, so global CPU/CUDA seeds remain deterministic.
                self.seed_strategy = "locked-global-torch-seed"
        self._seed_globally(seed)
        return self.model.generate(**inputs, **kwargs)

    def generate(
        self,
        image_bgr,
        prompt_message,
        task,
        seed=None,
        flag=None,
    ):
        """Generate one answer without reloading the model or adapters."""
        with self._lock, self._torch.inference_mode():
            self._activate(task)
            self.request_count += 1
            inputs = self._build_inputs(image_bgr, prompt_message)
            output = self._run_generate(inputs, seed)
            input_length = int(inputs["input_ids"].shape[1])
            generated = output[0][input_length:]
            raw_text = self.processor.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            if flag is not None:
                return raw_text
            return parse_point(raw_text), raw_text

    def generate_many(self, requests, batch_size=1):
        """Warm sequential generation for variable-size visual grids.

        True batching is deliberately not used: candidate grids and single
        visual checks have different image shapes and token counts, and
        padding them changes memory pressure on a 9B NF4 run. The persistent
        model and attached adapters are still reused for every request.
        """
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        outputs = []
        for request in requests:
            if isinstance(request, dict):
                outputs.append(self.generate(**request))
            else:
                image, prompt, task, *rest = request
                seed = rest[0] if rest else None
                flag = rest[1] if len(rest) > 1 else None
                outputs.append(
                    self.generate(
                        image,
                        prompt,
                        task,
                        seed=seed,
                        flag=flag,
                    )
                )
        return outputs


_SHARED_BACKEND = None
_SHARED_BACKEND_LOCK = threading.Lock()


def get_shared_backend():
    """Return one process-wide warm NF4 model with both adapters attached."""
    global _SHARED_BACKEND
    if _SHARED_BACKEND is None:
        with _SHARED_BACKEND_LOCK:
            if _SHARED_BACKEND is None:
                _SHARED_BACKEND = TransformersBackend()
    return _SHARED_BACKEND
