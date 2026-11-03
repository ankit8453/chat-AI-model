"""
QLoRA fine-tuning (Level 3): Qwen2.5-7B-Instruct in 4-bit + LoRA adapters, loss on the reply only.

Plain transformers + peft + bitsandbytes (no TRL / Unsloth): every example is tokenised here with the model's own
chat template, so a library update on Colab cannot silently change how examples are built. Settings that were
renamed between transformers versions are passed only if the installed version knows them.

Checkpoints (adapter + optimizer state) are pushed to a private HF model repo after every save, so a Colab
disconnect resumes from the last one. The finished adapter sits at the repo root.
"""
from __future__ import annotations

import inspect
import math
import os

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_config(mode: str, vram_gb: float) -> dict:
    """mode 'trial' = 3,000 examples x 1 epoch (does it learn?); 'full' = all examples x 2 epochs."""
    cfg = dict(mode=mode, base_model=BASE_MODEL, max_len=1536, lora_r=16, lora_alpha=32, lora_dropout=0.05,
               lr=2e-4, batch_size=4, grad_accum=4, warmup_share=0.03, eval_rows=200, seed=42)
    if vram_gb >= 39:                      # A100: bigger batches, same effective batch of 16
        cfg.update(batch_size=8, grad_accum=2)
    if mode == "trial":
        cfg.update(train_rows=3000, epochs=1, eval_steps=50, save_steps=50)
    elif mode == "full":
        cfg.update(train_rows=None, epochs=2, eval_steps=200, save_steps=200)
    else:
        raise ValueError(f"mode must be 'trial' or 'full', not {mode!r}")
    return cfg


# ---------------------------------------------------------------- data

def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def tokenize_example(row: dict, tok, max_len: int) -> dict | None:
    """input_ids = prompt + reply; labels = -100 on the prompt, so only the reply (and its end token) is learned."""
    msgs = row["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prompt):
        raise ValueError("chat template: the prompt is not a prefix of the full conversation")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


class Collator:
    """Right-pads a batch; padding is masked out of attention and loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        m = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (m - len(b["input_ids"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att), "labels": torch.tensor(lab)}


# ---------------------------------------------------------------- model

def _bf16() -> bool:
    import torch
    return torch.cuda.is_bf16_supported()


def load_model(cfg: dict):
    """4-bit NF4 base + fresh LoRA adapters. Fresh adapters change nothing (LoRA B starts at 0), so this same
    model object is the untrained baseline until training starts."""
    import torch
    import transformers
    from packaging.version import Version
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = torch.bfloat16 if _bf16() else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=dtype)
    dtype_kw = {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb, device_map={"": 0},
                                                 attn_implementation="sdpa", **dtype_kw)
    model.config.use_cache = False
    model.enable_input_require_grads()          # needed for gradient checkpointing with frozen 4-bit weights
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def load_trained_adapter(model, repo_id: str) -> None:
    """Put the finished adapter from the HF repo into the (fresh) PEFT model."""
    from huggingface_hub import hf_hub_download
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = hf_hub_download(repo_id, "adapter_model.safetensors")
    res = set_peft_model_state_dict(model, load_file(path))
    missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
    if missing:
        raise SystemExit(f"adapter load: {len(missing)} LoRA weights missing, e.g. {missing[:2]}")
    print(f"loaded trained adapter from {repo_id}")


# ---------------------------------------------------------------- trainer

def training_args(cfg: dict, output_dir: str, n_train: int, overrides: dict | None = None):
    from transformers import TrainingArguments
    params = inspect.signature(TrainingArguments.__init__).parameters
    steps = math.ceil(n_train / (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
    bf16 = _bf16()
    kw = dict(output_dir=output_dir, per_device_train_batch_size=cfg["batch_size"],
              per_device_eval_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
              learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"], lr_scheduler_type="cosine",
              warmup_steps=max(1, int(cfg["warmup_share"] * steps)), logging_steps=10,
              save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=2, eval_steps=cfg["eval_steps"],
              bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit", gradient_checkpointing=True,
              gradient_checkpointing_kwargs={"use_reentrant": False},
              report_to="none", seed=cfg["seed"], remove_unused_columns=False, dataloader_num_workers=2)
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    if "train_sampling_strategy" in params:          # transformers 5.x
        kw["train_sampling_strategy"] = "group_by_length"
    else:                                            # transformers 4.x
        kw["group_by_length"] = True
    kw.update(overrides or {})
    dropped = [k for k in kw if k not in params]
    if dropped:
        print("TrainingArguments: not supported by this transformers version, skipped:", dropped)
    print(f"optimizer steps: {steps}  (effective batch {cfg['batch_size'] * cfg['grad_accum']})")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def make_callbacks(repo_id: str):
    """Push every checkpoint to the HF repo (keep the last 2 there) + stop on a NaN / exploding loss."""
    from huggingface_hub import HfApi
    from transformers import TrainerCallback

    api = HfApi()

    class HubCheckpoints(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt):
                return
            api.upload_folder(folder_path=ckpt, path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                              repo_id=repo_id, commit_message=f"checkpoint {state.global_step}")
            for old in hub_checkpoint_steps(repo_id)[:-2]:
                try:
                    api.delete_folder(f"checkpoints/checkpoint-{old}", repo_id=repo_id,
                                      commit_message=f"drop old checkpoint {old}")
                except Exception:  # noqa: BLE001 — cleanup only
                    pass
            print(f"  checkpoint {state.global_step} pushed -> {repo_id}")

    class LossGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            loss = float(loss)
            if math.isnan(loss) or math.isinf(loss) or loss > 10:
                print(f"ABORT: training loss {loss} at step {state.global_step} — stopping.")
                control.should_training_stop = True

    return [HubCheckpoints(), LossGuard()]


def hub_checkpoint_steps(repo_id: str) -> list[int]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id)
    return sorted({int(f.split("/")[1].split("-")[1]) for f in files
                   if f.startswith("checkpoints/checkpoint-") and f.count("/") >= 2})


def download_latest_checkpoint(repo_id: str, local_dir: str) -> str | None:
    from huggingface_hub import snapshot_download
    steps = hub_checkpoint_steps(repo_id)
    if not steps:
        return None
    sub = f"checkpoints/checkpoint-{steps[-1]}"
    snapshot_download(repo_id, allow_patterns=[f"{sub}/*"], local_dir=local_dir)
    return os.path.join(local_dir, sub)


def adapter_finished(repo_id: str) -> bool:
    from huggingface_hub import HfApi
    return "adapter_model.safetensors" in HfApi().list_repo_files(repo_id)
