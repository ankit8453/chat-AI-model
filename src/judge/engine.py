"""
Inference engines for the judge. Same interface: generate(list_of_chat_messages) -> list[str].

- VllmEngine : fast batched inference (AWQ 4-bit). Preferred.
- HfEngine   : Hugging Face transformers + bitsandbytes 4-bit. Fallback when vLLM can't run on the GPU
               (newer vLLM releases may not support the T4's older architecture).
"""
from __future__ import annotations


def gpu_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"name": None, "vram_gb": 0.0}
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "vram_gb": round(p.total_memory / 1024**3, 1),
            "capability": f"{p.major}.{p.minor}"}


def auto_config() -> dict:
    """Pick model + batch settings for the detected GPU. Free Colab T4 = 15 GB."""
    g = gpu_info()
    vram = g["vram_gb"]
    if vram == 0:
        raise SystemExit("No GPU — Runtime > Change runtime type > T4 GPU.")
    cfg = {
        "gpu": g,
        "vllm_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
        "max_model_len": 3072,
        "gpu_memory_utilization": 0.90,
        "max_new_tokens": 120,
        "shard_size": 1000,          # pairs per checkpoint pushed to HF
        "hf_batch_size": 12,
    }
    if vram >= 39:                    # A100 40/80 GB (if ever available)
        cfg.update(vllm_model="Qwen/Qwen2.5-32B-Instruct-AWQ", shard_size=4000, hf_batch_size=32)
    elif vram >= 22:                  # L4 / A10 24 GB
        cfg.update(vllm_model="Qwen/Qwen2.5-14B-Instruct-AWQ", shard_size=2000, hf_batch_size=20)
    return cfg


class VllmEngine:
    name = "vllm"

    def __init__(self, cfg: dict):
        from vllm import LLM, SamplingParams
        self.model_id = cfg["vllm_model"]
        self.llm = LLM(model=self.model_id, dtype="half", max_model_len=cfg["max_model_len"],
                       gpu_memory_utilization=cfg["gpu_memory_utilization"])
        self.sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        outs = self.llm.chat(conversations, self.sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HfEngine:
    name = "transformers-4bit"

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.model_id = cfg["hf_model"]
        self.batch = cfg["hf_batch_size"]
        self.max_new = cfg["max_new_tokens"]
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16))
        self.model.eval()

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(conversations), self.batch):
            chunk = conversations[i:i + self.batch]
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2800).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out


def build_engine(cfg: dict, backend: str = "auto"):
    """backend: 'auto' (vLLM, fall back to transformers), 'vllm', or 'hf'."""
    if backend in ("auto", "vllm"):
        try:
            eng = VllmEngine(cfg)
            print(f"Engine: vLLM · {eng.model_id}")
            return eng
        except Exception as e:  # noqa: BLE001 — any vLLM start-up failure -> fallback
            if backend == "vllm":
                raise
            print(f"vLLM could not start on this GPU ({type(e).__name__}: {str(e)[:200]}).")
            print("Falling back to transformers 4-bit (slower, same results).")
    eng = HfEngine(cfg)
    print(f"Engine: transformers 4-bit · {eng.model_id} · batch {eng.batch}")
    return eng
