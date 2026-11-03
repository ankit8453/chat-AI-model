"""Batched greedy generation for the golden exam (same code path for the base model and every adapter)."""
from __future__ import annotations

import time


def generate_replies(model, tok, rows: list[dict], batch_size: int = 16, max_new_tokens: int = 256) -> list[str]:
    import torch
    tok.padding_side = "left"
    order = sorted(range(len(rows)), key=lambda i: sum(len(m["content"]) for m in rows[i]["messages"]))
    out: list[str] = [""] * len(rows)
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        texts = [tok.apply_chat_template(rows[i]["messages"], tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t.strip()
        done = b + len(idx)
        print(f"  {done}/{len(rows)} replies  ({time.time() - t0:.0f}s)", end="\r")
    print()
    model.config.use_cache = False
    tok.padding_side = "right"
    if was_training:
        model.train()
    return out
