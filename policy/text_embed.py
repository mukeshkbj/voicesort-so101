"""Frozen MiniLM-L6-v2 sentence embeddings for instruction conditioning.

Falls back to deterministic seeded random vectors if HF download fails.
"""
import numpy as np

EMB_DIM = 384


def embed_texts(texts):
    """Return (n_texts, 384) float32 semantic embeddings."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        mdl.eval()
        with torch.no_grad():
            x = tok(list(texts), return_tensors="pt", padding=True)
            h = mdl(**x).last_hidden_state          # (B, T, 384)
            mask = x["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1)    # mean pool
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.numpy().astype(np.float32)
    except Exception as e:
        print(f"[text_embed] MiniLM unavailable ({e}); using seeded random vectors")
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            v = rng.standard_normal(EMB_DIM)
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return np.stack(out)


def task_embeddings(task_list):
    """task_list: list of instruction strings -> (n,384) float32."""
    return embed_texts(task_list)
