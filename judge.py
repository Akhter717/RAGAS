"""Shared judge utilities: judge LLM (Groq) + embeddings. Independent of any RAG system."""
import os
from typing import Dict, List

import numpy as np
from groq import Groq


def judge_model() -> str: return os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
def embed_model_name() -> str: return os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


_emb = None


def embed_texts(texts: List[str]) -> np.ndarray:
    global _emb
    if _emb is None:
        from fastembed import TextEmbedding
        _emb = TextEmbedding(model_name=embed_model_name())
    v = np.array(list(_emb.embed(texts)), dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=5)
    r = client.chat.completions.create(model=model, temperature=temperature,
                                       messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()


def format_context(ctxs: List[Dict]) -> str:
    return "\n\n".join(f"[{c.get('source') or '-'} p.{c.get('page') or '-'}]\n{c['text']}" for c in ctxs)
