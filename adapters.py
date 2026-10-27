"""Adapters that let the evaluator test ANY RAG system.
Contract: rag_fn(question) -> {"answer": str, "contexts": [{"text","source","page"}, ...]}"""
import json
from typing import Dict, List

import requests


def _page(v):
    try:
        return int(float(v))
    except Exception:
        return None


def normalize_contexts(raw) -> List[Dict]:
    """Accepts a list of strings or dicts (text/content/page_content, optional source/page/metadata)."""
    out = []
    for c in raw or []:
        if isinstance(c, str):
            text, src, pg, sc = c, None, None, None
        elif isinstance(c, dict):
            meta = c.get("metadata") or {}
            text = c.get("text") or c.get("content") or c.get("page_content") or ""
            src, pg, sc = c.get("source") or meta.get("source"), c.get("page") or meta.get("page"), c.get("score")
        else:
            continue
        if str(text).strip():
            out.append({"text": str(text), "source": src or "-", "page": _page(pg), "score": sc})
    return out


def parse_contexts(cell) -> List[Dict]:
    """CSV cell -> contexts. JSON list, or chunks separated by |||"""
    s = str(cell or "").strip()
    if not s or s == "nan":
        return []
    if s.startswith("["):
        try:
            return normalize_contexts(json.loads(s))
        except Exception:
            pass
    return normalize_contexts([p.strip() for p in s.split("|||")])


class HttpRag:
    """Calls your own RAG API:  POST {q_field: question}  ->  JSON with answer + contexts."""
    def __init__(self, url, api_key="", q_field="question", a_field="answer", c_field="contexts", timeout=90):
        self.url, self.api_key, self.q, self.a, self.c, self.timeout = url, api_key, q_field, a_field, c_field, timeout

    def __call__(self, question: str) -> Dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(self.url, json={self.q: question}, headers=h, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return {"answer": str(d.get(self.a, "")), "contexts": normalize_contexts(d.get(self.c))}
