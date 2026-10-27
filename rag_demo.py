"""Built-in DEMO RAG system (PDF -> FAISS -> Groq). Replaceable: the evaluator never imports this file.
Contract of any RAG system:  rag_fn(question) -> {"answer": str, "contexts": [{"text","source","page"}, ...]}"""
import io
import os
from typing import Dict, List

import numpy as np
from pypdf import PdfReader

import judge

try:
    import faiss
except Exception:
    faiss = None

CHUNK_SIZE, CHUNK_OVERLAP = 800, 150
STRICT = ("You are a concise, accurate assistant. Answer ONLY from the provided context. "
          "If the answer is not in the context, say \"I don't know.\"")
LOOSE = ("You are a confident assistant. Use the context if it helps. If it does not contain the answer, "
         "answer from your own general knowledge anyway. Never say you do not know.")


def rag_model() -> str: return os.getenv("RAG_MODEL", "openai/gpt-oss-20b")


def chunk_pdf(data: bytes, name: str) -> List[Dict]:
    out = []
    for p, page in enumerate(PdfReader(io.BytesIO(data)).pages, 1):
        t = " ".join((page.extract_text() or "").split())
        for s in range(0, len(t), CHUNK_SIZE - CHUNK_OVERLAP):
            piece = t[s:s + CHUNK_SIZE].strip()
            if len(piece) > 50:
                out.append({"text": piece, "source": name, "page": p})
    return out


class RagIndex:
    def __init__(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("No text found in the PDF(s). Scanned PDFs need OCR.")
        self.chunks = chunks
        self.vecs = judge.embed_texts([c["text"] for c in chunks])
        self.dim = int(self.vecs.shape[1])
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.vecs)

    def retrieve(self, q: str, k: int) -> List[Dict]:
        k = max(1, min(k, len(self.chunks)))
        qv = judge.embed_texts([q])
        if self.index is not None:
            d, i = self.index.search(qv, k)
            return [{**self.chunks[j], "score": float(s)} for s, j in zip(d[0], i[0]) if j != -1]
        sims = self.vecs @ qv[0]
        return [{**self.chunks[j], "score": float(sims[j])} for j in np.argsort(-sims)[:k]]


class DemoRAG:
    def __init__(self, index: RagIndex, top_k: int = 4, strict: bool = True):
        self.index, self.top_k, self.strict = index, top_k, strict

    def __call__(self, question: str) -> Dict:
        ctxs = self.index.retrieve(question, self.top_k)
        ans = judge.chat(rag_model(), STRICT if self.strict else LOOSE,
                         f"Context:\n{judge.format_context(ctxs)}\n\nQuestion: {question}")
        return {"answer": ans, "contexts": ctxs}
