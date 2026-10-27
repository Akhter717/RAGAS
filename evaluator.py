"""RAG-agnostic evaluation. Input per item: question, answer, contexts (+ optional ground_truth / expected_page).
Never imports a RAG system. Metrics: RAGAS (5) + Hit Rate + Semantic Similarity + Hallucination Rate."""
import sys
import types

for _n, _a in (("langchain_community.chat_models.vertexai", "ChatVertexAI"),
               ("langchain_community.llms.vertexai", "VertexAI")):
    _m = types.ModuleType(_n)
    setattr(_m, _a, type(_a, (), {}))
    sys.modules[_n] = _m

import asyncio
import json
import math
import random
import re
from typing import Dict, List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithoutReference,
                           LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy)

import judge

NO_ANSWER = "INSUFFICIENT_CONTEXT"
METRICS = ["Context Precision", "Context Recall", "Hit Rate", "Faithfulness", "Answer Relevancy",
           "Factual Correctness", "Semantic Similarity", "Hallucination Rate"]
LOWER_IS_BETTER = {"Hallucination Rate"}
_IDK = ("don't know", "do not know", "dont know", "cannot find", "can't find", "not mentioned",
        "not provided", "no information", "not in the context", "not contain", "insufficient")


def is_idk(t: str) -> bool:
    t = (t or "").lower()
    return len(t) < 220 and any(p in t for p in _IDK)


def make_questions(texts: List[str], n: int) -> List[str]:
    pool = [t for t in texts if len(t) > 300] or texts
    qs = []
    for t in random.sample(pool, min(n, len(pool))):
        q = judge.chat(judge.judge_model(), "You write exam-style questions.",
                       "Write ONE clear, specific question answerable using only this passage. Do not mention "
                       f"'the passage'. Output only the question.\n\n{t}", temperature=0.7)
        qs.append(q.strip().strip('"'))
    return qs


def generate_reference(q: str, ctxs: List[Dict]) -> str:
    return judge.chat(judge.judge_model(), "You are an expert reader. Answer strictly from the context.",
                      f"Context:\n{judge.format_context(ctxs)}\n\nQuestion: {q}\n\nWrite a complete, concise answer "
                      f"using ONLY the context. If the context lacks the answer, reply exactly: {NO_ANSWER}")


def check_claims(answer: str, ctxs: List[Dict]) -> List[Dict]:
    out = judge.chat(judge.judge_model(), "You output only valid JSON.",
                     f"You are a strict fact checker.\n\nContext:\n{judge.format_context(ctxs)}\n\nAnswer:\n{answer}\n\n"
                     "Split the answer into atomic factual claims. For each, decide if the context FULLY supports it. "
                     'Reply ONLY with a JSON array: [{"claim": "...", "verdict": "supported" or "not_supported", '
                     '"evidence": "short quote from context or empty"}]')
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        raise ValueError("Judge did not return a JSON list.")
    return [{"claim": str(d.get("claim", "")),
             "verdict": "supported" if str(d.get("verdict", "")).lower().startswith("supported") else "not_supported",
             "evidence": str(d.get("evidence", ""))} for d in json.loads(m.group(0)) if isinstance(d, dict)]


class _Emb(Embeddings):
    def embed_documents(self, texts): return judge.embed_texts(list(texts)).tolist()
    def embed_query(self, text): return judge.embed_texts([text])[0].tolist()


async def _score(metric, sample, errors, name):
    try:
        v = float(await metric.single_turn_ascore(sample))
        return None if math.isnan(v) else v
    except Exception as e:
        errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        return None


async def _ragas(q, ans, ctx, ref, do_answer, errors):
    llm = LangchainLLMWrapper(ChatGroq(model=judge.judge_model(), temperature=0, max_retries=5))
    emb = LangchainEmbeddingsWrapper(_Emb())
    s = SingleTurnSample(user_input=q, response=ans, retrieved_contexts=ctx, reference=ref or None)
    plan = {}
    if ref:
        plan["Context Precision"] = LLMContextPrecisionWithReference(llm=llm)
        plan["Context Recall"] = LLMContextRecall(llm=llm)
    else:
        plan["Context Precision"] = LLMContextPrecisionWithoutReference(llm=llm)
    if do_answer:
        plan["Faithfulness"] = Faithfulness(llm=llm)
        plan["Answer Relevancy"] = ResponseRelevancy(llm=llm, embeddings=emb, strictness=1)  # Groq: n=1 only
        if ref:
            plan["Factual Correctness"] = FactualCorrectness(llm=llm)
    return {n: await _score(m, s, errors, n) for n, m in plan.items()}


def _cos(a: str, b: str) -> float:
    v = judge.embed_texts([a, b])
    return float(max(0.0, min(1.0, np.dot(v[0], v[1]))))


def evaluate_item(question: str, answer: str, contexts: List[Dict], ground_truth: str = "",
                  expected_page=None, wide_contexts=None, run_claims: bool = True) -> Dict:
    """contexts: [{"text","source","page"}]. wide_contexts: optional bigger context so the judge can write a
    reference when no ground_truth is given. Without either, reference-based metrics are skipped."""
    errors: List[str] = []
    if ground_truth:
        ref, src = ground_truth, "dataset"
    elif wide_contexts:
        ref, src = generate_reference(question, wide_contexts), "judge"
    else:
        ref, src = "", "none"
    ref_ok = bool(ref) and NO_ANSWER not in ref
    idk = is_idk(answer)
    sc = {m: None for m in METRICS}
    claims: List[Dict] = []

    if not contexts:
        errors.append("The RAG system returned no contexts, so retrieval and faithfulness metrics were skipped.")
    else:
        if run_claims and not idk:
            try:
                claims = check_claims(answer, contexts)
            except Exception as e:
                errors.append(f"Claim check: {type(e).__name__}: {str(e)[:120]}")
        sc.update(asyncio.run(_ragas(question, answer, [c["text"] for c in contexts],
                                     ref if ref_ok else "", not idk, errors)))
        if expected_page:
            sc["Hit Rate"] = 1.0 if any(c.get("page") == expected_page for c in contexts) else 0.0
        elif sc["Context Recall"] is not None:
            sc["Hit Rate"] = 1.0 if sc["Context Recall"] > 0 else 0.0
    if not idk and ref_ok:
        sc["Semantic Similarity"] = _cos(answer, ref)
    if idk:
        sc["Hallucination Rate"] = 0.0
    elif claims:
        sc["Hallucination Rate"] = sum(c["verdict"] != "supported" for c in claims) / len(claims)

    if idk and ref_ok:
        status = "Retrieval miss"
    elif idk:
        status = "Correct refusal" if src == "judge" else "Refusal"
    elif not ref_ok and src == "judge":
        status = "Possible hallucination"
    elif sc["Hallucination Rate"]:
        status = "Hallucination found"
    else:
        status = "OK"
    return {"question": question, "answer": answer, "reference": ref if ref_ok else "(none / not found)",
            "ref_source": src, "status": status, "scores": sc, "claims": claims, "errors": errors,
            "chunks": [(c.get("source"), c.get("page"), c.get("score"), c["text"]) for c in contexts]}
