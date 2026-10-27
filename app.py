import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Testing Framework", layout="wide")

try:
    for k in ("GROQ_API_KEY", "RAG_MODEL", "JUDGE_MODEL", "EMBED_MODEL"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k]).strip()
except Exception:
    pass
if not os.getenv("GROQ_API_KEY"):
    st.error('GROQ_API_KEY missing. Streamlit Cloud -> App settings -> Secrets -> add:\n\nGROQ_API_KEY = "gsk_..."')
    st.stop()

import adapters
import evaluator as ev
import judge
import rag_demo

M = ev.METRICS
INFO = {
    "Context Precision": "RAGAS - retrieved chunks useful & ranked well (vs reference if available)",
    "Context Recall": "RAGAS - chunks contain everything the reference needs (needs reference)",
    "Hit Rate": "Retrieval - 1 if right page/info retrieved (expected_page, else recall>0)",
    "Faithfulness": "RAGAS - answer claims supported by chunks",
    "Answer Relevancy": "RAGAS - answer addresses the question",
    "Factual Correctness": "RAGAS - answer claims match reference claims (needs reference)",
    "Semantic Similarity": "Embedding cosine: answer vs reference (needs reference)",
    "Hallucination Rate": "Unsupported claims / total claims (LOWER is better)",
}
MODES = ["Built-in demo RAG (PDF)", "My own RAG via HTTP API", "Upload pre-computed outputs (CSV)"]


def badge(m, v):
    if v is None or v != v:
        return "⚪"
    v = 1 - v if m in ev.LOWER_IS_BETTER else v
    return "🟢" if v >= 0.8 else ("🟡" if v >= 0.5 else "🔴")


st.title("RAG Testing Framework (RAGAS + Hallucination check)")
st.caption(f"Judge LLM: `{judge.judge_model()}` | Embeddings: `{judge.embed_model_name()}` | "
           "The evaluator is independent of the RAG system under test.")

with st.sidebar:
    st.header("RAG system under test")
    mode = st.radio("Source of answers", MODES)
    top_k, strict, api = 4, True, {}
    if mode == MODES[0]:
        top_k = st.slider("Top-k chunks to RAG LLM", 1, 10, 4)
        strict = st.radio("Answering style", ["Strict", "Loose (demo hallucination)"]).startswith("Strict")
        st.caption(f"RAG LLM: `{rag_demo.rag_model()}`")
    elif mode == MODES[1]:
        api["url"] = st.text_input("API URL (POST, JSON)")
        api["key"] = st.text_input("API key (optional, sent as Bearer)", type="password")
        api["q"] = st.text_input("Request field for question", "question")
        api["a"] = st.text_input("Response field for answer", "answer")
        api["c"] = st.text_input("Response field for contexts", "contexts")
    else:
        st.caption("CSV columns: question, answer, contexts (JSON list or chunks joined by |||), "
                   "ground_truth (optional), expected_page (optional)")
    st.header("Reference document")
    files = st.file_uploader("PDF(s)" + (" - required" if mode == MODES[0] else " - optional"),
                             type="pdf", accept_multiple_files=True,
                             help="Used as the demo RAG's knowledge base, to auto-generate questions, and so the "
                                  "judge can write reference answers when ground_truth is empty.")
    wide_k = st.slider("Chunks for judge reference", 4, 20, 10)
    run_claims = st.checkbox("Claim-by-claim hallucination check", True)

if files:
    sig = tuple((f.name, f.size) for f in files)
    if st.session_state.get("sig") != sig:
        with st.spinner("Chunking, embedding, building FAISS index..."):
            try:
                chunks = []
                for f in files:
                    chunks += rag_demo.chunk_pdf(f.getvalue(), f.name)
                st.session_state.index = rag_demo.RagIndex(chunks)
                st.session_state.sig = sig
                st.session_state.pop("results", None)
            except Exception as e:
                for k in ("index", "sig"):
                    st.session_state.pop(k, None)
                st.error(f"Could not process PDF: {e}")
else:
    for k in ("index", "sig"):
        st.session_state.pop(k, None)
index = st.session_state.get("index")

if mode == MODES[0] and not index:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()
if index:
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(index.chunks))
    c2.metric("Embedding dim", index.dim)
    c3.metric("Vector store", "FAISS" if index.index is not None else "NumPy")

# ------------------------------------------------------------ test dataset
st.subheader("1. Test dataset")
COLS = ["question", "ground_truth", "expected_page", "answer", "contexts"]
if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame({"question": [""], "ground_truth": [""], "expected_page": [None],
                                             "answer": [""], "contexts": [""]})
a, b, c = st.columns([2, 1, 1])
up = a.file_uploader("Upload test CSV", type="csv")
if up is not None and st.session_state.get("csv_name") != (up.name, up.size):
    try:
        d = pd.read_csv(up)
        for col in COLS:
            if col not in d.columns:
                d[col] = None
        st.session_state.dataset = d[COLS]
        st.session_state.csv_name = (up.name, up.size)
    except Exception as e:
        st.error(f"Bad CSV: {e}")
if index:
    n_gen = b.number_input("Auto-generate N questions", 1, 15, 5)
    if b.button("🎲 Generate from PDF"):
        with st.spinner("Judge LLM is writing questions..."):
            try:
                qs = ev.make_questions([x["text"] for x in index.chunks], int(n_gen))
                st.session_state.dataset = pd.DataFrame({"question": qs, "ground_truth": [""] * len(qs),
                                                         "expected_page": [None] * len(qs),
                                                         "answer": [""] * len(qs), "contexts": [""] * len(qs)})
                st.session_state.pop("csv_name", None)
            except Exception as e:
                st.error(f"Could not generate: {e}")
tmpl = ("question,ground_truth,expected_page,answer,contexts\n"
        'What is X?,X is ...,3,"X is ...","chunk one ||| chunk two"\n')
c.download_button("Download CSV template", tmpl, "test_dataset_template.csv")
st.caption("Empty ground_truth = the judge writes a reference from the PDF (if uploaded); with neither, "
           "reference-based metrics are skipped. expected_page is optional and makes Hit Rate exact.")
show = COLS if mode == MODES[2] else COLS[:3]
edited = st.data_editor(st.session_state.dataset, num_rows="dynamic", use_container_width=True,
                        column_order=show, key="editor")

# ------------------------------------------------------------ run
st.subheader("2. Run evaluation")


def clean(v):
    s = "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
    return "" if s == "nan" else s


if st.button("▶ Run evaluation on all questions", type="primary"):
    if mode == MODES[1] and not api.get("url"):
        st.warning("Enter the API URL in the sidebar.")
        st.stop()
    rag_fn = (rag_demo.DemoRAG(index, top_k, strict) if mode == MODES[0] else
              adapters.HttpRag(api["url"], api["key"], api["q"], api["a"], api["c"]) if mode == MODES[1] else None)
    items = [r for r in edited.to_dict("records") if clean(r.get("question"))]
    if not items:
        st.warning("Add at least one question.")
        st.stop()
    results, bar, msg = [], st.progress(0.0), st.empty()
    for i, r in enumerate(items):
        q, gt = clean(r["question"]), clean(r.get("ground_truth"))
        try:
            pg = int(float(r.get("expected_page"))) if clean(r.get("expected_page")) else None
        except Exception:
            pg = None
        msg.write(f"Evaluating {i + 1}/{len(items)}: {q[:80]}")
        try:
            if rag_fn:
                out = rag_fn(q)
                ans, ctxs = out["answer"], out["contexts"]
            else:
                ans, ctxs = clean(r.get("answer")), adapters.parse_contexts(r.get("contexts"))
            wide = index.retrieve(q, wide_k) if (index and not gt) else None
            results.append(ev.evaluate_item(q, ans, ctxs, gt, pg, wide, run_claims))
        except Exception as e:
            results.append({"question": q, "answer": "", "reference": "", "ref_source": "-", "status": "Failed",
                            "scores": {m: None for m in M}, "claims": [], "chunks": [],
                            "errors": [f"{type(e).__name__}: {str(e)[:200]}"]})
        bar.progress((i + 1) / len(items))
    msg.empty()
    st.session_state.results = results

res = st.session_state.get("results")
if not res:
    st.stop()

# ------------------------------------------------------------ report
st.subheader("3. Report")
df = pd.DataFrame([{"Question": r["question"], "Status": r["status"], **r["scores"]} for r in res])
cols = st.columns(4)
for i, m in enumerate(M):
    mean = df[m].astype(float).mean()
    cols[i % 4].metric(f"{badge(m, mean)} {m}", "n/a" if mean != mean else f"{mean:.2f}", help=INFO[m])
halluc = sum(r["status"] in ("Hallucination found", "Possible hallucination") for r in res)
st.markdown(f"**Questions:** {len(res)} | **Answers with hallucination:** {halluc} ({halluc / len(res):.0%}) | "
            f"**Refusals:** {sum(r['status'] in ('Correct refusal', 'Refusal') for r in res)} | "
            f"**Retrieval misses:** {sum(r['status'] == 'Retrieval miss' for r in res)}")
mean_scores = df[M].astype(float).mean().dropna()
if len(mean_scores):
    st.bar_chart(mean_scores)
st.dataframe(df.style.format({m: "{:.2f}" for m in M}, na_rep="-"), use_container_width=True)
st.download_button("Download report (CSV)", df.to_csv(index=False), "rag_eval_report.csv", "text/csv")

st.subheader("4. Per-question details")
for i, r in enumerate(res, 1):
    with st.expander(f"{i}. {r['status']} - {r['question'][:90]}"):
        st.markdown(f"**RAG answer:** {r['answer']}")
        st.markdown(f"**Reference ({r['ref_source']}):** {r['reference']}")
        if r["claims"]:
            st.table([{"Claim": c["claim"], "Verdict": "✅ supported" if c["verdict"] == "supported" else "❌ NOT supported",
                       "Evidence": c["evidence"]} for c in r["claims"]])
        st.write({m: (None if v is None else round(v, 3)) for m, v in r["scores"].items()})
        for e in r["errors"]:
            st.warning(e)
        for src, pg, s, t in r["chunks"]:
            st.caption(f"{src} p.{pg}" + (f" | similarity {s:.3f}" if isinstance(s, float) else ""))
            st.text(t[:400])
st.caption("Scores come from an LLM judge and vary slightly between runs - use them as signals. "
           "Checks are against the retrieved contexts, not the whole document.")
