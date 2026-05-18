"""
app.py — HybridSearch Bench  ·  Streamlit Dashboard
=====================================================
Hybrid RAG pipeline (BM25 + Dense Vector + RRF) with RAGAS auto-evaluation.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HybridSearch Bench",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)

def wait_for_ollama():
    import urllib.request
    import urllib.error
    from settings import get_settings
    settings = get_settings()
    url = settings.ollama_base_url.rstrip("/") + "/"
    
    # Do a quick check first without showing UI if it's already up
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=1.0) as response:
            if response.status == 200:
                return
    except (urllib.error.URLError, TimeoutError):
        pass

    status_placeholder = st.empty()
    retries = 30  # 30 * 5 = 150 seconds
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=3.0) as response:
                if response.status == 200:
                    status_placeholder.empty()
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        status_placeholder.warning(f"⏳ Waiting for Ollama (local LLM & embeddings) to initialize... (Attempt {i+1}/{retries})")
        time.sleep(5)
    
    status_placeholder.error("🚨 Ollama failed to initialize within the timeout period. Please check the `ollama-init` container logs.")
    st.stop()

wait_for_ollama()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
        color: white;
    }
    .hero h1 { font-size: 2rem; margin: 0 0 6px; font-weight: 700; }
    .hero p  { font-size: 0.95rem; margin: 0; opacity: 0.75; }

    /* ── Strategy cards ── */
    .card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
    }
    .card-bm25   { border-left: 4px solid #f97316; }
    .card-vector { border-left: 4px solid #3b82f6; }
    .card-hybrid { border-left: 4px solid #10b981; }

    /* ── Metric pills ── */
    .metric-pill {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #93c5fd;
        margin-right: 4px;
    }

    /* ── Chunk expanders ── */
    .chunk-badge {
        background: rgba(255,255,255,0.07);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        color: #9ca3af;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* ── Answer box ── */
    .answer-box {
        background: rgba(16, 185, 129, 0.07);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Colour palette (consistent across charts) ─────────────────────────────────
COLOURS = {
    "bm25":   "#f97316",   # orange
    "vector": "#3b82f6",   # blue
    "hybrid": "#10b981",   # green
}

METRIC_LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer Relevancy",
    "context_precision":  "Context Precision",
}


# ── Session-state defaults ────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "chunks": None,
        "bm25_index": None,
        "collection": None,
        "ingested_filename": None,
        "retrieval_results": None,
        "eval_results": None,
        "eval_history": [],   # list of {query, results, eval, timestamp}
        "last_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cached_ingest(file_bytes: bytes, filename: str):
    """Cache ingestion result keyed by file content + name."""
    from ingestion import ingest_pdf
    return ingest_pdf(file_bytes, filename)


def _score_bar(score: float, max_val: float = 1.0) -> str:
    pct = min(int((score / max_val) * 100), 100)
    return f"{'█' * (pct // 10)}{'░' * (10 - pct // 10)}  {score:.3f}"


def _strategy_icon(s: str) -> str:
    return {"bm25": "📄", "vector": "🔮", "hybrid": "⚡"}[s]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 HybridSearch Bench")
    st.markdown("---")

    # ── Ollama ──
    st.markdown("### 🦙 Ollama Config")
    st.info("Using local Ollama instance for LLM and Embeddings.")
    st.markdown("---")

    st.markdown("---")

    # ── PDF Upload ──
    st.markdown("### 📄 Upload PDF Corpus")
    uploaded_file = st.file_uploader(
        "Drag & drop a PDF",
        type=["pdf"],
        help="ArXiv papers, textbooks, reports — any research PDF works.",
    )

    if uploaded_file:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**{uploaded_file.name}**  \n`{uploaded_file.size / 1024:.0f} KB`")
        with col_b:
            ingest_btn = st.button("⚙️ Ingest", use_container_width=True, type="primary")

        if ingest_btn:
            progress_bar = st.progress(0.0)
            status_txt = st.empty()

            def _cb(pct: float, msg: str = "") -> None:
                progress_bar.progress(pct)
                if msg:
                    status_txt.info(msg)

            with st.spinner("Running ingestion pipeline…"):
                file_bytes = uploaded_file.read()

                from ingestion import ingest_pdf
                chunks, collection = ingest_pdf(file_bytes, uploaded_file.name, _cb)

                from retrieval import build_bm25_index
                bm25_idx = build_bm25_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.collection = collection
            st.session_state.bm25_index = bm25_idx
            st.session_state.ingested_filename = uploaded_file.name
            st.session_state.retrieval_results = None
            st.session_state.eval_results = None

            progress_bar.progress(1.0)
            status_txt.success(f"Ingested **{len(chunks)}** chunks ✓")

    st.markdown("---")

    # ── Retrieval Settings ──
    st.markdown("### ⚙️ Retrieval Settings")
    top_k = st.slider("Top-K documents", min_value=1, max_value=10, value=5)
    rrf_k = st.slider("RRF constant (k)", min_value=10, max_value=100, value=60, step=5,
                       help="Higher k reduces the weight of top ranks. Default 60 is standard.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>BM25 · Dense Vector · RRF Fusion  \n"
        "Eval: RAGAS faithfulness + relevancy</small>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>⚡ HybridSearch Bench</h1>
        <p>RAG pipeline · BM25 + Dense Vector + Reciprocal Rank Fusion · RAGAS auto-evaluation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_query, tab_eval, tab_history = st.tabs(
    ["🔎 Query & Compare", "📊 Evaluation Dashboard", "📜 History"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query & Compare
# ══════════════════════════════════════════════════════════════════════════════

with tab_query:
    if st.session_state.chunks is None:
        st.info("👈  Upload and ingest a PDF in the sidebar to get started.")
    else:
        st.success(
            f"Corpus: **{st.session_state.ingested_filename}**  ·  "
            f"**{len(st.session_state.chunks)}** chunks indexed",
            icon="📚",
        )

        # ── Query Input ──
        with st.form("query_form", clear_on_submit=False):
            query = st.text_input(
                "Enter a question about your corpus",
                placeholder="e.g. What are the main EEG preprocessing steps?",
                value=st.session_state.last_query,
            )
            run_col, eval_col, gt_col = st.columns([1, 1, 2])
            with run_col:
                run_btn = st.form_submit_button("🔍 Retrieve", use_container_width=True, type="primary")
            with eval_col:
                eval_btn = st.form_submit_button("📊 Retrieve + Evaluate", use_container_width=True)
            with gt_col:
                ground_truth = st.text_input(
                    "Ground truth (optional, for context_precision)",
                    placeholder="Ideal answer text…",
                )

        if (run_btn or eval_btn) and query.strip():
            from retrieval import run_all_strategies

            st.session_state.last_query = query
            st.session_state.eval_results = None

            with st.spinner("Running all retrieval strategies…"):
                results = run_all_strategies(
                    query,
                    st.session_state.bm25_index,
                    st.session_state.chunks,
                    st.session_state.collection,
                    top_k=top_k,
                )
            st.session_state.retrieval_results = results

            # Auto-run evaluation if requested
            if eval_btn:
                from evaluation import evaluate_all_strategies

                eval_progress = st.progress(0.0)
                eval_status = st.empty()

                def _eval_cb(pct: float, msg: str = "") -> None:
                    eval_progress.progress(pct)
                    if msg:
                        eval_status.markdown(msg)

                with st.spinner("Running RAGAS evaluation (this takes ~30 s)…"):
                    eval_res = evaluate_all_strategies(
                        query,
                        results,
                        ground_truth=ground_truth.strip() or None,
                        progress_callback=_eval_cb,
                    )
                    st.session_state.eval_results = eval_res
                    eval_progress.progress(1.0)
                    eval_status.success("RAGAS evaluation complete ✓")

                    # Save to history
                    st.session_state.eval_history.append(
                        {
                            "query": query,
                            "results": results,
                            "eval": eval_res,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                    )

        # ── Results columns ──
        if st.session_state.retrieval_results:
            results = st.session_state.retrieval_results
            eval_res = st.session_state.eval_results

            st.markdown("### Retrieved Documents")

            cols = st.columns(3)
            strategy_labels = {
                "bm25":   "📄 BM25",
                "vector": "🔮 Vector",
                "hybrid": "⚡ Hybrid (RRF)",
            }

            for col, strategy in zip(cols, ["bm25", "vector", "hybrid"]):
                with col:
                    colour = COLOURS[strategy]
                    st.markdown(
                        f"<div style='color:{colour}; font-weight:700; font-size:1.05rem;'>"
                        f"{strategy_labels[strategy]}</div>",
                        unsafe_allow_html=True,
                    )

                    # Show generated answer if eval ran
                    if eval_res and strategy in eval_res:
                        st.markdown(
                            f"<div class='answer-box'><b>Answer:</b><br>{eval_res[strategy]['answer']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("")

                        # Metric pills
                        pills = ""
                        for mk, ml in METRIC_LABELS.items():
                            if mk in eval_res[strategy]:
                                v = eval_res[strategy][mk]
                                pills += (
                                    f"<span class='metric-pill'>{ml[:4]}: {v:.2f}</span>"
                                )
                        st.markdown(pills, unsafe_allow_html=True)
                        st.markdown("")

                    # Chunk cards
                    for doc in results[strategy]:
                        score_label = (
                            f"RRF={doc['rrf_score']:.4f}"
                            if strategy == "hybrid" and doc["rrf_score"]
                            else f"Score={doc['score']:.4f}"
                        )
                        with st.expander(
                            f"#{doc['rank']}  ·  p.{doc['metadata'].get('page','?')}  ·  {score_label}"
                        ):
                            st.markdown(doc["text"])
                            st.markdown(
                                f"<span class='chunk-badge'>chunk_id: {doc['metadata']['chunk_id']}</span>",
                                unsafe_allow_html=True,
                            )

            # ── Score distribution bar chart ──
            if results:
                st.markdown("---")
                st.markdown("### Score Distribution Across Retrieved Chunks")

                rows = []
                for strategy, docs in results.items():
                    for d in docs:
                        rows.append(
                            {
                                "Strategy": strategy.upper(),
                                "Rank": f"#{d['rank']}",
                                "Score": d["score"],
                                "Chunk": d["text"][:60] + "…",
                            }
                        )

                df_scores = pd.DataFrame(rows)
                fig_scores = px.bar(
                    df_scores,
                    x="Rank",
                    y="Score",
                    color="Strategy",
                    barmode="group",
                    color_discrete_map={
                        "BM25": COLOURS["bm25"],
                        "VECTOR": COLOURS["vector"],
                        "HYBRID": COLOURS["hybrid"],
                    },
                    height=340,
                    hover_data=["Chunk"],
                )
                fig_scores.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5e7eb",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                fig_scores.update_xaxes(showgrid=False)
                fig_scores.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig_scores, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    eval_res = st.session_state.eval_results

    if eval_res is None:
        st.info(
            "Run a query with **Retrieve + Evaluate** in the Query tab to populate this dashboard."
        )
    else:
        st.markdown("### RAGAS Metrics — Strategy Comparison")

        # ── Metrics summary table ──
        table_rows = []
        for strategy, data in eval_res.items():
            row = {"Strategy": f"{_strategy_icon(strategy)} {strategy.upper()}"}
            for mk in METRIC_LABELS:
                row[METRIC_LABELS[mk]] = data.get(mk, float("nan"))
            table_rows.append(row)

        df_metrics = pd.DataFrame(table_rows).set_index("Strategy")
        st.dataframe(
            df_metrics.style.format("{:.3f}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1, axis=None
            ),
            use_container_width=True,
        )

        st.markdown("---")

        # ── Grouped bar chart ──
        metric_cols = [ml for ml in METRIC_LABELS.values() if ml in df_metrics.columns]
        if metric_cols:
            fig_metrics = go.Figure()
            for strategy in eval_res:
                fig_metrics.add_trace(
                    go.Bar(
                        name=strategy.upper(),
                        x=metric_cols,
                        y=[eval_res[strategy].get(mk, 0) for mk in METRIC_LABELS if METRIC_LABELS[mk] in metric_cols],
                        marker_color=COLOURS[strategy],
                    )
                )

            fig_metrics.update_layout(
                barmode="group",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)"),
            )
            fig_metrics.update_xaxes(showgrid=False)
            st.plotly_chart(fig_metrics, use_container_width=True)

        # ── Radar chart ──
        radar_metrics = [ml for ml in METRIC_LABELS.values() if ml in df_metrics.columns]
        if len(radar_metrics) >= 3:
            st.markdown("### Radar — Strategy Profiles")
            fig_radar = go.Figure()
            for strategy in eval_res:
                vals = [eval_res[strategy].get(mk, 0) for mk in METRIC_LABELS if METRIC_LABELS[mk] in radar_metrics]
                vals += [vals[0]]  # close the polygon
                categories = radar_metrics + [radar_metrics[0]]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=categories,
                        fill="toself",
                        name=strategy.upper(),
                        line_color=COLOURS[strategy],
                        fillcolor=COLOURS[strategy].replace(")", ", 0.15)").replace("rgb", "rgba")
                        if "rgb" in COLOURS[strategy]
                        else COLOURS[strategy] + "26",
                    )
                )

            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                ),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                margin=dict(l=20, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Generated answers side-by-side ──
        st.markdown("---")
        st.markdown("### Generated Answers")
        ans_cols = st.columns(3)
        for col, strategy in zip(ans_cols, eval_res):
            with col:
                colour = COLOURS[strategy]
                st.markdown(
                    f"<div style='color:{colour}; font-weight:700;'>"
                    f"{_strategy_icon(strategy)} {strategy.upper()}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='answer-box'>{eval_res[strategy].get('answer','—')}</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    history = st.session_state.eval_history

    if not history:
        st.info("Evaluation history will appear here once you run at least one query with evaluation.")
    else:
        st.markdown(f"### {len(history)} evaluated quer{'y' if len(history)==1 else 'ies'}")

        # ── Summary table across all queries ──
        rows = []
        for entry in history:
            for strategy, data in entry["eval"].items():
                rows.append(
                    {
                        "Time": entry["timestamp"],
                        "Query": entry["query"][:60] + ("…" if len(entry["query"]) > 60 else ""),
                        "Strategy": strategy.upper(),
                        "Faithfulness": data.get("faithfulness", float("nan")),
                        "Ans. Relevancy": data.get("answer_relevancy", float("nan")),
                        "Ctx. Precision": data.get("context_precision", float("nan")),
                    }
                )

        df_hist = pd.DataFrame(rows)
        st.dataframe(
            df_hist.style.format(
                {"Faithfulness": "{:.3f}", "Ans. Relevancy": "{:.3f}", "Ctx. Precision": "{:.3f}"}
            ).background_gradient(
                cmap="RdYlGn",
                subset=["Faithfulness", "Ans. Relevancy"],
                vmin=0,
                vmax=1,
            ),
            use_container_width=True,
            height=350,
        )

        # ── Trend line per metric per strategy (multi-query) ──
        if len(history) >= 2:
            st.markdown("---")
            st.markdown("### Metric Trends Over Queries")

            for metric_key, metric_label in METRIC_LABELS.items():
                fig_trend = go.Figure()
                for strategy in ["bm25", "vector", "hybrid"]:
                    y_vals = [
                        e["eval"].get(strategy, {}).get(metric_key, None) for e in history
                    ]
                    x_vals = [f"Q{i+1}" for i in range(len(history))]
                    valid = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
                    if not valid:
                        continue
                    xs, ys = zip(*valid)
                    fig_trend.add_trace(
                        go.Scatter(
                            x=list(xs),
                            y=list(ys),
                            name=strategy.upper(),
                            mode="lines+markers",
                            line=dict(color=COLOURS[strategy], width=2),
                            marker=dict(size=7),
                        )
                    )

                fig_trend.update_layout(
                    title=metric_label,
                    height=260,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5e7eb",
                    yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)"),
                    xaxis=dict(showgrid=False),
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        if st.button("🗑️ Clear history"):
            st.session_state.eval_history = []
            st.rerun()
