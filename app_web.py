"""
app_web.py - Streamlit Web UI for the Air-Gapped RAG Engine + ML Predictor.
Run:  streamlit run app_web.py

Two tabs:
  1. RAG Chat   — ask questions about metallurgical documents (unchanged)
  2. Predictor  — input experiment params → get Ts / Tm / Tm-Ts predictions

This file is a thin UI layer.
  - Retrieval/generation logic lives in pipeline/rag_pipeline.py
  - Prediction logic lives in ml/predictor.py
"""

# ── Suppress harmless WinError 10054 asyncio noise (Windows ProactorEventLoop) ──
import sys, asyncio, logging

if sys.platform == "win32":
    _orig_exc_handler = asyncio.BaseEventLoop.call_exception_handler.__func__ \
        if hasattr(asyncio.BaseEventLoop.call_exception_handler, "__func__") else None

    def _quiet_exception_handler(self, ctx):
        exc = ctx.get("exception")
        if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
            return   # swallow the WinError 10054 noise
        if _orig_exc_handler:
            _orig_exc_handler(self, ctx)
        else:
            self.default_exception_handler(ctx)

    asyncio.BaseEventLoop.call_exception_handler = _quiet_exception_handler

import streamlit as st


st.set_page_config(
    page_title="Metallurgical RAG + Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import hashlib
from pathlib import Path

from config.paths import DOCS_DIR, LLM_MODEL_DIR, ML_MODEL_DIR
from config.retrieval import TOP_K
from pipeline.rag_pipeline import answer_question, build_engine, load_model

MODEL_PATH = LLM_MODEL_DIR
TOP_N = TOP_K


# ---------- CSS ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0f1117 !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0a0c10 !important;
    border-right: 1px solid #1e2230;
}
[data-testid="stChatMessage"] {
    background-color: #1a1d28 !important;
    border-radius: 12px;
    padding: 14px 18px !important;
    margin-bottom: 8px;
    border: 1px solid #252a3a;
}
[data-testid="stChatInput"] textarea {
    background-color: #1a1d28 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a3050 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] * { color: #b0b8cc !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8ecf4 !important; }
[data-testid="stProgressBar"] > div { background-color: #10a37f !important; }
details {
    background-color: #12141c !important;
    border-radius: 8px;
    border: 1px solid #252a3a !important;
}
summary { color: #8892a8 !important; }
hr { border-color: #1e2230 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a3050; border-radius: 4px; }

/* ── Tab styling ────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500;
    font-size: 1rem;
    color: #6b7394 !important;
    border-bottom: 2px solid transparent;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e0e0e0 !important;
    border-bottom-color: #10a37f !important;
}
[data-testid="stTabs"] button:hover {
    color: #c0c8e0 !important;
}

/* ── Prediction card styling ────────────────────── */
.pred-card {
    background: linear-gradient(135deg, #1a1d28 0%, #15171f 100%);
    border: 1px solid #252a3a;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pred-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.pred-card.ts::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.pred-card.tm::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.pred-card.tmts::before { background: linear-gradient(90deg, #10b981, #34d399); }
.pred-label {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.pred-card.ts .pred-label   { color: #60a5fa; }
.pred-card.tm .pred-label   { color: #fbbf24; }
.pred-card.tmts .pred-label { color: #34d399; }
.pred-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f2f8;
    line-height: 1.1;
}
.pred-unit {
    font-size: 0.9rem;
    font-weight: 400;
    color: #6b7394;
    margin-left: 4px;
}

/* ── Consistency badge ──────────────────────────── */
.consistency-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 8px;
}
.consistency-badge.good {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.consistency-badge.warn {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

/* ── Input section header ───────────────────────── */
.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7394;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2230;
}

/* ── Summary block ──────────────────────────────── */
.summary-block {
    background: #12141c;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 20px 24px;
    font-family: 'Inter', monospace;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #b0b8cc;
    white-space: pre-wrap;
}
/* ── Confidence badge ───────────────────────────── */
.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 8px;
}
.conf-badge.high   { background: rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); }
.conf-badge.medium { background: rgba(245,158,11,0.12);  color:#fbbf24; border:1px solid rgba(245,158,11,0.25); }
.conf-badge.low    { background: rgba(239,68,68,0.12);   color:#f87171; border:1px solid rgba(239,68,68,0.25);  }

/* ── Warning chips ──────────────────────────────── */
.warn-chip {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.22);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 6px;
}

/* ── History row ────────────────────────────────── */
.hist-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr auto;
    gap: 8px;
    align-items: center;
    padding: 8px 14px;
    background: #12141c;
    border: 1px solid #1e2230;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.83rem;
    color: #b0b8cc;
}
.hist-row .val { color: #e0e0e0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------- Model (loaded once, never reloaded) -------------------------------
@st.cache_resource(show_spinner=False)
def cached_load_model(model_path):
    return load_model(model_path)


# ---------- Predictor (loaded once) -------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_load_predictor(model_dir, model_type):
    from ml.predictor import Predictor
    return Predictor(model_dir=model_dir, model_type=model_type)


# ---------- Docs fingerprint --------------------------------------------------
def docs_fingerprint(docs_dir):
    p = Path(docs_dir)
    if not p.exists():
        return ""
    items = sorted(
        (f.name, f.stat().st_size, f.stat().st_mtime)
        for f in p.glob("*.txt")
    )
    return hashlib.md5(str(items).encode()).hexdigest()[:8]


# ---------- Session state init ------------------------------------------------
def init_session():
    defaults = {
        "messages":           [],
        "last_query":         "",
        "last_matches":       [],
        "engine":             None,
        "chunks":             [],
        "engine_error":       None,
        "loaded_fingerprint": None,
        "pred_results":       None,
        "pred_history":       [],   # list of recent prediction dicts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# First-run: load docs automatically
if st.session_state.engine is None:
    try:
        eng, cks = build_engine(DOCS_DIR)
        st.session_state.engine             = eng
        st.session_state.chunks             = cks
        st.session_state.engine_error       = None
        st.session_state.loaded_fingerprint = docs_fingerprint(DOCS_DIR)
    except Exception as exc:
        st.session_state.engine_error = str(exc)


# ---------- Sidebar -----------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 Metallurgical Assistant")
    st.caption("RAG + ML Prediction · 100% Offline")
    st.divider()

    # -- Model status --
    st.markdown("### System")
    model_ok = False
    with st.spinner("Loading LLM..."):
        try:
            model_tokenizer, model_core = cached_load_model(MODEL_PATH)
            model_ok = True
        except Exception as exc:
            st.error(f"LLM load failed:\n{exc}")
            st.info(
                "Place these files in ./LocalModels/:\n"
                "- config.json\n"
                "- model.safetensors\n"
                "- tokenizer.json\n"
                "- tokenizer_config.json"
            )

    if model_ok:
        st.success("LLM ready")

    # -- ML models status --
    ml_ok = False
    try:
        predictor_default = cached_load_predictor(ML_MODEL_DIR, "best")
        ml_ok = True
        st.success("ML models ready")
    except Exception as exc:
        st.warning(f"ML models not loaded: {exc}")

    # -- Document index status --
    engine_ok   = st.session_state.engine is not None
    chunk_count = len(st.session_state.chunks)

    if st.session_state.engine_error:
        st.error(f"Index error: {st.session_state.engine_error}")
    elif engine_ok:
        st.success(f"{chunk_count} chunks indexed")
    else:
        st.warning("No documents loaded — add .txt files below")

    st.divider()

    # -- Documents section --
    st.markdown("### Documents")

    current_fp = docs_fingerprint(DOCS_DIR)
    loaded_fp  = st.session_state.loaded_fingerprint
    if current_fp != loaded_fp and loaded_fp is not None:
        st.info("Files changed on disk — click Reload Documents")

    uploaded_files = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
        help="Files are saved to ./docs and indexed immediately.",
    )
    if uploaded_files:
        docs_path = Path(DOCS_DIR)
        docs_path.mkdir(parents=True, exist_ok=True)
        saved = []
        for uf in uploaded_files:
            dest = docs_path / uf.name
            dest.write_bytes(uf.read())
            saved.append(uf.name)
        try:
            eng, cks = build_engine(DOCS_DIR)
            st.session_state.engine             = eng
            st.session_state.chunks             = cks
            st.session_state.engine_error       = None
            st.session_state.loaded_fingerprint = docs_fingerprint(DOCS_DIR)
            st.success(f"Saved {len(saved)} file(s) — {len(cks)} chunks indexed")
        except Exception as exc:
            st.session_state.engine_error = str(exc)
            st.error(f"Index failed after upload: {exc}")

    if st.button("Reload Documents", use_container_width=True):
        try:
            eng, cks = build_engine(DOCS_DIR)
            st.session_state.engine             = eng
            st.session_state.chunks             = cks
            st.session_state.engine_error       = None
            st.session_state.loaded_fingerprint = docs_fingerprint(DOCS_DIR)
            st.success(f"Reloaded — {len(cks)} chunks indexed")
        except Exception as exc:
            st.session_state.engine_error = str(exc)
            st.error(f"Reload failed: {exc}")

    with st.expander("Indexed files", expanded=False):
        if st.session_state.chunks:
            seen = set()
            for c in st.session_state.chunks:
                src = c.source
                if src not in seen:
                    st.markdown(f"- `{src}`")
                    seen.add(src)
        else:
            st.caption("No documents indexed yet.")

    st.divider()

    # -- Retrieval diagnostics panel --
    st.markdown("### Last Query — Sources")
    relevance_placeholder = st.empty()

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_query   = ""
        st.session_state.last_matches = []
        st.rerun()


# ---------- Main area ---------------------------------------------------------
st.markdown("## 🔬 Metallurgical RAG + Predictor")
st.caption("Ask questions about your documents or predict softening–melting behaviour. Fully offline.")
st.divider()

tab_rag, tab_pred = st.tabs(["🔍 RAG Chat", "🧮 Predictor"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAG CHAT (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def render_relevance(matches, placeholder):
    """
    [rank]  source file  |  SECTION  |  score
    progress bar
    first 120 chars of chunk content
    """
    if not matches:
        placeholder.caption("No query yet.")
        return

    with placeholder.container():
        max_score = matches[0][1] if matches else 1.0
        for i, (chunk, score) in enumerate(matches, 1):
            norm    = min(score / max_score, 1.0) if max_score > 0 else 0.0
            source  = chunk.source
            section = chunk.section.upper()

            st.markdown(
                f"**[{i}] {source}**&nbsp;&nbsp;"
                f"<span style='color:#8892a8'>|&nbsp;&nbsp;"
                f"<code style='font-size:0.78em'>{section}</code>"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;{score:.4f}</span>",
                unsafe_allow_html=True,
            )
            st.progress(norm)

            preview = chunk.content[:120].replace("\n", " ")
            st.markdown(
                f"<small style='color:#6b7394'>{preview}…</small>",
                unsafe_allow_html=True,
            )
            if i < len(matches):
                st.markdown("---")


with tab_rag:
    if not model_ok:
        st.error("LLM not ready — check the sidebar.")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    render_relevance(st.session_state.last_matches, relevance_placeholder)

    # -- Chat input --
    if user_prompt := st.chat_input("Ask something about your documents..."):

        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Rolling context for short follow-ups
        search_query = user_prompt
        if len(user_prompt.split()) <= 4 and st.session_state.last_query:
            search_query = f"{st.session_state.last_query} {user_prompt}"
        st.session_state.last_query = user_prompt

        if not st.session_state.engine:
            response = (
                "No documents are indexed yet. "
                "Upload .txt files or click Reload Documents in the sidebar."
            )
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        with st.spinner("Reading documents..."):
            response, matches = answer_question(
                question=user_prompt,
                search_query=search_query,
                engine=st.session_state.engine,
                tokenizer=model_tokenizer,
                model=model_core,
                top_n=TOP_N,
            )

        st.session_state.last_matches = matches
        render_relevance(matches, relevance_placeholder)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_pred:
    if not ml_ok:
        st.error(
            "ML models not loaded. Ensure trained `.pkl` files exist in "
            f"`{ML_MODEL_DIR}` (run `python ml/train.py` first)."
        )
    else:
        # ── Two-column layout ─────────────────────────────────────────────
        col_input, col_spacer, col_result = st.columns([4, 0.5, 5.5])

        with col_input:
            st.markdown('<div class="section-header">Experiment Parameters</div>',
                        unsafe_allow_html=True)

            with st.form("predictor_form", clear_on_submit=False):
                # ── Chemistry ─────────────────────────────────────────
                st.markdown("**Chemistry**")
                chem_c1, chem_c2 = st.columns(2)
                with chem_c1:
                    tfe   = st.number_input("T Fe %",   value=57.62, format="%.2f", step=0.1)
                    feo   = st.number_input("FeO %",    value=6.65,  format="%.2f", step=0.1)
                    sio2  = st.number_input("SiO₂ %",   value=4.23,  format="%.2f", step=0.1)
                    cao   = st.number_input("CaO %",    value=6.93,  format="%.2f", step=0.1)
                with chem_c2:
                    al2o3 = st.number_input("Al₂O₃ %",  value=2.51,  format="%.2f", step=0.1)
                    mgo   = st.number_input("MgO %",    value=1.89,  format="%.2f", step=0.1)
                    basicity = st.number_input("Basicity (CaO/SiO₂)", value=1.638, format="%.3f", step=0.01)

                st.markdown("---")
                st.markdown("**Gas Atmosphere & Burden Mix**")
                
                input_mode = st.radio("Condition Input Mode", ["Describe in plain English", "Manual Strings"], index=0)
                
                cond_text = ""
                test_condition = ""
                burden = ""
                test_type = "SO"
                
                if input_mode == "Describe in plain English":
                    cond_text = st.text_area(
                        "Describe conditions",
                        placeholder="e.g. CO=40%, H2=8%, N2=52%. Burden composition is 70% Sinter and 30% Ore.",
                        value="CO=40%, N2=60%. Burden is 100% sinter."
                    )
                else:
                    test_condition = st.text_input("Atmosphere String", value="CO= 40% & N2=60%")
                    burden = st.text_input("Burden String", value="S1-70%+O1-30%")
                    test_type = st.selectbox("Test Type", ["SO", "SOP", "P"], index=0)

                st.markdown("---")

                model_choice = st.selectbox(
                    "Model",
                    options=["Best (auto)", "RandomForest", "XGBoost"],
                    index=0,
                    help="'Best' auto-selects the model with lowest RMSE per target",
                )

                st.markdown("")
                submitted = st.form_submit_button(
                    "⚡ Predict",
                    use_container_width=True,
                    type="primary",
                )

            if submitted:
                # Map UI choice to Predictor model_type
                model_type_map = {
                    "Best (auto)":   "best",
                    "RandomForest":  "RandomForest",
                    "XGBoost":       "XGBoost",
                }
                chosen_type = model_type_map[model_choice]

                chemistry = {
                    "T Fe %":  tfe,
                    "FeO %":   feo,
                    "SiO2 %":  sio2,
                    "CaO %":   cao,
                    "Al2O3 %": al2o3,
                    "MgO%":    mgo,
                    "Basicity": basicity,
                }

                try:
                    # Initialize PredictionPipeline with the chosen type
                    from pipeline.prediction_pipeline import PredictionPipeline
                    pipe = PredictionPipeline(model_dir=ML_MODEL_DIR, model_type=chosen_type)
                    
                    # Run the pipeline, passing model/tokenizer if LLM parsing is needed
                    result_obj = pipe.run(
                        chemistry=chemistry,
                        test_condition=test_condition if input_mode == "Manual Strings" else None,
                        burden=burden if input_mode == "Manual Strings" else None,
                        test_type=test_type if input_mode == "Manual Strings" else None,
                        condition_text=cond_text if input_mode == "Describe in plain English" else None,
                        tokenizer=model_tokenizer if input_mode == "Describe in plain English" else None,
                        model=model_core if input_mode == "Describe in plain English" else None,
                    )
                    
                    st.session_state.pred_results = {
                        "predictions":     result_obj.predictions,
                        "chemistry":       chemistry,
                        "model":           model_choice,
                        "inputs":          result_obj.inputs,
                        "parser_warnings": result_obj.parser_warnings,
                    }
                    # Append to rolling history (keep last 10)
                    import time as _time
                    hist_entry = {
                        "ts_val":   result_obj.predictions.get("Ts"),
                        "tm_val":   result_obj.predictions.get("Tm"),
                        "tmt_val":  result_obj.predictions.get("Tm-Ts"),
                        "conf":     result_obj.predictions.get("confidence", "?"),
                        "dist":     result_obj.predictions.get("distance", 999.0),
                        "model":    model_choice,
                        "burden":   result_obj.inputs.get("burden") or "N/A",
                        "cond":     result_obj.inputs.get("test_condition") or "N/A",
                        "ts_stamp": _time.strftime("%H:%M:%S"),
                    }
                    hist = st.session_state.pred_history
                    hist.insert(0, hist_entry)
                    st.session_state.pred_history = hist[:10]

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.session_state.pred_results = None

        # ── Results column ────────────────────────────────────────────────
        with col_result:
            pr = st.session_state.pred_results

            if pr is None:
                # Empty state
                st.markdown("")
                st.markdown("")
                st.markdown(
                    "<div style='text-align:center; padding: 60px 20px; "
                    "color: #4a506a;'>"
                    "<p style='font-size: 3rem; margin-bottom: 12px;'>🧪</p>"
                    "<p style='font-size: 1.1rem; font-weight: 500;'>"
                    "Enter chemistry parameters and click Predict</p>"
                    "<p style='font-size: 0.85rem; margin-top: 8px;'>"
                    "The model will estimate softening temperature (Ts), "
                    "melting temperature (Tm), and the melting interval (Tm−Ts).</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                preds = pr["predictions"]
                ts_val  = preds.get("Ts",    None)
                tm_val  = preds.get("Tm",    None)
                tmt_val = preds.get("Tm-Ts", None)

                st.markdown('<div class="section-header">Prediction Results</div>',
                            unsafe_allow_html=True)
                st.caption(f"Model: **{pr['model']}**")

                # ── Metric cards ──────────────────────────────────────
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(
                        f'<div class="pred-card ts">'
                        f'<div class="pred-label">Softening Temp (Ts)</div>'
                        f'<div class="pred-value">{ts_val}<span class="pred-unit">°C</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with mc2:
                    st.markdown(
                        f'<div class="pred-card tm">'
                        f'<div class="pred-label">Melting Temp (Tm)</div>'
                        f'<div class="pred-value">{tm_val}<span class="pred-unit">°C</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with mc3:
                    st.markdown(
                        f'<div class="pred-card tmts">'
                        f'<div class="pred-label">Melting Interval (Tm−Ts)</div>'
                        f'<div class="pred-value">{tmt_val}<span class="pred-unit">°C</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Consistency check ─────────────────────────────────
                if ts_val is not None and tm_val is not None and tmt_val is not None:
                    computed_diff = round(tm_val - ts_val, 1)
                    delta = abs(computed_diff - tmt_val)

                    st.markdown("")
                    if delta <= 15:
                        st.markdown(
                            f'<div style="text-align:center">'
                            f'<span class="consistency-badge good">'
                            f'✓ Consistent — Tm − Ts = {computed_diff}°C '
                            f'(predicted Tm-Ts = {tmt_val}°C, Δ = {delta}°C)'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="text-align:center">'
                            f'<span class="consistency-badge warn">'
                            f'⚠ Mismatch — Tm − Ts = {computed_diff}°C '
                            f'but predicted Tm-Ts = {tmt_val}°C (Δ = {delta}°C)'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )

                # ── Confidence badge ──────────────────────────────────
                confidence = preds.get("confidence", "medium")
                nn_dist    = preds.get("distance", 999.0)
                conf_icons  = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                conf_labels = {
                    "high":   "High confidence",
                    "medium": "Medium confidence",
                    "low":    "Low confidence — extrapolation risk",
                }
                st.markdown("")
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<span class="conf-badge {confidence}">'
                    f'{conf_icons.get(confidence, "")} '
                    f'{conf_labels.get(confidence, confidence)}'
                    f'&nbsp;·&nbsp; NN dist = {nn_dist}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

                # ── Parser warnings ───────────────────────────────────
                parser_warnings = pr.get("parser_warnings", [])
                if parser_warnings:
                    st.markdown("")
                    st.markdown(
                        '<div class="section-header">Parser Warnings</div>',
                        unsafe_allow_html=True,
                    )
                    for w in parser_warnings:
                        st.markdown(
                            f'<div class="warn-chip">⚠ {w}</div>',
                            unsafe_allow_html=True,
                        )

                # ── Parsed condition preview (NL mode) ────────────────
                parsed = pr["inputs"].get("parsed_condition")
                if parsed:
                    with st.expander("🔍 Parsed Condition Detail", expanded=False):
                        a1, a2, a3 = st.columns(3)
                        a1.metric("CO %",  f"{parsed.get('CO_pct', 0):.1f}")
                        a2.metric("H₂ %",  f"{parsed.get('H2_pct', 0):.1f}")
                        a3.metric("N₂ %",  f"{parsed.get('N2_pct', 0):.1f}")
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Sinter %", f"{parsed.get('sinter_pct', 0):.1f}")
                        b2.metric("Ore %",    f"{parsed.get('ore_pct',    0):.1f}")
                        b3.metric("Pellet %", f"{parsed.get('pellet_pct', 0):.1f}")
                        b4.metric("Other %",  f"{parsed.get('other_pct',  0):.1f}")

                # ── Input summary ─────────────────────────────────────
                st.markdown("")
                st.markdown('<div class="section-header">Input Summary</div>',
                            unsafe_allow_html=True)

                chem = pr["chemistry"]
                inputs = pr.get("inputs", {})
                cond_val   = inputs.get("test_condition") or "N/A"
                burden_val = inputs.get("burden")         or "N/A"
                type_val   = inputs.get("test_type")      or "N/A"

                summary_lines = [
                    f"  Model        : {pr['model']}",
                    f"  Atmosphere   : {cond_val}",
                    f"  Burden       : {burden_val}",
                    f"  Test Type    : {type_val}",
                    "",
                    "  Chemistry:",
                ]
                for col_name, val in chem.items():
                    summary_lines.append(f"    {col_name:<12}: {val}")

                summary_lines += [
                    "",
                    "  Predictions:",
                    f"    Ts     = {ts_val} °C",
                    f"    Tm     = {tm_val} °C",
                    f"    Tm-Ts  = {tmt_val} °C",
                    f"    Conf.  = {confidence}  (NN dist = {nn_dist})",
                ]

                st.markdown(
                    f'<div class="summary-block">{chr(10).join(summary_lines)}</div>',
                    unsafe_allow_html=True,
                )

# ── Prediction history panel ───────────────────────────────────────────────────
with tab_pred:
    hist = st.session_state.get("pred_history", [])
    if hist:
        st.markdown("")
        with st.expander(f"📋 Prediction History  ({len(hist)} runs)", expanded=False):
            conf_colors = {"high": "#34d399", "medium": "#fbbf24", "low": "#f87171"}
            for h in hist:
                color = conf_colors.get(h["conf"], "#8892a8")
                st.markdown(
                    f'<div class="hist-row">'
                    f'<span>{h["ts_stamp"]}</span>'
                    f'<span>Ts=<span class="val">{h["ts_val"]}°C</span></span>'
                    f'<span>Tm=<span class="val">{h["tm_val"]}°C</span></span>'
                    f'<span>Tm-Ts=<span class="val">{h["tmt_val"]}°C</span></span>'
                    f'<span style="color:{color};font-weight:600;">{h["conf"].upper()}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear History", key="clear_hist"):
                st.session_state.pred_history = []
                st.rerun()

