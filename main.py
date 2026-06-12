"""
app_web.py — Streamlit Web UI for the Air-Gapped RAG Engine.
Run:  streamlit run app_web.py

Key stability fixes vs original:
  - Model loaded ONCE into st.cache_resource, completely isolated from doc changes
  - Docs stored in st.session_state (not cache), re-indexed only on explicit button press
  - No @st.cache_data wrapping doc loading — avoids cache-conflict crashes on file add
  - .streamlit/config.toml disables file watcher on ./docs to stop auto-reloads
"""

import streamlit as st

st.set_page_config(
    page_title="Local RAG Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import hashlib
from pathlib import Path
from ignore.rag_engine import load_and_chunk_data, RetrievalEngine, load_model, generate_answer

DOCS_DIR   = "./docs"
MODEL_PATH = "./LocalModels"
TOP_N      = 3

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #212121 !important;
    color: #ececec !important;
}
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2e2e2e;
}
[data-testid="stChatMessage"] {
    background-color: #2a2a2a !important;
    border-radius: 10px;
    padding: 12px 16px !important;
    margin-bottom: 8px;
    border: 1px solid #333;
}
[data-testid="stChatInput"] textarea {
    background-color: #2f2f2f !important;
    color: #ececec !important;
    border: 1px solid #444 !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] * { color: #ccc !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #fff !important; }
[data-testid="stProgressBar"] > div { background-color: #10a37f !important; }
details {
    background-color: #1e1e1e !important;
    border-radius: 8px;
    border: 1px solid #333 !important;
}
summary { color: #aaa !important; }
hr { border-color: #333 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #212121; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND — model is FULLY isolated from document state.
# cached_load_model() is called once and never touched again when docs change.
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def cached_load_model(model_path: str):
    """Loads the LLM once. Never re-runs, even when documents change."""
    return load_model(model_path)


def docs_fingerprint(docs_dir: str) -> str:
    """
    Returns a hash of all .txt filenames + sizes in docs_dir.
    Used only to DISPLAY whether new files exist — not passed to any cache key.
    """
    p = Path(docs_dir)
    if not p.exists():
        return ""
    items = sorted((f.name, f.stat().st_size) for f in p.glob("*.txt"))
    return hashlib.md5(str(items).encode()).hexdigest()[:8]


def build_engine_from_disk() -> tuple:
    """Reads docs and builds a fresh RetrievalEngine. No Streamlit caching."""
    chunks = load_and_chunk_data(DOCS_DIR)
    engine = RetrievalEngine(chunks) if chunks else None
    return engine, chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — documents live here, NOT in st.cache_data
# This means adding a file NEVER invalidates the model cache.
# ═══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "messages":      [],
        "last_query":    "",
        "last_matches":  [],
        "engine":        None,
        "chunks":        [],
        "engine_error":  None,
        "loaded_fingerprint": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# Load documents on first run (engine not built yet)
if st.session_state.engine is None:
    try:
        eng, cks = build_engine_from_disk()
        st.session_state.engine = eng
        st.session_state.chunks = cks
        st.session_state.engine_error = None
        st.session_state.loaded_fingerprint = docs_fingerprint(DOCS_DIR)
    except Exception as exc:
        st.session_state.engine_error = str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 RAG Engine")
    st.caption("100% Offline · Corporate Proxy Immune")
    st.divider()

    # ── Model status (loads once, silently) ───────────────────────────────────
    st.markdown("### System")
    model_ok = False

    with st.spinner("Loading model…"):
        try:
            model_tokenizer, model_core = cached_load_model(MODEL_PATH)
            model_ok = True
        except Exception as exc:
            st.error(f"Model load failed:\n{exc}")
            st.info(
                "Place these files in ./LocalModels/:\n"
                "- config.json\n- model.safetensors\n"
                "- tokenizer.json\n- tokenizer_config.json"
            )

    if model_ok:
        st.success("✅ Model ready")

    # ── Document index status ─────────────────────────────────────────────────
    engine_ok = st.session_state.engine is not None
    chunk_count = len(st.session_state.chunks)

    if st.session_state.engine_error:
        st.error(f"Index error: {st.session_state.engine_error}")
    elif engine_ok:
        st.success(f"✅ {chunk_count} chunks indexed")
    else:
        st.warning("⚠️ No documents loaded — add .txt files to ./docs")

    st.divider()

    # ── New-file detector + Reload button ────────────────────────────────────
    current_fp = docs_fingerprint(DOCS_DIR)
    loaded_fp  = st.session_state.loaded_fingerprint

    if current_fp != loaded_fp and loaded_fp is not None:
        st.info("📄 New documents detected in ./docs")

    st.markdown("### Documents")
    if st.button("🔄  Reload documents", use_container_width=True):
        # Re-index docs WITHOUT touching the model cache
        try:
            eng, cks = build_engine_from_disk()
            st.session_state.engine = eng
            st.session_state.chunks = cks
            st.session_state.engine_error = None
            st.session_state.loaded_fingerprint = docs_fingerprint(DOCS_DIR)
            st.success(f"Reloaded — {len(cks)} chunks indexed")
        except Exception as exc:
            st.session_state.engine_error = str(exc)
            st.error(f"Reload failed: {exc}")

    # ── Document list ─────────────────────────────────────────────────────────
    with st.expander("📂 Indexed files", expanded=False):
        if st.session_state.chunks:
            seen = set()
            for c in st.session_state.chunks:
                src = c["source"]
                if src not in seen:
                    st.markdown(f"- `{src}`")
                    seen.add(src)
        else:
            st.caption("No documents indexed yet.")

    st.divider()

    # ── Live relevance panel ──────────────────────────────────────────────────
    st.markdown("### Last Query — Sources")
    relevance_placeholder = st.empty()

    st.divider()

    # ── Clear conversation ────────────────────────────────────────────────────
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.last_query = ""
        st.session_state.last_matches = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════

if not model_ok:
    st.error("⚠️  Model not ready — check the sidebar.")
    st.stop()

st.markdown("## Local Air-Gapped RAG")
st.caption("Ask questions about your documents. All processing happens on your machine.")
st.divider()


def render_relevance(matches, placeholder):
    if not matches:
        placeholder.caption("No query yet.")
        return
    with placeholder.container():
        max_score = matches[0][1] if matches else 1.0
        for i, (chunk, score) in enumerate(matches, 1):
            norm = min(score / max_score, 1.0) if max_score > 0 else 0.0
            st.markdown(f"**[{i}] {chunk['source']}**")
            st.markdown(
                f"<small style='color:#888'>Score: {score:.4f}</small>",
                unsafe_allow_html=True,
            )
            st.progress(norm)
            preview = chunk["content"][:120].replace("\n", " ")
            st.markdown(
                f"<small style='color:#666'>\"{preview}...\"</small>",
                unsafe_allow_html=True,
            )
            if i < len(matches):
                st.markdown("---")


# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Render last relevance scores
render_relevance(st.session_state.last_matches, relevance_placeholder)


# ── Chat input ────────────────────────────────────────────────────────────────
if user_prompt := st.chat_input("Ask something about your documents…"):

    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Rolling context for short follow-ups
    search_query = user_prompt
    if len(user_prompt.split()) <= 4 and st.session_state.last_query:
        search_query = f"{st.session_state.last_query} {user_prompt}"
    st.session_state.last_query = user_prompt

    # Guard: engine must exist
    if not st.session_state.engine:
        response = (
            "No documents are indexed yet. "
            "Add .txt files to ./docs and click **Reload documents** in the sidebar."
        )
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    # Retrieve
    matches = st.session_state.engine.search(search_query, top_n=TOP_N)
    st.session_state.last_matches = matches
    render_relevance(matches, relevance_placeholder)

    # Generate
    if not matches:
        response = (
            "No relevant content found for that question. "
            "Try rephrasing, or make sure the relevant .txt file is in ./docs "
            "and click **Reload documents**."
        )
    else:
        context = "\n\n".join(m[0]["content"] for m in matches)
        with st.spinner("Reading documents…"):
            response = generate_answer(
                user_prompt, context, model_tokenizer, model_core
            )

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})