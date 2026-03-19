import streamlit as st
import time, json, os
from io import BytesIO
from pypdf import PdfReader

from src.retrieval import (
    create_vector_store, generate_answer,
    load_vector_store, NOT_IN_BOOK_MARKER,
)
from src.text_processing import clean_text, chunk_text
from src.image_extractor import extract_images_from_pdf, find_relevant_images
from src import cache as answer_cache


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_and_page_map(pdf_bytes: bytes):
    """Returns (full_text, {page_num: text}) for page-level image matching."""
    reader   = PdfReader(BytesIO(pdf_bytes))
    full     = ""
    page_map = {}
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        full += t + "\n"
        page_map[i] = t
    return full, page_map


def find_relevant_page_nums(context: str, page_map: dict) -> list[int]:
    """Return page numbers whose text overlaps significantly with the context."""
    ctx_words = set(context.lower().split())
    scored    = []
    for pnum, ptext in page_map.items():
        overlap = len(ctx_words & set(ptext.lower().split()))
        if overlap > 5:
            scored.append((overlap, pnum))
    scored.sort(reverse=True)
    return [p for _, p in scored[:3]]



def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{int(ms)}ms"


# ─────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduTutor",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* Force sidebar to always be visible and block collapse behavior */
.st-emotion-cache-18ni7ap, /* Sidebar wrapper */
.st-emotion-cache-vk3wp9, /* Alternative sidebar class */
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
    position: relative !important;
    left: auto !important;
    transform: none !important;
}

/* Hide all collapse/toggle buttons */
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Toggle sidebar"],
button[aria-label="Collapse sidebar"],
[data-testid="collapseSidebarButton"],
.st-emotion-cache-1kyxreq > button,
header > * button:first-child,
nav button {
    display: none !important;
}

/* Ensure sidebar is never hidden by state */
body.sidebar-closed [data-testid="stSidebar"],
body.sidebar-closed section[data-testid="stSidebar"],
[data-testid="stSidebar"][style*="display: none"],
[data-testid="stSidebar"][style*="left: -"],
[data-testid="stSidebar"][style*="transform: translate"] {
    display: flex !important;
    visibility: visible !important;
    left: 0 !important;
    transform: none !important;
    width: 340px !important;
}

/* Main app container - ensure it accounts for sidebar */
[data-testid="stAppViewContainer"] {
    margin-left: 0 !important;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important; font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }

[data-testid="stSidebar"] {
    background: #13151c !important; border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] * { color: #c9cfe0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #e8ecf5 !important; font-weight: 600; letter-spacing: -0.02em;
}
[data-testid="stFileUploader"] {
    background: #1a1d28 !important; border: 1.5px dashed #2e3348 !important;
    border-radius: 12px !important; padding: 8px !important;
}
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem 140px !important; max-width: 860px !important; margin: 0 auto !important;
}
h1 { color: #e8ecf5 !important; font-size: 1.6rem !important; font-weight: 600 !important;
     letter-spacing: -0.03em !important; margin-bottom: 0.25rem !important; }
h3 { color: #8892aa !important; font-size: 0.72rem !important; font-weight: 500 !important;
     letter-spacing: 0.12em !important; text-transform: uppercase !important;
     margin: 1.8rem 0 0.8rem !important; }

.chat-wrap { display: flex; flex-direction: column; gap: 20px; margin-bottom: 1.5rem; }

.bubble { max-width: 82%; padding: 13px 18px; border-radius: 18px;
          font-size: 0.93rem; line-height: 1.65; word-break: break-word;
          animation: popIn .25s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes popIn {
    from { opacity: 0; transform: scale(.92) translateY(6px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.bubble-user { background: #2f4de0; color: #eef1ff; align-self: flex-end;
               border-bottom-right-radius: 5px; box-shadow: 0 4px 20px rgba(47,77,224,.35); }
.bubble-ai   { background: #1a1d28; color: #c9cfe0; align-self: flex-start;
               border-bottom-left-radius: 5px; border: 1px solid #232740; }
.bubble-oob  { background: #1f1520; color: #c9a0c0; align-self: flex-start;
               border-bottom-left-radius: 5px; border: 1px solid #3a1f35; }
.bubble-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase; margin-bottom: 5px; opacity: 0.55; }

.metrics-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.metric-pill { background: #0d0f14; border: 1px solid #1e2130; border-radius: 8px;
               padding: 4px 10px; font-size: 0.7rem; color: #8892aa; font-family: 'DM Mono', monospace; }
.metric-pill b { color: #c9cfe0; }

.cache-badge { display: inline-block; background: #0f1e14; border: 1px solid #1D9E75;
               border-radius: 6px; padding: 2px 8px; font-size: 0.68rem; color: #1D9E75;
               margin-left: 6px; vertical-align: middle; }

.empty-state { text-align: center; padding: 3rem 1rem; color: #3a3f55; }
.empty-state .icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
.empty-state p { font-size: 0.88rem; line-height: 1.6; }

[data-testid="stBottom"] {
    background: #0d0f14 !important; border-top: 1px solid #1e2130 !important;
    padding: 14px 2.5rem 18px !important;
}
[data-testid="stChatInput"] textarea {
    background: #13151c !important; border: 1px solid #1e2130 !important;
    border-radius: 16px !important; color: #e8ecf5 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
    padding: 14px 18px !important; box-shadow: none !important; resize: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3a3f55 !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #2f4de0 !important; box-shadow: 0 0 0 3px rgba(47,77,224,.2) !important;
}
[data-testid="stChatInputSubmitButton"] button:not([disabled]):not([aria-disabled="true"]) {
    background: #1d6cf0 !important; border-radius: 10px !important;
}
hr { border-color: #1e2130 !important; }

/* ═════════════════════════════════════════════════════════════════════════════ */
/* LIGHT MODE SUPPORT - Automatic detection and proper contrast */
/* ═════════════════════════════════════════════════════════════════════════════ */
@media (prefers-color-scheme: light) {
    html, body, [data-testid="stAppViewContainer"] {
        background: #f8f9fa !important; color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] {
        background: #ffffff !important; border-right: 1px solid #e5e7eb !important;
    }
    
    [data-testid="stSidebar"] * { color: #374151 !important; }
    
    [data-testid="stSidebar"] button { color: #1f2937 !important; }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1f2937 !important; font-weight: 600; letter-spacing: -0.02em;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 2rem 2.5rem 140px !important; max-width: 860px !important; margin: 0 auto !important;
    }
    
    h1 { color: #1f2937 !important; font-size: 1.6rem !important; font-weight: 600 !important;
         letter-spacing: -0.03em !important; margin-bottom: 0.25rem !important; }
    
    h3 { color: #6b7280 !important; font-size: 0.72rem !important; font-weight: 500 !important;
         letter-spacing: 0.12em !important; text-transform: uppercase !important;
         margin: 1.8rem 0 0.8rem !important; }
    
    [data-testid="stFileUploader"] {
        background: #f5f7fa !important; border: 1.5px dashed #d1d5db !important;
        border-radius: 12px !important; padding: 8px !important;
    }
    
    .bubble-user { background: #3b82f6; color: #ffffff; align-self: flex-end;
                   border-bottom-right-radius: 5px; box-shadow: 0 4px 20px rgba(59,130,246,.25); }
    
    .bubble-ai   { background: #f3f4f6; color: #1f2937; align-self: flex-start;
                   border-bottom-left-radius: 5px; border: 1px solid #e5e7eb; }
    
    .bubble-oob  { background: #fdf2f8; color: #ec4899; align-self: flex-start;
                   border-bottom-left-radius: 5px; border: 1px solid #f5d4e6; }
    
    .bubble-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
                    text-transform: uppercase; margin-bottom: 5px; opacity: 0.6; color: #6b7280; }
    
    .metrics-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    
    .metric-pill { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
                   padding: 4px 10px; font-size: 0.7rem; color: #6b7280; font-family: 'DM Mono', monospace; }
    
    .metric-pill b { color: #374151; }
    
    .cache-badge { display: inline-block; background: #ecfdf5; border: 1px solid #10b981;
                   border-radius: 6px; padding: 2px 8px; font-size: 0.68rem; color: #059669;
                   margin-left: 6px; vertical-align: middle; }
    
    .empty-state { text-align: center; padding: 3rem 1rem; color: #9ca3af; }
    
    .empty-state .icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
    
    .empty-state p { font-size: 0.88rem; line-height: 1.6; color: #6b7280; }
    
    [data-testid="stBottom"] {
        background: #f8f9fa !important; border-top: 1px solid #e5e7eb !important;
        padding: 14px 2.5rem 18px !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important; border: 1px solid #d1d5db !important;
        border-radius: 16px !important; color: #1f2937 !important;
        font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
        padding: 14px 18px !important; box-shadow: none !important; resize: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; }
    
    [data-testid="stChatInput"] textarea:focus {
        border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
    }
    
    [data-testid="stChatInputSubmitButton"] button:not([disabled]):not([aria-disabled="true"]) {
        background: #3b82f6 !important; border-radius: 10px !important; color: #ffffff !important;
    }
    
    hr { border-color: #e5e7eb !important; }
}

/* ═════════════════════════════════════════════════════════════════════════════ */
/* SIDEBAR FIXES - Make sidebar permanently visible and disable collapse */
/* ═════════════════════════════════════════════════════════════════════════════ */

/* Force sidebar visibility in all states */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    position: relative !important;
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
    left: 0 !important;
    right: auto !important;
    transform: translateX(0) !important;
    opacity: 1 !important;
    z-index: 100 !important;
    transition: none !important;
}

/* Override collapsed state */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    width: 340px !important;
    min-width: 340px !important;
}

/* Hide all collapse/expand mechanisms */
button[aria-label*="sidebar"],
button[aria-label*="Sidebar"],
button[aria-label*="Close"],
button[aria-label*="Open"],
button[aria-label*="Toggle"],
[data-testid="collapseSidebarButton"],
[data-testid="stSidebar"] ~ button,
header button:first-of-type,
.st-emotion-cache-1kyxreq button,
nav button {
    display: none !important;
}

/* Prevent any animation that could hide the sidebar */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
.st-emotion-cache-18ni7ap,
.st-emotion-cache-vk3wp9 {
    animation: none !important;
    transition: none !important;
}

/* Ensure layout uses flexbox and sidebar stays visible */
.stApp {
    display: flex !important;
    flex-direction: row !important;
}

.stApp > section:first-of-type {
    display: block !important;
}

/* Prevent sidebar from being pushed off-screen */
@media (max-width: 1024px) {
    [data-testid="stSidebar"] {
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        width: 340px !important;
        z-index: 999999 !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        margin-left: 340px !important;
    }
}
</style>
<script>
// Prevent sidebar from collapsing - Active monitoring and enforcement
(function() {
    const preventCollapse = () => {
        // Find sidebar element
        const sidebar = document.querySelector('[data-testid="stSidebar"]') || 
                       document.querySelector('section[data-testid="stSidebar"]');
        
        if (sidebar) {
            // Force visibility
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.style.width = '340px';
            sidebar.style.left = '0';
            sidebar.style.position = 'relative';
            sidebar.style.transform = 'none';
            sidebar.style.opacity = '1';
        }
        
        // Hide collapse buttons
        const buttons = document.querySelectorAll('button[aria-label*="sidebar"], button[aria-label*="Sidebar"], button[aria-label*="Close"], button[aria-label*="Open"], button[aria-label*="Toggle"]');
        buttons.forEach(btn => btn.style.display = 'none');
    };
    
    // Run immediately
    preventCollapse();
    
    // Run periodically to counter any dynamic changes
    setInterval(preventCollapse, 500);
    
    // Also watch for attribute/style changes
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        const observer = new MutationObserver(preventCollapse);
        observer.observe(sidebar, {
            attributes: true,
            attributeFilter: ['style', 'class', 'aria-expanded'],
            subtree: false
        });
    }
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("messages",         []),
    ("index",            None),
    ("chunks",           None),
    ("source_label",     ""),
    ("uploaded_filename",None),
    ("pdf_bytes",        None),
    ("page_map",         {}),
    ("pdf_images",       []),
    ("active_pdfs",      {}),   # {name: {index, chunks, page_map, images}}
    ("selected_source",  "📚 Pre-loaded textbook"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-built vector store on first run
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["index"] is None:
    try:
        index, chunks = load_vector_store()
        st.session_state["index"]        = index
        st.session_state["chunks"]       = chunks
        st.session_state["source_label"] = f"📚 Pre-loaded textbook ({len(chunks)} chunks)"
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 EduTutor")
    st.markdown("---")

    if st.session_state["index"] is not None:
        st.markdown(
            f"<div style='font-size:.78rem;color:#4f6ef7;margin-bottom:.5rem;'>"
            f"✅ {st.session_state['source_label']}</div>",
            unsafe_allow_html=True,
        )

    # ── Multi-PDF upload ──────────────────────────────────────────────────────
    st.markdown("### Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        newly_added = None
        for uploaded_pdf in uploaded_files:
            name = uploaded_pdf.name
            if name not in st.session_state["active_pdfs"]:
                with st.spinner(f"Processing {name}…"):
                    uploaded_pdf.seek(0)
                    pdf_bytes = uploaded_pdf.read()

                    text, page_map = extract_text_and_page_map(pdf_bytes)
                    cleaned        = clean_text(text)
                    chunks         = chunk_text(cleaned)
                    index          = create_vector_store(chunks)
                    images         = extract_images_from_pdf(pdf_bytes)

                    st.session_state["active_pdfs"][name] = {
                        "index":    index,
                        "chunks":   chunks,
                        "page_map": page_map,
                        "images":   images,
                    }
                st.success(f"✅ {name} — {len(chunks)} chunks, {len(images)} images")
                newly_added = name

        # Auto-switch to the newly uploaded PDF immediately
        if newly_added:
            entry = st.session_state["active_pdfs"][newly_added]
            st.session_state["index"]             = entry["index"]
            st.session_state["chunks"]            = entry["chunks"]
            st.session_state["page_map"]          = entry["page_map"]
            st.session_state["pdf_images"]        = entry["images"]
            st.session_state["source_label"]      = f"📄 {newly_added}"
            st.session_state["uploaded_filename"] = newly_added
            st.session_state["selected_source"]   = newly_added
            st.session_state["messages"]          = []

    # ── PDF selector ─────────────────────────────────────────────────────────
    active  = st.session_state["active_pdfs"]
    options = ["📚 Pre-loaded textbook"] + list(active.keys())

    if len(options) > 1:
        st.markdown("### Active source")

        # Use persisted selection so rerender doesn't reset to index 0
        current = st.session_state.get("selected_source", "📚 Pre-loaded textbook")
        if current not in options:
            current = options[0]
        default_idx = options.index(current)

        chosen = st.selectbox(
            "Select textbook", options,
            index=default_idx,
            label_visibility="collapsed",
            key="source_selectbox",
        )
        st.session_state["selected_source"] = chosen

        if chosen != "📚 Pre-loaded textbook" and chosen in active:
            entry = active[chosen]
            st.session_state["index"]             = entry["index"]
            st.session_state["chunks"]            = entry["chunks"]
            st.session_state["page_map"]          = entry["page_map"]
            st.session_state["pdf_images"]        = entry["images"]
            st.session_state["source_label"]      = f"📄 {chosen}"
            st.session_state["uploaded_filename"] = chosen
        else:
            # Revert to pre-built
            try:
                index, chunks = load_vector_store()
                st.session_state["index"]             = index
                st.session_state["chunks"]            = chunks
                st.session_state["page_map"]          = {}
                st.session_state["pdf_images"]        = []
                st.session_state["source_label"]      = f"📚 Pre-loaded textbook ({len(chunks)} chunks)"
                st.session_state["uploaded_filename"] = None
            except FileNotFoundError:
                pass

    st.markdown("---")

    # ── Cache stats ───────────────────────────────────────────────────────────
    stats = answer_cache.stats()
    st.markdown(
        f"<div style='font-size:.75rem;color:#3a3f55;margin-bottom:8px;'>"
        f"💾 Cache: <b style='color:#4f6ef7'>{stats['total_entries']}</b> saved answers</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("� Clear cache", use_container_width=True):
            try:
                answer_cache.clear()
                st.success("✅ Cache cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")

    if st.button("🔄 Reload", use_container_width=True):
        try:
            index, chunks = load_vector_store()
            st.session_state.update({
                "index": index, "chunks": chunks,
                "page_map": {}, "pdf_images": [],
                "source_label": f"📚 Pre-loaded textbook ({len(chunks)} chunks)",
                "uploaded_filename": None, "messages": [],
            })
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────
st.title("Ask your textbook anything")

messages = st.session_state["messages"]
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not messages:
    no_store = st.session_state["index"] is None
    st.markdown(
        f"""<div class="empty-state">
            <div class="icon">{'⚠️' if no_store else '💬'}</div>
            <p>{'No textbook loaded. Upload a PDF in the sidebar.' if no_store
                else 'Textbook ready!<br>Ask any question — answers come only from the book.'}</p>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        role = msg["role"]

        if role == "user":
            st.markdown(
                f"""<div style="display:flex;justify-content:flex-end;">
                  <div class="bubble bubble-user">
                    <div class="bubble-label">You</div>{msg["text"]}
                  </div></div>""",
                unsafe_allow_html=True,
            )

        elif role == "ai":
            is_oob = msg.get("out_of_book", False)
            bubble_class = "bubble-oob" if is_oob else "bubble-ai"
            m = msg.get("metrics", {})

            cache_html = '<span class="cache-badge">⚡ cached</span>' if m.get("from_cache") else ""

            if is_oob:
                body = """<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="font-size:1.2rem;">📕</span>
                    <b style="color:#c9a0c0;">Content not found in the book</b>
                  </div>
                  <div style="font-size:0.85rem;opacity:0.75;">
                    This topic does not appear in the uploaded textbook.<br>
                    Try rephrasing, or check if a different chapter covers it.
                  </div>"""
            else:
                body = msg["text"]

            # Metrics pills
            pills = ""
            if m:
                pills = f"""
                <div class="metrics-row">
                  <div class="metric-pill">⏱ <b>{fmt_ms(m.get('total_time_ms',0))}</b> total</div>
                  <div class="metric-pill">🔍 <b>{fmt_ms(m.get('retrieval_time_ms',0))}</b> retrieval</div>
                  <div class="metric-pill">🤖 <b>{fmt_ms(m.get('llm_time_ms',0))}</b> LLM</div>
                  <div class="metric-pill">📄 <b>{m.get('chunks_scanned',0)}</b> chunks in book</div>
                  <div class="metric-pill">📌 <b>{m.get('chunks_retrieved',0)}</b> retrieved</div>
                  <div class="metric-pill">✂️ <b>{m.get('chunks_after_compress',0)}</b> after compress</div>
                  <div class="metric-pill">🔁 <b>{m.get('epochs',1)}</b> epoch(s)</div>
                </div>"""

            st.markdown(
                f"""<div style="display:flex;justify-content:flex-start;">
                  <div class="bubble {bubble_class}" style="max-width:88%;">
                    <div class="bubble-label">✦ Tutor {cache_html}</div>
                    {body}{pills}
                  </div></div>""",
                unsafe_allow_html=True,
            )

            # Images (shown outside bubble as Streamlit native images)
            for img_data in msg.get("images", []):
                import base64
                b64 = img_data["b64"]
                st.markdown(
                    f"""<div style="margin-left:16px;margin-top:8px;">
                      <div style="font-size:0.7rem;color:#3a3f55;margin-bottom:4px;">
                        📷 From page {img_data['page']}</div>
                      <img src="data:image/png;base64,{b64}"
                           style="max-width:320px;border-radius:8px;border:1px solid #1e2130;" />
                    </div>""",
                    unsafe_allow_html=True,
                )

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat input + answer pipeline
# ─────────────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your textbook…")

if question:
    if st.session_state["index"] is None:
        st.warning("⬅  Please upload a PDF first.")
        st.stop()

    question = question.strip()
    st.session_state["messages"].append({"role": "user", "text": question})

    pdf_name = st.session_state.get("uploaded_filename") or "preloaded"

    # ── Check cache first ─────────────────────────────────────────────────────
    cached = answer_cache.get(question, pdf_name)

    if cached:
        answer  = cached["answer"]
        metrics = cached["metrics"]
        metrics["from_cache"] = True
        images  = []
    else:
        with st.spinner("Searching the textbook…"):
            answer, metrics = generate_answer(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
            )

        # ── Find relevant images ──────────────────────────────────────────────
        images = []
        if answer != NOT_IN_BOOK_MARKER and st.session_state.get("pdf_images"):
            from src.context_compression import compress_chunks
            context, _ = compress_chunks(question, st.session_state["chunks"][:metrics["chunks_retrieved"]], return_count=True)
            page_nums   = find_relevant_page_nums(context, st.session_state.get("page_map", {}))
            images      = find_relevant_images(st.session_state["pdf_images"], page_nums)

        # ── Cache the result ──────────────────────────────────────────────────
        if answer != NOT_IN_BOOK_MARKER:
            answer_cache.set(question, pdf_name, answer, metrics)

    is_oob = (answer == NOT_IN_BOOK_MARKER)

    st.session_state["messages"].append({
        "role":        "ai",
        "text":        answer if not is_oob else "",
        "out_of_book": is_oob,
        "metrics":     metrics,
        "images":      images,
    })
    st.rerun()