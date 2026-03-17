import streamlit as st
from io import BytesIO
from pypdf import PdfReader

from retrieval import create_vector_store, generate_answer
from text_processing import clean_text, chunk_text


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF present as bytes."""
    reader = PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A", layout="wide", page_icon="📄")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] * { color: #c9cfe0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e8ecf5 !important;
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1d28 !important;
    border: 1.5px dashed #2e3348 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: #4f6ef7 !important; }

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

/* ── Main area — extra bottom padding so last bubble isn't hidden behind input ── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem 120px !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}

/* ── Page title ── */
h1 {
    color: #e8ecf5 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.25rem !important;
}

/* ── Section header ── */
h3 {
    color: #8892aa !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 1.8rem 0 0.8rem !important;
}

/* ── Chat container ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 1.5rem;
}

/* ── Bubbles ── */
.bubble {
    max-width: 78%;
    padding: 13px 18px;
    border-radius: 18px;
    font-size: 0.93rem;
    line-height: 1.65;
    word-break: break-word;
    animation: popIn .25s cubic-bezier(.34,1.56,.64,1) both;
}

@keyframes popIn {
    from { opacity: 0; transform: scale(.92) translateY(6px); }
    to   { opacity: 1; transform: scale(1)  translateY(0);   }
}

.bubble-user {
    background: #2f4de0;
    color: #eef1ff;
    align-self: flex-end;
    border-bottom-right-radius: 5px;
    box-shadow: 0 4px 20px rgba(47,77,224,.35);
}

.bubble-ai {
    background: #1a1d28;
    color: #c9cfe0;
    align-self: flex-start;
    border-bottom-left-radius: 5px;
    border: 1px solid #232740;
}

.bubble-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 5px;
    opacity: 0.55;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #3a3f55;
}
.empty-state .icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
.empty-state p { font-size: 0.88rem; line-height: 1.6; }

/* ── st.chat_input wrapper — Streamlit renders this in a fixed stBottom container ── */
[data-testid="stBottom"] {
    background: #0d0f14 !important;
    border-top: 1px solid #1e2130 !important;
    padding: 14px 2.5rem 18px !important;
}

[data-testid="stChatInput"] textarea {
    background: #13151c !important;
    border: 1px solid #1e2130 !important;
    border-radius: 16px !important;
    color: #e8ecf5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    box-shadow: none !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:invalid,
[data-testid="stChatInput"] textarea:required,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within {
    border-color: #1e2130 !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3a3f55 !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #2f4de0 !important;
    box-shadow: 0 0 0 3px rgba(47,77,224,.2) !important;
    outline: none !important;
}

[data-testid="stChatInputSubmitButton"] button {
    background: transparent !important;
    border-radius: 10px !important;
    transition: background .2s, transform .1s !important;
}
[data-testid="stChatInputSubmitButton"] button[data-testid="baseButton-primary"],
[data-testid="stChatInputSubmitButton"] button:not([disabled]):not([aria-disabled="true"]) {
    background: #007bff !important;
}
[data-testid="stChatInputSubmitButton"] button:not([disabled]):not([aria-disabled="true"]):hover {
    background: #0056b3 !important;
    transform: translateY(-1px) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #4f6ef7 !important; }

/* Divider */
hr { border-color: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Q&A")
    st.markdown("---")
    st.markdown("### Upload document")

    uploaded_pdf = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_pdf is not None:
        if st.session_state.get("uploaded_filename") != uploaded_pdf.name:
            with st.spinner("Processing PDF…"):
                uploaded_pdf.seek(0)
                pdf_bytes = uploaded_pdf.read()
                text = extract_text_from_pdf_bytes(pdf_bytes)
                cleaned = clean_text(text)
                chunks = chunk_text(cleaned)
                index = create_vector_store(chunks)

                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["uploaded_filename"] = uploaded_pdf.name
                st.session_state["messages"] = []

            st.success(f"✅ Ready — {len(chunks)} chunks indexed")
        else:
            st.success(f"✅ **{uploaded_pdf.name}** loaded")

    st.markdown("---")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if "uploaded_filename" in st.session_state:
        st.markdown(
            f"<div style='font-size:.75rem;color:#3a3f55;margin-top:1rem;'>"
            f"📎 {st.session_state['uploaded_filename']}</div>",
            unsafe_allow_html=True,
        )


# ── Main: header ──────────────────────────────────────────────────────────────
st.title("Ask your PDF anything")

# ── Chat history ──────────────────────────────────────────────────────────────
messages = st.session_state["messages"]

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">💬</div>
            <p>Upload a PDF in the sidebar, then ask<br>any question about its contents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="display:flex;justify-content:flex-end;">
                  <div class="bubble bubble-user">
                    <div class="bubble-label">You</div>
                    {msg["text"]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display:flex;justify-content:flex-start;">
                  <div class="bubble bubble-ai">
                    <div class="bubble-label">✦ Assistant</div>
                    {msg["text"]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("</div>", unsafe_allow_html=True)

# ── Scroll anchor — scrolls to latest message on every rerender ───────────────
st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
st.markdown("""
<script>
    const anchor = window.parent.document.getElementById("chat-bottom");
    if (anchor) anchor.scrollIntoView({ behavior: "smooth" });
</script>
""", unsafe_allow_html=True)

# ── Chat input — st.chat_input is rendered inside [data-testid="stBottom"]
#    which Streamlit itself positions as position:fixed at the bottom of the
#    viewport. No CSS hacks needed.
# ─────────────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your PDF…")

# ── Handle submission ─────────────────────────────────────────────────────────
if question:
    if "index" not in st.session_state or "chunks" not in st.session_state:
        st.warning("⬅  Please upload a PDF first.")
    else:
        st.session_state["messages"].append({"role": "user", "text": question.strip()})

        with st.spinner("Thinking…"):
            answer, timing = generate_answer(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
            )

        # Format timing information
        timing_text = (
            f"**⏱ Response time:** {timing['total']:.2f}s | "
            f"Retrieval: {timing['retrieval']:.2f}s | "
            f"Compression: {timing['compression']:.2f}s | "
            f"LLM: {timing['llm']:.2f}s"
        )
        
        answer_with_timing = f"{answer}\n\n<div style='font-size:0.75rem;color:#3a3f55;margin-top:1rem;'>{timing_text}</div>"

        st.session_state["messages"].append({"role": "ai", "text": answer_with_timing})
        st.rerun()