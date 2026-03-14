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


# Page configuration
st.set_page_config(page_title="PDF Q&A App", layout="wide")

# Title
st.title("📄 Ask Questions From Your PDF")

# Sidebar for PDF upload
st.sidebar.header("Upload PDF")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)

if uploaded_pdf is not None:
    st.sidebar.success("PDF uploaded successfully!")

    # Only process PDF if it's a new upload (not already processed)
    if st.session_state.get("uploaded_filename") != uploaded_pdf.name:
        with st.spinner("Processing PDF..."):
            uploaded_pdf.seek(0)
            pdf_bytes = uploaded_pdf.read()
            text = extract_text_from_pdf_bytes(pdf_bytes)

            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)

            index = create_vector_store(chunks)

            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["uploaded_filename"] = uploaded_pdf.name
        
        st.success(f"✅ PDF processed! ({len(chunks)} chunks)")

# Main section
st.subheader("Ask a Question")

# Use form for Enter key submission
with st.form("question_form"):
    question = st.text_input(
        "Enter your question based on the uploaded PDF:"
    )
    
    submit = st.form_submit_button("Get Answer", use_container_width=True)

# Output placeholder
if submit:
    if "index" not in st.session_state or "chunks" not in st.session_state:
        st.warning("Please upload a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Retrieve the vector store for the current session
        index = st.session_state["index"]
        chunks = st.session_state["chunks"]

        with st.spinner("Processing your question..."):
            answer = generate_answer(question, index, chunks)

        st.subheader("Answer")
        st.write(answer)