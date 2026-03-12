import streamlit as st
from retrieval import generate_answer

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

# Main section
st.subheader("Ask a Question")

# Question input
question = st.text_input(
    "Enter your question based on the uploaded PDF:"
)

# Submit button
submit = st.button("Get Answer")

# Output placeholder
if submit:
    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Placeholder for backend response
        with st.spinner("Processing your question..."):
            answer = generate_answer(question)

        st.subheader("Answer")
        st.write(answer)