import pdfplumber
import os

DATA_PATH = "data"


def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def load_all_pdfs():
    documents = {}

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):

            path = os.path.join(DATA_PATH, file)

            print(f"Loading {file}...")

            text = extract_text_from_pdf(path)

            documents[file] = text

    return documents


if __name__ == "__main__":

    docs = load_all_pdfs()

    for name, content in docs.items():
        print(f"\n{name} loaded.")
        print("Total characters:", len(content))