# use pypdf (formerly PyPDF2) for PDF parsing
from pypdf import PdfReader
import os

DATA_PATH = "data"


def extract_text_from_pdf(pdf_path):
    """Extract text from every page of a PDF using pypdf.

    Prints simple progress information so users can see which page is
    currently being processed.
    """

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    text = ""

    for idx, page in enumerate(reader.pages, start=1):
        print(f"  reading page {idx}/{num_pages}...", end="\r")
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # clear the progress line after finishing
    print("", end="\r")
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