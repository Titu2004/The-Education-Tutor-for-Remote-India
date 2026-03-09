from src.load_pdfs import load_all_pdfs
from src.text_processing import process_documents


docs = load_all_pdfs()

chunks = process_documents(docs)

print("Total chunks created:", len(chunks))

print("\nExample chunk:\n")
print(chunks[0]["content"])