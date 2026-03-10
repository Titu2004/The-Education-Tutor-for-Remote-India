Step-by-Step Procedure for the Project
1️⃣ Understand the Problem

Goal:
Students ask questions from state-board textbooks
AI gives accurate answers
Low API cost
Low data transfer
Works even with large PDFs

So the system must:
Read textbooks once
Store knowledge efficiently
Retrieve only relevant parts when answering

2️⃣ Collect Dataset (Textbooks)

Download State Board textbooks (PDF format).

Examples:
NCERT
WB Board
ICSE

Subjects you can start with:
Science
Mathematics
History

Store them in a folder:

/data
   class8_science.pdf
   class9_history.pdf
   class10_math.pdf
3️⃣ Convert PDF → Text

Extract text from the PDFs.

Tools:
Python
PyPDF
pdfplumber

Example process:
PDF → Raw Text

Output:
photosynthesis is the process by which green plants...
4️⃣ Clean the Text

Remove:
Page numbers
Headers
Footers
Unwanted symbols

Example:
Before
Page 32
Chapter 3: Photosynthesis

After
Photosynthesis is the process by which plants...
5️⃣ Split the Text into Chunks

Large textbooks must be split.

Example chunk size:
300 – 500 words

Example:
Chunk 1 → Photosynthesis introduction
Chunk 2 → Chlorophyll explanation
Chunk 3 → Process diagram explanation

Why?
Because AI cannot process entire books at once.

6️⃣ Create Embeddings

Convert text chunks into vector embeddings.

Tools:
Sentence Transformers
OpenAI Embeddings
InstructorXL

Example:
Text Chunk → Vector Numbers
Photosynthesis → [0.25, -0.11, 0.89 ...]
7️⃣ Store Embeddings in a Vector Database

Use a vector database for fast search.

Best options for student projects:
FAISS
ChromaDB
Pinecone

Structure:
Vector Database
   ├ Chunk 1
   ├ Chunk 2
   ├ Chunk 3

Now the system can quickly search relevant knowledge.

8️⃣ Build the Retrieval System

When a student asks a question:

Example:
"What is photosynthesis?"

Process:
Question → Embedding → Search Vector DB

Top relevant chunks are retrieved.

Example:
Chunk 12: Photosynthesis definition
Chunk 15: Process explanation
Chunk 17: Chemical equation
9️⃣ Implement Prompt Compression

This is the main innovation of your project.

Instead of sending large text to the AI model:

Normal RAG
User Question + 3000 tokens context
Compressed RAG
User Question + summarized context

Example compression:

Before:

Photosynthesis is the process by which plants produce food using sunlight...
(large paragraph)

After compression:

Photosynthesis: plants convert sunlight, CO2, and water into glucose and oxygen.

This reduces API cost by 50-80%.

🔟 Generate Final Answer

Now send to the LLM:

Prompt =

Student Question
+
Compressed Context

Example:

Question: What is photosynthesis?

Context:
Photosynthesis: plants convert sunlight, CO2, water into glucose.

Answer:
Photosynthesis is the process by which plants make food using sunlight...
1️⃣1️⃣ Build the AI Tutor Interface

Create a simple interface.

Options:

Web App

Use:

Flask

Streamlit

React

Example UI:

Ask your question from the textbook

[Type here]

Output:

Answer:
Photosynthesis is the process...
1️⃣2️⃣ Add Personalization

Make it behave like a tutor.

Add features:

Explain in simple language

Provide examples

Show important formulas

Example:

Explain like I am class 8 student
1️⃣3️⃣ Measure Cost Reduction

Compare two systems:

Baseline

Normal RAG

Your Model

Compressed RAG

Metrics:

Metric	Baseline	Your Model
Tokens used	3000	900
Cost	High	Low
Response speed	Medium	Fast

This is important for your project evaluation.

1️⃣4️⃣ Add Extra Features (Optional but impressive)

You can add:

📚 Chapter wise learning
🧠 Quiz generator
📝 Homework solver
🔊 Voice question answering
📶 Offline lightweight model

1️⃣5️⃣ Final Project Architecture
Student Question
       ↓
Text Embedding
       ↓
Vector Database Search
       ↓
Relevant Chunks
       ↓
Prompt Compression
       ↓
LLM
       ↓
Answer
Tech Stack for Your Project
Component	Tool
Language	Python
PDF parsing	pdfplumber
Embeddings	Sentence Transformers
Vector DB	FAISS
LLM	OpenAI / Mistral
Backend	Flask
UI	Streamlit
