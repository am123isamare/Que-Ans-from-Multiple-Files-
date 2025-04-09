import streamlit as st
import PyPDF2
import docx
from io import StringIO
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load models
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="QA Retrieval App", layout="centered")
st.title("📚 QA from Multiple Files ")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from TXT
def extract_text_from_txt(file):
    stringio = StringIO(file.getvalue().decode("utf-8"))
    return stringio.read()

# Function to split text into chunks
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

uploaded_files = st.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
question = st.text_input("Ask a question based on the uploaded files:")

if uploaded_files:
    all_text = ""

    # Read all files and extract text
    for file in uploaded_files:
        if file.type == "application/pdf":
            all_text += extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            all_text += extract_text_from_docx(file)
        elif file.type == "text/plain":
            all_text += extract_text_from_txt(file)
        all_text += "\n"

    # Split and embed
    with st.spinner("Processing documents..."):
        chunks = split_text(all_text)
        embeddings = embedder.encode(chunks)

        # Ensure embeddings are float32 for FAISS
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index and add embeddings
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

    if question:
        # Get embedding for the question
        question_embedding = embedder.encode([question])

        # Ensure question embedding is float32
        question_embedding = np.array(question_embedding, dtype=np.float32)

        # Perform search for similar chunks
        D, I = index.search(question_embedding, k=3)

        # Get the context based on the retrieved indices
        context = " ".join([chunks[i] for i in I[0]])

        # QA
        result = qa_model(question=question, context=context)

        st.subheader("📢 Answer")
        st.write(result["answer"])
