import os, json, pickle
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ======================
# CONFIG
# ======================
DATA_DIR = "data/pdfs"
CHROMA_DIR = "storage/chroma_db"
TRIPLET_PATH = "storage/triplets.json"

# Load embedding model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

import json
from google import genai

# Initialize your Gemini model client
client = genai.Client()

def extract_triplets(text):
    """
    Uses LLM to extract knowledge triplets from text.
    Returns a list of dicts with keys: subject, relation, object
    """
    prompt = f"""
    Extract factual triplets from the following text in JSON format.
    Each triplet should describe a relationship (subject, relation, object).
    
    Example format:
    [
      {{"subject": "Bangalore", "relation": "is known for", "object": "IT industry"}},
      {{"subject": "Cubbon Park", "relation": "is located in", "object": "Bengaluru"}}
    ]

    Text:
    {text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        raw_output = response.text.strip()

        # Try to clean and load JSON safely
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[-1].split("```")[0].strip()

        triplets = json.loads(raw_output)
        if isinstance(triplets, dict):  # Sometimes single object
            triplets = [triplets]

        # Ensure valid structure
        clean_triplets = []
        for t in triplets:
            if all(k in t for k in ("subject", "relation", "object")):
                clean_triplets.append({
                    "subject": t["subject"].strip(),
                    "relation": t["relation"].strip(),
                    "object": t["object"].strip()
                })

        return clean_triplets

    except Exception as e:
        print("⚠️ Triplet extraction failed:", e)
        return []


def build_vector_store():
    all_texts = []
    all_triplets = []

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        chunks = splitter.split_text(text)
        all_texts.extend(chunks)

        # Extract triplets
        all_triplets.extend(extract_triplets(text))

    # Create embeddings and save in Chroma
    db = Chroma.from_texts(all_texts, embedding=embedder, persist_directory=CHROMA_DIR)
    db.persist()

    # Save triplets
    with open(TRIPLET_PATH, "w", encoding="utf-8") as f:
        json.dump(all_triplets, f, indent=4)

    print(f"✅ Ingestion completed. {len(all_texts)} chunks stored in Chroma.")
    print(f"✅ {len(all_triplets)} triplets saved to {TRIPLET_PATH}")

if __name__ == "__main__":
    build_vector_store()
