import fitz  # PyMuPDF
import io
import numpy as np
from PIL import Image
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import uuid

# ------------------- CONFIG --------------------
PDF_PATH = "data/sample.pdf"   # your pdf
CHROMA_PATH = "chroma_db"
MODEL_NAME = "gemini-1.5-flash"  # Vision + Text
EMBED_MODEL = "models/text-embedding-004"
# ------------------------------------------------

genai.configure(api_key="YOUR_GEMINI_API_KEY")

# ---------- Step 1: Extract text and images ----------
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    images = []

    for page in doc:
        text = page.get_text("text")
        all_text.append(text)
        
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
    
    return " ".join(all_text), images


# ---------- Step 2: Caption images ----------
def caption_image(image):
    """Describe image content with Gemini Vision model"""
    try:
        response = genai.generate_content(
            model=MODEL_NAME,
            contents=[
                "Describe this image in factual, concise sentences suitable for a knowledge base.",
                image
            ]
        )
        return response.text.strip()
    except Exception as e:
        print("Error captioning image:", e)
        return ""


# ---------- Step 3: Extract triplets from combined text ----------
def extract_triplets(text):
    """Use Gemini to extract (subject, relation, object) triplets in JSON"""
    try:
        prompt = f"""
        Extract all factual triplets (subject, relation, object) from this text in JSON list format.
        Only return JSON.
        Text:
        {text}
        """
        response = genai.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text
    except Exception as e:
        print("Triplet extraction error:", e)
        return "[]"


# ---------- Step 4: Embed and store in Chroma ----------
def embed_and_store(chroma_path, texts, metadatas):
    chroma_client = chromadb.Client(Settings(persist_directory=chroma_path))
    collection = chroma_client.get_or_create_collection(name="city_knowledge")

    embeddings = genai.embed_content(model=EMBED_MODEL, content=texts)["embedding"]
    if not isinstance(embeddings[0], (list, np.ndarray)):
        embeddings = [embeddings]

    ids = [str(uuid.uuid4()) for _ in texts]
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    print(f"✅ Stored {len(texts)} entries in Chroma.")


# ---------- Step 5: Orchestrate entire ingestion ----------
def ingest_pdf(pdf_path):
    print(f"📄 Processing PDF: {pdf_path}")
    text, images = extract_text_and_images(pdf_path)

    print(f"🖼️ Found {len(images)} images, captioning them...")
    captions = [caption_image(img) for img in images if img is not None]

    combined_text = text + "\n\n" + "\n".join(captions)
    print(f"✅ Combined text length: {len(combined_text)} characters")

    triplets_json = extract_triplets(combined_text)
    print(f"📊 Extracted triplets: {triplets_json[:300]}...")

    embed_and_store(
        chroma_path=CHROMA_PATH,
        texts=[combined_text],
        metadatas=[{"source": pdf_path, "triplets": triplets_json}]
    )
    print("🚀 Ingestion complete.")


# ------------------- RUN --------------------
if __name__ == "__main__":
    ingest_pdf(PDF_PATH)
