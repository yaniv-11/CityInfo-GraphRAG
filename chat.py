import json
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai

# ======================
# CONFIG
# ======================
CHROMA_DIR = "storage/chroma_db"
TRIPLET_PATH = "storage/triplets.json"
MODEL = "gemini-1.5-flash"

# Initialize
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
triplets = json.load(open(TRIPLET_PATH, "r", encoding="utf-8"))

# Build hybrid retriever
docs = db.get()["documents"]
bm25_retriever = BM25Retriever.from_texts(docs)
vector_retriever = db.as_retriever(search_kwargs={"k": 3})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# Gemini client
client = genai.Client()

def augment_with_triplets(query, retrieved_docs, triplets):
    text_context = "\n".join([doc.page_content for doc in retrieved_docs])
    related = []
    for t in triplets:
        if any(e.lower() in text_context.lower() for e in [t["subject"], t["object"]]):
            related.append(f"{t['subject']} {t['relation']} {t['object']}")
    return text_context + "\n\nRelated Info:\n" + "\n".join(related[:10])

def answer_query(query):
    retrieved_docs = hybrid_retriever.get_relevant_documents(query)
    if not retrieved_docs:
        return "No relevant info found."

    augmented = augment_with_triplets(query, retrieved_docs, triplets)

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Answer the question based on the context:\n{augmented}\n\nQ: {query}"
    )

    return response.text

if __name__ == "__main__":
    while True:
        q = input("\n🧠 Ask something about city info: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n🤖", answer_query(q))
