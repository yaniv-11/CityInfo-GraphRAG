import re
import pickle
import pandas as pd
from chromadb import PersistentClient
from google import generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------ CONFIG ------------------
CHROMA_PATH = "chroma_db"                 # path to your Chroma database
TRIPLETS_PATH = "triplets.json"           # saved JSON from knowledge graph
EMBEDDING_MODEL = "text-embedding-004"    # Gemini embedding model
GENAI_MODEL = "gemini-1.5-flash"          # Gemini LLM model
TOP_K = 5                                 # number of top results for retrieval
# --------------------------------------------

genai.configure(api_key="YOUR_API_KEY_HERE")

# ------------------ LOADERS ------------------
def load_triplets(path=TRIPLETS_PATH):
    try:
        df = pd.read_json(path)
        return df
    except Exception as e:
        print("Error loading triplets:", e)
        return pd.DataFrame(columns=["subject", "relation", "object"])

def load_chroma_collection():
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="pdf_knowledge")
    return collection

# ------------------ HYBRID SEARCH ------------------
def hybrid_search(query, collection, df_triplets):
    # Vector search (semantic)
    query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)

    # Keyword search (lexical)
    docs = results.get("documents", [[]])[0]
    all_text = " ".join(docs)
    keywords = [w for w in re.findall(r'\b\w+\b', query.lower()) if len(w) > 3]
    keyword_hits = [doc for doc in docs if any(k in doc.lower() for k in keywords)]

    # Merge semantic + keyword results
    unique_docs = list(dict.fromkeys(docs + keyword_hits))

    # Enrich with triplets
    augmented_docs = augment_with_triplets(query, unique_docs, df_triplets)

    return augmented_docs

# ------------------ TRIPLET AUGMENT ------------------
def augment_with_triplets(query, documents, df_triplets):
    augmented_chunks = []
    triplet_entities = set(df_triplets["subject"].astype(str).tolist() + df_triplets["object"].astype(str).tolist())
    entity_patterns = [(e, re.compile(r"\b" + re.escape(e) + r"\b", re.IGNORECASE)) for e in triplet_entities]

    for doc in documents:
        identified = set()
        for entity, pattern in entity_patterns:
            if pattern.search(doc):
                identified.add(entity)

        if identified:
            related = df_triplets[
                df_triplets["subject"].isin(identified) | df_triplets["object"].isin(identified)
            ].head(5)
            triplet_text = "\n".join([f"- {r.subject} {r.relation} {r.object}" for r in related.itertuples()])
            doc += "\n\n📘 Related Knowledge Graph Info:\n" + triplet_text

        augmented_chunks.append(doc)
    return augmented_chunks

# ------------------ LLM RESPONSE ------------------
def generate_final_answer(query, augmented_docs):
    context = "\n\n".join(augmented_docs)
    prompt = f"""
You are a city information assistant. 
Use the following context to answer the question accurately and concisely.

Question: {query}

Context:
{context}

Answer:
"""
    response = genai.GenerativeModel(GENAI_MODEL).generate_content(prompt)
    return response.text

# ------------------ MAIN CHAT LOOP ------------------
def chat():
    df_triplets = load_triplets()
    collection = load_chroma_collection()

    print("✅ RAG Chat System Ready! Type 'exit' to stop.")
    while True:
        query = input("\n🧑‍💻 Ask: ")
        if query.lower() == "exit":
            break

        docs = hybrid_search(query, collection, df_triplets)
        answer = generate_final_answer(query, docs)
        print("\n🤖 Answer:\n", answer)

# ------------------ RUN ------------------
if __name__ == "__main__":
    chat()
