🏙️ City Info — Multi-Modal RAG Knowledge System
 Advanced AI System | RAG | Multi-Modal | Knowledge Graph | ChromaDB | Gemini LLM | GraphRAG

 🧩 Problem It Solves

City information is often scattered across multiple government reports, policy PDFs, and public datasets — making it hard for citizens, analysts, or developers to find reliable, up-to-date answers. Traditional keyword search fails to understand context, relationships, or semantics, especially when users ask natural questions like "what are the latest traffic fines structure","electricity outrages in doddaballapura"

🌟 Overview

The City Info project is a multi-modal Retrieval-Augmented Generation (RAG) system that extracts, structures, and answers questions from city-related documents (PDFs) containing text, tables, and images.

It automatically builds a knowledge base by:

Extracting all types of data (text, tables, and images) from PDFs.

Generating triplets (subject–relation–object) using an LLM.

Storing semantic embeddings in ChromaDB for retrieval.

Combining semantic + keyword search for hybrid retrieval.

Enriching answers with knowledge graph relations.

Using Gemini LLM for contextual final answers.

The system functions as a self-building, multi-modal city knowledge assistant — capable of understanding and answering complex queries using structured and unstructured data.

🧩 Architecture
1. Data Ingestion & Extraction

Input: City information PDFs (e.g., government reports, district profiles).

Extraction tools:

📝 Text: pdfplumber

📊 Tables: Extracted using camelot-py and stored as text

🖼️ Images: Extracted and processed using pdf2image + pytesseract OCR

🧠 Image Captioning: Vision-LLM generates meaningful captions for images 

Output: Cleaned ocr_texts list with text, table, and image captions.

2. Knowledge Graph Generation

Model: Gemini / OpenAI LLM

Method: Each page’s text is passed to the LLM to generate JSON triplets in format:

{
  "subject": "Bangalore Palace",
  "relation": "is located in",
  "object": "Bengaluru"
}


Storage: Triplets saved as triplets.json and later used for graph-based reasoning.

3. Text Cleaning & Preprocessing

Remove noise from OCR (special characters, misaligned text, etc.)

Chunk text page-wise (each PDF page = 1 chunk)

Maintain alignment with original PDF structure

4. Embedding & Vector Store Creation

Model: sentence-transformers (all-MiniLM-L6-v2)

Process:

Convert each chunk into embeddings

Save embeddings locally as ocr_embeddings.pkl

Then push them into ChromaDB persistent store (chroma.sqlite3)

Example:

vectorstore.add_texts(texts=chunks, embeddings=embeddings)

5. Hybrid Retrieval (Semantic + Keyword)

Combines semantic similarity search (via embeddings)
and keyword-based search (via regex/token match).

Ensures both context-rich and literal matches are retrieved.

Hybrid Flow:
User Query ➜
Semantic Search (Chroma) ➜
Keyword Match (Regex) ➜
Combine + Rank ➜
Augment with Triplets ➜
Final Context to LLM

6. Graph-Augmented Context Enrichment

Extract entities from retrieved text using regex.

Match with triplet entities (subjects/objects).

Enrich retrieved chunks with related triplets for deeper reasoning.

Example output:

Related Information from Knowledge Graph:
- Bangalore Palace is located in Bengaluru
- Lalbagh Botanical Garden covers 240 acres

7. LLM Integration & Response Generation

Model Used: Gemini 1.5 Flash

Combines top retrieved chunks + graph knowledge → sends to LLM.

Generates context-aware natural language response.

response = gemini_model.generate_content(context + query)
print(response.text)


Example:

User: “What are the key tourist spots in Bengaluru Urban?”
System: “Top spots include Lalbagh Botanical Garden, Tipu Sultan’s Summer Palace, Bannerghatta National Park, and Bangalore Palace. Lalbagh spans 240 acres and was commissioned by Hyder Ali.”

8. End-to-End Pipeline Automation

✅ PDF ingestion
✅ Text + image + table extraction
✅ Triplet generation
✅ Embedding creation
✅ ChromaDB storage
✅ Hybrid retrieval
✅ Knowledge graph augmentation
✅ Gemini response

All automated in a single workflow:

process_pdf("data/Bengaluru_Urban.pdf")
chat_with_city_info("What are the key industries in Bengaluru?")

🧠 Key Features
Feature	Description
Multi-Modal Extraction	Handles text, tables, and images
OCR + Captioning	Extracts and captions images contextually
Graph + Vector Fusion	Combines embeddings with knowledge graph
Hybrid Retrieval	Semantic + keyword search
LLM-Powered Answers	Uses Gemini for coherent responses
Persistent Storage	Embeddings stored in chroma.sqlite3
Extensible Design	Can scale to other cities or domains
📊 Example Query Flow

Query: “What is the land utilization pattern in Bengaluru Urban district?”
Pipeline:

Retrieve related chunks (from Chroma)

Identify entities like “land”, “forest”, “cultivation”

Append triplets like “Forests occupy 1.49% of total area”

Send to Gemini for final synthesis

Final Response:

Bengaluru Urban district has 1.49% forest area, 40.76% uncultivated land, and 14.09% net sown area out of a total of 2,17,410 hectares.

⚙️ Technologies Used
Category	Tools
Extraction	pdfplumber, pdf2image, pytesseract, camelot-py
Embedding	SentenceTransformers
Vector Store	ChromaDB
Knowledge Graph	LLM (triplet extraction)
Retrieval Logic	Hybrid (semantic + keyword + graph)
LLM Response	Gemini 1.5 Flash
Orchestration	Python (Jupyter Notebook / .py pipeline)
🧩 Challenges & Solutions
Challenge	Solution
Extracting noisy OCR text	Used regex-based cleanup and manual correction
Image context missing	Integrated image caption LLM for visual semantics
PDF tables misaligned	Used Camelot’s lattice and stream mode extraction
Triplet redundancy	Filtered and ranked top 10 per entity
Large embedding data	Used persistent ChromaDB with optimized batching
Mixed modalities	Unified all outputs into ocr_texts pipeline


 Example Results

Q: “List famous tourist places in Bengaluru Urban district.”
A:

Famous spots include Lalbagh Botanical Garden (240 acres, 1000+ species), Tipu Sultan’s Summer Palace (built in 1791), and Bannerghatta National Park (home to white tigers and India’s first butterfly park).

Q: “Describe forest and land use in the district.”
A:

Forests cover 1.49%, uncultivated land 40.76%, and net sown area 14.09% of the district’s 2,17,410 hectares.


This project reflects months of effort, research, and experimentation. It represents a full end-to-end AI system that blends natural language understanding, information retrieval, and knowledge representation. Every step — from raw PDF to intelligent answers — was crafted manually and deeply optimized for accuracy.

— Yaniv
