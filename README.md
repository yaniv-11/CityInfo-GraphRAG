# CivicGraph Bengaluru

Evaluated retrieval, graph augmentation, and grounded answer generation for Bengaluru city information.

CivicGraph Bengaluru is a document-grounded RAG system built from Bengaluru district reports. It combines
multi-modal ingestion, knowledge-graph triplets, Chroma vector retrieval, OpenRouter generation, DeepEval
quality analysis, and adversarial safety testing.

## Repository name

Recommended repository and product name: **CivicGraph Bengaluru**

Short description:

> A reproducible, evaluated Graph-RAG system for grounded answers about Bengaluru city information.

GitHub topics:

```text
rag graph-rag deepeval chromadb openrouter bengaluru civic-data llm-evaluation prompt-injection
```

## Current capabilities

- Extracts text, tables, images, and image captions from city documents.
- Creates subject-relation-object knowledge graph triplets.
- Stores document chunks in a persistent Chroma collection.
- Combines dense and BM25 retrieval, then reranks the merged candidates with a cross-encoder.
- Sends only chunks with a sigmoid-normalized reranker score of at least `0.7` to the graph augmentation and answer stages.
- Generates interactive chat answers through Gemini and uses OpenRouter models in evaluation workflows.
- Uses DeepEval for retrieval, generator, workflow, and safety evaluation.
- Records claims, retrieved chunks, metric scores, explanations, and pass/fail states in JSON reports.
- Includes prompt-injection boundaries and adversarial safety fixtures.

## Project structure

```text
chat.py                         # Interactive chat application
hybrid_retrieval.py             # Shared hybrid retrieval, reranking, and triplet augmentation
ingest.py                       # PDF text and triplet ingestion
ingest-caption.py               # Image caption and triplet ingestion
pipeline-rag.py                 # Alternative Graph-RAG pipeline
evals/
  fixtures/
    city_rag_golden.json       # 15 golden dense retrieval questions
    safety_cases.json          # Adversarial safety cases
  evaluate_dense_retrieval.py
  evaluate_generator.py
  evaluate_workflow.py
  evaluate_safety.py
  config/metrics.json
  README.md
tests/
  test_hybrid_retrieval.py
```

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m pip install chromadb python-dotenv deepeval openai sentence-transformers langchain-community rank-bm25 google-genai
Copy-Item .env.example .env
```

Configure `.env`:

```dotenv
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
GOOGLE_API_KEY=your_key_here
```

`GOOGLE_API_KEY` is used by the interactive chat generator. Never commit `.env` or a real API key.

## Run

Start the interactive entry point:

```powershell
python main.py
```

Chat retrieves up to five chunks from each of dense search and BM25, deduplicates
the candidates, and scores each question/chunk pair with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. The cross-encoder emits logits; the
application applies a sigmoid and retains scores `>= 0.7`. Matching knowledge
graph triplets are added only after this filtering, and only the retained,
triplet-augmented context is sent to Gemini. If no chunks meet the threshold,
the application abstains without calling the generator. The cross-encoder and
embedding models are downloaded on first use.

Run the evaluation suites:

```powershell
python evals/evaluate_dense_retrieval.py --judge deepeval
python evals/evaluate_generator.py
python evals/evaluate_workflow.py
python evals/evaluate_safety.py
```

Run focused reranking tests with `python -m pytest tests/test_hybrid_retrieval.py`.
The workflow evaluation follows the same dense + BM25, cross-encoder threshold,
and retained-chunk triplet augmentation path as chat.

The lexical dense-retrieval judge can run without an API key:

```powershell
python evals/evaluate_dense_retrieval.py --judge lexical
```

## Evaluation coverage

### Retriever

- Dense precision and recall at k
- Hit rate
- Mean reciprocal rank
- Contextual precision
- Contextual recall
- Contextual relevancy

### Generator

- Answer relevancy
- Faithfulness
- Answer correctness

### Workflow

- Answer relevancy
- Faithfulness
- Context recall
- Hallucination resistance
- Context precision
- End-to-end answer correctness

### Safety

- Prompt injection
- Jailbreak resistance
- Indirect prompt injection
- Instruction-hierarchy attacks
- Credential extraction
- PII leakage
- Toxic content
- Benign control behavior

## Safety boundary

The application treats user questions, retrieved documents, OCR output, captions, and generated triplets
as untrusted data. Trusted system instructions remain separate, and generated answers should use only the
provided source context.

See [SECURITY-REVIEW.md](SECURITY-REVIEW.md) for the current security review and remediation tracking.
