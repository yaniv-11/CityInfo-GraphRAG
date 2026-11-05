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
- Supports dense retrieval and graph-augmented retrieval workflows.
- Generates answers through configurable OpenRouter models.
- Uses DeepEval for retrieval, generator, workflow, and safety evaluation.
- Records claims, retrieved chunks, metric scores, explanations, and pass/fail states in JSON reports.
- Includes prompt-injection boundaries and adversarial safety fixtures.

## Project structure

```text
chat.py                         # Interactive chat application
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
```

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m pip install chromadb python-dotenv deepeval openai sentence-transformers
Copy-Item .env.example .env
```

Configure `.env`:

```dotenv
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Never commit `.env` or commit a real API key.

## Run

Start the interactive entry point:

```powershell
python main.py
```

Run the evaluation suites:

```powershell
python evals/evaluate_dense_retrieval.py --judge deepeval
python evals/evaluate_generator.py
python evals/evaluate_workflow.py
python evals/evaluate_safety.py
```

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
