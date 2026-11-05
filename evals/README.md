# Dense Retriever Evaluation

This first evaluation focuses only on dense retrieval from the persisted `city_info_embeddings` Chroma collection.

## Run

DeepEval judge with Gemini:

```powershell
Copy-Item .env.example .env
# Add your OpenRouter key and model to .env
python evals/evaluate_dense_retrieval.py --judge deepeval
```

```powershell
python evals/evaluate_dense_retrieval.py --judge lexical
python evals/evaluate_dense_retrieval.py --judge gemini
```

## Generator evaluation

```powershell
python evals/evaluate_generator.py
```

The generator uses dense retrieval context and OpenRouter, then DeepEval measures answer relevancy,
faithfulness, and answer correctness.

## Workflow evaluation

```powershell
python evals/evaluate_workflow.py
```

The workflow runner retrieves context, generates an answer, and reports:

- Answer relevancy
- Faithfulness
- Context recall
- Hallucination
- Context precision
- End-to-end answer correctness

## Safety evaluation

```powershell
python evals/evaluate_safety.py
```

The safety suite tests direct and indirect prompt injection, jailbreak attempts, instruction-hierarchy
attacks, credential extraction, PII requests, executable instructions, and a benign control case.

The default judge is Gemini. Set a Google API key before running it:

```powershell
$env:GOOGLE_API_KEY = "..."
```

The `lexical` judge is an offline deterministic fallback useful for smoke tests. It is not equivalent to an LLM judge.

## Metrics

- Contextual precision: retrieved claims also supported by the ideal answer.
- Contextual recall: ideal-answer claims supported by retrieved context.
- Contextual relevancy: retrieved claims judged relevant to the user's question.
- Retrieval precision@k and recall@k: retrieved chunk IDs compared with required chunk IDs.
- Hit rate: whether any required chunk was retrieved.
- Reciprocal rank: rank of the first required chunk.

Results are written to `evals/dense_retrieval_results.json`. The JSON includes claims, support decisions, scores, retrieved chunk IDs, and aggregate metrics.
