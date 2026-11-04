# Hybrid RAG Evaluation Specification

## 1. Purpose

Create a reproducible evaluation framework for the city information Hybrid
Graph-RAG application. The evaluation suite must measure whether:

- the correct source context is retrieved;
- knowledge graph information improves, rather than harms, answers;
- generated answers are grounded in the supplied context;
- the complete application works consistently from question to answer; and
- changes to retrieval, graph augmentation, prompts, models, or data can be
  compared against a fixed baseline.

This specification intentionally does not define final metric thresholds, API
provider settings, credentials, or judge models.

## 2. Scope

The evaluation suite will cover four layers.

### 2.1 Component-level ingestion tests

Verify that the existing city documents are converted into useful searchable
units.

The suite should test:

- PDF text extraction does not lose important facts;
- text is split into appropriately sized chunks;
- chunk overlap preserves surrounding meaning;
- empty or unusable chunks are excluded;
- generated embeddings are compatible with the vector index; and
- extracted subject-relation-object triplets have the required structure.

This layer will use fixtures when possible so that it can detect regressions
without invoking Gemini.

### 2.2 Component-level retrieval tests

Evaluate dense retrieval, BM25 retrieval, hybrid retrieval, and graph
augmentation independently.

The suite should test:

- dense retrieval precision and recall;
- BM25 retrieval precision and recall;
- hybrid retrieval precision and recall;
- retrieval ranking quality;
- graph triplet retrieval and relevance; and
- the effect of duplicate, unrelated, or conflicting context.

Retriever metrics should be calculated against expected source chunk IDs or
stable source chunk fingerprints.

### 2.3 Component-level generator tests

Evaluate answer generation independently by supplying a fixed question and
fixed retrieved context.

The suite should test:

- the answer is grounded in the supplied context;
- unsupported facts are not invented;
- the answer addresses the complete question;
- conflicts between context and world knowledge are handled as specified; and
- answers have acceptable completeness and relevance.

### 2.4 Workflow-level tests

Evaluate the complete retrieval-and-generation workflow:

```text
question
  -> query normalization
  -> dense retrieval
  -> BM25 retrieval
  -> hybrid fusion
  -> knowledge graph augmentation
  -> final prompt
  -> generated answer
```

Workflow tests should verify that retrieved context and final answers are
consistent across repeated runs where the underlying data and model settings
are pinned.

### 2.5 Application-level tests

Evaluate the application as a user-facing system.

For the current implementation, application tests should cover:

- ingestion entry point behavior from [main.py](C:/Users/vinays/Desktop/Graph-RAG/main.py);
- chat entry point behavior from [main.py](C:/Users/vinays/Desktop/Graph-RAG/main.py);
- loading persisted Chroma data;
- loading persisted triplets;
- handling valid questions;
- handling unknown topics;
- handling empty or malformed questions; and
- graceful handling of missing data, API failures, and unavailable services.

Application tests should interact with the scripts using a controlled test
fixture rather than the user's live Gemini account.

## 3. Golden Dataset

The initial dataset will contain 15 medium-to-hard questions based only on the
city information available in the project documents.

Each record should have:

| Field | Required | Description |
|---|---:|---|
| `id` | Yes | Stable unique case identifier |
| `question` | Yes | User-facing question |
| `ideal_answer` | Yes | Concise authoritative answer |
| `acceptable_answer_points` | Yes | Facts that an answer must contain |
| `required_context_ids` | Yes | Source chunks needed for a complete answer |
| `optional_context_ids` | Yes | Useful supporting chunks |
| `relevant_triplets` | Yes | Knowledge graph facts required or useful for the answer |
| `category` | Yes | Retrieval or reasoning category |
| `difficulty` | Yes | Medium or hard |
| `source_document` | Recommended | PDF or data source |
| `notes` | Recommended | Ambiguity, expected reasoning, or disallowed facts |

### 3.1 Question categories

The dataset should cover, where supported by the documents:

1. direct factual questions;
2. multi-fact questions requiring several chunks;
3. comparison questions;
4. trend or change-over-time questions;
5. questions involving a document table;
6. questions requiring entities from triplets;
7. multi-hop graph questions;
8. questions that combine document and graph context;
9. answers that require careful distinction between related entities; and
10. unsupported questions that should trigger an honest no-answer response.

### 3.2 Dataset creation rule

Questions must be grounded in the source data. They must not be invented from
model knowledge or general city facts that are absent from the documents.

The dataset should use one or more of:

- stable source chunk IDs;
- section/page identifiers;
- source document plus page;
- exact text spans; or
- generated chunk fingerprints.

The 15-question dataset is a development set for initial implementation.
Additional holdout cases should be created before declaring the application
stable.

## 4. Retrieval Metrics

The following metrics are required unless an item is explicitly marked
optional.

### 4.1 Precision

Precision measures whether retrieved context is relevant:

```text
relevant retrieved items / all retrieved items
```

Precision may be calculated at a configurable retrieval cutoff, such as `k=3`,
`k=5`, or `k=10`.

### 4.2 Recall

Recall measures whether expected context was retrieved:

```text
relevant retrieved items / all relevant gold items
```

### 4.3 Hit rate

Hit rate measures whether at least one relevant gold chunk appears in the
retrieved results.

### 4.4 Mean reciprocal rank

MRR measures how highly the first relevant result is ranked.

### 4.5 Normalized discounted cumulative gain

NDCG may be used when gold context has graded relevance. It rewards relevant
context being retrieved near the top.

### 4.6 Per-component breakdown

Metrics must be reported separately for:

- dense-only retrieval;
- BM25-only retrieval;
- ensemble hybrid retrieval;
- graph-only triplet retrieval; and
- hybrid retrieval plus graph augmentation.

Reported metrics should include both aggregate and per-category results.

## 5. RAG Triad

The standard DeepEval RAG evaluation should include:

### 5.1 Answer relevancy

Measures whether the final answer directly answers the user question.

Required inputs:

- actual output;
- user question;
- retrieved context, when the metric supports it.

### 5.2 Faithfulness

Measures whether answer claims are supported by retrieved context.

Required inputs:

- actual output;
- retrieved context.

This should catch unsupported facts, hallucinations, and incorrect graph
relationships.

### 5.3 Context recall

Measures whether the retrieved context contains the information required for
the ideal answer.

Required inputs:

- actual output or supporting expected answer facts;
- ideal answer;
- retrieved context.

### 5.4 Context precision

Measures whether retrieved context consists primarily of relevant information.

Required inputs:

- user question;
- retrieved context;
- expected/relevant context labels.

## 6. Additional Generator Metrics

In addition to the RAG triad, DeepEval or equivalent deterministic checks
should assess:

- answer correctness against the ideal answer;
- completeness based on required answer points;
- hallucination detection;
- instruction following;
- unknown-answer behavior; and
- consistency across repeated executions.

LLM-based metrics should be supplemented by deterministic checks. Judge models
can disagree or be probabilistic, so expected answer facts should remain
observable in test logs.

## 7. Evaluation Data Format

A JSON fixture should contain one question per record. Recommended locations:

```text
evals/
  fixtures/
    city_rag_golden.json
  config/
    metrics.example.json
  test_component_retrieval.py
  test_component_generator.py
  test_workflow.py
  test_application.py
```

A project runner may use a separate location if this structure conflicts with
the repository conventions, but the golden dataset and metric configuration
must remain separate.

## 8. Test Isolation

Evaluation tests must:

- use fixed seed and retrieval settings;
- use test-specific Chroma collections;
- avoid overwriting production indexes;
- use pinned embeddings or snapshot equivalent results;
- avoid live API calls where a deterministic fixture can represent the result;
- record prompts, retrieved context, triplets, and outputs; and
- clean temporary test data after execution.

Live-model tests should be opt-in so that unit and fixture tests can run
without external credentials.

## 9. Expected Observability

Each test result should include:

- case ID;
- question;
- ideal answer;
- actual answer;
- retrieved chunk IDs and source locations;
- retrieved context text;
- applied triplets;
- dense scores;
- BM25 scores;
- fused ranks;
- selected components and model configuration;
- metric values;
- threshold evaluation;
- elapsed time;
- exception details if applicable.

Results should be available as:

- terminal output;
- machine-readable JSON; and
- a report suitable for CI review.

## 10. Functional Acceptance

### 10.1 Dataset readiness

- 15 cases must exist.
- Every case must be grounded in project data.
- Every case must have an ideal answer.
- Every case must identify required context.
- Multi-hop questions must identify intermediate entities or triplets.

### 10.2 Retrieval readiness

- Dense, BM25, and hybrid results are individually measurable.
- Precision and recall are reported at declared cutoff values.
- Graph augmentation is evaluated independently from vector and BM25 search.
- Empty retrieval sets are reported correctly rather than as pipeline crashes.

### 10.3 Generator readiness

- Generator tests run with fixed context.
- Faithfulness, relevancy, context recall, and context precision can run
  independently.
- Unsupported-answer cases are treated correctly.

### 10.4 Workflow readiness

- Workflow tests exercise the same retrieval functions used by
  [chat.py](C:/Users/vinays/Desktop/Graph-RAG/chat.py).
- Test paths must not diverge from production behavior.
- Result artifacts expose all intermediate stages.

### 10.5 Application readiness

- Valid questions produce answers or an explicit unsupported-answer response.
- Missing indexes and malformed data produce defined errors.
- API failures do not produce misleading successful responses.
- The application returns the same result contract for an evaluation request
  and normal chat request.

## 11. Deferred Decisions

The implementation must not proceed without final decisions on:

1. exact metric threshold values;
2. judge model/provider;
3. API credential method;
4. whether DeepEval judge calls use the same model as the application;
5. cutoff values for precision and recall;
6. the desired answer verbosity and formatting;
7. treatment of temporal, numerical, and comparative questions;
8. whether unsupported answers must say "not found" in a fixed phrase;
9. whether tests run against existing indexes or rebuild fixtures each time;
10. CI behavior for live-model tests;
11. whether scores must pass every case or a configured aggregate;
12. retention of detailed prompts and context in reports.

## 12. Recommended Evaluation Sequence

1. Finalize and validate the 15-case golden dataset.
2. Add deterministic ingestion fixtures.
3. Implement component retrieval metrics.
4. Implement generator and RAG-triad metrics.
5. Implement the workflow harness.
6. Add application-level tests using fixtures.
7. Run tests and inspect failures per layer.
8. Define and approve thresholds.
9. Add CI gates.
10. Create a holdout dataset for periodic regression testing.

## 13. Definition of Success

The evaluation suite is complete when it can:

- reproduce a known baseline;
- distinguish dense, BM25, hybrid, and graph contributions;
- detect context retrieval regressions;
- detect answer grounding regressions;
- identify unsupported-answer failures;
- run locally without damaging live indexes;
- produce machine-readable and human-readable reports; and
- apply approved gates consistently in CI.
