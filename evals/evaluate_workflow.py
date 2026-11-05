from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "evals" / "fixtures" / "city_rag_golden.json"
DEFAULT_CONFIG = ROOT / "evals" / "config" / "metrics.json"
DEFAULT_OUTPUT = ROOT / "evals" / "workflow_results.json"

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the complete dense RAG workflow.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "city_info_chroma")
    parser.add_argument("--collection", default="city_info_embeddings")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def openrouter_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env.")

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url), model


def generate_answer(client: OpenAI, model: str, question: str, context: list[str]) -> str:
    prompt = (
        "Answer the question using only the supplied context. "
        "If the context does not contain enough information, state that explicitly.\n\n"
        f"Question: {question}\n\nContext:\n\n" + "\n\n".join(context)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def evaluate_workflow(case: dict, answer: str, context: list[str], model: str, config: dict) -> dict:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.models import OpenAIModel
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    judge = OpenAIModel(model=model, api_key=api_key, base_url=base_url)
    threshold = float(config.get("threshold", 0.5))
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=answer,
        expected_output=case["ideal_answer"],
        retrieval_context=context,
        context=context,
    )

    correctness = GEval(
        name="End-to-End Answer Correctness",
        criteria=(
            "Compare the actual answer with the expected answer. Award a high score only when "
            "the answer contains the required facts accurately and completely, without conflicting facts."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=judge,
        async_mode=False,
    )
    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=threshold, model=judge, async_mode=False, include_reason=True
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=threshold, model=judge, async_mode=False, include_reason=True
        ),
        "context_recall": ContextualRecallMetric(
            threshold=threshold, model=judge, async_mode=False, include_reason=True
        ),
        "hallucination": HallucinationMetric(
            threshold=threshold, model=judge, async_mode=False, include_reason=True
        ),
        "context_precision": ContextualPrecisionMetric(
            threshold=threshold, model=judge, async_mode=False, include_reason=True
        ),
        "answer_correctness": correctness,
    }

    output = {"scores": {}, "reasons": {}, "successes": {}}
    for name, metric in metrics.items():
        metric.measure(test_case)
        output["scores"][name] = float(metric.score or 0.0)
        output["reasons"][name] = metric.reason
        output["successes"][name] = bool(metric.success)
    return output


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = json.loads(args.fixture.read_text(encoding="utf-8"))
    top_k = int(config.get("top_k", 5))
    openrouter, workflow_model = openrouter_client()
    judge_model = os.getenv("OPENROUTER_MODEL", config.get("judge_model", "openai/gpt-4o-mini"))

    temp_dir = Path(tempfile.mkdtemp(prefix="workflow-rag-eval-"))
    try:
        isolated_chroma = temp_dir / "chroma"
        shutil.copytree(args.chroma_path, isolated_chroma)
        chroma_client = chromadb.PersistentClient(path=str(isolated_chroma))
        collection = chroma_client.get_collection(args.collection)
        embedder = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))

        results = []
        for case in cases:
            embedding = embedder.encode(case["question"], normalize_embeddings=True).tolist()
            hits = collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, collection.count()),
            )
            context = hits["documents"][0]
            answer = generate_answer(openrouter, workflow_model, case["question"], context)
            metrics = evaluate_workflow(case, answer, context, judge_model, config)
            results.append({
                "id": case["id"],
                "question": case["question"],
                "ideal_answer": case["ideal_answer"],
                "actual_output": answer,
                "retrieved_chunk_ids": hits["ids"][0],
                "dense_distances": hits["distances"][0],
                "retrieved_context": context,
                "deepeval": metrics,
            })

        collection = None
        chroma_client = None
        gc.collect()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    metric_names = (
        "answer_relevancy",
        "faithfulness",
        "context_recall",
        "hallucination",
        "context_precision",
        "answer_correctness",
    )
    aggregate = {
        metric: mean([result["deepeval"]["scores"][metric] for result in results])
        for metric in metric_names
    }
    report = {
        "configuration": {
            "provider": "openrouter",
            "workflow_model": workflow_model,
            "judge_model": judge_model,
            "top_k": top_k,
        },
        "aggregate": aggregate,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "aggregate": aggregate}, indent=2))
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
