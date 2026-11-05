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
DEFAULT_OUTPUT = ROOT / "evals" / "generator_results.json"

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer generation with DeepEval and OpenRouter.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "city_info_chroma")
    parser.add_argument("--collection", default="city_info_embeddings")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def openrouter_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env to run generator evaluation.")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url), model


def generate_answer(client: OpenAI, model: str, question: str, context: list[str]) -> str:
    context_text = "\n\n".join(context)
    prompt = (
        "Answer the question using only the supplied context. "
        "If the context does not contain enough information, say so explicitly.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def evaluate_answer(case: dict, actual_output: str, context: list[str], model: str, config: dict) -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.models import OpenAIModel
    from deepeval.test_case import LLMTestCase
    from deepeval.test_case import LLMTestCaseParams

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    judge = OpenAIModel(model=model, api_key=api_key, base_url=base_url)
    threshold = float(config.get("threshold", 0.5))
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["ideal_answer"],
        retrieval_context=context,
    )
    correctness = GEval(
        name="Answer Correctness",
        criteria=(
            "Evaluate whether the actual answer correctly and completely matches the expected answer. "
            "The response should contain the expected facts and should not introduce conflicting facts."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
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
    client, generator_model = openrouter_client()
    judge_model = os.getenv("OPENROUTER_MODEL", config.get("judge_model", "openai/gpt-4o-mini"))

    temp_dir = Path(tempfile.mkdtemp(prefix="generator-rag-eval-"))
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
            retrieved_context = hits["documents"][0]
            actual_output = generate_answer(client, generator_model, case["question"], retrieved_context)
            deepeval_result = evaluate_answer(
                case,
                actual_output,
                retrieved_context,
                judge_model,
                config,
            )
            results.append({
                "id": case["id"],
                "question": case["question"],
                "ideal_answer": case["ideal_answer"],
                "actual_output": actual_output,
                "retrieved_chunk_ids": hits["ids"][0],
                "retrieved_context": retrieved_context,
                "deepeval": deepeval_result,
            })

        collection = None
        chroma_client = None
        gc.collect()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    aggregate = {
        metric: mean([
            result["deepeval"]["scores"][metric]
            for result in results
        ])
        for metric in ("answer_relevancy", "faithfulness", "answer_correctness")
    }
    report = {
        "configuration": {
            "provider": "openrouter",
            "generator_model": generator_model,
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
