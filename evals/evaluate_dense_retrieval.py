from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "evals" / "fixtures" / "city_rag_golden.json"
DEFAULT_CONFIG = ROOT / "evals" / "config" / "metrics.json"
DEFAULT_OUTPUT = ROOT / "evals" / "dense_retrieval_results.json"

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense Chroma retrieval with an LLM claim judge.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "city_info_chroma")
    parser.add_argument("--collection", default="city_info_embeddings")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge", choices=("deepeval", "gemini", "lexical"), default="deepeval")
    return parser.parse_args()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = []
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip(" -\t")
        if len(tokenize(sentence)) >= 3:
            claims.append(sentence)
    return claims[:12] or ([text.strip()] if text.strip() else [])


def lexical_supported(claim: str, evidence_claims: Iterable[str]) -> bool:
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return False
    for evidence in evidence_claims:
        evidence_tokens = tokenize(evidence)
        overlap = claim_tokens & evidence_tokens
        if len(overlap) >= max(2, math.ceil(0.45 * len(claim_tokens))):
            return True
    return False


class ClaimJudge:
    def __init__(self, mode: str, model: str) -> None:
        self.mode = mode
        self.model = model
        self.client = None
        if mode == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to use the Gemini claim judge.")
            try:
                from google import genai

                self.client = genai.Client(api_key=api_key)
            except ImportError:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)

    def _generate_json(self, prompt: str) -> object:
        if hasattr(self.client, "models"):
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            text = response.text
        else:
            response = self.client.generate_content(prompt)
            text = response.text
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def claims(self, text: str) -> list[str]:
        if self.mode == "lexical":
            return lexical_claims(text)
        prompt = (
            "Break the text below into complete, non-overlapping factual claims. "
            "Return only a JSON array of strings. Ignore headings, navigation text, and duplicate facts.\n\n"
            f"TEXT:\n{text}"
        )
        value = self._generate_json(prompt)
        if not isinstance(value, list):
            raise RuntimeError("Judge returned claims that were not a JSON array.")
        return [str(claim).strip() for claim in value if str(claim).strip()]

    def compare_claims(self, claims: list[str], evidence_claims: list[str], question: str) -> list[bool]:
        if self.mode == "lexical":
            return [lexical_supported(claim, evidence_claims) for claim in claims]
        prompt = (
            "Question: " + question + "\n\n"
            "For each IDEAL claim, decide whether it is supported by the RETRIEVED evidence claims. "
            "Return only a JSON array of booleans in the same order, with true meaning supported.\n\n"
            "IDEAL claims:\n" + json.dumps(claims, ensure_ascii=False) + "\n\n"
            "RETRIEVED evidence claims:\n" + json.dumps(evidence_claims, ensure_ascii=False)
        )
        value = self._generate_json(prompt)
        if not isinstance(value, list) or len(value) != len(claims):
            raise RuntimeError("Judge returned an invalid claim-support array.")
        return [bool(item) for item in value]

    def retrieval_claim_relevance(self, claims: list[str], question: str, ideal_claims: list[str]) -> list[bool]:
        if self.mode == "lexical":
            return [lexical_supported(claim, ideal_claims) for claim in claims]
        prompt = (
            "For the QUESTION below, decide whether each RETRIEVED claim is relevant and non-contradictory. "
            "A claim that supports answering the question is relevant even if it is not identical to the ideal answer. "
            "Return only a JSON array of booleans in the same order.\n\n"
            f"QUESTION:\n{question}\n\nRETRIEVED claims:\n" + json.dumps(claims, ensure_ascii=False)
        )
        value = self._generate_json(prompt)
        if not isinstance(value, list) or len(value) != len(claims):
            raise RuntimeError("Judge returned an invalid retrieval-relevance array.")
        return [bool(item) for item in value]


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def deepeval_retrieval_metrics(
    case: dict,
    retrieved_docs: list[str],
    model: str,
    config: dict,
) -> dict[str, float | str | None]:
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.models import OpenAIModel
    from deepeval.test_case import LLMTestCase

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env to use DeepEval with OpenRouter.")

    judge_model = OpenAIModel(
        model=model,
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    threshold = float(config.get("threshold", 0.5))
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=case["ideal_answer"],
        expected_output=case["ideal_answer"],
        retrieval_context=retrieved_docs,
    )
    metrics = {
        "contextual_precision": ContextualPrecisionMetric(
            threshold=threshold, model=judge_model, async_mode=False, include_reason=True
        ),
        "contextual_recall": ContextualRecallMetric(
            threshold=threshold, model=judge_model, async_mode=False, include_reason=True
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=threshold, model=judge_model, async_mode=False, include_reason=True
        ),
    }
    output = {}
    reasons = {}
    successes = {}
    for name, metric in metrics.items():
        metric.measure(test_case)
        output[name] = float(metric.score or 0.0)
        reasons[name] = metric.reason
        successes[name] = bool(metric.success)
    return {
        "scores": output,
        "reasons": reasons,
        "successes": successes,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = json.loads(args.fixture.read_text(encoding="utf-8"))
    top_k = int(config.get("top_k", 5))

    temp_dir = Path(tempfile.mkdtemp(prefix="dense-rag-eval-"))
    try:
        isolated_chroma = temp_dir / "chroma"
        shutil.copytree(args.chroma_path, isolated_chroma)
        client = chromadb.PersistentClient(path=str(isolated_chroma))
        collection = client.get_collection(args.collection)
        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        embedder = SentenceTransformer(embedding_model)
        claim_judge = ClaimJudge(args.judge, config.get("judge_model", "gemini-2.0-flash")) if args.judge != "deepeval" else None
        judge = args.judge

        results = []
        for case in cases:
            query_embedding = embedder.encode(case["question"], normalize_embeddings=True).tolist()
            hits = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))
            retrieved_ids = hits["ids"][0]
            retrieved_docs = hits["documents"][0]
            retrieved = "\n\n".join(retrieved_docs)

            if args.judge == "deepeval":
                deepeval_result = deepeval_retrieval_metrics(
                    case,
                    retrieved_docs,
                    config.get("judge_model", "gemini-2.0-flash"),
                    config,
                )
                contextual_precision = deepeval_result["scores"]["contextual_precision"]
                contextual_recall = deepeval_result["scores"]["contextual_recall"]
                contextual_relevancy = deepeval_result["scores"]["contextual_relevancy"]
                ideal_claims = []
                retrieved_claims = []
                ideal_supported_by_retrieved = []
                retrieved_supported_by_ideal = []
                retrieved_relevant = []
            else:
                ideal_claims = claim_judge.claims(case["ideal_answer"])
                retrieved_claims = claim_judge.claims(retrieved)
                ideal_supported_by_retrieved = claim_judge.compare_claims(ideal_claims, retrieved_claims, case["question"])
                retrieved_supported_by_ideal = claim_judge.compare_claims(retrieved_claims, ideal_claims, case["question"])
                retrieved_relevant = claim_judge.retrieval_claim_relevance(retrieved_claims, case["question"], ideal_claims)
                contextual_precision = (
                    sum(retrieved_supported_by_ideal) / len(retrieved_claims) if retrieved_claims else 0.0
                )
                contextual_recall = (
                    sum(ideal_supported_by_retrieved) / len(ideal_claims) if ideal_claims else 0.0
                )
                contextual_relevancy = (
                    sum(retrieved_relevant) / len(retrieved_claims) if retrieved_claims else 0.0
                )
                deepeval_result = None

            required_ids = set(case.get("required_context_ids", []))
            relevant_retrieved = [chunk_id for chunk_id in retrieved_ids if chunk_id in required_ids]
            retrieval_precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
            retrieval_recall = len(relevant_retrieved) / len(required_ids) if required_ids else 0.0

            results.append({
                "id": case["id"],
                "question": case["question"],
                "ideal_answer": case["ideal_answer"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "retrieved_chunk_ids": retrieved_ids,
                "dense_distances": hits["distances"][0],
                "retrieved_context": retrieved_docs,
                "ideal_claims": ideal_claims,
                "retrieved_claims": retrieved_claims,
                "ideal_claim_support": ideal_supported_by_retrieved,
                "retrieved_claim_support": retrieved_supported_by_ideal,
                "retrieved_claim_relevance": retrieved_relevant,
                "deepeval": deepeval_result,
                "contextual_precision": round(contextual_precision, 4),
                "contextual_recall": round(contextual_recall, 4),
                "contextual_relevancy": round(contextual_relevancy, 4),
                "retrieval_precision_at_k": round(retrieval_precision, 4),
                "retrieval_recall_at_k": round(retrieval_recall, 4),
                "hit_rate": float(bool(relevant_retrieved)),
                "reciprocal_rank": reciprocal_rank(retrieved_ids, required_ids),
            })

        collection = None
        client = None
        gc.collect()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    aggregate = {
        metric: mean([row[metric] for row in results])
        for metric in (
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "retrieval_precision_at_k",
            "retrieval_recall_at_k",
            "hit_rate",
            "reciprocal_rank",
        )
    }
    report = {
        "configuration": {
            "top_k": top_k,
            "embedding_model": embedding_model,
            "judge": args.judge,
            "judge_model": config.get("judge_model"),
            "collection": args.collection,
        },
        "aggregate": aggregate,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dense retrieval evaluation: {len(results)} cases")
    print(json.dumps(aggregate, indent=2))
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
