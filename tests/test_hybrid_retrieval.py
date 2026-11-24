import math

from langchain_core.documents import Document

from hybrid_retrieval import (
    augment_with_triplets,
    build_bm25_retriever,
    merge_documents,
    rerank_documents,
)


class FixedReranker:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits

    def predict(self, sentences: list[list[str]]) -> list[float]:
        return self.logits


def test_bm25_retriever_returns_lexically_matching_chunk() -> None:
    documents = [
        Document(page_content="needleterm Bengaluru urban district"),
        Document(page_content="Mysuru district agricultural profile"),
        Document(page_content="Bengaluru economy report"),
    ]

    retrieved = build_bm25_retriever(documents, top_k=1).invoke("needleterm")

    assert retrieved[0].page_content == documents[0].page_content


def test_rerank_keeps_scores_equal_to_threshold_and_sorts() -> None:
    equal_threshold_logit = math.log(0.7 / 0.3)
    documents = [
        Document(page_content="lower ranked"),
        Document(page_content="exact threshold"),
    ]

    results = rerank_documents(
        "question",
        documents,
        FixedReranker([0.0, equal_threshold_logit]),
    )

    assert [document.page_content for document, _ in results] == ["exact threshold"]
    assert results[0][1] == 0.7


def test_merge_documents_deduplicates_by_chunk_id_and_keeps_first_order() -> None:
    dense = [Document(page_content="dense", metadata={"chunk_id": "1"})]
    bm25 = [
        Document(page_content="duplicate", metadata={"chunk_id": "1"}),
        Document(page_content="bm25", metadata={"chunk_id": "2"}),
    ]

    merged = merge_documents(dense, bm25)

    assert [document.page_content for document in merged] == ["dense", "bm25"]


def test_triplets_augment_only_retained_chunk_evidence() -> None:
    documents = [Document(page_content="The city authority is Bruhat Bengaluru Mahanagara Palike.")]
    triplets = [
        {
            "subject": "Bruhat Bengaluru Mahanagara Palike",
            "relation": "published",
            "object": "Performance Report",
        },
        {"subject": "Unrelated Entity", "relation": "has", "object": "Unrelated Fact"},
    ]

    augmented = augment_with_triplets(documents, triplets)

    assert len(augmented) == 1
    assert "Bruhat Bengaluru Mahanagara Palike published Performance Report" in augmented[0]
    assert "Unrelated Entity" not in augmented[0]
