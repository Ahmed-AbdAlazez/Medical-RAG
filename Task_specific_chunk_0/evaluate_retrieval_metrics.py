from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import config
from embedding_models import JinaSmallEmbedding, TfidfEmbedding
from ingest import SimpleVectorIndex, chunk_documents, load_pdfs


@dataclass(frozen=True)
class EvalQuestion:
    key: str
    text: str
    relevant_citations: set[str]


CHUNK_SETTINGS = [
    (400, 80),
    (600, 100),
]

QUESTIONS = [
    EvalQuestion(
        key="question_1",
        text="What recommendations did the committee make to ensure responsible use of ADHD medication?",
        relevant_citations={"ADHD.pdf:p46"},
    ),
    EvalQuestion(
        key="question_2",
        text="How can parents create better home and school environments for a child with ADHD?",
        relevant_citations={"parenting2015.pdf:p1"},
    ),
    EvalQuestion(
        key="question_3",
        text="What discipline methods should parents learn for children with ADHD?",
        relevant_citations={"parenting2015.pdf:p3"},
    ),
]

MODELS = [
    (
        "JinaSmallEmbedding",
        lambda: JinaSmallEmbedding(
            model_name=config.JINA_MODEL_NAME,
            fallback_model=TfidfEmbedding(max_features=config.TFIDF_MAX_FEATURES),
        ),
    ),
    ("TfidfEmbedding", lambda: TfidfEmbedding(max_features=config.TFIDF_MAX_FEATURES)),
]


def precision_recall_at_k(
    retrieved_citations: list[str], relevant_citations: set[str]
) -> tuple[float, float]:
    relevant_hits = [
        citation for citation in retrieved_citations if citation in relevant_citations
    ]
    precision = len(relevant_hits) / len(retrieved_citations) if retrieved_citations else 0.0
    recall = (
        len(set(relevant_hits)) / len(relevant_citations)
        if relevant_citations
        else 0.0
    )
    return precision, recall


def evaluate() -> list[dict[str, str | int | float]]:
    rows = []
    original_chunk_size = config.CHUNK_SIZE
    original_chunk_overlap = config.CHUNK_OVERLAP
    try:
        for chunk_size, chunk_overlap in CHUNK_SETTINGS:
            config.CHUNK_SIZE = chunk_size
            config.CHUNK_OVERLAP = chunk_overlap
            pages = load_pdfs(config.DATA_DIR)
            chunks = chunk_documents(pages)

            for model_name, model_factory in MODELS:
                embedding_model = model_factory()
                index = SimpleVectorIndex(embedding_model, chunks)
                runtime_name = getattr(embedding_model, "runtime_name", lambda: model_name)()
                row: dict[str, str | int | float] = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "number_of_chunks": len(chunks),
                    "embedding_model": runtime_name,
                }

                precisions = []
                recalls = []
                for question in QUESTIONS:
                    results = index.hybrid_search_with_relevance_scores(
                        question.text,
                        k=config.TOP_K,
                        semantic_weight=config.HYBRID_SEMANTIC_WEIGHT,
                        keyword_weight=config.HYBRID_KEYWORD_WEIGHT,
                    )
                    top_score = results[0][1] if results else 0.0
                    retrieved_citations = [
                        str(doc.metadata.get("citation", "")) for doc, _ in results
                    ]
                    precision, recall = precision_recall_at_k(
                        retrieved_citations, question.relevant_citations
                    )

                    row[f"{question.key}_score"] = round(top_score, 3)
                    row[f"{question.key}_citations"] = " ".join(retrieved_citations)
                    row[f"{question.key}_precision_at_3"] = round(precision, 3)
                    row[f"{question.key}_recall_at_3"] = round(recall, 3)
                    precisions.append(precision)
                    recalls.append(recall)

                row["avg_precision_at_3"] = round(sum(precisions) / len(precisions), 3)
                row["avg_recall_at_3"] = round(sum(recalls) / len(recalls), 3)
                rows.append(row)
    finally:
        config.CHUNK_SIZE = original_chunk_size
        config.CHUNK_OVERLAP = original_chunk_overlap
    return rows


def write_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    fieldnames = [
        "chunk_size",
        "chunk_overlap",
        "number_of_chunks",
        "embedding_model",
        "question_1_score",
        "question_1_citations",
        "question_1_precision_at_3",
        "question_1_recall_at_3",
        "question_2_score",
        "question_2_citations",
        "question_2_precision_at_3",
        "question_2_recall_at_3",
        "question_3_score",
        "question_3_citations",
        "question_3_precision_at_3",
        "question_3_recall_at_3",
        "avg_precision_at_3",
        "avg_recall_at_3",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = evaluate()
    output_path = Path("embedding_chunk_comparison.csv")
    write_csv(rows, output_path)
    print(f"Wrote {output_path}")
    for row in rows:
        print(
            f"{row['chunk_size']}/{row['chunk_overlap']} {row['embedding_model']} "
            f"precision@3={row['avg_precision_at_3']} recall@3={row['avg_recall_at_3']}"
        )


if __name__ == "__main__":
    main()
