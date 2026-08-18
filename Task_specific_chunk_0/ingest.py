from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import config
from embedding_models import JinaSmallEmbedding, TfidfEmbedding, normalize_tokens


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class PyPDFLoader:
    """Small PyPDFLoader-compatible fallback used by the notebook."""

    def __init__(self, file_path: str):
        self.file_path = str(file_path)

    def load(self) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required to parse PDFs. Install requirements.txt first."
            ) from exc

        reader = PdfReader(self.file_path)
        pages: List[Document] = []
        for page_index, page in enumerate(reader.pages):
            pages.append(
                Document(
                    page_content=clean_text(page.extract_text() or ""),
                    metadata={"source": self.file_path, "page": page_index},
                )
            )
        return pages


class RecursiveCharacterTextSplitter:
    """Dependency-light splitter with the LangChain method used in the notebook."""

    def __init__(
        self,
        chunk_size: int = 1600,
        chunk_overlap: int = 320,
        separators: Sequence[str] | None = None,
    ):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.separators = list(separators or ["\n\n", "\n", ". ", " ", ""])

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            for text in self._split_text(doc.page_content):
                if text.strip():
                    chunks.append(Document(text.strip(), dict(doc.metadata)))
        return chunks

    def _split_text(self, text: str) -> List[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks: List[str] = []
        start = 0
        while start < len(text):
            hard_end = min(start + self.chunk_size, len(text))
            end = self._best_boundary(text, start, hard_end)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _best_boundary(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)
        window = text[start:hard_end]
        minimum = int(len(window) * 0.55)
        for sep in self.separators:
            if sep == "":
                continue
            pos = window.rfind(sep)
            if pos >= minimum:
                return start + pos + len(sep)
        return hard_end


class SimpleVectorIndex:
    def __init__(self, embedding_function, documents: List[Document]):
        self.embedding_function = embedding_function
        self.documents = documents
        texts = [d.page_content for d in documents]
        fit = getattr(embedding_function, "fit", None)
        if callable(fit):
            fit(texts)
        self.vectors = embedding_function.embed_documents(texts)
        self.document_tokens = [normalize_tokens(d.page_content) for d in documents]
        self.idf = self._build_idf(self.document_tokens)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 3
    ) -> List[Tuple[Document, float]]:
        query_vector = self.embedding_function.embed_query(query)
        scored = [
            (doc, cosine_similarity(query_vector, vector))
            for doc, vector in zip(self.documents, self.vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def hybrid_search_with_relevance_scores(
        self,
        query: str,
        k: int = 3,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ) -> List[Tuple[Document, float]]:
        query_vector = self.embedding_function.embed_query(query)
        query_tokens = normalize_tokens(query)
        semantic_scores = [
            cosine_similarity(query_vector, vector) for vector in self.vectors
        ]
        keyword_scores = [
            self._keyword_score(query_tokens, tokens) for tokens in self.document_tokens
        ]
        max_keyword = max(keyword_scores) if keyword_scores else 0.0
        if max_keyword > 0:
            keyword_scores = [score / max_keyword for score in keyword_scores]

        scored = []
        for doc, semantic_score, keyword_score in zip(
            self.documents, semantic_scores, keyword_scores
        ):
            hybrid_score = (
                semantic_weight * semantic_score + keyword_weight * keyword_score
            )
            doc.metadata["semantic_score"] = round(semantic_score, 3)
            doc.metadata["keyword_score"] = round(keyword_score, 3)
            scored.append((doc, hybrid_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def _build_idf(self, tokenized_documents: Sequence[Sequence[str]]) -> dict[str, float]:
        document_count = len(tokenized_documents)
        document_frequency: dict[str, int] = {}
        for tokens in tokenized_documents:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        return {
            token: math.log((1 + document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

    def _keyword_score(
        self, query_tokens: Sequence[str], document_tokens: Sequence[str]
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        document_counts: dict[str, int] = {}
        for token in document_tokens:
            document_counts[token] = document_counts.get(token, 0) + 1
        score = 0.0
        for token in set(query_tokens):
            term_frequency = document_counts.get(token, 0)
            if term_frequency:
                score += (1 + math.log(term_frequency)) * self.idf.get(token, 1.0)
        return score

    def persist(self, persist_directory: Path | str = config.CHROMA_DIR) -> None:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        (persist_path / "vector_index.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        persist_directory: Path | str = config.CHROMA_DIR,
        embedding_function=None,
    ) -> "SimpleVectorIndex":
        payload = json.loads(
            (Path(persist_directory) / "vector_index.json").read_text(encoding="utf-8")
        )
        documents = [
            Document(item["page_content"], item.get("metadata", {})) for item in payload
        ]
        return cls(embedding_function or get_embedding_function(), documents)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?i)\bwatermark\b", "", text)
    text = re.sub(r"© NICE 2026\. All rights reserved\..*", "", text)
    text = re.sub(r"Subject to Notice of rights.*", "", text)
    text = re.sub(r"(?m)^Page\s+\d+\s+(of\s+\d+)?\s*$", "", text)
    text = re.sub(r"(?m)^help4adhd\.org\s*\|\s*\d+\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_section(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if re.match(
            r"^(\d+(\.\d+)*\.?|recommendation|remarks|background|parenting|setting|help)\b",
            candidate,
            re.I,
        ):
            return candidate[:120]
    return "Unspecified section"


def load_pdfs(data_dir: Path | str = config.DATA_DIR) -> List[Document]:
    data_path = Path(data_dir)
    pdf_paths = sorted(data_path.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_path}")

    pages: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        for page in loader.load():
            if not page.page_content.strip():
                continue
            page_index = int(page.metadata.get("page", 0))
            page.metadata.update(
                {
                    "source": str(pdf_path),
                    "document_name": pdf_path.name,
                    "page_number": page_index + 1,
                    "citation": f"{pdf_path.name}:p{page_index + 1}",
                    "section": infer_section(page.page_content),
                }
            )
            pages.append(page)
    return pages


def chunk_documents(pages: Sequence[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", "; ", ": ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    per_document_counts: dict[str, int] = {}
    for chunk in chunks:
        document_name = chunk.metadata.get("document_name", "document")
        per_document_counts[document_name] = per_document_counts.get(document_name, 0) + 1
        chunk.metadata["chunk_id"] = (
            f"{Path(document_name).stem}-p{chunk.metadata.get('page_number', '?')}-"
            f"c{per_document_counts[document_name]:04d}"
        )
        chunk.metadata["citation"] = (
            f"{document_name}:p{chunk.metadata.get('page_number', '?')}"
        )
        chunk.metadata.setdefault("section", infer_section(chunk.page_content))
    return chunks


def get_embedding_function():
    return JinaSmallEmbedding(
        model_name=getattr(config, "JINA_MODEL_NAME", "jina-embeddings-v2-small-en"),
        fallback_model=TfidfEmbedding(
            max_features=getattr(config, "TFIDF_MAX_FEATURES", 512)
        ),
        api_key=config.JINA_API_KEY
    )


def build_index(chunks: Sequence[Document]) -> SimpleVectorIndex:
    vectordb = SimpleVectorIndex(get_embedding_function(), list(chunks))
    vectordb.persist(config.CHROMA_DIR)
    return vectordb


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    pages = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(pages)
    index = build_index(chunks)
    model_name = getattr(index.embedding_function, "runtime_name", lambda: "embedding")()
    print(f"Loaded {len(pages)} pages from {config.DATA_DIR}")
    print(f"Built {len(chunks)} chunks at {config.CHROMA_DIR}")
    print(f"Embedding model: {model_name}")
    for doc, score in index.hybrid_search_with_relevance_scores(
        "How can parents create better home and school environments for a child with ADHD?",
        k=config.TOP_K,
    ):
        print(
            f"{score:.3f} | semantic={doc.metadata.get('semantic_score')} | "
            f"keyword={doc.metadata.get('keyword_score')} | "
            f"{doc.metadata.get('citation')} | {doc.metadata.get('section')}"
        )
