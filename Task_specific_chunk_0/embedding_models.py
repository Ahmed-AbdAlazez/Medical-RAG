from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Sequence


class JinaSmallEmbedding:
    """Jina small embedding wrapper with a local TF-IDF fallback for offline runs."""

    def __init__(
        self,
        model_name: str = "jina-embeddings-v2-small-en",
        fallback_model: "TfidfEmbedding | None" = None,
        api_key: str | None = None,
        endpoint: str = "https://api.jina.ai/v1/embeddings",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.endpoint = endpoint
        self.fallback_model = fallback_model or TfidfEmbedding(max_features=512)
        self.fallback_used = not bool(self.api_key)

    @property
    def dimensions(self) -> int:
        if self.fallback_used:
            return self.fallback_model.dimensions
        return 512

    def fit(self, texts: Sequence[str]) -> None:
        if self.fallback_used:
            self.fallback_model.fit(texts)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.api_key:
            try:
                return self._embed_with_jina_api(texts, task="retrieval.passage")
            except RuntimeError as exc:
                print(f"Jina API unavailable, using TF-IDF fallback: {exc}")
                self.fallback_used = True
                self.fallback_model.fit(texts)
        return self.fallback_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.api_key and not self.fallback_used:
            try:
                return self._embed_with_jina_api([text], task="retrieval.query")[0]
            except RuntimeError as exc:
                print(f"Jina API unavailable for query, using TF-IDF fallback: {exc}")
                self.fallback_used = True
        return self.fallback_model.embed_query(text)

    def runtime_name(self) -> str:
        if self.fallback_used:
            return f"JinaSmallEmbedding({self.model_name}) [TF-IDF fallback]"
        return f"JinaSmallEmbedding({self.model_name})"

    def _embed_with_jina_api(self, texts: Sequence[str], task: str) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": list(texts),
            "task": task,
            "normalized": True,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        try:
            return [item["embedding"] for item in body["data"]]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Jina response: {str(body)[:300]}") from exc


class TfidfEmbedding:
    """Corpus-aware TF-IDF embedding used for local retrieval experiments."""

    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vocabulary: list[str] = []
        self.idf: dict[str, float] = {}

    @property
    def dimensions(self) -> int:
        return len(self.vocabulary) or self.max_features

    def fit(self, texts: Sequence[str]) -> None:
        tokenized_texts = [normalize_tokens(text) for text in texts]
        document_count = len(tokenized_texts)
        document_frequency: Counter[str] = Counter()
        corpus_frequency: Counter[str] = Counter()

        for tokens in tokenized_texts:
            document_frequency.update(set(tokens))
            corpus_frequency.update(tokens)

        ranked_terms = sorted(
            corpus_frequency,
            key=lambda token: (document_frequency[token], corpus_frequency[token], token),
            reverse=True,
        )
        self.vocabulary = ranked_terms[: self.max_features]
        self.idf = {
            token: math.log((1 + document_count) / (1 + document_frequency[token])) + 1
            for token in self.vocabulary
        }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.vocabulary:
            self.fit(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def runtime_name(self) -> str:
        return "TfidfEmbedding"

    def _embed(self, text: str) -> list[float]:
        counts = Counter(normalize_tokens(text))
        vector = []
        for token in self.vocabulary:
            term_frequency = counts.get(token, 0)
            value = 0.0
            if term_frequency:
                value = (1 + math.log(term_frequency)) * self.idf.get(token, 1.0)
            vector.append(value)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def normalize_tokens(text: str) -> list[str]:
    text = text.lower()
    replacements = {
        "attention deficit hyperactivity disorder": "adhd",
        "attention-deficit/hyperactivity disorder": "adhd",
        "attention deficit disorder": "adhd",
        "medicines": "medication",
        "medicine": "medication",
        "drug treatment": "medication treatment",
        "baseline assessment": "baseline_assessment",
        "physical health": "physical_health",
        "mental health": "mental_health",
        "blood pressure": "blood_pressure",
        "heart rate": "pulse",
        "parent training": "parent_training",
        "parent-training": "parent_training",
        "behaviour management": "behavior_management",
        "behavior management": "behavior_management",
        "time outs": "time_outs",
        "school environments": "school_environment",
        "home environments": "home_environment",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    tokens = re.findall(r"[a-z][a-z0-9_]+", text)
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "of",
        "to",
        "in",
        "with",
        "is",
        "are",
        "be",
        "by",
        "on",
        "as",
        "from",
        "what",
        "why",
        "how",
        "should",
        "patient",
        "patients",
        "person",
        "people",
        "child",
        "children",
    }
    return [token for token in tokens if token not in stop_words]
