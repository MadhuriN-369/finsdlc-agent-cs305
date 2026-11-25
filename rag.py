"""
rag.py

Lightweight local vector store for domain-specific Retrieval-Augmented
Generation (RAG). Uses the Gemini Embedding API (REST) for embeddings and
numpy for cosine-similarity search -- no Docker, no native DLLs, no
external vector database required.

Usage:
    from rag import LocalVectorStore

    store = LocalVectorStore("banking", api_key="AIza...")
    store.build_index()                       # one-time, cached to disk
    chunks = store.query("loan origination")  # top-k relevant chunks
    context = store.format_context(chunks)     # ready to inject into prompt
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)
BATCH_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
)

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
INDEX_DIR = KNOWLEDGE_BASE_DIR / "_indices"


def _index_path(domain: str) -> Path:
    return INDEX_DIR / f"{domain}.{EMBED_MODEL}.json"

CHUNK_TARGET_WORDS = 300   # target words per chunk
CHUNK_OVERLAP_WORDS = 40   # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str) -> list[dict]:
    """Split markdown into paragraph-level chunks.

    Each "## Section" is split into paragraphs; every chunk keeps its section
    heading in front of the text so retrieval and citations stay meaningful.
    """
    chunks = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = block.partition("\n")
        heading = heading.strip()
        for para in re.split(r"\n\s*\n", body.strip()):
            para = para.strip()
            if para:
                chunks.append({
                    "text": f"{heading}: {para}",
                    "source": f"{source} > {heading}",
                })
    return chunks


# ---------------------------------------------------------------------------
# Gemini Embedding API (REST, no SDK)
# ---------------------------------------------------------------------------

def _post_with_retry(url: str, api_key: str, payload: dict, timeout: int = 90):
    """POST with retries on busy/rate-limited/timeouts."""
    resp = None
    for attempt in range(4):
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            resp = None
            time.sleep(min(2 ** attempt, 8))
            continue
        if resp.status_code not in (429, 500, 503):
            break
        time.sleep(min(2 ** attempt, 8))
    if resp is None:
        raise RuntimeError("Embedding API did not respond (timeouts).")
    resp.raise_for_status()
    return resp


def _embed_single(text: str, api_key: str) -> list[float]:
    """Get the embedding for a query via the Gemini REST API."""
    url = EMBED_URL.format(model=EMBED_MODEL)
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    return _post_with_retry(url, api_key, payload).json()["embedding"]["values"]


def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    """Get embeddings for document chunks (max 100 per call)."""
    url = BATCH_EMBED_URL.format(model=EMBED_MODEL)
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                }
                for t in batch
            ]
        }
        data = _post_with_retry(url, api_key, payload).json()
        all_embeddings.extend(emb["values"] for emb in data["embeddings"])
    return all_embeddings


# ---------------------------------------------------------------------------
# Local Vector Store
# ---------------------------------------------------------------------------

class LocalVectorStore:
    """A simple file-backed vector store for domain-specific RAG.

    - Each domain corresponds to one or more .md/.txt files in
      knowledge_base/<domain>.md (single file) or knowledge_base/<domain>/*.md
    - Embeddings are cached to knowledge_base/_indices/<domain>.json
    - Queries use cosine similarity via numpy
    """

    def __init__(self, domain: str, api_key: str):
        self.domain = domain
        self.api_key = api_key
        self._chunks: list[dict] = []       # {"text": ..., "source": ...}
        self._embeddings: np.ndarray = None  # (n_chunks, embed_dim)

    # ----- Discovery -----

    @staticmethod
    def list_domains() -> list[str]:
        """Return available domains (files/dirs in knowledge_base/)."""
        if not KNOWLEDGE_BASE_DIR.exists():
            return []
        domains = set()
        for entry in KNOWLEDGE_BASE_DIR.iterdir():
            if entry.name.startswith(("_", ".")):
                continue
            if entry.is_file() and entry.suffix in (".md", ".txt"):
                domains.add(entry.stem)
            elif entry.is_dir():
                domains.add(entry.name)
        return sorted(domains)

    @staticmethod
    def index_exists(domain: str) -> bool:
        """Check whether a cached index exists for this domain."""
        return _index_path(domain).exists()

    # ----- Index building -----

    def _load_domain_texts(self) -> list[dict]:
        """Load and chunk all text files for the domain."""
        chunks = []

        # Check for single-file domain (knowledge_base/banking.md)
        single_file = KNOWLEDGE_BASE_DIR / f"{self.domain}.md"
        if not single_file.exists():
            single_file = KNOWLEDGE_BASE_DIR / f"{self.domain}.txt"
        if single_file.exists():
            text = single_file.read_text(encoding="utf-8")
            chunks.extend(_chunk_text(text, single_file.name))

        # Check for directory domain (knowledge_base/banking/*.md)
        domain_dir = KNOWLEDGE_BASE_DIR / self.domain
        if domain_dir.is_dir():
            for fpath in sorted(domain_dir.iterdir()):
                if fpath.suffix in (".md", ".txt") and not fpath.name.startswith("."):
                    text = fpath.read_text(encoding="utf-8")
                    chunks.extend(_chunk_text(text, fpath.name))

        return chunks

    def build_index(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> int:
        """Build the vector index: chunk files, embed, save to disk.

        Returns the number of chunks indexed.
        """
        if progress_callback:
            progress_callback("Loading and chunking domain knowledge...")
        self._chunks = self._load_domain_texts()
        if not self._chunks:
            raise ValueError(
                f"No knowledge-base files found for domain '{self.domain}'. "
                f"Add .md or .txt files to {KNOWLEDGE_BASE_DIR}/{self.domain}/"
            )

        if progress_callback:
            progress_callback(
                f"Embedding {len(self._chunks)} chunks via Gemini API..."
            )
        texts = [c["text"] for c in self._chunks]
        embeddings = _embed_batch(texts, self.api_key)
        self._embeddings = np.array(embeddings, dtype=np.float32)

        # Save to disk
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_data = {
            "domain": self.domain,
            "embed_model": EMBED_MODEL,
            "chunks": [
                {"text": c["text"], "source": c["source"]}
                for c in self._chunks
            ],
            "embeddings": self._embeddings.tolist(),
        }
        index_path = _index_path(self.domain)
        index_path.write_text(json.dumps(index_data), encoding="utf-8")

        if progress_callback:
            progress_callback(
                f"Index built: {len(self._chunks)} chunks saved to {index_path.name}"
            )
        return len(self._chunks)

    # ----- Index loading -----

    def load_index(self) -> bool:
        """Load a previously built index from disk. Returns True on success."""
        index_path = _index_path(self.domain)
        if not index_path.exists():
            return False
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self._chunks = data["chunks"]
        self._embeddings = np.array(data["embeddings"], dtype=np.float32)
        return True

    # ----- Querying -----

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Find the top-k most relevant chunks for the given query text.

        Returns a list of dicts with 'text', 'source', and 'score' keys.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            if not self.load_index():
                return []

        # Embed the query
        query_emb = np.array(_embed_single(text, self.api_key), dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        similarities = self._embeddings @ query_emb / norms

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self._chunks[idx]["text"],
                "source": self._chunks[idx]["source"],
                "score": float(similarities[idx]),
            })
        return results

    @staticmethod
    def format_context(chunks: list[dict], max_chars: int = 3000) -> str:
        """Format retrieved chunks into a context string for prompt injection."""
        if not chunks:
            return ""
        lines = ["Relevant domain knowledge (retrieved from knowledge base):"]
        total = 0
        for i, chunk in enumerate(chunks, 1):
            entry = f"\n[{i}] (source: {chunk['source']}, relevance: {chunk['score']:.0%})\n{chunk['text']}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n".join(lines)