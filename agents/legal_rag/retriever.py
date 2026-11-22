
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agents.legal_rag.schemas import RetrievedLegalChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"

COLLECTION_NAME = "dpdp_legal_sources"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED_METADATA = (
    "source_file",
    "document_type",
    "authority",
    "pdf_page",
    "provision",
)


def _explicit_provisions(query: str) -> set[tuple[str, str]]:
    """Extract explicit Section and Rule references from a query."""
    pattern = r"\b(section|rule)\s+(\d+[a-z]?)\b"

    return {
        (kind.lower(), number.lower())
        for kind, number in re.findall(pattern, query, re.IGNORECASE)
    }


def _matches_explicit_provision(
    provision: str,
    references: set[tuple[str, str]],
) -> bool:
    """Check whether a provision matches an explicit query reference."""
    for kind, number in references:
        pattern = rf"^\s*{kind}\s+{re.escape(number)}(?:\b|[:.(])"

        if re.search(pattern, provision, re.IGNORECASE):
            return True

    return False


def _validate_metadata(metadata: dict) -> None:
    """Reject retrieved documents with incomplete source metadata."""
    missing = [
        key
        for key in REQUIRED_METADATA
        if key not in metadata
        or metadata[key] is None
        or (
            isinstance(metadata[key], str)
            and not metadata[key].strip()
        )
    ]

    if missing:
        raise ValueError(
            f"Retrieved legal document is missing metadata: {missing}"
        )


def _chunk_key(
    source_file: str,
    pdf_page: int,
    provision: str,
    text: str,
) -> tuple[str, int, str, str]:
    """Create a normalized identity for duplicate detection."""
    normalized_text = " ".join(text.split())

    return (
        source_file,
        pdf_page,
        provision,
        normalized_text,
    )


def _make_chunk(document, distance: float) -> RetrievedLegalChunk:
    """Convert a retrieved document into a traceable legal chunk."""
    metadata = document.metadata

    _validate_metadata(metadata)

    source_file = str(metadata["source_file"])
    pdf_page = int(metadata["pdf_page"])
    provision = str(metadata["provision"])
    document_type = str(metadata["document_type"])
    authority = str(metadata["authority"])

    identity = "|".join(
        [
            source_file,
            str(pdf_page),
            provision,
            " ".join(document.page_content.split()),
        ]
    )

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return RetrievedLegalChunk(
        source_id=f"LEGAL-{digest}",
        text=document.page_content,
        source_file=source_file,
        document_type=document_type,
        authority=authority,
        pdf_page=pdf_page,
        provision=provision,
        distance=float(distance),
    )


class LegalRetriever:
    """Retrieve relevant authoritative DPDP Act and Rules passages."""

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
        relevance_threshold: float = 1.0,
    ) -> None:
        if relevance_threshold < 0:
            raise ValueError("relevance_threshold cannot be negative.")

        self.relevance_threshold = float(relevance_threshold)
        self.persist_directory = Path(persist_directory)

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self.persist_directory}"
            )

        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
    ) -> list[RetrievedLegalChunk]:
        """Retrieve explicit legal provisions before semantic evidence."""
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query = query.strip()
        references = _explicit_provisions(query)

        chunks: list[RetrievedLegalChunk] = []
        seen: set[tuple[str, int, str, str]] = set()

        def add_results(results, *, explicit: bool = False) -> None:
            for document, distance in results:
                metadata = document.metadata

                _validate_metadata(metadata)

                if metadata["authority"] != "primary":
                    continue

                distance = float(distance)

                if not explicit and distance > self.relevance_threshold:
                    continue

                chunk = _make_chunk(document, distance)

                key = _chunk_key(
                    chunk.source_file,
                    chunk.pdf_page,
                    chunk.provision,
                    chunk.text,
                )

                if key in seen:
                    continue

                seen.add(key)
                chunks.append(chunk)

        # Retrieve explicitly requested provisions first.
        if references:
            collection = self.vector_store.get(
                where={"authority": "primary"},
                include=["metadatas"],
            )

            metadata_records = collection.get("metadatas") or []

            matching_provisions = {
                str(metadata["provision"])
                for metadata in metadata_records
                if metadata
                and metadata.get("authority") == "primary"
                and metadata.get("provision")
                and _matches_explicit_provision(
                    str(metadata["provision"]),
                    references,
                )
            }

            provision_results = []

            for provision in sorted(matching_provisions):
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=max(3, k),
                    filter={
                        "$and": [
                            {"authority": "primary"},
                            {"provision": provision},
                        ]
                    },
                )

                provision_results.append(results)

            # Round-robin retrieval gives each requested provision
            # an opportunity to appear before taking extra chunks.
            max_results = max(
                (len(results) for results in provision_results),
                default=0,
            )

            for index in range(max_results):
                for results in provision_results:
                    if index < len(results):
                        add_results([results[index]], explicit=True)

        # Retrieve additional semantic evidence.
        semantic_results = self.vector_store.similarity_search_with_score(
            query,
            k=max(k * 3, k),
            filter={"authority": "primary"},
        )

        add_results(semantic_results)

        return chunks[:k]
