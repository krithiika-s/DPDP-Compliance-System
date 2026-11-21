from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agents.legal_rag.schemas import RetrievedLegalChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"

COLLECTION_NAME = "dpdp_legal_sources"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
        """Return primary-source passages within the configured distance."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        results = self.vector_store.similarity_search_with_score(
            query.strip(),
            k=k,
            filter={"authority": "primary"},
        )

        chunks = []

        for index, (document, distance) in enumerate(results, start=1):
            metadata = document.metadata

            required = (
                "source_file",
                "document_type",
                "authority",
                "pdf_page",
                "provision",
            )
            missing = [
                key for key in required
                if key not in metadata or metadata[key] is None
                or (isinstance(metadata[key], str) and not metadata[key].strip())
            ]
            if missing:
                raise ValueError(
                    f"Retrieved legal document is missing metadata: {missing}"
                )

            if metadata["authority"] != "primary":
                continue

            distance = float(distance)
            if distance > self.relevance_threshold:
                continue

            chunks.append(
                RetrievedLegalChunk(
                    source_id=(
                        f"LEGAL-{metadata['source_file']}-"
                        f"P{int(metadata['pdf_page'])}-R{index}"
                    ),
                    text=document.page_content,
                    source_file=str(metadata["source_file"]),
                    document_type=str(metadata["document_type"]),
                    authority=str(metadata["authority"]),
                    pdf_page=int(metadata["pdf_page"]),
                    provision=str(metadata["provision"]),
                    distance=distance,
                )
            )

        return chunks
