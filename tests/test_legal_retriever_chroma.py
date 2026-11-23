
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from agents.legal_rag.retriever import LegalRetriever


class DeterministicEmbeddings(Embeddings):
    """Small local embeddings for testing ChromaDB without model downloads."""

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 1.0]


def test_real_chroma_retrieves_multiple_explicit_primary_provisions(tmp_path):
    retriever = LegalRetriever(
        persist_directory=tmp_path,
        collection_name="legal-retrieval-test",
        embeddings=DeterministicEmbeddings(),
        relevance_threshold=0.5,
    )

    documents = [
        Document(
            page_content="Consent must be free, specific and informed.",
            metadata={
                "source_file": "dpdp-act.pdf",
                "document_type": "DPDP Act, 2023",
                "authority": "primary",
                "pdf_page": 4,
                "provision": "Section 6: Consent",
            },
        ),
        Document(
            page_content="The data principal may exercise the right to correction.",
            metadata={
                "source_file": "dpdp-rules.pdf",
                "document_type": "DPDP Rules, 2025",
                "authority": "primary",
                "pdf_page": 12,
                "provision": "Rule 5: Rights",
            },
        ),
        Document(
            page_content="Secondary commentary about consent.",
            metadata={
                "source_file": "commentary.pdf",
                "document_type": "Commentary",
                "authority": "secondary",
                "pdf_page": 2,
                "provision": "Section 6: Consent",
            },
        ),
    ]

    retriever.vector_store.add_documents(documents)

    results = retriever.retrieve(
        "Explain Section 6 and Rule 5",
        k=2,
    )

    provisions = {chunk.provision for chunk in results}

    assert len(results) == 2
    assert provisions == {
        "Section 6: Consent",
        "Rule 5: Rights",
    }

    assert all(chunk.authority == "primary" for chunk in results)

    # The deterministic embeddings deliberately produce a distance
    # above the threshold. Explicit references must still be returned.
    assert all(chunk.distance > 0.5 for chunk in results)
