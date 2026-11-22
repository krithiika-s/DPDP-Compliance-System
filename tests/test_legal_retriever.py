
import pytest
from langchain_core.documents import Document

from agents.legal_rag.retriever import LegalRetriever


@pytest.fixture
def retriever(tmp_path):
    instance = LegalRetriever.__new__(LegalRetriever)
    instance.persist_directory = tmp_path
    instance.relevance_threshold = 0.5
    return instance


def metadata(**overrides):
    result = {
        "source_file": "dpdp-act.pdf",
        "document_type": "DPDP Act, 2023",
        "authority": "primary",
        "pdf_page": 4,
        "provision": "Section 6: Consent",
    }
    result.update(overrides)
    return result




class FakeVectorStore:
    """A lightweight fake that simulates the ChromaDB methods we use."""

    def __init__(self, results):
        self.results = results
        self.kwargs = None
        self.calls = []

    def get(self, where=None, include=None):
        metadatas = []

        for document, _ in self.results:
            if where and document.metadata.get("authority") != "primary":
                continue

            metadatas.append(document.metadata)

        return {"metadatas": metadatas}

    def similarity_search_with_score(self, query, k=5, filter=None):
        self.kwargs = ((query,), {"k": k, "filter": filter})
        self.calls.append((query, k, filter))

        results = self.results

        if filter:
            conditions = filter.get("$and", [filter])

            for condition in conditions:
                results = [
                    (document, distance)
                    for document, distance in results
                    if all(
                        document.metadata.get(key) == value
                        for key, value in condition.items()
                    )
                ]

        return sorted(results, key=lambda item: item[1])[:k]

def test_retriever_filters_to_primary_authority(retriever):
    retriever.vector_store = FakeVectorStore([
        (Document(page_content="Consent provision", metadata=metadata()), 0.2),
    ])

    results = retriever.retrieve("consent")

    assert len(results) == 1
    assert retriever.vector_store.kwargs[1]["filter"] == {
        "authority": "primary"
    }


def test_retriever_returns_correct_chunk_metadata(retriever):
    retriever.vector_store = FakeVectorStore([
        (Document(page_content="Consent provision", metadata=metadata()), 0.2),
    ])

    result = retriever.retrieve("consent")[0]

    assert result.text == "Consent provision"
    assert result.source_file == "dpdp-act.pdf"
    assert result.document_type == "DPDP Act, 2023"
    assert result.authority == "primary"
    assert result.pdf_page == 4
    assert result.provision == "Section 6: Consent"
    assert result.distance == 0.2


def test_retriever_excludes_results_above_distance_threshold(retriever):
    retriever.vector_store = FakeVectorStore([
        (Document(page_content="Relevant", metadata=metadata()), 0.2),
        (Document(page_content="Irrelevant", metadata=metadata()), 0.8),
    ])

    results = retriever.retrieve("consent")

    assert len(results) == 1
    assert results[0].text == "Relevant"


def test_retriever_rejects_missing_metadata(retriever):
    incomplete = metadata()
    del incomplete["source_file"]

    retriever.vector_store = FakeVectorStore([
        (Document(page_content="Missing source", metadata=incomplete), 0.2),
    ])

    with pytest.raises(ValueError, match="missing metadata"):
        retriever.retrieve("consent")


def test_retriever_rejects_blank_metadata(retriever):
    incomplete = metadata(provision="   ")

    retriever.vector_store = FakeVectorStore([
        (Document(page_content="Blank provision", metadata=incomplete), 0.2),
    ])

    with pytest.raises(ValueError, match="missing metadata"):
        retriever.retrieve("consent")


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_retriever_rejects_empty_query(retriever, query):
    retriever.vector_store = FakeVectorStore([])

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)


def test_retriever_rejects_invalid_k(retriever):
    retriever.vector_store = FakeVectorStore([])

    with pytest.raises(ValueError, match="k"):
        retriever.retrieve("consent", k=0)


def test_retriever_rejects_negative_threshold(tmp_path):
    with pytest.raises(ValueError, match="cannot be negative"):
        LegalRetriever(
            persist_directory=tmp_path,
            embeddings=object(),
            relevance_threshold=-0.1,
        )


def test_retriever_deduplicates_identical_chunks(retriever):
    document = Document(
        page_content="Consent must be free, specific and informed.",
        metadata=metadata(),
    )

    retriever.vector_store = FakeVectorStore([
        (document, 0.2),
        (document, 0.3),
    ])

    results = retriever.retrieve("consent", k=5)

    assert len(results) == 1
    assert results[0].text == document.page_content


def test_explicit_section_retrieval_bypasses_threshold(retriever):
    document = Document(
        page_content="Consent must be free, specific and informed.",
        metadata=metadata(),
    )

    retriever.vector_store = FakeVectorStore([
        (document, 0.8),
    ])

    results = retriever.retrieve(
        "What conditions must consent satisfy under Section 6?"
    )

    assert len(results) == 1
    assert results[0].provision == "Section 6: Consent"
    assert results[0].distance == 0.8


def test_explicit_rule_retrieval(retriever):
    document = Document(
        page_content="The data fiduciary must publish the required information.",
        metadata=metadata(
            provision="Rule 5: Notice",
            pdf_page=12,
            source_file="dpdp-rules.pdf",
            document_type="DPDP Rules, 2025",
        ),
    )

    retriever.vector_store = FakeVectorStore([
        (document, 0.2),
    ])

    results = retriever.retrieve("Explain Rule 5")

    assert len(results) == 1
    assert results[0].provision == "Rule 5: Notice"
    assert results[0].document_type == "DPDP Rules, 2025"


def test_multiple_explicit_provisions_are_balanced(retriever):
    retriever.vector_store = FakeVectorStore([
        (
            Document(
                page_content="Consent must be free and informed.",
                metadata=metadata(provision="Section 6: Consent"),
            ),
            0.2,
        ),
        (
            Document(
                page_content="Consent must be unambiguous.",
                metadata=metadata(
                    provision="Section 6: Consent",
                    pdf_page=5,
                ),
            ),
            0.3,
        ),
        (
            Document(
                page_content="The data principal has the right to correction.",
                metadata=metadata(
                    provision="Rule 5: Rights",
                    pdf_page=12,
                    source_file="dpdp-rules.pdf",
                    document_type="DPDP Rules, 2025",
                ),
            ),
            0.1,
        ),
        (
            Document(
                page_content="The data principal can request erasure.",
                metadata=metadata(
                    provision="Rule 5: Rights",
                    pdf_page=13,
                    source_file="dpdp-rules.pdf",
                    document_type="DPDP Rules, 2025",
                ),
            ),
            0.2,
        ),
    ])

    results = retriever.retrieve(
        "Explain Section 6 and Rule 5",
        k=2,
    )

    provisions = {chunk.provision for chunk in results}

    assert provisions == {
        "Section 6: Consent",
        "Rule 5: Rights",
    }


def test_citation_source_ids_are_unique_for_distinct_chunks(retriever):
    retriever.vector_store = FakeVectorStore([
        (
            Document(
                page_content="Consent must be free and informed.",
                metadata=metadata(),
            ),
            0.2,
        ),
        (
            Document(
                page_content="Consent must also be unambiguous.",
                metadata=metadata(),
            ),
            0.3,
        ),
    ])

    results = retriever.retrieve("consent", k=5)

    source_ids = [chunk.source_id for chunk in results]

    assert len(source_ids) == 2
    assert len(set(source_ids)) == 2


def test_citation_source_ids_are_deterministic(retriever):
    document = Document(
        page_content="Consent must be free and informed.",
        metadata=metadata(),
    )

    retriever.vector_store = FakeVectorStore([
        (document, 0.2),
    ])

    first_result = retriever.retrieve("consent")[0]
    second_result = retriever.retrieve("consent")[0]

    assert first_result.source_id == second_result.source_id
