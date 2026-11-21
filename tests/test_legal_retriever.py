from pathlib import Path

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
    def __init__(self, results):
        self.results = results
        self.kwargs = None

    def similarity_search_with_score(self, *args, **kwargs):
        self.kwargs = (args, kwargs)
        return self.results


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
