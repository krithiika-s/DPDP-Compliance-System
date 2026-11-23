from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from agents.legal_rag import ingest


def make_page(text, page=1, document_type="DPDP Act, 2023"):
    return Document(
        page_content=text,
        metadata={
            "source_file": "sample.pdf",
            "document_type": document_type,
            "authority": "primary",
            "pdf_page": page,
        },
    )


def test_stable_chunk_id_is_deterministic():
    chunk = make_page("Consent is required.")
    chunk.metadata["start_index"] = 10
    chunk.metadata["provision"] = "Section 6: Consent"

    assert ingest.stable_chunk_id(chunk) == ingest.stable_chunk_id(chunk)


def test_stable_chunk_id_changes_when_content_changes():
    first = make_page("Original legal text.")
    second = make_page("Changed legal text.")

    assert ingest.stable_chunk_id(first) != ingest.stable_chunk_id(second)


def test_stable_chunk_id_changes_when_page_changes():
    first = make_page("Same text", page=1)
    second = make_page("Same text", page=2)

    assert ingest.stable_chunk_id(first) != ingest.stable_chunk_id(second)


def test_find_act_section_heading():
    text = (
        "PREAMBLE TEXT\n"
        "6. Consent by Data Principal\n"
        "(1) Consent shall be free and informed.\n"
    )

    markers = ingest._find_provision_markers(text, "DPDP Act, 2023")

    assert any(label.startswith("Section 6:") for _, label in markers)


def test_find_rules_heading():
    text = (
        "PREAMBLE TEXT\n"
        "3. Notice by Data Fiduciary\n"
        "(1) The notice shall contain information.\n"
    )

    markers = ingest._find_provision_markers(text, "DPDP Rules, 2025")

    assert any(label.startswith("Rule 3:") for _, label in markers)


def test_numbered_parenthetical_subitems_are_not_main_headings():
    text = (
        "6. Consent by Data Principal\n"
        "(1) The first condition.\n"
        "(2) The second condition.\n"
    )

    markers = ingest._find_provision_markers(text, "DPDP Act, 2023")

    assert len(markers) == 1
    assert markers[0][1].startswith("Section 6:")


def test_split_legal_pages_preserves_provision_metadata():
    page = make_page(
        "Preamble text.\n\n"
        "6. Consent by Data Principal\n"
        "Consent must be free and informed.\n\n"
        "7. Certain legitimate uses\n"
        "Personal data may be processed for lawful purposes."
    )

    chunks = ingest.split_legal_pages([page])

    assert chunks
    assert all("provision" in chunk.metadata for chunk in chunks)
    assert any(
        chunk.metadata["provision"].startswith("Section 6:")
        for chunk in chunks
    )
    assert any(
        chunk.metadata["provision"].startswith("Section 7:")
        for chunk in chunks
    )


def test_split_legal_pages_preserves_source_and_page():
    page = make_page("Preamble text.\n\n6. Consent by Data Principal\nConsent text.")

    chunks = ingest.split_legal_pages([page])

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "sample.pdf"
        assert chunk.metadata["pdf_page"] == 1
        assert chunk.metadata["authority"] == "primary"


def test_validate_rejects_empty_pages():
    source = {
        "filename": "sample.pdf",
        "document_type": "DPDP Act, 2023",
    }

    with pytest.raises(RuntimeError, match="No usable pages"):
        ingest.validate_source_chunks(source, [], [])


def test_validate_rejects_unidentified_chunks():
    source = {
        "filename": "sample.pdf",
        "document_type": "DPDP Act, 2023",
    }
    page = make_page("Some text")
    chunk = make_page("Some text")
    chunk.metadata["provision"] = "not_identified"

    with pytest.raises(RuntimeError, match="Unidentified"):
        ingest.validate_source_chunks(source, [page], [chunk])


def test_validate_rejects_non_contiguous_pages():
    source = {
        "filename": "sample.pdf",
        "document_type": "DPDP Act, 2023",
    }
    pages = [make_page("Page one", 1), make_page("Page three", 3)]
    chunk = make_page("Some legal text", 1)
    chunk.metadata["provision"] = "Preamble"
    chunks = [chunk]

    with pytest.raises(RuntimeError, match="Non-contiguous"):
        ingest.validate_source_chunks(source, pages, chunks)


class FakeVectorStore:
    def __init__(self, existing_ids=None, fail_add=False):
        self.existing_ids = set(existing_ids or [])
        self.fail_add = fail_add
        self.events = []

    def get(self, **kwargs):
        return {"ids": list(self.existing_ids)}

    def add_documents(self, documents, ids):
        self.events.append(("add", list(ids)))
        if self.fail_add:
            raise RuntimeError("simulated add failure")
        self.existing_ids.update(ids)

    def delete(self, ids):
        self.events.append(("delete", list(ids)))
        self.existing_ids.difference_update(ids)


def test_safe_replace_adds_new_chunks_before_deleting_old(monkeypatch):
    chunk = make_page("New legal text")
    old_id = "old-chunk"
    new_id = "new-chunk"

    monkeypatch.setattr(ingest, "stable_chunk_id", lambda document: new_id)

    store = FakeVectorStore(existing_ids={old_id})

    ingest.safely_replace_source_chunks(store, "sample.pdf", [chunk])

    assert store.events[0] == ("add", [new_id])
    assert store.events[1] == ("delete", [old_id])
    assert store.existing_ids == {new_id}


def test_safe_replace_rolls_back_new_ids_when_add_fails(monkeypatch):
    chunk = make_page("New legal text")
    old_id = "old-chunk"
    new_id = "new-chunk"

    monkeypatch.setattr(ingest, "stable_chunk_id", lambda document: new_id)

    store = FakeVectorStore(existing_ids={old_id}, fail_add=True)

    with pytest.raises(RuntimeError, match="simulated add failure"):
        ingest.safely_replace_source_chunks(store, "sample.pdf", [chunk])

    assert store.events[0] == ("add", [new_id])
    assert store.events[1] == ("delete", [new_id])
    assert old_id in store.existing_ids


def test_safe_replace_rejects_duplicate_chunk_ids(monkeypatch):
    chunks = [make_page("First"), make_page("Second")]

    monkeypatch.setattr(ingest, "stable_chunk_id", lambda document: "same-id")

    store = FakeVectorStore()

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        ingest.safely_replace_source_chunks(store, "sample.pdf", chunks)

    assert store.events == []


def test_extract_pdf_pages_reads_all_pages_and_preserves_page_numbers(
    tmp_path, monkeypatch
):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def get_text(self, mode):
            assert mode == "text"
            return self.text

    class FakePDF:
        def __init__(self):
            self.pages = [
                FakePage("First page text"),
                FakePage("Second page text"),
                FakePage("Third page text"),
            ]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()

    monkeypatch.setattr(ingest, "LEGAL_DOCS_DIR", tmp_path)
    monkeypatch.setattr(ingest.pymupdf, "open", lambda path: FakePDF())

    source = {
        "filename": "sample.pdf",
        "document_type": "DPDP Act, 2023",
        "authority": "primary",
        "pages": None,
    }

    pages = ingest.extract_pdf_pages(source)

    assert len(pages) == 3
    assert [page.metadata["pdf_page"] for page in pages] == [1, 2, 3]
    assert pages[0].page_content == "First page text"


def test_extract_pdf_pages_rejects_missing_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "LEGAL_DOCS_DIR", tmp_path)

    source = {
        "filename": "missing.pdf",
        "document_type": "DPDP Act, 2023",
        "authority": "primary",
        "pages": None,
    }

    with pytest.raises(FileNotFoundError):
        ingest.extract_pdf_pages(source)

def test_numbered_schedule_paragraphs_are_not_detected_as_rules():
    text = (
        "23. Calling for information from Data Fiduciary or intermediary.\n"
        "The Central Government may require information.\n\n"
        "FIRST SCHEDULE\n"
        "[See rule 4]\n"
        "PART A\n"
        "1. The applicant is a company incorporated in India.\n"
        "2. The applicant has sufficient capacity to fulfil its obligations.\n"
        "3. The financial condition of the applicant is sound.\n"
    )

    markers = ingest._find_provision_markers(
        text,
        "DPDP Rules, 2025",
    )

    labels = [label for _, label in markers]

    assert any(label.startswith("Rule 23:") for label in labels)
    assert "First Schedule" in labels
    assert not any(label.startswith("Rule 1:") for label in labels)
    assert not any(label.startswith("Rule 2:") for label in labels)
    assert not any(label.startswith("Rule 3:") for label in labels)


def test_schedule_classification_continues_across_pages():
    first_page = make_page(
        "23. Calling for information from Data Fiduciary or intermediary.\n"
        "The Central Government may require information.\n\n"
        "FIRST SCHEDULE\n"
        "[See rule 4]\n"
        "PART A\n"
        "Conditions for registration.",
        page=9,
        document_type="DPDP Rules, 2025",
    )

    second_page = make_page(
        "1. The applicant is a company incorporated in India.\n"
        "2. The applicant has sufficient capacity to fulfil its obligations.\n"
        "3. The financial condition of the applicant is sound.",
        page=10,
        document_type="DPDP Rules, 2025",
    )

    chunks = ingest.split_legal_pages([first_page, second_page])

    second_page_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["pdf_page"] == 10
    ]

    assert second_page_chunks
    assert all(
        chunk.metadata["provision"] == "First Schedule"
        for chunk in second_page_chunks
    )
    assert not any(
        chunk.metadata["provision"].startswith("Rule ")
        for chunk in second_page_chunks
    )
