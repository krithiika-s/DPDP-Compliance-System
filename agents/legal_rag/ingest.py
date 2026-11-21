
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGAL_DOCS_DIR = PROJECT_ROOT / "knowledge_base" / "legal_docs"
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"

COLLECTION_NAME = "dpdp_legal_sources"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SCHEDULE_NAMES = (
    "FIRST",
    "SECOND",
    "THIRD",
    "FOURTH",
    "FIFTH",
    "SIXTH",
    "SEVENTH",
)

LEGAL_SOURCES = [
    {
        "filename": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "document_type": "DPDP Act, 2023",
        "authority": "primary",
        "pages": None,
    },
    {
        "filename": "DPDP_Rules_2025_English_only (1).pdf",
        "document_type": "DPDP Rules, 2025",
        "authority": "primary",
        "pages": None,
    },
]


def extract_pdf_pages(source: dict) -> list[Document]:
    """Extract every requested PDF page without silently skipping pages."""
    pdf_path = LEGAL_DOCS_DIR / source["filename"]

    if not pdf_path.exists():
        raise FileNotFoundError(f"Legal document not found: {pdf_path}")

    documents: list[Document] = []

    try:
        with pymupdf.open(pdf_path) as pdf:
            total_pages = len(pdf)

            if total_pages == 0:
                raise RuntimeError(
                    f"PDF contains no pages: {source['filename']}"
                )

            selected_pages = source["pages"]

            page_indexes = (
                list(range(total_pages))
                if selected_pages is None
                else list(selected_pages)
            )

            if not page_indexes:
                raise RuntimeError(
                    f"No pages selected for {source['filename']}"
                )

            if len(page_indexes) != len(set(page_indexes)):
                raise ValueError(
                    f"Duplicate page indexes in {source['filename']}"
                )

            print(f"PDF pages available: {total_pages}")
            print(f"Pages requested: {len(page_indexes)}")

            for page_index in page_indexes:
                if page_index < 0 or page_index >= total_pages:
                    raise ValueError(
                        f"Invalid page index {page_index} "
                        f"for {source['filename']}"
                    )

                try:
                    text = pdf[page_index].get_text("text").strip()
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not read {source['filename']}, "
                        f"page {page_index + 1}: {exc}"
                    ) from exc

                if not text:
                    raise RuntimeError(
                        f"No extractable text on {source['filename']}, "
                        f"page {page_index + 1}. "
                        "Ingestion stopped to prevent silent page loss. "
                        "OCR or manual inspection may be required."
                    )

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_file": source["filename"],
                            "document_type": source["document_type"],
                            "authority": source["authority"],
                            "pdf_page": page_index + 1,
                            "pdf_total_pages": total_pages,
                            "language": "English",
                        },
                    )
                )

    except (FileNotFoundError, ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open or extract {source['filename']}: {exc}"
        ) from exc

    return documents


def _find_provision_markers(
    page_text: str,
    document_type: str,
    active_provision: str = "not_identified",
) -> list[tuple[int, str]]:
    """Find main Act Sections, Rules, and Schedule headings."""
    markers: list[tuple[int, str]] = []
    is_act = "Act" in document_type

    if is_act and re.search(
        r"(?im)^\s*THE\s+SCHEDULE\s*$",
        page_text,
    ):
        return [(0, "Schedule")]

    in_schedule = active_provision.endswith("Schedule")

    schedule_pattern = re.compile(
        r"^(?:THE\s+)?("
        + "|".join(SCHEDULE_NAMES)
        + r")\s+SCHEDULE$",
        re.IGNORECASE,
    )
    heading_pattern = re.compile(r"^(\d{1,2})\.\s*(\S.*)$")

    offset = 0

    for raw_line in page_text.splitlines(keepends=True):
        line = raw_line.strip()

        if line:
            schedule_match = schedule_pattern.fullmatch(line)

            if schedule_match:
                schedule_name = schedule_match.group(1).title()
                markers.append((offset, f"{schedule_name} Schedule"))
                in_schedule = True

            elif not in_schedule:
                heading_match = heading_pattern.match(line)

                if heading_match:
                    number = int(heading_match.group(1))
                    heading = heading_match.group(2).strip()[:120]

                    if is_act and 1 <= number <= 44:
                        markers.append(
                            (offset, f"Section {number}: {heading}")
                        )
                    elif not is_act and 1 <= number <= 23:
                        markers.append(
                            (offset, f"Rule {number}: {heading}")
                        )

        offset += len(raw_line)

    return markers


def identify_provision(
    page_text: str,
    document_type: str,
    chunk_start: int,
    active_provision: str = "not_identified",
) -> str:
    """Identify the provision governing a position in a page."""
    markers = _find_provision_markers(
        page_text,
        document_type,
        active_provision,
    )

    provision = active_provision

    for position, label in markers:
        if position > chunk_start:
            break
        provision = label

    return provision


def split_legal_pages(pages: list[Document]) -> list[Document]:
    """
    Split each page at provision boundaries first, then into chunks.

    This prevents a single chunk from being labelled with only one
    provision when it actually contains multiple Section or Rule
    headings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    active_provisions: dict[str, str] = {}

    for page in pages:
        document_type = page.metadata["document_type"]
        active = active_provisions.get(
            document_type,
            "not_identified",
        )

        markers = _find_provision_markers(
            page.page_content,
            document_type,
            active,
        )

        if (
            active == "not_identified"
            and (not markers or markers[0][0] > 0)
        ):
            active = "Preamble"

        boundaries: list[tuple[int, str]] = [(0, active)]
        boundaries.extend(markers)

        position_labels: dict[int, str] = {}

        for position, label in boundaries:
            position_labels[position] = label

        ordered_boundaries = sorted(position_labels.items())
        page_length = len(page.page_content)

        for index, (start, provision) in enumerate(ordered_boundaries):
            end = (
                ordered_boundaries[index + 1][0]
                if index + 1 < len(ordered_boundaries)
                else page_length
            )

            segment_text = page.page_content[start:end].strip()

            if not segment_text:
                continue

            segment = Document(
                page_content=segment_text,
                metadata=dict(page.metadata),
            )

            segment_chunks = splitter.split_documents([segment])

            for chunk in segment_chunks:
                local_start = chunk.metadata.get("start_index", 0)

                chunk.metadata["start_index"] = start + local_start
                chunk.metadata["provision"] = provision

                chunks.append(chunk)

        if markers:
            active = markers[-1][1]

        active_provisions[document_type] = active

    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def stable_chunk_id(document: Document) -> str:
    """Create a deterministic ID from chunk content and its metadata."""
    identity = "|".join(
        [
            document.metadata["source_file"],
            str(document.metadata["pdf_page"]),
            str(document.metadata.get("start_index", 0)),
            str(sorted(document.metadata.items())),
            hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest(),
        ]
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def get_existing_source_ids(
    vector_store: Chroma,
    filename: str,
) -> set[str]:
    """Get existing vector IDs for one source PDF."""
    result = vector_store.get(
        where={"source_file": filename},
        include=[],
    )

    return set(result.get("ids", []))


def safely_replace_source_chunks(
    vector_store: Chroma,
    filename: str,
    chunks: list[Document],
) -> None:
    """
    Add replacement chunks before deleting stale chunks.

    If adding fails, remove newly introduced IDs and preserve
    the previous source vectors.
    """
    old_ids = get_existing_source_ids(vector_store, filename)
    new_ids = [stable_chunk_id(chunk) for chunk in chunks]

    if len(new_ids) != len(set(new_ids)):
        raise ValueError(
            f"Duplicate chunk IDs generated for {filename}"
        )

    try:
        vector_store.add_documents(
            documents=chunks,
            ids=new_ids,
        )
    except Exception:
        rollback_ids = [
            chunk_id
            for chunk_id in new_ids
            if chunk_id not in old_ids
        ]

        if rollback_ids:
            try:
                vector_store.delete(ids=rollback_ids)
            except Exception as rollback_error:
                print(
                    "WARNING: Rollback failed. "
                    f"Some new vectors may remain: {rollback_error}"
                )

        raise

    stale_ids = sorted(old_ids - set(new_ids))

    if stale_ids:
        vector_store.delete(ids=stale_ids)

    print(f"Previous chunks found: {len(old_ids)}")
    print(f"Replacement chunks added: {len(new_ids)}")
    print(f"Old chunks removed: {len(stale_ids)}")


def report_source(
    source: dict,
    pages: list[Document],
    chunks: list[Document],
) -> None:
    """Report extracted pages and provision labels."""
    page_numbers = sorted(
        page.metadata["pdf_page"] for page in pages
    )

    provision_counts = Counter(
        chunk.metadata.get("provision", "not_identified")
        for chunk in chunks
    )

    print(f"Pages extracted: {len(pages)}")
    print(f"PDF page numbers extracted: {page_numbers}")
    print(f"Chunks generated: {len(chunks)}")
    print("Provision labels:")

    for provision, count in sorted(provision_counts.items()):
        print(f"  {provision}: {count} chunks")



def validate_source_chunks(
    source: dict,
    pages: list[Document],
    chunks: list[Document],
) -> None:
    """Reject incomplete, unreadable, or unidentified legal extraction."""
    if not pages or not chunks:
        raise RuntimeError(
            f"No usable pages or chunks extracted from {source['filename']}"
        )

    # Real PDF extraction records the actual total page count.
    # Older test fixtures may not contain this metadata.
    total_page_values = {
        int(
            page.metadata.get(
                "pdf_total_pages",
                max(int(item.metadata["pdf_page"]) for item in pages),
            )
        )
        for page in pages
    }

    if len(total_page_values) != 1:
        raise RuntimeError(
            f"Inconsistent PDF page counts in {source['filename']}"
        )

    total_pdf_pages = total_page_values.pop()

    page_numbers = sorted(
        int(page.metadata["pdf_page"]) for page in pages
    )

    selected_pages = source.get("pages")

    expected_page_numbers = (
        list(range(1, total_pdf_pages + 1))
        if selected_pages is None
        else sorted(int(index) + 1 for index in selected_pages)
    )

    if page_numbers != expected_page_numbers:
        missing = sorted(set(expected_page_numbers) - set(page_numbers))
        unexpected = sorted(set(page_numbers) - set(expected_page_numbers))

        raise RuntimeError(
            f"Non-contiguous or incomplete PDF pages in "
            f"{source['filename']}. "
            f"Missing pages: {missing}; unexpected pages: {unexpected}"
        )

    if len(page_numbers) != len(set(page_numbers)):
        raise RuntimeError(
            f"Duplicate extracted pages in {source['filename']}"
        )

    empty_pages = [
        page.metadata["pdf_page"]
        for page in pages
        if not page.page_content.strip()
    ]

    if empty_pages:
        raise RuntimeError(
            f"Empty extracted pages in {source['filename']}: {empty_pages}"
        )

    labels = {
        chunk.metadata.get("provision", "not_identified")
        for chunk in chunks
    }

    if "not_identified" in labels:
        raise RuntimeError(
            f"Unidentified legal chunks remain in {source['filename']}"
        )

    if "Preamble" not in labels:
        raise RuntimeError(
            f"Preamble was not identified in {source['filename']}"
        )

    document_type = source["document_type"]

    if "Act" in document_type:
        expected = {f"Section {number}:" for number in range(1, 45)}

        actual = {
            label.split(":", 1)[0] + ":"
            for label in labels
            if label.startswith("Section ")
        }

        missing = expected - actual

        if missing:
            raise RuntimeError(
                f"Missing Act sections in {source['filename']}: "
                f"{sorted(missing)}"
            )

        if "Schedule" not in labels:
            raise RuntimeError(
                f"Act Schedule not detected in {source['filename']}"
            )

    elif "Rules" in document_type:
        expected = {f"Rule {number}:" for number in range(1, 24)}

        actual = {
            label.split(":", 1)[0] + ":"
            for label in labels
            if label.startswith("Rule ")
        }

        missing = expected - actual

        if missing:
            raise RuntimeError(
                f"Missing Rules provisions in {source['filename']}: "
                f"{sorted(missing)}"
            )

        expected_schedules = {
            "First Schedule",
            "Second Schedule",
            "Third Schedule",
            "Fourth Schedule",
            "Fifth Schedule",
            "Sixth Schedule",
            "Seventh Schedule",
        }

        missing_schedules = expected_schedules - labels

        if missing_schedules:
            raise RuntimeError(
                f"Missing Rules schedules in {source['filename']}: "
                f"{sorted(missing_schedules)}"
            )

    print(f"Validation passed: {document_type}")
    print(f"  PDF pages: {total_pdf_pages}")
    print(f"  Pages extracted: {len(pages)}")
    print(f"  Chunks: {len(chunks)}")


def ingest_legal_documents() -> None:
    # Extract and validate EVERY source before changing the vector database.
    prepared_sources = []

    for source in LEGAL_SOURCES:
        print(f"\nPreparing: {source['document_type']}")

        pages = extract_pdf_pages(source)
        chunks = split_legal_pages(pages)

        validate_source_chunks(source, pages, chunks)
        report_source(source, pages, chunks)

        prepared_sources.append((source, chunks))

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    total_chunks = 0

    for source, chunks in prepared_sources:
        filename = source["filename"]

        print(f"\nReplacing vectors for: {filename}")

        safely_replace_source_chunks(
            vector_store,
            filename,
            chunks,
        )

        total_chunks += len(chunks)

    print("\n" + "=" * 60)
    print("LEGAL DOCUMENT INGESTION COMPLETED")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks processed: {total_chunks}")
    print(
        "Total vectors in collection: "
        f"{vector_store._collection.count()}"
    )
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_legal_documents()
