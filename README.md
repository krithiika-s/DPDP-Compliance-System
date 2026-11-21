# DPDP-Compliance-System
General-purpose multi-agent system for automated DPDP compliance assessment

## Legal RAG Setup (Person 3)

The Legal RAG component retrieves relevant provisions from the official
Digital Personal Data Protection Act, 2023 and DPDP Rules, 2025.

### 1. Install dependencies

From the repository root, run:

```powershell
pip install -r requirements.txt
```

### 2. Check legal documents

Place the following official PDF documents inside:

`knowledge_base/legal_docs/`

- `2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf` — DPDP Act, 2023
- `DPDP_Rules_2025_English_only (1).pdf` — DPDP Rules, 2025

### 3. Build the legal vector index

From the repository root, run:

```powershell
python -m agents.legal_rag.ingest
```

This command:
- Extracts text from the official legal PDFs.
- Splits the text into smaller chunks.
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores the embeddings in a ChromaDB collection named `dpdp_legal_sources`.

The generated database is stored at:

`knowledge_base/chroma_db/`

The first run may download the Hugging Face embedding model if it is not already cached.

### 4. Git tracking

The generated ChromaDB directory is excluded from Git through `.gitignore`:

`knowledge_base/chroma_db/`

Each developer must build the local vector index before using the real legal retriever. The generated database is not committed to the repository.

### 5. Legal retrieval behavior

The retriever uses similarity search against primary legal sources.

- Default retrieval count: 5 passages.
- Returns up to 5 nearest passages, depending on availability.
- Filters results to primary-authority sources.
- No minimum relevance-distance threshold is currently applied.

Therefore, the retriever may return passages even when their relevance is weak. An empty result is possible if the collection has no matching documents.

