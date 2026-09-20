import io
import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote

import faiss
import gdown
import numpy as np
import requests
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# -----------------------------
# 1. Document extraction
# -----------------------------

def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text.strip(),
                "filename": filename,
                "page": page_number,
            })

    return pages


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "filename": filename,
        "page": None,
    }]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="replace").strip()

    if not text:
        return []

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }]


def extract_md(file_bytes, filename):
    # Markdown is kept as text. Formatting markers are preserved because
    # they can contain useful document structure.
    text = file_bytes.decode("utf-8", errors="replace").strip()

    if not text:
        return []

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }]


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# 2. Text chunking
# -----------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        current_words = []
        current_length = 0
        index = start

        while index < len(words):
            word = words[index]
            extra_length = len(word) + (1 if current_words else 0)

            if current_words and current_length + extra_length > chunk_size:
                break

            current_words.append(word)
            current_length += extra_length
            index += 1

        chunks.append(" ".join(current_words))

        if index >= len(words):
            break

        # Convert character overlap approximately into a word overlap.
        overlap_words = max(1, overlap // 6)
        start = max(start + 1, index - overlap_words)

    return chunks


def create_chunks(extracted_pages):
    all_chunks = []

    for item in extracted_pages:
        for chunk in chunk_text(item["text"]):
            all_chunks.append({
                "text": chunk,
                "filename": item["filename"],
                "page": item["page"],
            })

    return all_chunks


# -----------------------------
# 3. Embeddings + FAISS
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner="Creating document embeddings...")
def build_vector_index(chunk_texts):
    model = load_embedding_model()

    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# 4. Keyword search
# -----------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "that",
    "the", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "with", "you", "your"
}


def important_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 2}


def keyword_score(query, chunk):
    query_words = important_words(query)
    chunk_words = important_words(chunk)

    if not query_words:
        return 0.0

    matches = query_words.intersection(chunk_words)
    return len(matches) / len(query_words)


# -----------------------------
# 5. Hybrid search
# -----------------------------

def hybrid_search(question, chunks, index, model, top_k=TOP_K):
    if not chunks:
        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(len(chunks), max(top_k * 3, 10))
    semantic_scores, semantic_indices = index.search(question_embedding, search_k)

    candidates = set(int(i) for i in semantic_indices[0] if i >= 0)

    # Keyword search adds chunks that may contain exact important terms
    # even when their semantic similarity is lower.
    keyword_scores = [
        keyword_score(question, chunk["text"]) for chunk in chunks
    ]

    keyword_indices = np.argsort(keyword_scores)[::-1][:search_k]
    candidates.update(int(i) for i in keyword_indices if keyword_scores[i] > 0)

    results = []

    for idx in candidates:
        semantic_score = 0.0
        for rank, candidate_idx in enumerate(semantic_indices[0]):
            if int(candidate_idx) == idx:
                semantic_score = float(semantic_scores[0][rank])
                break

        k_score = float(keyword_scores[idx])

        # Both components are normalized to roughly 0-1.
        semantic_score = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))
        hybrid_score = (0.70 * semantic_score) + (0.30 * k_score)

        result = dict(chunks[idx])
        result["semantic_score"] = semantic_score
        result["keyword_score"] = k_score
        result["hybrid_score"] = hybrid_score
        results.append(result)

    results.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return results[:top_k]


# -----------------------------
# 6. Google Drive loading
# -----------------------------

# Native Google files cannot be downloaded as-is, so they are exported.
# kind -> (export format, extension to save as)
GOOGLE_EXPORTS = {
    "document": ("docx", ".docx"),
    "presentation": ("pdf", ".pdf"),
    "spreadsheets": ("csv", ".txt"),
}


def parse_drive_url(url):
    """Return (kind, file_id) for a Google Drive/Docs link, else (None, None)."""
    url = url.strip()

    match = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
    if match:
        return "folder", match.group(1)

    match = re.search(
        r"/(document|presentation|spreadsheets)/d/([A-Za-z0-9_-]+)", url
    )
    if match:
        return match.group(1), match.group(2)

    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if match:
        return "file", match.group(1)

    # Old-style links such as drive.google.com/open?id=... or uc?id=...
    # can point to either a file or a folder.
    match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if match:
        return "file_or_folder", match.group(1)

    return None, None


def detect_extension(path):
    """Return a supported extension for the file, or None if unsupported."""
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS:
        return suffix

    # A file with an unrecognised extension (.xlsx, .png, ...) is unsupported.
    # Only files with NO extension are identified by their content.
    if suffix:
        return None

    with open(path, "rb") as handle:
        head = handle.read(2048)

    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"PK"):
        return ".docx"

    lowered = head.lstrip().lower()
    if lowered.startswith((b"<!doctype html", b"<html")):
        return None  # an HTML error/login page, not a document

    if head and b"\x00" not in head:
        return ".txt"

    return None


def _safe_stem(name, fallback):
    stem = re.sub(r"[^\w\-. ]", "_", Path(name).stem).strip(" .")
    return stem or fallback


def _download_file(file_id, out_dir):
    # Passing a directory (trailing separator) lets gdown keep the REAL
    # file name, including its extension, from Google Drive.
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(
        url,
        output=str(out_dir) + os.sep,
        quiet=True,
        use_cookies=False,
    )

    if not result:
        raise RuntimeError(
            "Google Drive did not return the file. Set sharing to "
            "'Anyone with the link' (Viewer) and try again."
        )


def _download_folder(folder_id, out_dir):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    result = gdown.download_folder(
        url,
        output=str(out_dir),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,  # do not fail on folders with more than 50 files
    )

    if not result:
        raise RuntimeError(
            "Could not read the folder. Set sharing to 'Anyone with the "
            "link' (Viewer) and make sure the folder is not empty."
        )


def _download_google_export(kind, file_id, out_dir):
    export_format, extension = GOOGLE_EXPORTS[kind]
    url = (
        f"https://docs.google.com/{kind}/d/{file_id}/export"
        f"?format={export_format}"
    )

    response = requests.get(url, timeout=60)
    content_type = response.headers.get("Content-Type", "")

    if response.status_code != 200 or "text/html" in content_type:
        raise RuntimeError(
            "Could not export the Google file. Set sharing to 'Anyone "
            "with the link' (Viewer) and try again."
        )

    # Try to keep the original document title as the file name.
    disposition = response.headers.get("Content-Disposition", "")
    name = ""
    match = re.search(r"filename\*=UTF-8''([^;]+)", disposition)
    if match:
        name = unquote(match.group(1))
    else:
        match = re.search(r'filename="([^"]+)"', disposition)
        if match:
            name = match.group(1)

    stem = _safe_stem(name, f"google_{kind}_{file_id[:8]}")
    (out_dir / f"{stem}{extension}").write_bytes(response.content)


def load_from_google_drive(url):
    kind, file_id = parse_drive_url(url)

    if not kind:
        return [], (
            "This does not look like a Google Drive or Google Docs link. "
            "Copy the link from the Share dialog."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="drive_docs_"))

    try:
        if kind == "folder":
            _download_folder(file_id, temp_dir)
        elif kind in GOOGLE_EXPORTS:
            _download_google_export(kind, file_id, temp_dir)
        elif kind == "file":
            _download_file(file_id, temp_dir)
        else:  # "file_or_folder"
            try:
                _download_file(file_id, temp_dir)
            except Exception as file_error:
                try:
                    _download_folder(file_id, temp_dir)
                except Exception:
                    raise file_error
    except Exception as exc:
        return [], str(exc)

    files = []
    skipped = []

    for path in sorted(temp_dir.rglob("*")):
        if not path.is_file():
            continue

        extension = detect_extension(path)

        if extension is None:
            skipped.append(path.name)
            continue

        if path.suffix.lower() != extension:
            path = path.rename(path.with_name(path.name + extension))

        files.append(path)

    if not files:
        message = (
            "The link opened, but no supported files (PDF, DOCX, TXT, MD) "
            "were found."
        )
        if skipped:
            message += " Skipped: " + ", ".join(skipped[:5])
        return [], message

    return files, None


def process_files(file_items):
    extracted = []

    for file_item in file_items:
        filename = file_item["filename"]
        file_bytes = file_item["bytes"]

        try:
            pages = extract_document(file_bytes, filename)
        except Exception as exc:
            st.warning(f"Could not read {filename}: {exc}")
            continue

        extracted.extend(pages)

    return extracted


# -----------------------------
# 7. Groq answer generation
# -----------------------------

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    client = get_groq_client()

    if client is None:
        return (
            "Add GROQ_API_KEY to Streamlit secrets before asking questions.",
            None,
        )

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {number}] {chunk['filename']} | {page_text}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question ONLY using the provided document context.
Do not use outside knowledge.
If the answer is not contained in the context, say:
"I couldn't find that information in the provided documents."

Keep answers clear and concise.
Do not invent facts, citations, page numbers, or details.
"""

    try:
        response = client.chat.completions.create(
            model=st.secrets.get(
                "GROQ_MODEL",
                "llama-3.1-8b-instant",
            ),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    ),
                },
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content, None

    except Exception as exc:
        return None, str(exc)


# -----------------------------
# 8. Streamlit app
# -----------------------------

st.title("AI Document Assistant")
st.caption("Upload documents or load supported files from a public Google Drive link.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    drive_url = st.text_input(
        "Google Drive file or folder link",
        placeholder="Paste a public/shared Drive link",
    )

    process_button = st.button("Process Documents", type="primary")

    st.divider()

    st.write("Supported:")
    st.write("• PDF")
    st.write("• DOCX")
    st.write("• TXT")
    st.write("• MD")
    st.write("• Public/shared Google Drive files or folders")
    st.write("• Google Docs, Slides and Sheets links")


if process_button:
    source_files = []

    # Local uploads
    if uploaded_files:
        for uploaded in uploaded_files:
            source_files.append({
                "filename": uploaded.name,
                "bytes": uploaded.getvalue(),
            })

    # Google Drive files
    if drive_url.strip():
        with st.spinner("Loading files from Google Drive..."):
            drive_paths, drive_error = load_from_google_drive(drive_url.strip())

        if drive_error:
            st.warning(
                "Google Drive could not be loaded. Make sure the link is "
                "public/shared and points to a supported file or folder."
            )
            st.caption(drive_error)

        for path in drive_paths:
            source_files.append({
                "filename": path.name,
                "bytes": path.read_bytes(),
            })

    if not source_files:
        st.warning("Add at least one local document or a Google Drive link.")
    else:
        # A content-based signature prevents rebuilding the same documents
        # when the user clicks Process Documents again.
        signature_parts = []

        for item in source_files:
            signature_parts.append(
                hashlib.sha256(item["bytes"]).hexdigest()
            )

        signature = hashlib.sha256(
            "|".join(sorted(signature_parts)).encode()
        ).hexdigest()

        if signature == st.session_state.document_signature:
            st.info("These documents are already processed.")
        else:
            with st.spinner("Extracting and chunking documents..."):
                extracted_pages = process_files(source_files)
                chunks = create_chunks(extracted_pages)

            if not chunks:
                st.error("No readable text was found in the documents.")
            else:
                chunk_texts = [chunk["text"] for chunk in chunks]

                index, embeddings = build_vector_index(chunk_texts)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.embeddings = embeddings
                st.session_state.documents = source_files
                st.session_state.document_signature = signature

                st.success(
                    f"Processed {len(source_files)} document(s) and created "
                    f"{len(chunks)} chunks."
                )


# -----------------------------
# 9. Document information
# -----------------------------

if st.session_state.chunks:
    st.subheader("Document Information")

    document_names = [
        item["filename"] for item in st.session_state.documents
    ]

    cols = st.columns(3)
    cols[0].metric("Documents", len(document_names))
    cols[1].metric("Chunks", len(st.session_state.chunks))
    cols[2].metric("Embedding size", st.session_state.embeddings.shape[1])

    for name in document_names:
        matching = [
            chunk for chunk in st.session_state.chunks
            if chunk["filename"] == name
        ]

        pages = sorted({
            chunk["page"]
            for chunk in matching
            if chunk["page"] is not None
        })

        page_info = (
            f"{len(pages)} page(s)"
            if pages
            else "Page information not available"
        )

        st.write(f"**{name}** — {len(matching)} chunks — {page_info}")


# -----------------------------
# 10. Question answering
# -----------------------------

st.divider()
st.subheader("Ask Your Documents")

question = st.text_input(
    "Question",
    placeholder="Ask something about the uploaded documents...",
)

ask_button = st.button("Ask")

if ask_button:
    if not st.session_state.chunks:
        st.warning("Process at least one document first.")
    elif not question.strip():
        st.warning("Enter a question first.")
    else:
        model = load_embedding_model()

        with st.spinner("Searching documents..."):
            retrieved = hybrid_search(
                question=question,
                chunks=st.session_state.chunks,
                index=st.session_state.index,
                model=model,
                top_k=TOP_K,
            )

        with st.spinner("Generating answer..."):
            answer, error = answer_question(question, retrieved)

        if error:
            st.error(f"Groq error: {error}")
        else:
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved Sources")

            for number, source in enumerate(retrieved, start=1):
                page_label = (
                    f"Page {source['page']}"
                    if source["page"] is not None
                    else "Page not available"
                )

                with st.expander(
                    f"{number}. {source['filename']} — {page_label}"
                ):
                    st.caption(
                        f"Hybrid score: {source['hybrid_score']:.3f} | "
                        f"Semantic: {source['semantic_score']:.3f} | "
                        f"Keyword: {source['keyword_score']:.3f}"
                    )
                    st.write(source["text"])
else:
    st.info(
        "Upload documents or add a Google Drive link, then click "
        "\"Process Documents\"."
    )
