# AI Document Assistant

A simple Streamlit document question-answering app.

It supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Public/shared Google Drive file or folder links
- Text extraction
- Overlapping text chunks
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid retrieval
- Groq-powered answers
- Retrieved source chunks with filename and page information
- Streamlit session state and caching so document embeddings are not recreated for every question
  
##   Live Demo

  **Try the application here:**

  [Doc-Asistant](https://docassistantai-bo7xport7hxgety5oqkkuw.streamlit.app/)

## Project files

```text
document-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add your Groq API key

Do not put the API key directly in `app.py`.

For local Streamlit development, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

`GROQ_MODEL` is optional. If it is omitted, the app uses the default model in `app.py`.

For Streamlit Community Cloud:

1. Open your app.
2. Go to **Settings → Secrets**.
3. Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

## 3. Run

```bash
streamlit run app.py
```

## How the app works

```text
Documents
   ↓
Text extraction
   ↓
Overlapping chunks
   ↓
Sentence Transformers embeddings
   ↓
FAISS index
   ↓
                  User question
                       ↓
              Question embedding
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
       Semantic search      Keyword search
             └─────────┬─────────┘
                       ↓
                  Hybrid ranking
                       ↓
              Relevant text chunks
                       ↓
                     Groq
                       ↓
                    Answer
                       ↓
              Retrieved sources
```

## Document processing

Each extracted section keeps:

- `filename`
- `page` when the source provides page information
- `text`

PDF files are extracted page by page.

DOCX, TXT and MD files do not have reliable page information during normal text extraction, so their page value is stored as `None`.

The extracted text is split into overlapping chunks. Each chunk keeps its original filename and page metadata.

## Search

The app combines:

- **Semantic search:** Sentence Transformers + FAISS
- **Keyword search:** important-word matching

The hybrid score currently uses:

```text
70% semantic score
30% keyword score
```

The top retrieved chunks are sent to Groq as context.

The model is instructed to answer only from that context. If the answer is not available, it should say that the information could not be found in the provided documents.

## Reusing embeddings

Document embeddings are created when documents are processed, not when every question is asked.

The app uses:

- `st.session_state` to keep the processed chunks, FAISS index and embeddings available during the session.
- `st.cache_resource` for the Sentence Transformer model.
- `st.cache_data` for the document embedding/index-building step.

A content hash is also used to avoid rebuilding the same set of documents when the **Process Documents** button is pressed again.

## Google Drive

Paste a public/shared Google Drive file or folder link into the sidebar.

The app uses `gdown` to download accessible Drive files and then sends them through the exact same:

```text
extraction → chunking → embedding → FAISS → hybrid search
```

pipeline as local uploads.

The Drive link must be accessible without an interactive Google login, and the actual files must be PDF, DOCX, TXT or MD.

## Important limitation

Google Drive support here is intentionally simple. It is designed for publicly accessible/shared files and folders rather than a full Google Drive OAuth integration.

For a private Drive account or files requiring login, a Google Drive API/OAuth integration would be needed.

## GitHub + Streamlit deployment

Upload:

```text
app.py
requirements.txt
README.md
```

to GitHub.

Then deploy the repository on Streamlit Community Cloud.

Add `GROQ_API_KEY` under the app's Streamlit Secrets. Never commit `.streamlit/secrets.toml` or expose your API key in GitHub.

## Example questions

After uploading a document, try:

```text
What is the main purpose of this document?
```

```text
What are the key findings?
```

```text
What does the document say about [specific topic]?
```

```text
Which section discusses [specific term]?
```
