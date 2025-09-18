## RAG based Chat Bot (Merged)

An end‑to‑end Streamlit app that lets you:

- Upload PDFs and chat with their content using Retrieval‑Augmented Generation (RAG) backed by FAISS and Sentence Transformers.
- Use Google Gemini (primary) or fall back to Hugging Face Inference API automatically.
- Persist conversation history in ChromaDB (vector database) and browse it in the sidebar.
- Toggle Private Chat mode to avoid persisting any messages.
- Export your current chat history to a JSON file.

### Key Features

- **RAG over PDFs**: Splits PDFs into chunks, embeds with `sentence-transformers/all-MiniLM-L6-v2`, retrieves top‑k by FAISS similarity.
- **LLM Failover**: Primary: Gemini 1.5 Flash via `langchain-google-genai`. Fallback: `mistralai/Mixtral-8x7B-Instruct-v0.1` via `huggingface_hub`.
- **Memory with ChromaDB**: Stores user/assistant messages as embeddings for sidebar review.
- **Private Chat**: When enabled, no messages are written to ChromaDB.
- **Export Chat**: Download current session messages as JSON.
- **Modern UI**: Streamlit UI enhanced with Tailwind CSS.

---

## Architecture Overview

- `app.py`: Streamlit UI, state, PDF upload, RAG vs. pure LLM decision, Chroma persistence, export.
- `document_loader.py`: Loads PDFs with `PyPDFLoader` and splits via `RecursiveCharacterTextSplitter`.
- `faiss_vector_store.py`: Creates FAISS vector store and performs similarity search with scores.
- `llm_handler.py`: Initializes LLM (Gemini or HF) and generates responses for RAG or non‑RAG prompts.
- `chroma_store.py`: Thin wrapper around Chroma client for storing and retrieving chat history pairs.
- `chat_history.py`: Sidebar renderer for stored chat history.
- `private_chat.py`: UI control for Private Chat toggle.
- `export_chat.py`: Exports current `st.session_state.messages` as JSON.

Data flow at a glance:

1. User uploads PDFs → chunks are embedded and indexed in FAISS.
2. User asks a question → similar chunks retrieved (if any). 
3. If relevant context found → RAG prompt; else → pure LLM prompt.
4. Response displayed → optionally stored in ChromaDB unless Private Chat is enabled.
5. Sidebar shows persisted history; export button saves current session to JSON.

---

## Requirements

Python 3.10+ (the project’s virtual environment is pinned to 3.10 based on `venv` contents).

System packages are managed via `requirements.txt`:

```text
streamlit>=1.18.0
google-generativeai
langchain
langchain-google-genai
langchain-huggingface
langchain-community
pypdf
faiss-cpu
sentence-transformers
python-dotenv
huggingface_hub
chromadb
```

---

## Setup

### 1) Clone and enter the project

```bash
cd "C:\Users\moune\Desktop\final project"
```

### 2) Create and activate a virtual environment (Windows PowerShell)

If you don’t have one yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you prefer the existing `venv` in the repo, activate it instead:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you may need (admin PowerShell):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_generative_ai_key
HF_API_KEY=your_huggingface_api_token_optional
```

- **GEMINI_API_KEY**: Required for primary LLM (Gemini 1.5 Flash). If missing or invalid, the app tries to fall back.
- **HF_API_KEY**: Optional fallback for `mistralai/Mixtral-8x7B-Instruct-v0.1`. If both are missing, the app will raise an error.

The app loads these via `python-dotenv` in `app.py`.

---

## Running the App

From an activated virtual environment:

```bash
streamlit run app.py
```

Streamlit will open in your browser (default `http://localhost:8501`).

---

## Usage Guide

1. **Upload PDFs**: Use the “Upload PDFs” widget to add one or more `.pdf` files. You should see a success message once the FAISS index is created.
2. **Ask a question**: Type in the chat input. If relevant PDF chunks exist, the app builds a RAG prompt. Otherwise, it uses the LLM without context.
3. **Private Chat**: Enable the checkbox “Private Chat (not stored in memory)” to prevent saving your messages to ChromaDB.
4. **Clear Chat History**: Resets in‑memory messages and reinitializes the Chroma store.
5. **Export Chat**: Click “Export Chat” then “Download JSON” to save your current session’s messages.
6. **Sidebar History**: The sidebar lists pairs of stored user/assistant messages from ChromaDB for quick review.

---

## Implementation Details

- **Document loading**: `PyPDFLoader` reads uploaded PDFs from a temporary file, then `RecursiveCharacterTextSplitter` chunks them at 1000 chars with 200 overlap.
- **Embeddings**: `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`.
- **Retrieval**: `FAISS.similarity_search_with_score(query, k=5)`; docs are filtered by a score threshold (default `>= 0.5`).
- **Prompting**:
  - RAG prompt includes recent chat history and concatenated context from retrieved docs.
  - Pure LLM prompt includes recent chat history only.
- **LLM selection**:
  - Tries Gemini first; validates by calling `llm.invoke("Test")`.
  - On invalid key/model/rate limits/other errors, falls back to HF Inference if `HF_API_KEY` is set.
- **Persistence**: When not in Private mode, each user/assistant turn is added to ChromaDB as two documents with simple metadata.

---

## Troubleshooting

- **No valid API keys**: You’ll see an error like “No Gemini API key provided and no Hugging Face API key available.” Set `GEMINI_API_KEY` or `HF_API_KEY` in `.env`.
- **Gemini key/model errors**: The app logs the exception and falls back if possible.
- **Rate limits**: Gemini `ResourceExhausted` triggers fallback to HF if configured.
- **PDF parsing issues**: Ensure files are valid PDFs. Consider pre‑processing scanned PDFs (OCR) outside the app; `PyPDFLoader` expects extractable text.
- **FAISS score threshold**: If you often get pure LLM responses, try lowering the threshold in `retrieve_relevant_docs()`.
- **Chroma growth**: Clearing chat history reinitializes the in‑memory handle. For long‑term stores, configure Chroma persistence separately.

---

## Folder Structure

```text
final project/
├─ app.py
├─ document_loader.py
├─ faiss_vector_store.py
├─ llm_handler.py
├─ chroma_store.py
├─ chat_history.py
├─ private_chat.py
├─ export_chat.py
├─ requirements.txt
└─ venv/ (optional, existing virtual environment)
```

---

## Security Notes

- Do not commit `.env` with real API keys.
- Treat exported chat JSON as sensitive if it contains private data.
- Private Chat prevents writing to ChromaDB but still holds messages in ephemeral Streamlit session state until cleared.

---

## Roadmap Ideas

- Add persistent ChromaDB storage directory and metadata (timestamps, session IDs).
- Support additional file types (DOCX, HTML, TXT) and multi‑modal inputs.
- Reranking and better prompt templates per domain.
- Per‑user auth and encrypted secrets management.
- Adjustable retrieval hyperparameters from the UI.

---

## OUTPUT

<img width="1033" height="906" alt="Screenshot 2025-09-18 203512" src="https://github.com/user-attachments/assets/c5486275-0dc3-4aa6-ae57-dee4a6500843" />

<img width="1916" height="911" alt="Screenshot 2025-09-18 203541" src="https://github.com/user-attachments/assets/3f3acde3-8e44-416c-9620-6d5f7b91b3c0" />

<img width="1917" height="907" alt="Screenshot 2025-09-18 203606" src="https://github.com/user-attachments/assets/9f387a25-256f-45ed-bed9-6912f15baa34" />

<img width="1167" height="902" alt="Screenshot 2025-09-18 203836" src="https://github.com/user-attachments/assets/fc0d7cee-3c1b-46e6-9413-ba6a51c2f483" />


