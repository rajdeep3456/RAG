#Ultimate Summarizer & QA App

A Streamlit-based application that ingests various document sources (PDF, text, YouTube videos, and websites), preprocesses content, and provides both summarization and question-answering capabilities using a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain and Groq AI.

---

## Features

* **Multi-source Input:** Upload PDF or text files, enter websites or YouTube URLs.
* **RAG Pipeline:** Splits documents into overlapping text chunks, stores embeddings in a Chroma vector store, and retrieves relevant passages.
* **Summarization:** Uses a map-reduce chain to generate structured summaries with titles and numbered points.
* **Question Answering:** Retrieves top-k chunks and returns precise, context-aware answers.
* **Streamlit UI:** Intuitive, interactive interface with wide layout and input selectors.

---

## Architecture & Workflow

1. **Input:** User selects source (PDF, Text, Website, YouTube).
2. **Processing:** Utility modules (`utils/pdf_utils.py`, `utils/text_utils.py`, etc.) load and parse content into LangChain `Document` objects.
3. **RAG Initialization:** `initialize_rag` splits text into chunks via `RecursiveCharacterTextSplitter`, embeds using HuggingFace (`all-MiniLM-L6-v2`), and persists into a Chroma store.
4. **Summarize / QA:**

   * **Summarize:** Map-reduce chain (`load_summarize_chain`) with custom prompts to generate cohesive summaries.
   * **QA:** `RetrievalQA` chain retrieves top-k similar chunks and composes answers via the Groq-powered LLM.

---

## Folder Structure

```
ultimate-summarizer-qa/
├── app.py               # Streamlit frontend
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── utils/               # Utility modules for different inputs
│   ├── pdf_utils.py     # PDF processing
│   ├── text_utils.py    # Plain text processing
│   ├── web_utils.py     # Website scraping
│   ├── yt_utils.py      # YouTube transcript loading
│   └── rag_pipeline.py  # RAG initialization, summarization, QA
└── chroma_db/           # Persisted vector store directory
```

---

## Utilities & Modules

* **`utils/pdf_utils.py`**: Loads PDFs via a temporary file and `PyPDFLoader`.
* **`utils/text_utils.py`**: Loads text files via `TextLoader`.
* **`utils/web_utils.py`**: Scrapes web pages with `WebBaseLoader`.
* **`utils/yt_utils.py`**: Extracts YouTube transcripts & metadata.
* **`utils/rag_pipeline.py`**: All RAG functions:

  * `initialize_rag`
  * `summarize_doc`
  * `answer_query`

---

*Happy summarizing and querying!*
