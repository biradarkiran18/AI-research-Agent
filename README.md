# Ollama Research Agent

A private, local RAG (Retrieval-Augmented Generation) agent to chat with your documents. This application runs 100% locally, ensuring your data remains private. Ingest PDF files to build a searchable knowledge base and get answers from a local LLM, enriched with live web searches.

## Features

* **100% Private:** All components, from embeddings to the LLM, run locally. No data leaves your machine.
* **PDF Ingestion:** Upload PDFs to build and expand your knowledge base.
* **Semantic Search:** Uses vector embeddings to find relevant chunks of text, understanding meaning, not just keywords.
* **Hybrid Search:** Performs web searches if local data is insufficient.
* **Grounded Generation:** LLM answers are based only on provided document context, reducing hallucinations.
* **Simple UI:** Streamlit-based web interface for easy use.

## How It Works (RAG Flow)

1. **Ingestion:** PDFs are converted into text, split into overlapping chunks, and embedded using a Sentence Transformer. Embeddings are stored in a FAISS index.
2. **Retrieval:** Queries are embedded and compared to the FAISS index to retrieve the `k` most similar chunks.
3. **Enrichment:** If local results are sparse, the agent performs live Wikipedia searches for additional context.
4. **Generation:** Retrieved chunks and web results form a prompt sent to a local LLM via Ollama, which generates the final answer.

## Tech Stack

* **Backend:** FastAPI, Uvicorn  
* **Frontend:** Streamlit  
* **AI & Data Processing:**  
  * **LLM Service:** Ollama  
  * **Embeddings:** `sentence-transformers`  
  * **Vector Index:** FAISS (`faiss-cpu`)  
  * **Metadata DB:** SQLite  
  * **PDF Parsing:** PyMuPDF  

## Setup and Installation

**Prerequisites:**

* Python 3.9+  
* [Ollama](https://ollama.com/) installed and running  

**1. Clone the Repository:**

```bash
git clone <your-repository-url>
cd agent-ollama-research
```

**2. Create and Activate a Virtual Environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
**3. Install Dependencies:

```bash
pip install -r requirements.txt
```

**4. Download an Ollama Model:
```bash
# Recommended for 8GB RAM
ollama pull phi3:mini
```

```bash
# Recommended for 16GB+ RAM
ollama pull gemma:9b
```
**Running the Application

**Open two terminals:

Backend Server:
```bash
uvicorn backend.server:app --reload
```

Frontend App:
```bash
streamlit run frontend/streamlit_app.py
```

**A browser tab will open with the Streamlit application.

**Configuration

**To change the LLM model:
```bash
# backend/ollama_client.py
class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = "phi3:mini") -> None:
        ...
```

**Make sure the model you specify is downloaded with ollama pull <model-name>.
