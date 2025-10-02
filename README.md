# 📚 Book-QA System with RAG

This project is an **educational demonstration of Retrieval-Augmented Generation (RAG)**.  
It shows how to build a system that can answer questions about any uploaded PDF book.  
Instead of relying on the language model’s memory alone, the system retrieves relevant passages from the book and uses them to generate answers with citations.

---

## 🎯 What is RAG?

Large Language Models (LLMs) like GPT are powerful but limited:  
- They don’t automatically know about your specific documents.  
- They can sometimes “hallucinate” answers.  

**RAG (Retrieval-Augmented Generation)** fixes this by combining:  
1. **Retrieval** – find the most relevant text passages from a database.  
2. **Generation** – feed those passages plus the question into GPT to write a grounded answer.  

---

## 🏗️ How this project works

```

PDF → Chunking → Embeddings → Qdrant Vector DB
↘
Retriever ← Streamlit UI ← User Question
↘
GPT (o4-mini via MetisAI) → Answer + Sources

````

### Key Steps
- **PDF Parsing (PyPDF2)** – extracts raw text from each page.  
- **Chunking (LangChain)** – splits long text into ~1000-token pieces with overlap.  
- **Embeddings (MetisAI/OpenAI)** – converts chunks into vectors (`text-embedding-3-small`).  
- **Vector Database (Qdrant)** – stores vectors for semantic search.  
- **Retriever (LangChain)** – finds top matching chunks for each question.  
- **Generator (GPT o4-mini)** – produces a concise answer, citing the book.  
- **Web UI (Streamlit)** – simple interface to upload a PDF and ask questions.  

---

## 🛠️ Why this approach?

- **RAG instead of fine-tuning** – new books work instantly, no retraining required.  
- **Qdrant** – open-source, optimized for vector search, runs easily in Docker.  
- **LangChain** – simplifies chunking, retrieval, and prompt design.  
- **Streamlit** – provides a fast, interactive web interface.  
- **MetisAI proxy** – enables access to GPT where direct OpenAI API isn’t available.  

---

## ⚙️ Setup

### 1. Clone repo
```bash
git clone https://github.com/yourname/Book-QA-system-RAG-project.git
cd Book-QA-system-RAG-project
````

### 2. Virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```bash
OPENAI_API_KEY=your_metisai_api_key
OPENAI_BASE_URL=https://api.metisai.ir/openai/v1

EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=o4-mini

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=book_chunks
```

### 4. Start Qdrant

```powershell
docker compose -f docker\docker-compose.yml up -d
```

Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## ▶️ Usage

### Web App

```powershell
streamlit run src/ui_streamlit.py
```

* Upload a PDF
* Ask questions
* See answers + source passages

### CLI (optional)

```powershell
python src/rag_chain.py
```

Ask questions directly in the terminal.

---

## 📊 Example

```
Question: What is this book about?

🤖 Answer: This book introduces fundamental concepts in data science and explains how they apply to real-world problems.

📚 Sources:
- Page 1: "Data science is the discipline that combines statistics..."
- Page 2: "In this book we will explore..."
```

---

## 🗺️ Roadmap

* [ ] Embedding visualization (t-SNE) in Streamlit sidebar
* [ ] Multi-book support
* [ ] Deploy Qdrant + Streamlit together with Docker Compose
* [ ] Option to use cloud-hosted Qdrant

---

## 📜 License

MIT License – free to use and adapt for educational purposes.

```
```
