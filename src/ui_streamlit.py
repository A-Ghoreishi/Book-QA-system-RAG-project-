import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load env ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.metisai.ir/openai/v1")
COLLECTION = os.getenv("QDRANT_COLLECTION", "book_chunks")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --- Helpers ---
def load_pdf(file):
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append({"page": i + 1, "text": text})
    return texts

def chunk_texts(texts, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for t in texts:
        splits = splitter.split_text(t["text"])
        for i, s in enumerate(splits):
            chunks.append({
                "page": t["page"],
                "chunk_id": f"{t['page']}_{i}",
                "content": s
            })
    return chunks

def ingest_into_qdrant(chunks):
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=API_KEY,
        base_url=BASE_URL
    )
    client = QdrantClient(url=QDRANT_URL)

    # Reset collection
    client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )

    # Embed and upsert
    vectors, payloads = [], []
    for i, ch in enumerate(chunks):
        vec = embeddings.embed_query(ch["content"])
        vectors.append(vec)
        payloads.append(ch)

    client.upsert(
        collection_name=COLLECTION,
        points=models.Batch(
            ids=list(range(len(vectors))),
            vectors=vectors,
            payloads=payloads
        )
    )
    return len(vectors)

def get_retriever():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=API_KEY,
        base_url=BASE_URL
    )
    client = QdrantClient(url=QDRANT_URL)
    qdrant = Qdrant(
        client=client,
        collection_name=COLLECTION,
        embeddings=embeddings,
        content_payload_key="content"
    )
    return qdrant.as_retriever(search_kwargs={"k": 3})

def get_qa_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "o4-mini"),
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.2
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following book excerpts to answer."
        "\n\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources by page."
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

# --- Streamlit UI ---
st.set_page_config(page_title="📚 Book QA", page_icon="📖")
st.title("📚 Book Question Answering with RAG")

uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_file:
    st.info("⏳ Processing your PDF...")
    texts = load_pdf(uploaded_file)
    chunks = chunk_texts(texts)
    count = ingest_into_qdrant(chunks)
    st.success(f"✅ Ingested {count} chunks from your PDF")

    qa = get_qa_chain()
    query = st.text_input("💬 Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            res = qa.invoke({"query": query})
            st.subheader("🤖 Answer")
            st.write(res["result"])

            with st.expander("📚 Source Chunks"):
                for doc in res["source_documents"]:
                    st.markdown(f"**Page {doc.metadata.get('page')}**")
                    st.write(doc.page_content[:400] + "...")
                    st.divider()
