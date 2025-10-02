import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from PyPDF2 import PdfReader

def load_pdf(path: str):
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append({"page" : i + 1, "text": text})
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
                "page" : t["page"],
                "chunk_id" : f"{t['page']}_{i}",
                "text" : s
            })

    return chunks

def main():
    load_dotenv()
    # Settings
    pdf_path = "notebooks/sample.pdf"
    collection_name = os.getenv("QDRANT_COLLETION", "book_chunks")
    qdrant_url = os.getenv("QDRANT_URL","http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None

    # Load PDF & chunk
    raw_texts = load_pdf(pdf_path)
    print(f"📖 Loaded {len(raw_texts)} pages from {pdf_path}")
    chunks = chunk_texts(raw_texts)
    print(f"✂️ Split into {len(chunks)} chunks")

    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, api_key=api_key)

    # Create collection if not exists
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )

    client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )


    # Insert data
    vectors = []
    payloads = []
    for i, ch in enumerate(chunks):
        vec = embeddings.embed_query(ch["text"])
        vectors.append(vec)
        payloads.append({
            "page": ch["page"],
            "chunk_id": ch["chunk_id"],
            "content": ch["text"]
        })

    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=list(range(len(vectors))),
            vectors=vectors,
            payloads=payloads
        )
    )

    print(f"✅ Stored {len(vectors)} chunks in Qdrant collection `{collection_name}`")

    # NEW: confirm count from Qdrant
    count = client.count(collection_name=collection_name).count
    print(f"📊 Qdrant reports {count} vectors in `{collection_name}`")


if __name__ == "__main__":
    main()
