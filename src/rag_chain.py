import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

def get_retriever():
    load_dotenv(".env")

    # Settings
    collection_name = os.getenv("QDRANT_COLLECTION", "book_chunks")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=api_key,
        base_url=base_url
    )

    # Connect to Qdrant via LangChain wrapper
    qdrant = Qdrant(
        client=QdrantClient(url=qdrant_url, api_key=api_key),
        collection_name=collection_name,
        embeddings=embeddings,
        content_payload_key="content"
    )

    return qdrant.as_retriever(search_kwargs={"k": 3})

def get_qa_chain():
    retriever = get_retriever()

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "o4-mini"),
        temperature=0.2,
        api_key=api_key,
        base_url=base_url
    )

    # Simple prompt template with citation requirement
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following book excerpts to answer."
        "\n\n{context}\n\nQuestion: {question}\n\nAnswer with a concise explanation and cite sources by page."
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,      
    )
    return qa

if __name__ == "__main__":
    qa = get_qa_chain()
    while True:
        q = input("Ask a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        res = qa.invoke({"query": q})   # use .invoke instead of __call__
        print("\n🤖 Answer:", res["result"])
        print("📚 Sources:")
        for doc in res["source_documents"]:
            print(f" - Page {doc.metadata.get('page')}: {doc.page_content[:200]}...")
