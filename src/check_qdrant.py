import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def main():
    load_dotenv()
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    client = QdrantClient(url=url, api_key=api_key)

    cols = client.get_collections().collections
    names = [c.name for c in cols]
    print("✅ Qdrant reachable at", url)
    print("Existing collections:", names or "[]")

if __name__ == "__main__":
    main()
