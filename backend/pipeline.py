import os
from sentence_transformers import SentenceTransformer
import chromadb

# === CONFIG ===
VECTORDB_PATH = os.path.join("backend", "vectorDB", "chromadb_store")
COLLECTION_NAME = "hyderabad_guides"  # <- make sure this matches upload
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

# === Init ChromaDB ===
client = chromadb.PersistentClient(path=VECTORDB_PATH)

# Debug: List existing collections
print("✅ Available collections:", client.list_collections())

# Load collection
collection = client.get_collection(name=COLLECTION_NAME)

# === Load embedding model ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def retrieve_context(query: str, top_k: int = 3):
    formatted_query = f"query: {query}"
    query_embedding = model.encode(formatted_query, convert_to_numpy=True)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metadatas))

# === Example run ===
if __name__ == "__main__":
    query = "What are some top co-working spaces in Hyderabad?"
    results = retrieve_context(query)

    print("\n🔍 Top Results:\n")
    for i, (doc, meta) in enumerate(results, 1):
        print(f"{i}. {doc[:250]}...")
        print(f"   Metadata: {meta}\n")
