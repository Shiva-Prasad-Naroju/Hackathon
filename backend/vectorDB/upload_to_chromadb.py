import json
import uuid
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "all_structured.json")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chromadb_store")
COLLECTION_NAME = "hyderabad_guides"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

# === Load structured JSON ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

# === Load embedding model ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Init ChromaDB (new syntax) ===
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# === Prepare documents ===
texts, metadatas, ids = [], [], []

for doc in documents:
    raw_text = doc.get("content", "").strip()
    if not raw_text:
        continue
    embed_text = f"passage: {raw_text}"  # BAAI format
    texts.append(embed_text)
    metadatas.append(doc.get("metadata", {}))
    ids.append(str(uuid.uuid4()))

# === Generate embeddings ===
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# === Upload to Chroma ===
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Successfully uploaded {len(texts)} documents to ChromaDB using 'BAAI/bge-large-en'")
