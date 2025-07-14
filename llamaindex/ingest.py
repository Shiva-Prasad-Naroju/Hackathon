import json
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from chromadb import PersistentClient

def load_json_docs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    documents = []
    for data in data_list:
        metadata = {
            "url": data.get("url", ""),
            "topic": data.get("topic", ""),
            "subtopic": data.get("subtopic", ""),
            "source": data.get("metadata", {}).get("source", ""),
            "chunk_id": data.get("metadata", {}).get("chunk_id", "")
        }

        documents.append(Document(text=data.get("content", ""), metadata=metadata))

    return documents

def build_index():
    docs = load_json_docs("data/all_structured.json")

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    )

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    chroma_client = PersistentClient(path="./chroma_db")
    
    # Get or create a collection
    collection = chroma_client.get_or_create_collection("documents")
    
    vector_store = ChromaVectorStore(
        chroma_collection=collection  # Pass the collection directly
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    index.storage_context.persist()
    print("✅ Index built and stored in ./chroma_db")

if __name__ == "__main__":
    build_index()