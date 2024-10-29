from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions
from text_splitter import split_text
import json
import os


llama = Llama.from_pretrained(
    repo_id="QuantFactory/Llama-3.2-1B-GGUF", filename="Llama-3.2-1B.Q2_K.gguf"
)


markdown_path = "data/moby-dick-output.md"
with open(markdown_path, "r", encoding="utf-8") as file:
    content = file.read()


chunks_path = "data/moby_dick_chunks.json"

if os.path.exists(chunks_path):
    # print("it exist!")
    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)
else:
    # print("create chunks")
    chunks = split_text(content)
    with open(chunks_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file)


# print("First 3 chunks:")
# for i, chunk in enumerate(chunks[13:16]):
#     print(f"Chunk {i+1}:{chunk}\n")


chroma_client = chromadb.Client()
collection_name = "moby_dick"

if collection_name in chroma_client.list_collections():
    print("already a collection")
    collection = chroma_client.get_collection(name=collection_name)
else:
    print("createa new collection")
    collection = chroma_client.create_collection(name=collection_name)


embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

count = collection.count()

if count > 0:
    print(f"{count} greater than zero")
    existing_ids = set(collection.get()["ids"])
else:
    print("create empty set")
    existing_ids = set()
for id, chunk in enumerate(chunks):
    chunk_id = f"chunk_{id}"
    if chunk_id not in existing_ids:
        embedding = embedding_function([chunk])[0]

        collection.add(
            documents=[chunk],
            metadatas=[{"source": "moby_dick"}],
            ids=[chunk_id],
            embeddings=[embedding],
        )


results = collection.get(ids=["chunk_13", "chunk_14", "chunk_15"])
for i, document in enumerate(results["documents"]):
    print(f"Retrieved Chunk {i+1}: {document}")
