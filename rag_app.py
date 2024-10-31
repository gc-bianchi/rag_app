import multiprocessing

multiprocessing.set_start_method("spawn")

from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
from text_splitter import split_text
import json
import os


def main():
    gpt2 = pipeline("text-generation", model="gpt2")

    markdown_path = "data/moby-dick-output.md"
    with open(markdown_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks_path = "data/moby_dick_chunks.json"

    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)
    else:
        chunks = split_text(content)
        with open(chunks_path, "w", encoding="utf-8") as file:
            json.dump(chunks, file)

    chroma_client = chromadb.Client()
    collection_name = "moby_dick"

    collections = [col.name for col in chroma_client.list_collections()]
    if collection_name in collections:
        print("already a collection")
        collection = chroma_client.get_collection(name=collection_name)
    else:
        print("create a new collection")
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

    query = "What is the whale's significance in Moby-Dick?"
    response = generate_response(query, collection, gpt2)
    print(response)

    chroma_client.close()


def generate_response(query, collection, gpt2):
    results = collection.query(query_texts=[query], n_results=1)

    context = "\n".join(results["documents"])

    prompt = f"Based on the following text:\n{context}\nAnswer the question: {query}"
    response = gpt2(prompt, max_length=100)

    generated_text = response[0]["generated_text"]
    return generated_text


if __name__ == "__main__":
    main()
