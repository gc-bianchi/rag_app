# import multiprocessing

# multiprocessing.set_start_method("spawn")

from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
from text_splitter import split_text
import json
import os


def main():
    # Load GPT-2 model using transformers library
    gpt_neo = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

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

    # List all existing collections to check persistence
    existing_collections = chroma_client.list_collections()
    print("Existing collections:")
    for col in existing_collections:
        print(f"- {col.name}")
    collection_name = "moby_dick"

    # Check if collection exists by listing collections
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
    print(f"count: ${count}")

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

    countAfterLoop = collection.count()
    print(f"count: ${countAfterLoop}")

    query = "What is the whale's significance in Moby-Dick?"
    response = generate_response(query, collection, gpt_neo)
    print(response)


def generate_response(query, collection, gpt_neo):
    results = collection.query(query_texts=[query], n_results=3)

    context = "\n".join(
        [
            " ".join(doc) if isinstance(doc, list) else doc
            for doc in results["documents"]
        ]
    )

    prompt = f"The following passage is from Moby-Dick:\n{context}\nPlease provide an answer to the following question based on the passage: {query}"
    response = gpt_neo(
        prompt, max_new_tokens=150, truncation=True, return_full_text=True
    )

    generated_text = response[0]["generated_text"]
    return generated_text


if __name__ == "__main__":
    main()
