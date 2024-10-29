from llama_cpp import Llama

# import chromadb
# from chromadb.utils import embedding_functions
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


# chroma_client = chromadb.Client()
# collection = chroma_client.create_collection(name="moby_dick")


# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="all-MiniLM-L6-v2"
# )


# for id, chunk in enumerate(chunks):
#     collection.add(
#         documents=[chunk],
#         metadatas=[{"source": "moby_dick"}],
#         ids=[f"chunk_{id}"],
#         embedding_function=embedding_function,
#     )


# count = collection.count()
# print(count)


# # results = collection.get(ids=["chunk_0", "chunk_1", "chunk_2"])
# # for i, document in enumerate(results["documents"]):
# #     print(f"Retrieved Chunk {i+1}:
# # {document}\n")
