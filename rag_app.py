import json
import os
import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# file paths for pdf and where I saved output as json files
pdf_path = "data/herman-melville-moby-dick.pdf"
docs_json_path = "data/llama_docs.json"
nodes_json_path = "data/llama_nodes.json"
embeddings_json_path = "data/llama_embeddings.json"

if os.path.exists(docs_json_path):
    with open(docs_json_path, "r") as f:
        llama_docs = [Document.from_dict(doc) for doc in json.load(f)]
    print("Documents loaded from llama_docs.json")
else:
    llama_reader = pymupdf4llm.LlamaMarkdownReader()
    llama_docs = llama_reader.load_data(pdf_path)

    with open(docs_json_path, "w") as f:
        json.dump([doc.to_dict() for doc in llama_docs], f)
    print("Documents saved to llama_docs.json")

embedding_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
splitter = SemanticSplitterNodeParser(
    embed_model=embedding_model,
    breakpoint_percentile_threshold=95,
    buffer_size=1,
    include_metadata=True,
)

if os.path.exists(nodes_json_path):
    with open(nodes_json_path, "r") as f:
        nodes = [Document.from_dict(node) for node in json.load(f)]
    print("Nodes loaded from llama_nodes.json")
else:
    nodes = splitter.get_nodes_from_documents(llama_docs)

    with open(nodes_json_path, "w") as f:
        json.dump([node.to_dict() for node in nodes], f)
    print("Nodes saved to llama_nodes.json")

# for i, node in enumerate(nodes[20:23]):
#     print(f"Node {i+1} Text: {node.text}")
#     print(f"Node {i+1} Metadata: {node.metadata}")

if os.path.exists(embeddings_json_path):
    with open(embeddings_json_path, "r") as f:
        embeddings = json.load(f)
    print("Embeddings loaded from llama_embeddings.json")
else:
    print("Starting to create embeddings for nodes...")
    embeddings = [embedding_model.get_text_embedding(node.text) for node in nodes]
    print(f"Embeddings created")

    with open(embeddings_json_path, "w") as f:
        json.dump(embeddings, f)
    print("Embeddings saved to llama_embeddings.json")

# for i, embedding in enumerate(embeddings[20:23]):
#     print(f"Embedding {i+1}: {embedding}")

# Save embeddings in a ChromaDB collection
chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
collection_name = "moby_dick"

# Check if collection exists by listing collections
collections = [col.name for col in chroma_client.list_collections()]
if collection_name in collections:
    print("collection exists")
    collection = chroma_client.get_collection(name=collection_name)
else:
    print("creating collection")
    collection = chroma_client.create_collection(name=collection_name)

count = collection.count()
if count > 0:
    existing_ids = set(collection.get()["ids"])
else:
    existing_ids = set()

for id, (node, embedding) in enumerate(zip(nodes, embeddings)):
    chunk_id = f"chunk_{id}"
    if chunk_id not in existing_ids:
        cleaned_metadata = {
            k: (v if v is not None else "") for k, v in node.metadata.items()
        }
        collection.add(
            documents=[node.text],
            metadatas=[cleaned_metadata],
            ids=[chunk_id],
            embeddings=[embedding],
        )
print("Embeddings saved to ChromaDB collection")

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chatbot():
    print("Starting chatbot. Type 'exit' to quit.")
    while True:
        query = input("Ask a question about Moby-Dick: ")
        if query.lower() == "exit":
            break

        query_embedding = embedding_model.get_text_embedding(query)

        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=5)
            # print("Query results:", results)

            relevant_texts = results.get("documents", [[]])[0]
            if not relevant_texts:
                print("No relevant texts found.")
                continue
        except Exception as e:
            print(f"Error querying the collection: {e}")
            continue

        context = "\n".join(relevant_texts)

        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model="gpt-4",
            max_tokens=150,
            temperature=0.7,
        )

        # Print the response
        answer = response.choices[0].message.content.strip()
        print(f"Answer: {answer}")
        # print(f"Answer: {response}")


chatbot()
