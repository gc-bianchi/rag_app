import json
import os
import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

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


for i, embedding in enumerate(embeddings[20:23]):
    print(f"Embedding {i+1}: {embedding}")
