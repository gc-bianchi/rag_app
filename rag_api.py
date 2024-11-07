from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
import chromadb
from transformers import pipeline

app = FastAPI()

embedding_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
chroma_client = chromadb.Client()
collection_name = "moby_dick"

collections = [col.name for col in chroma_client.list_collections()]
if collection_name in collections:
    collection = chroma_client.get_collection(name=collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)

response_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")


class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Moby-Dick RAG API"}


@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        query_embedding = embedding_model.get_text_embedding(request.query)

        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        relevant_texts = results.get("documents", [[]])[0]
        if not relevant_texts:
            return {"response": "No relevant texts found."}

        context = "\n".join(relevant_texts)
        prompt = f"Context: {context}\nQuestion: {request.query}\nAnswer:"

        response = response_generator(
            prompt, max_new_tokens=150, truncation=True, return_full_text=True
        )
        answer = response[0]["generated_text"].split("Answer:")[-1].strip()

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
