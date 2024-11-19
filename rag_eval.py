import json
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
import chromadb
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
collection = chroma_client.get_collection(name="moby_dick")

with open("data/test_set.json", "r") as f:
    test_set = json.load(f)


def generate_response(query):
    return chatbot(query)


def chatbot(query):
    query_embedding = embedding_model.get_text_embedding(query)

    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        relevant_texts = results.get("documents", [[]])[0]
        if not relevant_texts:
            return "No relevant texts found."
    except Exception as e:
        return f"Error querying the collection: {e}"

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

    answer = response.choices[0].message.content.strip()
    return answer


responses = []
for item in test_set:
    question = item["question"]
    response = generate_response(question)
    responses.append(
        {
            "query": question,
            "context": item["expected_answer"],
            "response": response,
        }
    )


def evaluate_responses(responses):
    evaluation_results = []
    for response in responses:
        query = response["query"]
        context = response["context"]
        generated_answer = response["response"]

        context_embedding = embedding_model.get_text_embedding(context)
        answer_embedding = embedding_model.get_text_embedding(generated_answer)
        similarity_score = cosine_similarity([context_embedding], [answer_embedding])[
            0
        ][0]

        recall_score = (
            1.0 if any(word in context for word in generated_answer.split()) else 0.0
        )

        evaluation_results.append(
            {
                "query": query,
                "generated_answer": generated_answer,
                "similarity_score": similarity_score,
                "context_recall": recall_score,
            }
        )
    return evaluation_results


evaluation_results = evaluate_responses(responses)


with open("data/evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("Evaluation results saved to data/evaluation_results.json")
