import json
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
import chromadb
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

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
        results = collection.query(query_embeddings=[query_embedding], n_results=8)
        relevant_texts = results.get("documents", [[]])[0]
        if not relevant_texts:
            return "No relevant texts found."
    except Exception as e:
        return f"Error querying the collection: {e}"

    context = "\n".join(relevant_texts)
    prompt = f"Based on the context provided, answer the following question as accurately as possible. If the answer is not clear, provide your best inference.\nContext: {context}\nQuestion: {query}\nAnswer:"

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
        context = response["context"].lower().strip()
        generated_answer = response["response"].lower().strip()

        context_embedding = embedding_model.get_text_embedding(context)
        answer_embedding = embedding_model.get_text_embedding(generated_answer)
        similarity_score = cosine_similarity([context_embedding], [answer_embedding])[
            0
        ][0]

        context_words = re.findall(r"\w+", context)
        generated_words = re.findall(r"\w+", generated_answer)
        recall_score = (
            sum(1 for word in generated_words if word in context_words)
            / len(generated_words)
            if generated_words
            else 0.0
        )

        exact_match_score = 1.0 if context == generated_answer else 0.0

        evaluation_results.append(
            {
                "query": query,
                "generated_answer": response["response"],
                "similarity_score": similarity_score,
                "context_recall": recall_score,
                "exact_match_score": exact_match_score,
            }
        )
    return evaluation_results


evaluation_results = evaluate_responses(responses)


with open("data/evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("Evaluation results saved to data/evaluation_results.json")
