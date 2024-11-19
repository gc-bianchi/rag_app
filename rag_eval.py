import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
import chromadb
from openai import OpenAI
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
)


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

eval_llm = ChatOpenAI(model="gpt-4", temperature=0.0)
embedding_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
collection = chroma_client.get_collection(name="moby_dick")


context_recall_metric = LLMContextRecall(llm=eval_llm)
faithfulness_metric = Faithfulness(llm=eval_llm)
factual_correctness_metric = FactualCorrectness(llm=eval_llm)
semantic_similarity_metric = SemanticSimilarity(llm=eval_llm)

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
            "question": question,
            "expected_answer": item["expected_answer"],
            "generated_answer": response,
        }
    )


for response in responses:
    question = response["question"]
    expected_answer = response["expected_answer"]
    generated_answer = response["generated_answer"]

    context_recall_score = context_recall_metric.evaluate(
        query=question, context=expected_answer, response=generated_answer
    )

    faithfulness_score = faithfulness_metric.evaluate(
        query=question, context=expected_answer, response=generated_answer
    )

    factual_correctness_score = factual_correctness_metric.evaluate(
        query=question, context=expected_answer, response=generated_answer
    )

    semantic_similarity_score = semantic_similarity_metric.evaluate(
        response=generated_answer, reference=expected_answer
    )

with open("data/evaluation_results.json", "w") as f:
    json.dump(responses, f, indent=4)

print("Evaluation results saved to data/evaluation_results.json")
