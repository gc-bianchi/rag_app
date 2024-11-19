import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rag_app import chatbot
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Instantiate the language model for evaluation
eval_llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# Instantiate the metrics for evaluation
context_recall_metric = LLMContextRecall(llm=eval_llm)
faithfulness_metric = Faithfulness(llm=eval_llm)
factual_correctness_metric = FactualCorrectness(llm=eval_llm)
semantic_similarity_metric = SemanticSimilarity(llm=eval_llm)

# Load the test set
with open("data/test_set.json", "r") as f:
    test_set = json.load(f)


# Generate responses for the test set
def generate_response(query):
    return chatbot(query)


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

# Evaluate the responses
for response in responses:
    question = response["question"]
    expected_answer = response["expected_answer"]
    generated_answer = response["generated_answer"]

    # Evaluate LLMContextRecall
    context_recall_score = context_recall_metric.evaluate(
        query=question, context=expected_answer, response=generated_answer
    )

    # Evaluate Faithfulness
    faithfulness_score = faithfulness_metric.evaluate(
        query=question, context=expected_answer, response=generated_answer
    )

    # Evaluate Factual Correctness
    factual_correctness_score = factual_correctness_metric.evaluate(
        query=question, context=expected_answer, response=generated_answer
    )

    # Evaluate Semantic Similarity
    semantic_similarity_score = semantic_similarity_metric.evaluate(
        response=generated_answer, reference=expected_answer
    )

    # Print out the scores
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Generated Answer: {generated_answer}")
    print(f"LLMContextRecall Score: {context_recall_score}")
    print(f"Faithfulness Score: {faithfulness_score}")
    print(f"Factual Correctness Score: {factual_correctness_score}")
    print(f"Semantic Similarity Score: {semantic_similarity_score}")
    print("-------------------------------------------------")
