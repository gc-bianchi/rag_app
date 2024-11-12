import json
import random
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import chromadb


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def extract_random_chunks(collection, num_chunks=5):
    all_documents = collection.get()["documents"]
    selected_chunks = random.sample(all_documents, min(num_chunks, len(all_documents)))
    return selected_chunks


def generate_question_and_answer(context):
    qa_prompt = (
        f"Based on the following context, generate a question and provide the answer:\n{context}\n"
        "\nPlease output the question and answer in the following way:\n"
        "Question: {question generated}\nAnswer: {answer generated}"
    )
    response_generator = ChatOpenAI(model="gpt-4", temperature=0.7)

    response = response_generator.invoke(qa_prompt)
    response_text = response.content

    split_response = response_text.split("Answer:")
    if len(split_response) == 2:
        question_part = split_response[0].replace("Question:", "").strip()
        answer_part = split_response[1].strip()
    else:
        question_part = ""
        answer_part = ""

    return question_part, answer_part


def create_test_set(collection, output_file="data/test_set.json", num_samples=10):
    selected_chunks = extract_random_chunks(collection, num_samples)

    test_set = []
    for chunk in selected_chunks:
        question, answer = generate_question_and_answer(chunk)
        if question and answer:
            test_set.append({"question": question, "expected_answer": answer})

    with open(output_file, "w") as f:
        json.dump(test_set, f, indent=4)

    print(f"Test set saved to {output_file}")


chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
collection = chroma_client.get_collection(name="moby_dick")
create_test_set(collection)
