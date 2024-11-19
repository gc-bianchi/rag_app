import numpy as np
import json


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def calculate_metrics(data):

    similarity_scores = np.array([item["similarity_score"] for item in data])
    context_recalls = np.array([item["context_recall"] for item in data])
    exact_match_scores = np.array([item["exact_match_score"] for item in data])

    metrics = {
        "average_similarity_score": np.mean(similarity_scores),
        "average_context_recall": np.mean(context_recalls),
        "max_similarity_score": np.max(similarity_scores),
        "min_similarity_score": np.min(similarity_scores),
        "std_similarity_score": np.std(similarity_scores),
        "exact_match_count": int(np.sum(exact_match_scores == 1.0)),
    }
    return metrics


def save_metrics(metrics, output_path):

    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved")


if __name__ == "__main__":

    input_file_path = "data/evaluation_results.json"
    output_file_path = "data/output_metrics_updated_prompt.json"

    data = load_data(input_file_path)
    metrics = calculate_metrics(data)
    save_metrics(metrics, output_file_path)
