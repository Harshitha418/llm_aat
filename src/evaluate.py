import csv
import json
import time
from pathlib import Path

from spacy_ner import load_spacy_model, extract_entities as extract_spacy
from transformer_ner import (
    load_transformer_model,
    extract_entities as extract_transformer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ner_test.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Convert model labels into a shared label scheme.
SPACY_LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LANGUAGE": "MISC",
}

TRANSFORMER_LABEL_MAP = {
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
}


def load_dataset():
    """Load and validate the annotated test dataset."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    for index, item in enumerate(data):
        text = item["text"]

        for entity in item["entities"]:
            start = entity["start"]
            end = entity["end"]

            if not (0 <= start < end <= len(text)):
                raise ValueError(
                    f"Invalid entity offsets in sentence {index + 1}: "
                    f"{entity}"
                )

            actual_text = text[start:end]
            if not actual_text:
                raise ValueError(
                    f"Empty entity in sentence {index + 1}"
                )

    return data


def normalize_entities(entities, label_map):
    """Convert predictions to comparable (start, end, label) tuples."""
    normalized = set()

    for entity in entities:
        original_label = entity["label"]
        normalized_label = label_map.get(original_label)

        # Ignore labels that are not included in the shared label scheme.
        if normalized_label is None:
            continue

        normalized.add(
            (
                entity["start"],
                entity["end"],
                normalized_label,
            )
        )

    return normalized


def calculate_metrics(reference_entities, predicted_entities):
    """Calculate strict entity-level precision, recall and F1."""
    true_positives = len(reference_entities & predicted_entities)
    false_positives = len(predicted_entities - reference_entities)
    false_negatives = len(reference_entities - predicted_entities)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate_model(data, model_name, model, extractor, label_map):
    """Evaluate one model on the same annotated examples."""
    all_reference = set()
    all_predicted = set()
    errors = []
    inference_seconds = 0.0
    prediction_rows = []

    for sentence_id, item in enumerate(data, start=1):
        text = item["text"]

        reference = {
            (entity["start"], entity["end"], entity["label"])
            for entity in item["entities"]
        }

        start_time = time.perf_counter()
        raw_predictions = extractor(text, model)
        inference_seconds += time.perf_counter() - start_time

        predicted = normalize_entities(raw_predictions, label_map)

        all_reference.update(
            (sentence_id, start, end, label)
            for start, end, label in reference
        )
        all_predicted.update(
            (sentence_id, start, end, label)
            for start, end, label in predicted
        )
        missing = reference - predicted
        extra = predicted - reference

        for start, end, label in sorted(missing):
            errors.append({
                "model": model_name,
                "sentence_id": sentence_id,
                "text": text,
                "error_type": "False Negative",
                "entity_text": text[start:end],
                "expected_label": label,
                "predicted_label": "",
            })

        for start, end, label in sorted(extra):
            errors.append({
                "model": model_name,
                "sentence_id": sentence_id,
                "text": text,
                "error_type": "False Positive",
                "entity_text": text[start:end],
                "expected_label": "",
                "predicted_label": label,
            })

        prediction_rows.append({
            "model": model_name,
            "sentence_id": sentence_id,
            "text": text,
            "reference_entities": json.dumps(
                sorted(
                    [
                        {
                            "text": text[start:end],
                            "label": label,
                            "start": start,
                            "end": end,
                        }
                        for start, end, label in reference
                    ],
                    key=lambda entity: (entity["start"], entity["end"]),
                )
            ),
            "predicted_entities": json.dumps(
                sorted(
                    [
                        {
                            "text": text[start:end],
                            "label": label,
                            "start": start,
                            "end": end,
                        }
                        for start, end, label in predicted
                    ],
                    key=lambda entity: (entity["start"], entity["end"]),
                )
            ),
        })

    overall = calculate_metrics(all_reference, all_predicted)

    labels = sorted(
        {entity[3] for entity in all_reference}
        | {entity[3] for entity in all_predicted}
    )

    per_label_metrics = {}

    for label in labels:
        reference_for_label = {
            entity for entity in all_reference if entity[3] == label
        }
        predicted_for_label = {
            entity for entity in all_predicted if entity[3] == label
        }

        per_label_metrics[label] = calculate_metrics(
            reference_for_label,
            predicted_for_label,
        )

    overall["model"] = model_name
    overall["sentences"] = len(data)
    overall["reference_entity_count"] = len(all_reference)
    overall["predicted_entity_count"] = len(all_predicted)
    overall["inference_seconds"] = inference_seconds
    overall["average_inference_ms"] = (
        inference_seconds / len(data) * 1000
    )

    return overall, per_label_metrics, errors, prediction_rows


def save_csv(path, rows):
    """Save a list of dictionaries as a CSV file."""
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()

    print(f"Loaded {len(data)} sentences.")
    print(
        "Reference entities:",
        sum(len(item["entities"]) for item in data),
    )

    print("\nLoading spaCy model...")
    spacy_model = load_spacy_model()

    print("Loading Transformer model...")
    transformer_model = load_transformer_model()

    models = [
        (
            "spaCy",
            spacy_model,
            extract_spacy,
            SPACY_LABEL_MAP,
        ),
        (
            "Transformer",
            transformer_model,
            extract_transformer,
            TRANSFORMER_LABEL_MAP,
        ),
    ]

    summary_rows = []
    per_label_rows = []
    all_errors = []
    all_predictions = []

    for name, model, extractor, label_map in models:
        print(f"\nEvaluating {name}...")

        metrics, per_label, errors, predictions = evaluate_model(
            data,
            name,
            model,
            extractor,
            label_map,
        )

        summary_rows.append(metrics)
        all_errors.extend(errors)
        all_predictions.extend(predictions)

        for label, label_metrics in per_label.items():
            per_label_rows.append({
                "model": name,
                "label": label,
                **label_metrics,
            })

        print(f"\n{name} results")
        print("-" * 40)

        for metric in (
            "true_positives",
            "false_positives",
            "false_negatives",
        ):
            print(f"{metric}: {metrics[metric]}")

        for metric in ("precision", "recall", "f1_score"):
            print(f"{metric}: {metrics[metric]:.4f}")

        print(
            "Average inference time: "
            f"{metrics['average_inference_ms']:.2f} ms/sentence"
        )

    save_csv(RESULTS_DIR / "overall_metrics.csv", summary_rows)
    save_csv(RESULTS_DIR / "per_label_metrics.csv", per_label_rows)
    save_csv(RESULTS_DIR / "error_analysis.csv", all_errors)
    save_csv(RESULTS_DIR / "predictions.csv", all_predictions)

    print("\nResults saved in:", RESULTS_DIR)
    print("- overall_metrics.csv")
    print("- per_label_metrics.csv")
    print("- error_analysis.csv")
    print("- predictions.csv")


if __name__ == "__main__":
    main()