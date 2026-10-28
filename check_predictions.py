import json
from pathlib import Path

import pandas as pd

results_path = Path("results/predictions.csv")
df = pd.read_csv(results_path)

required_columns = {
    "model",
    "sentence_id",
    "text",
    "reference_entities",
    "predicted_entities",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for _, row in df.iterrows():
    text = row["text"]
    reference = json.loads(row["reference_entities"])
    predicted = json.loads(row["predicted_entities"])

    # Validate that every entity span matches its sentence text.
    for source_name, entities in (
        ("reference", reference),
        ("prediction", predicted),
    ):
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            extracted = text[start:end]

            if not (0 <= start < end <= len(text)):
                print(
                    f"INVALID BOUNDS | {row['model']} | "
                    f"Sentence {row['sentence_id']} | "
                    f"{source_name} | {entity}"
                )
            elif extracted != entity["text"]:
                print(
                    f"TEXT MISMATCH | {row['model']} | "
                    f"Sentence {row['sentence_id']} | "
                    f"{source_name} | "
                    f"stored={entity['text']!r}, "
                    f"actual={extracted!r}"
                )

print("\nChecking exact entity matches...")

for model in sorted(df["model"].unique()):
    subset = df[df["model"] == model]
    total_reference = 0
    total_predicted = 0
    total_matched = 0

    for _, row in subset.iterrows():
        reference = json.loads(row["reference_entities"])
        predicted = json.loads(row["predicted_entities"])

        reference_set = {
            (e["start"], e["end"], e["label"])
            for e in reference
        }
        predicted_set = {
            (e["start"], e["end"], e["label"])
            for e in predicted
        }

        total_reference += len(reference_set)
        total_predicted += len(predicted_set)
        total_matched += len(reference_set & predicted_set)

    print(f"\nModel: {model}")
    print(f"Reference entities: {total_reference}")
    print(f"Predicted entities: {total_predicted}")
    print(f"Exact matches: {total_matched}")

    precision = (
        total_matched / total_predicted if total_predicted else 0
    )
    recall = (
        total_matched / total_reference if total_reference else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")