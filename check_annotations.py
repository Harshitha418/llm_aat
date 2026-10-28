import json
from pathlib import Path

dataset_path = Path("data/ner_test.json")

with dataset_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Total sentences: {len(data)}\n")

for index, item in enumerate(data, start=1):
    text = item.get("text", item.get("sentence", ""))

    print(f"Sentence {index}: {text}")

    for entity in item.get("entities", []):
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        extracted_text = text[start:end]

        print(
            f"  {start}:{end} | "
            f"{label} | "
            f"{extracted_text!r}"
        )

    print()