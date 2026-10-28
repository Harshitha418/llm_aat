import json
from pathlib import Path

DATASET_PATH = Path("data/ner_test.json")
BACKUP_PATH = Path("data/ner_test_backup.json")

# Specify the intended entity text and label for each sentence.
# Character offsets will be calculated automatically.
CORRECT_ENTITIES = [
    [("Sundar Pichai", "PER"), ("Google", "ORG"), ("California", "LOC")],
    [("Microsoft", "ORG"), ("Bill Gates", "PER"), ("Paul Allen", "PER")],
    [("Apple", "ORG"), ("Cupertino", "LOC")],
    [("Narendra Modi", "PER"), ("Bengaluru", "LOC")],
    [("Amazon", "ORG"), ("London", "LOC")],
    [("Barack Obama", "PER"), ("Hawaii", "LOC")],
    [("Tesla", "ORG"), ("Elon Musk", "PER"), ("Paris", "LOC")],
    [("United Nations", "ORG"), ("New York", "LOC")],
    [("Sachin Tendulkar", "PER"), ("Mumbai", "LOC")],
    [("OpenAI", "ORG"), ("ChatGPT", "MISC")],
    [("Eiffel Tower", "LOC"), ("France", "LOC")],
    [("Infosys", "ORG"), ("Bengaluru", "LOC"), ("India", "LOC")],
    [
        ("Cristiano Ronaldo", "PER"),
        ("Al Nassr", "ORG"),
        ("Saudi Arabia", "LOC"),
    ],
    [("European Union", "ORG"), ("Brussels", "LOC")],
    [("Ratan Tata", "PER"), ("Tata Group", "ORG"), ("India", "LOC")],
]


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if len(data) != len(CORRECT_ENTITIES):
        raise ValueError(
            f"Expected {len(CORRECT_ENTITIES)} sentences, "
            f"but found {len(data)}."
        )

    # Preserve the original annotations before making changes.
    if not BACKUP_PATH.exists():
        with BACKUP_PATH.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"Backup saved to {BACKUP_PATH}")

    total_entities = 0

    for index, item in enumerate(data):
        text = item.get("text", item.get("sentence", ""))

        if not text:
            raise ValueError(
                f"Sentence {index + 1} has no recognized text field."
            )

        corrected = []

        for entity_text, label in CORRECT_ENTITIES[index]:
            start = text.find(entity_text)

            if start == -1:
                raise ValueError(
                    f"Cannot find {entity_text!r} in sentence "
                    f"{index + 1}: {text}"
                )

            end = start + len(entity_text)

            # Ensure the entity occurs only once, avoiding ambiguity.
            if text.find(entity_text, end) != -1:
                raise ValueError(
                    f"{entity_text!r} occurs more than once in "
                    f"sentence {index + 1}; manual review required."
                )

            corrected.append({
                "start": start,
                "end": end,
                "label": label,
            })

            total_entities += 1

        item["entities"] = corrected

    with DATASET_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print("\nAnnotations corrected successfully!")
    print(f"Sentences: {len(data)}")
    print(f"Reference entities: {total_entities}")
    print("\nDates such as Monday and November were excluded.")
    print("Please run check_annotations.py to verify the result.")


if __name__ == "__main__":
    main()