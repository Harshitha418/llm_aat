from transformers import pipeline


MODEL_NAME = "dslim/bert-base-NER"


def load_transformer_model():
    """Load the pretrained Transformer NER model."""
    return pipeline(
        "token-classification",
        model=MODEL_NAME,
        aggregation_strategy="max",
    )


def extract_entities(text, ner_pipeline=None):
    """Extract named entities using the Transformer model."""
    if ner_pipeline is None:
        ner_pipeline = load_transformer_model()

    predictions = ner_pipeline(text)

    return [
        {
            "text": entity["word"],
            "label": entity["entity_group"],
            "start": entity["start"],
            "end": entity["end"],
            "score": float(entity["score"]),
        }
        for entity in predictions
    ]


if __name__ == "__main__":
    ner = load_transformer_model()

    sample_text = (
        "Sundar Pichai visited Google headquarters in California."
    )

    print("Input:", sample_text)
    print("\nEntities detected by Transformer:")

    for entity in extract_entities(sample_text, ner):
        print(
            f"Text: {entity['text']:<20} "
            f"Label: {entity['label']:<8} "
            f"Confidence: {entity['score']:.4f}"
        )