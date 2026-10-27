import spacy


MODEL_NAME = "en_core_web_sm"


def load_spacy_model():
    """Load spaCy's pretrained English NER model."""
    return spacy.load(MODEL_NAME)


def extract_entities(text, nlp=None):
    """Extract named entities using spaCy."""
    if nlp is None:
        nlp = load_spacy_model()

    doc = nlp(text)

    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]


if __name__ == "__main__":
    nlp = load_spacy_model()

    sample_text = (
        "Sundar Pichai visited Google headquarters in California."
    )

    print("Input:", sample_text)
    print("\nEntities detected by spaCy:")

    for entity in extract_entities(sample_text, nlp):
        print(
            f"Text: {entity['text']:<20} "
            f"Label: {entity['label']:<8} "
            f"Start: {entity['start']:<3} "
            f"End: {entity['end']}"
        )