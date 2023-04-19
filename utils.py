import spacy

nlp = spacy.load("zh_core_web_sm")

def utils_preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not (token.is_punct or token.is_space)]
    return " ".join(tokens)
