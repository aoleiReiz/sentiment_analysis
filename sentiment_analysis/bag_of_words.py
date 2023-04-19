import os
import pickle

import spacy
import pandas as pd
from sklearn import feature_extraction

from settings import PathSettings
import utils

nlp = spacy.load("zh_core_web_sm")

data_file = os.path.join(PathSettings.DATA_FOLDER,"bag_data.pkl")
if not os.path.exists(data_file):
    train = pd.read_csv(os.path.join(PathSettings.DATA_FOLDER, "train_split.csv"))
    valid = pd.read_csv(os.path.join(PathSettings.DATA_FOLDER, "valid_split.csv"))
    test = pd.read_csv(os.path.join(PathSettings.DATA_FOLDER, "test_split.csv"))

    train = train[['Utterance', 'Sentiment']]
    valid = valid[['Utterance', 'Sentiment']]
    test = test[['Utterance', 'Sentiment']]

    train["cleanText"] = train["Utterance"].apply(lambda x: utils.utils_preprocess_text(x))
    valid["cleanText"] = valid["Utterance"].apply(lambda x: utils.utils_preprocess_text(x))
    test["cleanText"] = test["Utterance"].apply(lambda x: utils.utils_preprocess_text(x))

    with open(os.path.join(PathSettings.DATA_FOLDER,"bag_data.pkl"), "wb") as w:
        pickle.dump({
            "train": train,
            "valid": valid,
            "test": test
        }, w)
else:
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        train = data["train"]
        valid = data["valid"]
        test = data["test"]

bag_vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))