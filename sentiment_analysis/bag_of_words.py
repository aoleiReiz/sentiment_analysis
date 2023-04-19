import os
import pickle

import spacy
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from settings import PathSettings
import utils

nlp = spacy.load("zh_core_web_sm")

data_file = os.path.join(PathSettings.DATA_FOLDER, "bag_data.pkl")
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

    with open(os.path.join(PathSettings.DATA_FOLDER, "bag_data.pkl"), "wb") as w:
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

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train["Sentiment"])
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
corpus = train["cleanText"].tolist()
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
chi2, p = chi2(X_train, y_train)
p_score_limit = 0.95
feature_names_all = vectorizer.get_feature_names_out()
selected_features = []
for i, pv in enumerate(p):
    if 1 - pv >= p_score_limit:
        selected_features.append(feature_names_all[i])

vectorizer = TfidfVectorizer(vocabulary=selected_features)
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)

model_name = "gradient_boosting"
classifier = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0
)

## train classifier
model = Pipeline([("vectorizer", vectorizer),
                  ("classifier", classifier)])

model["classifier"].fit(X_train, y_train)
## test
X_test = valid["cleanText"].values
y_test = label_encoder.transform(valid["Sentiment"])
predicted = model.predict(X_test)
acc = accuracy_score(y_test, predicted)
clf_report = classification_report(y_test, predicted)

with open(os.path.join(PathSettings.RESULT_FOLDER, f"{model_name}.txt"), "w", encoding="utf-8") as w:
    w.write(f"{acc}\n")
    w.write(clf_report)
