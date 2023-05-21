import pickle
import os

from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import pipeline
import jsonlines

from settings import PathSettings

id2label = {0: "neutral", 1: "positive", 2: 'negative'}
label2id = {"neutral": 0, "positive": 1, "negative": 2}
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = TFAutoModelForSequenceClassification.from_pretrained(
    r"D:\Downloads\bert-base-chinese-fine-tuned-on-cped-sa", num_labels=3, id2label=id2label, label2id=label2id
)

p = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

data_file = os.path.join(PathSettings.DATA_FOLDER, "bag_data.pkl")

w = jsonlines.open("sentiment_result.jsonl", "w")
with open(data_file, "rb") as f:
    data = pickle.load(f)
    test = data["test"]

test_predicts = []
texts = test["Utterance"].tolist()
batch_size = 30
for i in range(0, len(texts), batch_size):
    batched_texts = texts[i: i + batch_size]
    preds = p.predict(batched_texts)
    for pred in preds:
        test_predicts.append(pred)

test["prediction"] = test_predicts
test.to_csv("bert-fine-tuned.csv", index_label=False, index=False)
