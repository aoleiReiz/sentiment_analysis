import pandas as pd
import json

df = pd.read_csv("../results/bert-fine-tuned.csv")
n = len(df)
acc_count = 0
for _, row in df.iterrows():
    acc_count += (eval(row["prediction"])["label"] == row["Sentiment"])

print(acc_count/n)