import os
import pickle
import jsonlines

from sklearn.preprocessing import LabelEncoder

from settings import PathSettings

API_KEY = 'EYieuTm5GRyYwK4dnGv2kFyG'
SECRET_KEY = 'OaDUmCTmVtW4950iqDoKGlkNeZKwE8pg'

import requests
import json


def get_access_token():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}"

    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()["access_token"]

def main():

    data_file = os.path.join(PathSettings.DATA_FOLDER, "bag_data.pkl")

    w = jsonlines.open("sentiment_result.jsonl", "w")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        test = data["test"]

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    sentiment_map = {
        2: "positive",
        1: "neutral",
        0: "negative"
    }
    access_token = get_access_token()
    for _, row in test.iterrows():
        url = f'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={access_token}'
        text = row["Utterance"]
        payload = {"text": text}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload)).json()
        sentiment = response["items"][0]["sentiment"]
        w.write({"text":text, "sentiment":sentiment})

if __name__ == '__main__':
    main()