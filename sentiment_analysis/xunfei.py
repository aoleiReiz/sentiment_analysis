#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import time
import urllib.request
import urllib.parse
import json
import hashlib
import base64
import jsonlines

from sentiment_analysis.bag_of_words import data_file
from settings import PathSettings

#接口地址
url ="http://ltpapi.xfyun.cn/v2/sa"
#开放平台应用ID
x_appid = "14cedfd4"
#开放平台应用接口秘钥
api_key = "ffb67a4350d506300b8522244368c459"
#语言文本

id2label = {0: "neutral", 1: "positive", -1: 'negative'}

def detect(text):
    body = urllib.parse.urlencode({'text': text}).encode('utf-8')
    param = {"type": "dependent"}
    x_param = base64.b64encode(json.dumps(param).replace(' ', '').encode('utf-8'))
    x_time = str(int(time.time()))
    x_checksum = hashlib.md5(api_key.encode('utf-8') + str(x_time).encode('utf-8') + x_param).hexdigest()
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    req = urllib.request.Request(url, body, x_header)
    result = urllib.request.urlopen(req)
    result = result.read()
    sentiment = json.loads(result.decode("utf-8"))["data"]["sentiment"]
    label = id2label[sentiment]
    return label


if __name__ == '__main__':
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        test = data["test"]
    total_count = 0
    acc_count = 0
    for _, row in test.iterrows():
        text = row["Utterance"]
        try:
            label = detect(text)
            total_count += 1
            if label == row["Sentiment"]:
                acc_count += 1
            print(acc_count, total_count)
        except Exception as e:
            print(e)

