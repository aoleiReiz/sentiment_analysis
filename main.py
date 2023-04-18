import xml.etree.ElementTree as ET
import pandas as pd


columns = ["weiboId", "sentence", "category0","category1", "category2"]
rows = []
cur_id = 1
for file in ["data/Training data for Emotion Classification.xml","data/Training data for Emotion Expression Identification.xml"]:
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:
        for sub_child in child:
            if sub_child.attrib['opinionated'] == 'N':
                rows.append([cur_id, sub_child.text, "N", "none", "none"])
            else:
                rows.append([cur_id, sub_child.text, "Y", sub_child.attrib.get("emotion-1-type", "none"), sub_child.attrib.get("emotion-2-type", "none")])

df = pd.DataFrame(rows, columns=columns)
df.to_csv("train.csv", index_label=False, index=False, encoding="utf-8")