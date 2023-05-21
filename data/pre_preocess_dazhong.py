import pandas as pd
import numpy as np


def convert_label(star):
    if star >= 4:
        return 1
    elif star == 3:
        return 0
    else:
        return -1


data = pd.read_csv("大众点评data.csv")
print(data[['stars', 'cus_comment']].head(3))
print(data.columns)

data = data.loc[np.random.permutation(data.index)].copy()
data = data[(data.stars.notnull()) & data.cus_comment.notnull()].copy()
data["label"] = data.stars.apply(lambda x: convert_label(x))

train_size = 0.7
val_size = 0.15
n_train = int(len(data) * train_size)
n_val = int(len(data) * val_size)
n_test = len(data) - n_train - n_val

train = data.head(n_train)
val = data.iloc[n_train: n_train + n_val, :]
test = data.tail(n_test)

train[["label","cus_comment"]].to_csv("train_dp.csv", index_label=False, index=False)
val[["label","cus_comment"]].to_csv("val_dp.csv", index_label=False, index=False)
test[["label","cus_comment"]].to_csv("test_dp.csv", index_label=False, index=False)