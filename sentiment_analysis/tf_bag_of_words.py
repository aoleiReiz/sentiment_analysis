import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from settings import PathSettings

train = pd.read_csv(os.path.join(PathSettings.DATA_FOLDER, "train_dp.csv"))
valid = pd.read_csv(os.path.join(PathSettings.DATA_FOLDER, "val_dp.csv"))
test = pd.read_csv(os.path.join(PathSettings.DATA_FOLDER, "test_dp.csv"))

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train["label"])
y_train = tf.keras.utils.to_categorical(y_train)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
corpus = train["cus_comment"].tolist()
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
X_train = X_train.toarray()

X_val = vectorizer.transform(valid["cus_comment"])
X_val = X_val.toarray()
y_val = tf.keras.utils.to_categorical(label_encoder.transform(valid["label"]))

model_save_path = os.path.join(PathSettings.MODEL_FOLDER, "tf_bow.h5")
if not os.path.exists(model_save_path):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation="softmax"))
    print(model.summary())
    model.compile("adam", loss='categorical_crossentropy', metrics=["accuracy"])

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=my_callbacks
    )

    model.save(model_save_path)
else:
    model = tf.keras.models.load_model(model_save_path)

model_name = "tf_embedding_word"
X_test = vectorizer.transform(test["cus_comment"]).toarray()
y_test = label_encoder.fit_transform(test["label"])
y_test = tf.keras.utils.to_categorical(y_test)
predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
predicted = tf.keras.utils.to_categorical(predicted)

acc = accuracy_score(y_test, predicted)
clf_report = classification_report(y_test, predicted)

with open(os.path.join(PathSettings.RESULT_FOLDER, f"{model_name}.txt"), "w", encoding="utf-8") as w:
    w.write(f"{acc}\n")
    w.write(clf_report)
