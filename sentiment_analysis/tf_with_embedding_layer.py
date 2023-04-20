import os
import pickle

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

vectorizer = tf.keras.layers.TextVectorization(max_tokens=20000, output_sequence_length=200)

from settings import PathSettings

data_file = os.path.join(PathSettings.DATA_FOLDER, "bag_data.pkl")
label_encoder = LabelEncoder()

with open(data_file, "rb") as f:
    data = pickle.load(f)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

text_ds = tf.data.Dataset.from_tensor_slices(train["cleanText"].tolist()).batch(128)
vectorizer.adapt(text_ds)
print(vectorizer.get_vocabulary()[:5])

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))
num_tokens = len(voc) + 2
embedding_dim = 100

embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim
)

model_save_path = os.path.join(PathSettings.MODEL_FOLDER, "tf_embedding.h5")
label_encoder.fit(train["Sentiment"])

if not os.path.exists(model_save_path):
    int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu")(embedded_sequences)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    preds = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(int_sequences_input, preds)
    print(model.summary())

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )

    x_train = vectorizer(np.array([[s] for s in train["cleanText"].tolist()])).numpy()
    x_val = vectorizer(np.array([[s] for s in valid["cleanText"].tolist()])).numpy()

    y_train = tf.keras.utils.to_categorical(label_encoder.fit_transform(train["Sentiment"]))
    y_val = tf.keras.utils.to_categorical(label_encoder.transform(valid["Sentiment"]))

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
    ]
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val), callbacks=my_callbacks)
    model.save(model_save_path)
else:
    model = tf.keras.models.load_model(model_save_path)

x_test = vectorizer(np.array([[s] for s in test["cleanText"].tolist()])).numpy()
y_test = tf.keras.utils.to_categorical(label_encoder.transform(test["Sentiment"]))
int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
predicted = model(x_test).numpy()
predicted = np.argmax(predicted, axis=1)
predicted = tf.keras.utils.to_categorical(predicted)


acc = accuracy_score(y_test, predicted)
clf_report = classification_report(y_test, predicted)

with open(os.path.join(PathSettings.RESULT_FOLDER, f"tf_embedding.txt"), "w", encoding="utf-8") as w:
    w.write(f"{acc}\n")
    w.write(clf_report)
