import tensorflow as tf
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder


from settings import PathSettings
import utils

data_file = os.path.join(PathSettings.DATA_FOLDER, "bag_data.pkl")

with open(data_file, "rb") as f:
    data = pickle.load(f)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train["Sentiment"])
y_train = tf.keras.utils.to_categorical(y_train)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
corpus = train["cleanText"].tolist()
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)

X_val = vectorizer.transform(valid["cleanText"])
y_val = tf.keras.utils.to_categorical(label_encoder.transform(valid["Sentiment"]))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation="softmax"))
print(model.summary())
model.compile("adam", loss='categorical_crossentropy', metrics=["accuracy"])


history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val)
)
