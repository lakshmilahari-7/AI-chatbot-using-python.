import json
import pickle
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
stemmer = PorterStemmer()

def tokenize(sentence):
    """Split a sentence into simple word tokens."""
    return re.findall(r"[a-zA-Z']+", sentence.lower())

def preprocess(sentence):
    """Lowercase, tokenize and stem each word."""
    return [stemmer.stem(word) for word in tokenize(sentence)]

def build_training_data():
    with (BASE_DIR / "intents.json").open("r", encoding="utf-8") as file:
        data = json.load(file)

    all_words = []
    tags = []
    documents = []

    for intent in data["intents"]:
        tag = intent["tag"]
        if tag == "fallback":
            continue
        for pattern in intent["patterns"]:
            words = preprocess(pattern)
            all_words.extend(words)
            documents.append((words, tag))
        if tag not in tags:
            tags.append(tag)

    vocabulary = sorted(set(all_words))
    tags = sorted(tags)

    X, y = [], []
    for words, tag in documents:
        word_set = set(words)
        X.append([1 if word in word_set else 0 for word in vocabulary])
        y.append(tags.index(tag))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), vocabulary, tags

def train():
    tf.keras.utils.set_random_seed(42)
    X, y, vocabulary, tags = build_training_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(vocabulary),)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(tags), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X, y, epochs=250, batch_size=8, verbose=0)

    model.save(BASE_DIR / "chatbot_model.keras")
    with (BASE_DIR / "words.pkl").open("wb") as file:
        pickle.dump(vocabulary, file)
    with (BASE_DIR / "classes.pkl").open("wb") as file:
        pickle.dump(tags, file)

    print("Training complete. Model and vocabulary files saved.")

if __name__ == "__main__":
    train()
