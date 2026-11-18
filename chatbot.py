import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent

class Chatbot:
    def __init__(self):
        model_path = BASE_DIR / "chatbot_model.keras"
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run: python train.py")

        self.model = tf.keras.models.load_model(model_path)
        with (BASE_DIR / "words.pkl").open("rb") as file:
            self.words = pickle.load(file)
        with (BASE_DIR / "classes.pkl").open("rb") as file:
            self.classes = pickle.load(file)
        with (BASE_DIR / "intents.json").open("r", encoding="utf-8") as file:
            self.intents = json.load(file)["intents"]

        self.stemmer = PorterStemmer()

    def bag_of_words(self, sentence):
        tokens = re.findall(r"[a-zA-Z']+", sentence.lower())
        tokens = [self.stemmer.stem(word) for word in tokens]
        token_set = set(tokens)
        return np.array([[1 if word in token_set else 0 for word in self.words]], dtype=np.float32)

    def get_response(self, message):
        if not message.strip():
            return "Please type a message."

        probabilities = self.model.predict(self.bag_of_words(message), verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        predicted_tag = self.classes[best_index]

        # Confidence threshold helps avoid random answers to unrelated messages.
        if confidence < 0.45:
            fallback = next((item for item in self.intents if item["tag"] == "fallback"), None)
            if fallback:
                return random.choice(fallback["responses"])
            return "I'm not sure I understood. Could you rephrase?"

        for intent in self.intents:
            if intent["tag"] == predicted_tag:
                return random.choice(intent["responses"])

        return "Sorry, I could not find a suitable response."

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot ready. Type 'quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower().strip() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        print("Bot:", bot.get_response(message))
