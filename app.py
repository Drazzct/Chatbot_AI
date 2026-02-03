from flask import Flask, request, jsonify
from flask_cors import CORS
import random, json, pickle, numpy as np
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
import os

import nltk
nltk.download("punkt")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = load_model("chatbot_model.h5")

with open("words.pkl", "rb") as f:
    words = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def bag_of_words(sentence):
    tokens = word_tokenize(sentence.lower())
    return np.array([1 if w in tokens else 0 for w in words])

def predict(sentence):
    bow = bag_of_words(sentence).reshape(1,1,-1)
    intent = labels[np.argmax(model.predict(bow, verbose=0))]
    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

@app.route("/")
def home():
    return {"status": "ok", "message": "Chatbot API running"}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"response": "Bạn hãy nhập nội dung nhé 😊"})

    try:
        return jsonify({"response": predict(text)})
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"response": "Bot gặp lỗi, thử lại sau nhé 😢"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
