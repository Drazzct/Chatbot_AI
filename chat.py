import random
import json
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# load model
model = load_model("chatbot_model.h5")

# load words & labels
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# load intents
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def bag_of_words(sentence):
    tokens = word_tokenize(sentence.lower())
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag)

def predict_intent(sentence):
    bow = bag_of_words(sentence)
    bow = bow.reshape(1, 1, len(bow))

    result = model.predict(bow, verbose=0)[0]
    intent_index = np.argmax(result)
    return labels[intent_index]

def chatbot_response(sentence):
    intent = predict_intent(sentence)

    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

    return "Mình chưa hiểu câu này 🤔"

print("Chatbot đã sẵn sàng (gõ 'quit' để thoát)")

while True:
    msg = input("Bạn: ")
    if msg.lower() == "quit":
        break

    print("Bot:", chatbot_response(msg))
