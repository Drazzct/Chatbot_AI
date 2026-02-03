import json
import pickle
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

nltk.download('punkt')

# load intents
with open('intents.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = word_tokenize(pattern.lower())
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = sorted(set(words))
labels = sorted(labels)

# vectorization
training = []
output = []
out_empty = [0] * len(labels)

for i, doc in enumerate(docs_x):
    bag = [1 if w in doc else 0 for w in words]

    output_row = out_empty[:]
    output_row[labels.index(docs_y[i])] = 1

    training.append(bag)
    output.append(output_row)

X = np.array(training)
y = np.array(output)

X = X.reshape(X.shape[0], 1, X.shape[1])

# build model
model = Sequential()
model.add(LSTM(128, input_shape=(1, X.shape[2])))
model.add(Dense(len(labels), activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# save model & data
model.save("chatbot_model.h5")

with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Training completed & model saved!")
