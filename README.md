# 🤖 Simple AI Chatbot using LSTM

A simple AI chatbot built with **Python** and **LSTM (Long Short-Term Memory)** for intent classification and response generation.  
This project demonstrates the full NLP pipeline from text preprocessing to model training, saving, and inference.

---

## 📌 Project Overview

The chatbot works by classifying user input into predefined **intents** and returning a corresponding response.  
It is designed as an educational project to understand how traditional NLP pipelines and LSTM-based models work.

---

## 🧠 Model Pipeline

User Input
→ Tokenization (NLTK)
→ Bag of Words Vectorization
→ LSTM Model
→ Intent Prediction
→ Predefined Response

---

## ⚙️ Features

- Text preprocessing using **NLTK**
- Bag-of-Words feature encoding
- LSTM-based intent classification
- Model persistence (save & load trained model)
- Separation of training and inference pipelines
- Easy to extend with new intents and responses

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NLTK
- NumPy

---

## 📂 Project Structure


Chatbot_AI/
│
├── intents.json          # Training data (intents & responses)
├── train.py              # Train and save the LSTM model
├── chat.py               # Run chatbot using the trained model
├── chatbot_model.h5      # Saved LSTM model
├── words.pkl             # Saved vocabulary
├── labels.pkl            # Saved intent labels
├── requirements.txt      # Python dependencies
└── README.md

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
````

---

### 2️⃣ Train the model (run once)

```bash
python train.py
```

This will generate:

* `chatbot_model.h5`
* `words.pkl`
* `labels.pkl`

---

### 3️⃣ Run the chatbot

```bash
python chat.py
```

Type your message in the terminal and start chatting.
Type `quit` to exit.

---

## 💬 Example

```
You: xin chào
Bot: Chào bạn 👋

You: bạn tên gì
Bot: Mình là chatbot LSTM 🤖
```

---

## ⚠️ Limitations

* The chatbot does **not generate new sentences**
* It only classifies input into predefined intents
* No conversation memory
* Limited semantic understanding (Bag of Words based)

---

## 📈 Possible Improvements

* Replace Bag of Words with **Embedding layer**
* Improve Vietnamese tokenization (e.g. underthesea, pyvi)
* Add confidence threshold for intent prediction
* Deploy as a REST API using Flask
* Integrate with web or messaging platforms

---

## 📚 Learning Outcomes

* Understanding NLP preprocessing steps
* Applying LSTM for text classification
* Managing ML workflows (train → save → load → inference)
* Strengthening Python and machine learning fundamentals

---

## 📄 License

This project is for educational purposes.
