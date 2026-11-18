# AI Chatbot Using Python and NLP

An intent-based AI chatbot built with Python, Natural Language Processing (NLP), TensorFlow, and Streamlit. The chatbot processes user messages, predicts the user's intent, and provides a suitable response.

## Features

- Interactive chatbot interface built with Streamlit.
- Text preprocessing using NLTK stemming.
- Intent classification using a TensorFlow/Keras neural network.
- Bag-of-Words text representation.
- JSON-based intents, example questions, and responses.
- Fallback responses for questions the chatbot does not recognize.
- Conversation history during the current app session.

## Technologies Used

- Python
- NLTK
- TensorFlow / Keras
- NumPy
- Streamlit
- JSON

## Project Structure

- `app.py` — runs the Streamlit chatbot interface.
- `chatbot.py` — loads the trained model and generates responses.
- `train.py` — preprocesses the training data and trains the neural network.
- `intents.json` — stores intent tags, example phrases, and responses.
- `requirements.txt` — lists the required Python packages.

## Installation and Setup

1. Install Python 3.10 or 3.11.
2. Open this project folder in VS Code.
3. Open the VS Code terminal.
4. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

5. Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the Chatbot

Run the following command to train the model and create the required model files:

```bash
python train.py
```

## Run the Application

Start the chatbot web application:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to chat with the application.

## Customize the Chatbot

Open `intents.json` and add new intent entries. Each intent should contain:

- `tag` — a unique intent name.
- `patterns` — example messages users may type.
- `responses` — possible chatbot replies.

After changing the dataset, train the model again using `python train.py`, then restart the Streamlit application.

## How It Works

1. The user enters a message in the Streamlit interface.
2. NLTK tokenizes and stems the words.
3. The chatbot converts the text into a Bag-of-Words feature vector.
4. The trained TensorFlow model predicts the most likely intent.
5. The chatbot selects and displays a response associated with that intent.

## Note

This is an educational, intent-based chatbot. It responds using the intents and example phrases provided in its dataset; it is not a general-purpose generative AI assistant.
