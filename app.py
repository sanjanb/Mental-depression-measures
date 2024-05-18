from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import re
import string

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('mental_health.csv')

# Data preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = nltk.word_tokenize(text)
    text = ' '.join(tokens)
    return text

nltk.download('punkt')
df['text'] = df['text'].apply(preprocess_text)
# Renaming the target column
df.rename(columns={'label': 'poisonous'}, inplace=True)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['poisonous']

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Function to predict of user input
def predict_toxicity(user_input):
    preprocessed_input = preprocess_text(user_input)
    input_vec = vectorizer.transform([preprocessed_input])
    prediction = model.predict(input_vec)
    confidence = np.max(model.predict_proba(input_vec))
    return prediction[0], confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    prediction, confidence = predict_toxicity(user_input)

    if prediction == 1:
        negative = f" You don't have to go through this alone, Lets figure out How can I help!"
        return render_template('negative.html', negative=negative)
    else:
        positive = f"Thats fantastic news, I am so happy for you, Would you like to evolve with us"
        return render_template('positive.html', positive=positive)
    
@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/counselor')
def counselor():
    return render_template('counselor.html')

@app.route('/games')
def games():
    return render_template('games.html')


if __name__ == '__main__':
    app.run(debug=True)
