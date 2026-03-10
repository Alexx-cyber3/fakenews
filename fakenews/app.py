from flask import Flask, render_template, request, jsonify
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

app = Flask(__name__)

# Load Model and Vectorizer
model_path = 'model/fake_news_model.pkl'
tfidf_path = 'model/tfidf_vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(tfidf_path):
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
else:
    model = None
    tfidf_vectorizer = None

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Credibility Database (Simplified for Project)
CREDIBILITY_DB = {
    "bbc.com": "High Credibility - Reliable international news source.",
    "nytimes.com": "High Credibility - Established mainstream media.",
    "reuters.com": "High Credibility - Global news agency with strict standards.",
    "theonion.com": "Satire - This is a parody/satirical site, not real news.",
    "abcnews.com.co": "Low Credibility - Known for spreading misinformation.",
    "infowars.com": "Low Credibility - Frequently promotes conspiracy theories.",
    "thegatewaypundit.com": "Questionable - Often cited for biased or unverified claims."
}

def check_credibility(url):
    url = url.lower()
    for domain, info in CREDIBILITY_DB.items():
        if domain in url:
            return info
    return "Unknown - This source is not in our database. Please verify independently."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tfidf_vectorizer is None:
        error_msg = 'Model not trained yet. Please run train_model.py first.'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 400
        return jsonify({'error': error_msg})

    news_text = request.form.get('news_text')
    if not news_text:
        error_msg = 'Please enter some news text.'
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': error_msg}), 400
        return jsonify({'error': error_msg})

    # Preprocess input
    processed_text = preprocess_text(news_text)
    # Vectorize
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(vectorized_text)[0]
    # Confidence Score
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = max(probabilities) * 100
    
    result = {
        'prediction': prediction,
        'confidence': f"{confidence:.2f}%",
        'input_text': news_text
    }

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(result)

    return render_template('index.html', **result)

@app.route('/check_source', methods=['POST'])
def check_source():
    url = request.form.get('news_url')
    if not url:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'error': 'Please enter a URL.'}), 400
        return render_template('index.html', source_result="Please enter a URL.")
    
    result_text = check_credibility(url)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'source_result': result_text})

    return render_template('index.html', source_result=result_text, input_url=url)

if __name__ == '__main__':
    app.run(debug=True)
