import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

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

def main():
    # 1. Load Dataset
    dataset_path = 'dataset/fake_or_real_news.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    print("Dataset Loaded Successfully.")

    # Combine Title and Text for better features
    df['content'] = df['title'] + " " + df['text']

    # 2. Text Preprocessing
    print("Preprocessing text...")
    df['content'] = df['content'].apply(preprocess_text)

    # 3. Features and Labels
    X = df['content']
    y = df['label']

    # 4. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. TF-IDF Vectorization
    print("Vectorizing text...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # 6. Model Training (Logistic Regression)
    print("Training Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # 7. Model Evaluation
    y_pred = model.predict(X_test_tfidf)
    print("\nModel Evaluation:")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Save Model and Vectorizer
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')
    print("\nModel and Vectorizer saved in 'model/' directory.")

if __name__ == "__main__":
    main()
