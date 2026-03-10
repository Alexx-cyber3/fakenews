# AI-Based Fake News Detection System

This project detects whether a news article is **Fake or Real** based on its text content using Machine Learning (Logistic Regression) and Natural Language Processing (TF-IDF).

## Project Structure

- `dataset/`: Contains the CSV dataset.
- `model/`: Stores the trained model and vectorizer.
- `templates/`: HTML frontend files.
- `static/`: CSS styling files.
- `train_model.py`: Script to preprocess data and train the ML model.
- `app.py`: Flask backend to serve the web application.
- `requirements.txt`: List of dependencies.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python train_model.py
   ```
   This will generate `fake_news_model.pkl` and `tfidf_vectorizer.pkl` in the `model/` folder.

3. **Run the Web App:**
   ```bash
   python app.py
   ```
   Open your browser and go to `http://127.0.0.1:5000/`.

## Example Input & Output

- **Input:** "NASA confirms Moon is actually made of cheese."
- **Output:** Prediction: **FAKE**, Confidence Score: ~90%

- **Input:** "The Federal Reserve announces interest rate hike."
- **Output:** Prediction: **REAL**, Confidence Score: ~85%

## Libraries Used
- **Pandas/Numpy:** For data manipulation.
- **Scikit-learn:** For Logistic Regression and TF-IDF Vectorization.
- **NLTK:** For text preprocessing (stopwords, tokenization).
- **Flask:** For the web interface.
- **Joblib:** For saving and loading the model.
