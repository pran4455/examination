import numpy as np
import pandas as pd
import nltk
import string
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download dataset
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Load IMDB dataset from NLTK
def load_imdb_data():
    documents = [(movie_reviews.raw(fileid), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    df = pd.DataFrame(documents, columns=["review", "sentiment"])
    return df

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

if __name__ == "__main__":
    # Load dataset
    df = load_imdb_data()
    
    # Preprocess reviews
    df["cleaned_review"] = df["review"].apply(preprocess_text)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df["cleaned_review"], df["sentiment"], test_size=0.2, random_state=42)

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Use top 5000 words
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier(criterion="entropy", max_depth=20, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Example: Predict sentiment of a new review
    def predict_sentiment(review):
        review_cleaned = preprocess_text(review)
        review_vectorized = vectorizer.transform([review_cleaned])
        prediction = model.predict(review_vectorized)[0]
        return prediction

    # Test on a new movie review
    new_review = "This movie was absolutely fantastic! The acting and storyline were superb."
    print("\nNew Review Sentiment:", predict_sentiment(new_review))
