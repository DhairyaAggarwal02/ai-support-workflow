import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


DATA_PATH = "data/processed/bitext_sample.csv"
MODEL_PATH = "data/processed/intent_classifier.joblib"


def train_intent_classifier():
    df = pd.read_csv(DATA_PATH)

    X = df["instruction"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


def predict_intent(text):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities)

    return {
        "query": text,
        "predicted_intent": prediction,
        "confidence": round(float(confidence), 3)
    }


if __name__ == "__main__":
    train_intent_classifier()

    test_query = "I want to return my order and get a refund"
    result = predict_intent(test_query)

    print("\nTest Prediction:")
    print(result)