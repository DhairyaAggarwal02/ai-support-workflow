import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed/bitext_full.csv")
MODEL_PATH = Path("data/processed/intent_classifier.joblib")
REPORT_PATH = Path("reports/intent_classifier_report.txt")
CONFUSION_MATRIX_PATH = Path("reports/intent_confusion_matrix.csv")


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
            max_features=20000
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    report = classification_report(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("\nClassification Report:")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("Intent Classifier Evaluation Report\n")
        file.write("=" * 50 + "\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        file.write(report)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


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