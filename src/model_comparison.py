import pandas as pd
from pathlib import Path
import time
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed/bitext_full.csv")
REPORT_PATH = Path("reports/model_comparison.csv")
BEST_MODEL_PATH = Path("data/processed/intent_classifier.joblib")


def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )),
        ("classifier", model)
    ])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_time = time.time()

    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    train_time = time.time() - start_time

    prediction_start = time.time()
    predictions = pipeline.predict(X_test)
    inference_time = time.time() - prediction_start

    result = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "macro_f1": round(f1_score(y_test, predictions, average="macro"), 4),
        "weighted_f1": round(f1_score(y_test, predictions, average="weighted"), 4),
        "training_time_seconds": round(train_time, 2),
        "inference_time_seconds": round(inference_time, 4)
    }

    return result, pipeline


def main():
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

    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )),

        # LinearSVC does not provide probabilities by default,
        # so we calibrate it to support confidence scores.
        ("Calibrated Linear SVM", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"),
            cv=3
        )),

        ("Naive Bayes", MultinomialNB()),

        ("Random Forest", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]

    results = []
    trained_pipelines = {}

    for name, model in models:
        print(f"\nTraining {name}...")
        result, pipeline = evaluate_model(name, model, X_train, X_test, y_train, y_test)

        results.append(result)
        trained_pipelines[name] = pipeline

        print(result)

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False
    )

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(REPORT_PATH, index=False)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    print("\nModel Comparison Results:")
    print(results_df)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved comparison report to {REPORT_PATH}")
    print(f"Saved best model to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()