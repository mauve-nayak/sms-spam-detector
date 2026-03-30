# main.py
# Requirements: pandas scikit-learn joblib
# pip install pandas scikit-learn joblib

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "spam_detector.joblib"
DATA_PATH = "spam.csv"
ENCODING = "latin-1"

def clear_console():
    """Clear console in a cross-platform way."""
    os.system("cls" if os.name == "nt" else "clear")

def load_dataset(path=DATA_PATH, encoding=ENCODING):
    df = pd.read_csv(path, encoding=encoding)
    if {'v1', 'v2'}.issubset(df.columns):
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    else:
        df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})[['label', 'message']]
    df = df.dropna(subset=['label', 'message']).reset_index(drop=True)
    return df

def build_model():
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2)),
        ("char_tfidf", TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2))
    ])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    model = make_pipeline(feature_union, clf)
    return model

def train_and_save_model(df, model, save_path=MODEL_PATH):
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model (this may take a little while)...")
    model.fit(X_train, y_train)

    # quick evaluation (printed so user sees model quality)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel trained. Test accuracy: {acc:.4f}")
    print("Spam class report (brief):")
    print(classification_report(y_test, y_pred, digits=4))
    # Save
    joblib.dump(model, save_path)
    print(f"\nModel saved to: {os.path.abspath(save_path)}")
    return model

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            print("Failed to load saved model (will retrain). Error:", e)
    # if not present or failed to load -> train
    df = load_dataset()
    model = build_model()
    model = train_and_save_model(df, model)
    return model

def predict_text(model, text):
    text_list = [text]
    try:
        pred = model.predict(text_list)[0]
    except Exception as e:
        return None, None, f"Prediction failed: {e}"

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(text_list)[0]
            # classifier classes are accessible from the last pipeline step
            last_step_name = list(model.named_steps.keys())[-1]
            classes = model.named_steps[last_step_name].classes_
            # find probability for the predicted class
            prob = float(probs[list(classes).index(pred)])
        except Exception:
            prob = None
    return pred, prob, None

def main():
    clear_console()

    # short one-line description
    print("SMS Spam Detector — classify an SMS as 'spam' or 'ham'.\n")

    # read SMS from user
    sms = input("Enter SMS message to classify (or press Enter to quit): ").strip()
    if sms == "":
        print("No input provided. Exiting.")
        sys.exit(0)

    # load or train model
    model = load_or_train_model()

    # predict
    pred, prob, err = predict_text(model, sms)
    if err:
        print(err)
        sys.exit(1)

    # show result cleanly
    if prob is None:
        print(f"\nPrediction: {pred}")
    else:
        if (pred == "ham"):
            print(f"\nPrediction: Not Spam  (confidence: {prob:.2%})")
        else:
            print(f"\nPrediction: {pred}  (confidence: {prob:.2%})")


if __name__ == "__main__":
    main()
