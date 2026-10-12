"""
Email Spam Classifier
---------------------
Usage:
    python spam_classifier.py                      # demo with built-in sample data
    python spam_classifier.py --data emails.csv    # CSV with columns: label,text
    python spam_classifier.py --data emails.csv --predict "Win a free iPhone now!!!"

CSV labels can be "spam"/"ham" (or 1/0). For the SMS/UCI datasets
(e.g. spam.csv with v1,v2 columns) pass --label-col v1 --text-col v2.
"""
import argparse
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SAMPLE = [
    ("spam", "WINNER!! You have been selected for a free $1000 gift card. Click now to claim"),
    ("spam", "Congratulations! You won a lottery. Send your bank details to claim prize"),
    ("spam", "Cheap meds online, no prescription needed. Buy now and save 80%"),
    ("spam", "URGENT: your account is suspended. Verify your password at http://bit.ly/x9"),
    ("spam", "Make money fast from home!!! Earn $5000 weekly, limited offer"),
    ("spam", "Free entry in our weekly draw. Text WIN to 80085 to claim your reward"),
    ("spam", "Hot singles in your area want to meet you. Click here"),
    ("spam", "You've been pre-approved for a loan with 0% interest. Apply today"),
    ("spam", "Claim your FREE iPhone now, offer expires tonight!!!"),
    ("spam", "Increase your income. Work from home. No experience required. Act now"),
    ("ham", "Hi Priya, are we still meeting for lunch tomorrow at 1pm?"),
    ("ham", "Please find attached the quarterly report. Let me know your feedback"),
    ("ham", "Reminder: team standup moved to 10:30 AM on Thursday"),
    ("ham", "Can you send me the notes from yesterday's lecture?"),
    ("ham", "Thanks for your help with the project, really appreciate it"),
    ("ham", "Dinner at our place on Saturday? Mom is making biryani"),
    ("ham", "Your invoice for October is attached. Payment is due by the 30th"),
    ("ham", "Meeting agenda: budget review, hiring plan, roadmap updates"),
    ("ham", "Happy birthday! Hope you have a wonderful day with family"),
    ("ham", "I pushed the fix to the repo, can you review the pull request?"),
]


def clean(text: str) -> str:
    """Light normalisation: lowercase, mark URLs/numbers, strip extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\$\s?\d[\d,\.]*", " moneytoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path, label_col, text_col):
    if path is None:
        df = pd.DataFrame(SAMPLE, columns=["label", "text"])
        print("No --data given: using tiny built-in sample (demo only).")
    else:
        df = pd.read_csv(path, encoding="latin-1")[[label_col, text_col]]
        df.columns = ["label", "text"]
    df = df.dropna()
    df["y"] = df["label"].astype(str).str.lower().map(
        {"spam": 1, "1": 1, "ham": 0, "0": 0, "not spam": 0}
    )
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    return df


def build_model(kind):
    clf = MultinomialNB(alpha=0.1) if kind == "nb" else LogisticRegression(
        max_iter=1000, class_weight="balanced"
    )
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2),
                                  stop_words="english", sublinear_tf=True)),
        ("clf", clf),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="emails.csv")
    ap.add_argument("--label-col", default="spam")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model", choices=["nb", "lr"], default="nb")
    ap.add_argument("--predict", help="Classify a single email text")
    ap.add_argument("--save", default="spam_model.joblib")
    args = ap.parse_args()

    df = load_data(args.data, args.label_col, args.text_col)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["y"], test_size=0.25, random_state=42, stratify=df["y"]
    )
    model = build_model(args.model).fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(f"\nModel: {'Naive Bayes' if args.model == 'nb' else 'Logistic Regression'}")
    print(classification_report(y_te, pred, target_names=["ham", "spam"], zero_division=0))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))

    joblib.dump(model, args.save)
    print(f"\nSaved model to {args.save}")

    tests = [args.predict] if args.predict else [
        "Free gift card!!! Click http://win.example now to claim $500",
        "Can we move our call to 4pm? I'll share the slides before.",
    ]
    for t in tests:
        p = model.predict_proba([t])[0][1]
        print(f"\n[{'SPAM' if p >= 0.5 else 'HAM '}] spam prob={p:.2f} | {t}")


if __name__ == "__main__":
    main()