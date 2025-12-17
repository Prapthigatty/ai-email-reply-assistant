import pandas as pd
import string
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

STOP_WORDS = set(stopwords.words("english"))

# Load dataset
data = pd.read_csv("data/emails.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

data["clean_email"] = data["email"].apply(clean_text)

X = data["clean_email"]
y = data["intent"]


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)

X_tfidf = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("intent_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))




scores = cross_val_score(
    model,
    X_tfidf,
    y,
    cv=4,
    scoring="accuracy"
)

print("Cross-validation accuracies:", scores)
print("Mean CV accuracy:", scores.mean())


print("Improved model trained and saved")
