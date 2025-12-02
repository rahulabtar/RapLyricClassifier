import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------
# Load JSONL properly
# ---------------------
data = []
with open("tokenized_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# ---------------------
# Prepare training data
# ---------------------
X_texts = []
y_labels = []

for entry in data:
    # Convert token IDs -> "108 909 1334 ..."
    token_string = " ".join(str(t) for t in entry["tokens"])
    
    X_texts.append(token_string)
    y_labels.append(entry["artist"])

# ---------------------
# TF-IDF over token IDs
# ---------------------
vectorizer = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"\d+",
    min_df=2
)

X = vectorizer.fit_transform(X_texts)

# ---------------------
# Train/test split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# ---------------------
# Train SVM classifier
# ---------------------
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

# ---------------------
# Evaluate
# ---------------------
preds = svm.predict(X_test)
print(classification_report(y_test, preds))
