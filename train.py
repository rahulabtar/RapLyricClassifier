from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess_data import load_tokenized_data, preprocess_sequences, print_preprocessing_stats

# ---------------------
# Load and preprocess data
# ---------------------
print("Loading tokenized data...")
data = load_tokenized_data("tokenized_dataset.jsonl")

print("Preprocessing sequences...")
X_texts, y_labels, stats = preprocess_sequences(data)

print("\n" + "="*80)
print("PREPROCESSING STATISTICS")
print("="*80)
print_preprocessing_stats(stats)

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
print("Training SVM classifier...")
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

# ---------------------
# Evaluate
# ---------------------
print("Evaluating SVM classifier...")
preds = svm.predict(X_test)
print(classification_report(y_test, preds))
