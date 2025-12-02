import sklearn.linear_model as lm
import sklearn.metrics as metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory for plots
os.makedirs("naive_bayes_results", exist_ok=True)

# Load JSONL properly
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
    # Skip empty entries
    if not entry.get("tokens") or len(entry["tokens"]) <= 2:  # Skip if only [CLS] and [SEP]
        continue
    # Convert token IDs -> "108 909 1334 ..." (space-separated string)
    token_string = " ".join(str(t) for t in entry["tokens"])
    X_texts.append(token_string)
    y_labels.append(entry["artist"])

# ---------------------
# Convert token ID strings to count vectors
# ---------------------
vectorizer = CountVectorizer(
    analyzer="word",
    token_pattern=r"\d+",  # Match sequences of digits (token IDs)
    min_df=2  # Ignore tokens that appear in fewer than 2 documents
)

X = vectorizer.fit_transform(X_texts)

print(f"Data shape: {X.shape}")
print(f"Number of samples: {len(y_labels)}")
print(f"Number of unique artists: {len(set(y_labels))}")

# ---------------------
# Train/test split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# Get unique classes for multi-class ROC
classes = sorted(list(set(y_labels)))
n_classes = len(classes)

print(f"\nClasses: {classes}")
print(f"Number of classes: {n_classes}")

# ---------------------
# Train Naive Bayes classifier with different alpha values
# ---------------------

print("\nTraining Multinomial Naive Bayes classifier with different alpha values...")

# Binarize labels for multi-class ROC
y_test_binarized = label_binarize(y_test, classes=classes)

for i in range(11):
    alpha = i / 10
    print(f"\n{'-'*80}")
    print(f"Alpha = {alpha:.1f}")
    print(f"{'-'*80}")
    
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=classes)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Alpha = {alpha:.1f})', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"naive_bayes_results/confusion_matrix_alpha_{alpha:.1f}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: naive_bayes_results/confusion_matrix_alpha_{alpha:.1f}.png")
    
    # ROC Curve (multi-class: one-vs-rest)
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for class_idx in range(n_classes):
        fpr[class_idx], tpr[class_idx], _ = roc_curve(
            y_test_binarized[:, class_idx], 
            pred_proba[:, class_idx]
        )
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_binarized.ravel(), 
        pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for class_idx, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[class_idx], 
            tpr[class_idx], 
            color=color, 
            lw=2,
            label=f'{classes[class_idx]} (AUC = {roc_auc[class_idx]:.2f})'
        )
    
    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"], 
        tpr["micro"],
        color='deeppink', 
        linestyle='--', 
        linewidth=2,
        label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})'
    )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Multi-class (Alpha = {alpha:.1f})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"naive_bayes_results/roc_curve_alpha_{alpha:.1f}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: naive_bayes_results/roc_curve_alpha_{alpha:.1f}.png")
    
    # Print AUC scores
    print(f"\nAUC Scores:")
    for class_idx in range(n_classes):
        print(f"  {classes[class_idx]}: {roc_auc[class_idx]:.4f}")
    print(f"  Micro-average: {roc_auc['micro']:.4f}")

print(f"\n{'='*80}")
print("All results saved to 'naive_bayes_results/' directory")
print(f"{'='*80}")



