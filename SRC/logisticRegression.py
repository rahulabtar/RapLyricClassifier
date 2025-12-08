import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)
import time

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
# Multi-class setup for ROC
# ---------------------
classes = sorted(list(set(y_labels)))
y_bin = label_binarize(y_labels, classes=classes)
n_classes = y_bin.shape[1]

# ---------------------
# 5-Fold Cross Validation
# ---------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []
fold_macro_f1s = []
fold_aucs = []
fold_train_times = []

# For ROC aggregation
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(8,6))

fold = 1
for train_idx, test_idx in skf.split(X, y_labels):

    print(f"\n======================")
    print(f"      Fold {fold}")
    print(f"======================\n")

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = [y_labels[i] for i in train_idx]
    y_test = [y_labels[i] for i in test_idx]

    y_test_bin = label_binarize(y_test, classes=classes)

    # ---------------------
    # Train Logistic Regression
    # ---------------------
    start_time = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        random_state=42
    )
    lr.fit(X_train, y_train)
    train_time = time.time() - start_time

    fold_train_times.append(train_time)

    preds = lr.predict(X_test)

    print(classification_report(y_test, preds))

    # Metrics for averages
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    fold_accuracies.append(acc)
    fold_macro_f1s.append(macro_f1)

    # ---------------------
    # ROC/AUC per fold
    # ---------------------
    y_scores = lr.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC for this fold
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr_fold = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr_fold += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr_fold /= n_classes

    fold_auc = auc(all_fpr, mean_tpr_fold)
    fold_aucs.append(fold_auc)

    # Add to global mean ROC computation
    interp_tpr = np.interp(mean_fpr, all_fpr, mean_tpr_fold)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(fold_auc)

    fold += 1

# ---------------------
# Final Summary
# ---------------------
print("\n======================")
print(" Cross-Validation Summary")
print("======================\n")

print(f"Average Accuracy:     {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Average Macro F1:      {np.mean(fold_macro_f1s):.4f} ± {np.std(fold_macro_f1s):.4f}")
print(f"Average Macro AUC:     {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"Training Time (sec):   {np.mean(fold_train_times):.2f} ± {np.std(fold_train_times):.2f}")

# ---------------------
# Plot Mean ROC Curve
# ---------------------
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

std_tpr = np.std(tprs, axis=0)

plt.plot(mean_fpr, mean_tpr,
         label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
         lw=2)

plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve (Logistic Regression, 5-Fold CV)")
plt.legend(loc="lower right")
plt.show()
