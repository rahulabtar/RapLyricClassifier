import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)

# ---------------------
# Load JSONL properly
# ---------------------
data = []
with open("tokenized_dataset_nodre.jsonl", "r", encoding="utf-8") as f:
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
# 5-Fold Cross Validation with metrics and ROC
# ---------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []
fold_macro_f1s = []

# For ROC plotting
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

    # Train SVM (probability=True for ROC)
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    # Metrics
    fold_acc = accuracy_score(y_test, preds)
    fold_f1 = f1_score(y_test, preds, average="macro")
    fold_accuracies.append(fold_acc)
    fold_macro_f1s.append(fold_f1)

    # Compute ROC per fold (OvR)
    y_score = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)

    # Interpolate for mean plot across folds
    tprs.append(np.interp(mean_fpr, all_fpr, mean_tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc_macro)

    fold += 1

# ---------------------
# Report averages
# ---------------------
print("\n======================")
print("Cross-Validation Summary")
print("======================\n")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Average Macro F1: {np.mean(fold_macro_f1s):.4f} ± {np.std(fold_macro_f1s):.4f}")
print(f"Average Macro AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ---------------------
# Plot mean ROC
# ---------------------
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',
         label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

std_tpr = np.std(tprs, axis=0)
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='b', alpha=0.2)

plt.plot([0,1], [0,1], linestyle='--', color='r', lw=2)
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve Across 5 Folds")
plt.legend(loc="lower right")
plt.show()
