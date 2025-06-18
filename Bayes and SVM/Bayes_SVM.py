import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Step 1: Load Preprocessed File
df = pd.read_csv("../data/file_preprocessed.csv")
df = df.dropna(subset=["tweets"])
df = df[df["tweets"].str.strip() != ""]

# Step 2: Feature/Label
X = df["tweets"]
y = df["labels"]

# Binarize for ROC later
y_bin = label_binarize(y, classes=[0, 1, 2])

# Step 3: Train-test Split
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=42, stratify=y
)

# Step 4: TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ========= Model 1: Multinomial Naive Bayes ========= #
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

print("🔍 Naive Bayes Classification Report:")
print(classification_report(y_test, nb_preds, target_names=["bad", "neutral", "good"]))

# Confusion Matrix
nb_cm = confusion_matrix(y_test, nb_preds)
sns.heatmap(nb_cm, annot=True, cmap="Oranges", fmt="d", xticklabels=["bad", "neutral", "good"], yticklabels=["bad", "neutral", "good"])
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# # ========= Model 2: Linear SVM ========= #
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)

print("🔍 SVM Classification Report:")
print(classification_report(y_test, svm_preds, target_names=["bad", "neutral", "good"]))

# Confusion Matrix
svm_cm = confusion_matrix(y_test, svm_preds)
sns.heatmap(svm_cm, annot=True, cmap="Blues", fmt="d", xticklabels=["bad", "neutral", "good"], yticklabels=["bad", "neutral", "good"])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ========= Cross Validation (Optional) ========= #
# cv_score = cross_val_score(svm_model, X_train_vec, y_train, cv=5, scoring="f1_macro")
# print(f"✅ SVM 5-Fold CV F1 Score: {cv_score.mean():.4f}")

# ========= ROC Curve For Multi-Class ========= #
colors = ['darkorange', 'green', 'blue']
labels = ['bad', 'neutral', 'good']

# ROC for nb
nb_probs = nb_model.predict_proba(X_test_vec)

# Compute ROC and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], nb_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 5))
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title("ROC Curves for Naive Bayes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ROC for SVM
y_score_svm = svm_model.decision_function(X_test_vec)  # shape: [n_samples, n_classes]

# Compute ROC and AUC
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_svm[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 5))
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title("ROC Curves for Linear SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# obviously print the accuracy
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_preds):.4f}")
print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds):.4f}")


# PR curve
# from sklearn.metrics import precision_recall_curve, average_precision_score

# precision, recall, pr_auc = {}, {}, {}
# for i in range(3):
#     precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], nb_probs[:, i])
#     pr_auc[i] = average_precision_score(y_test_bin[:, i], nb_probs[:, i])
