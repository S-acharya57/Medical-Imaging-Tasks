import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn 
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score
)
import os 


df_train = pd.read_csv("train_set.csv")
df_test = pd.read_csv("test_set.csv")
blind  = pd.read_csv("blinded_test_set.csv")

X_train_raw = df_train.drop(columns=['ID', 'CLASS']).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
y_train = df_train['CLASS']
X_test_raw = df_test.drop(columns=['ID', 'CLASS']).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
y_test = df_test['CLASS']

X_blind_raw = blind.drop(columns=["ID"]).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
blind_ids = blind["ID"]


with open('random_forest_features.txt', 'r') as f:
    content = f.read()
    features = content.split(',')


X_train_corr = X_train_raw[features]
X_test_corr = X_test_raw[features]
X_blind_corr = X_blind_raw[features]

# scaling for better approximation of the data
scaler = StandardScaler()
X_train_corr_scaled = scaler.fit_transform(X_train_corr)
X_test_corr_scaled = scaler.transform(X_test_corr)
X_blind_corr_scaled = scaler.transform(X_blind_corr)


def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return acc, auc, recall, specificity, f1


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_corr_scaled, y_train)

prob_train = clf.predict_proba(X_train_corr_scaled)[:, 1]
prob_test = clf.predict_proba(X_test_corr_scaled)[:, 1]
prob_blind = clf.predict_proba(X_blind_corr_scaled)[:, 1]

pred_train = clf.predict(X_train_corr_scaled)
pred_test = clf.predict(X_test_corr_scaled)

results = []

name = "LogisticRegression"
os.makedirs("predictions", exist_ok=True)

# Evaluate
train_metrics = evaluate(y_train, pred_train, prob_train)
test_metrics = evaluate(y_test, pred_test, prob_test)
results.append([name, "Train", *train_metrics])
results.append([name, "Test", *test_metrics])

# Save CSVs
pd.DataFrame({
    "ID": df_train["ID"],
    "Prob_CLASS_1": prob_train,
    "Prob_CLASS_0": 1 - prob_train
}).to_csv(f"predictions/train_preds_{name.replace(' ', '_')}.csv", index=False)

pd.DataFrame({
    "ID": df_test["ID"],
    "Prob_CLASS_1": prob_test,
    "Prob_CLASS_0": 1 - prob_test
}).to_csv(f"predictions/test_preds_{name.replace(' ', '_')}.csv", index=False)

pd.DataFrame({
    "ID": blind_ids,
    "Prob_CLASS_1": prob_blind,
    "Prob_CLASS_0": 1 - prob_blind
}).to_csv(f"predictions/blind_preds_{name.replace(' ', '_')}.csv", index=False)

# final dataframe
results_df = pd.DataFrame(results, columns=["clf", "Dataset", "Accuracy", "AUROC", "Recall", "Specificity", "F1"])
results_df.to_csv("predictions/metrics_summary.csv", index=False)
print("\n=== Summary ===\n", results_df)