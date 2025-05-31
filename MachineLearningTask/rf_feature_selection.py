import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)

df_train = pd.read_csv("train_set.csv")
df_test = pd.read_csv("test_set.csv")

X_train_raw = df_train.drop(columns=['ID', 'CLASS']).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
y_train = df_train['CLASS']
X_test_raw = df_test.drop(columns=['ID', 'CLASS']).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
y_test = df_test['CLASS']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns)


# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train_scaled_df, y_train) 

# feature importances
importances = model.feature_importances_

feature_importances = pd.Series(importances, index=X_train_scaled_df.columns).sort_values(ascending=False)

print("\nRandom Forest Feature Importances:")
print(feature_importances)

selected_features = feature_importances[:51].keys()
print(selected_features)

# writing them to .txt
# with open('random_forest_features.txt', 'w') as f:
#     f.write(','.join(str(feature) for feature in selected_features))