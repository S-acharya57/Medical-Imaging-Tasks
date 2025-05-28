import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df_train = pd.read_csv("train_set.csv")
df_test = pd.read_csv("test_set.csv")

# replacing the inf values first, and NaN values 
X_train_raw = df_train.drop(columns=['ID', 'CLASS']).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
y_train = df_train['CLASS']
X_test_raw = df_test.drop(columns=['ID', 'CLASS']).replace([np.inf, -np.inf], np.nan).replace(np.nan, 0)
y_test = df_test['CLASS']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# L1 (Lasso) Regularization: Adds an L1 penalty to the loss function
# can drive some feature coefficients to exactly zero

# Verify scaling (mean ~ 0, std ~ 1)
print("X_train_scaled mean:", np.mean(X_train_scaled, axis=0).mean())
print("X_train_scaled std:", np.std(X_train_scaled, axis=0).mean())

feature_names = X_train_raw.columns

# Narrowed alpha range to avoid very small values
alphas = np.logspace(-3, 1, 50)  # From 0.001 to 10
lasso = Lasso(max_iter=100000, tol=1e-3, random_state=42)  

# Cross-validation
lasso_scores = []
for alpha in alphas:
    lasso.set_params(alpha=alpha)
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    lasso_scores.append(-scores.mean())

# best alpha
best_alpha_lasso = alphas[np.argmin(lasso_scores)]
print(f"Best Lasso alpha: {best_alpha_lasso}")

lasso = Lasso(alpha=best_alpha_lasso, max_iter=100000, tol=1e-3, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Evaluate
y_pred_lasso = lasso.predict(X_test_scaled)
print(f"Lasso MSE: {mean_squared_error(y_test, y_pred_lasso)}")
print(f"Lasso R²: {r2_score(y_test, y_pred_lasso)}")

# Important features as per lasso
lasso_features = np.where(lasso.coef_ != 0)[0]
selected_feature_names = feature_names[lasso_features].tolist()
print(f"Number of features selected by Lasso: {len(lasso_features)}")
print(f"Selected feature names: {selected_feature_names}")

###### TRYING ELASTIC NET #####
# Combines L1 and L2 penalties
# balancing sparsity (from L1) and coefficient shrinkage (from L2)
# useful when features are correlated.

param_grid = {
    'alpha': np.logspace(-3, 1, 50),
    'l1_ratio': [0.5, 0.7, 0.9, 1.0]
}

# lower tolerance for easier search
elastic_net = ElasticNet(max_iter=100000, tol=1e-2, random_state=42)
grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# best model search 
best_elastic_net = grid_search.best_estimator_
print(f"Best ElasticNet parameters: {grid_search.best_params_}")

y_pred_enet = best_elastic_net.predict(X_test_scaled)
print(f"ElasticNet MSE: {mean_squared_error(y_test, y_pred_enet)}")
print(f"ElasticNet R²: {r2_score(y_test, y_pred_enet)}")

# important features
enet_features = np.where(best_elastic_net.coef_ != 0)[0]
enet_feature_names = feature_names[enet_features].tolist()
print(f"Number of features selected by ElasticNet: {len(enet_features)}")
print(f"Selected feature names: {enet_feature_names}")

X_train_lasso = X_train_scaled[:, lasso_features]
X_test_lasso = X_test_scaled[:, lasso_features]
X_train_enet = X_train_scaled[:, enet_features]
X_test_enet = X_test_scaled[:, enet_features]


log_reg_lasso = LogisticRegression(random_state=42, max_iter=1000)
log_reg_lasso.fit(X_train_lasso, y_train)

y_pred_log_lasso = log_reg_lasso.predict(X_test_lasso)

print(f"Accuracy on Lasso-selected features: {accuracy_score(y_test, y_pred_log_lasso):.4f}")
# gives 65%

print("\nClassification Report (Lasso Features):")
print(classification_report(y_test, y_pred_log_lasso))

log_reg_enet = LogisticRegression(random_state=42, max_iter=1000)
log_reg_enet.fit(X_train_enet, y_train)

y_pred_log_enet = log_reg_enet.predict(X_test_enet)

print(f"Accuracy on ElasticNet-selected features: {accuracy_score(y_test, y_pred_log_enet):.4f}")
# gives 68%

print("\nClassification Report (ElasticNet Features):")
print(classification_report(y_test, y_pred_log_enet))
