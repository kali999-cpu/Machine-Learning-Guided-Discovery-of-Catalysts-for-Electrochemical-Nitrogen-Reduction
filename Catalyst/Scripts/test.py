import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# Load dataset parts from future_prediction_v3.py
with open("c:/Users/rafif/Desktop/New folder (2)/future_prediction_v3.py", "r", encoding="utf-8") as f:
    code = f.read()

# Execute only the dataset building part
exec_code = ""
active = False
for line in code.split("\n"):
    if "FEATURES =" in line:
        active = True
    if "print(" in line and "STEP 2" in line:
        break
    if active and "print(" not in line:
        exec_code += line + "\n"

namespace = {"pd": pd, "np": np}
exec(exec_code, namespace)
X = namespace["X"]
y = namespace["y"]

# Try different scalers
scalers = {
    "Robust": RobustScaler(),
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler(),
    "Quantile": QuantileTransformer(output_distribution='normal', n_quantiles=50)
}

models = {
    "Ridge(1)": Ridge(alpha=1.0),
    "Ridge(0.1)": Ridge(alpha=0.1),
    "Lasso(0.01)": Lasso(alpha=0.01, max_iter=10000),
    "RF(200, None)": RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=42, n_jobs=-1),
    "ET(200, sqrt)": ExtraTreesRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=42, n_jobs=-1),
    "ET(300, 0.6)": ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features=0.6, random_state=42, n_jobs=-1),
    "SVR(rbf, C=10)": SVR(C=10.0, epsilon=0.02, gamma='scale'),
    "SVR(rbf, C=50)": SVR(C=50.0, epsilon=0.01, gamma='scale'),
    "SVR(rbf, C=100)": SVR(C=100.0, epsilon=0.01, gamma='scale'),
    "GBR(100)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
}

loo = LeaveOneOut()

for s_name, scaler in scalers.items():
    print(f"\n--- Scaler: {s_name} ---")
    X_s = scaler.fit_transform(X)
    for name, mdl in models.items():
        preds = np.zeros(len(y))
        for tr_i, te_i in loo.split(X_s):
            m = type(mdl)(**mdl.get_params())
            m.fit(X_s[tr_i], y[tr_i])
            preds[te_i] = m.predict(X_s[te_i])
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        
        # also check 5-fold CV to make sure it's stable
        kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
        m = type(mdl)(**mdl.get_params())
        kf5_r2 = cross_val_score(m, X_s, y, cv=kf5, scoring="r2").mean()
        
        print(f"{name:16} LOO R2: {r2:6.4f}  LOO MAE: {mae:6.4f}  5F R2: {kf5_r2:6.4f}")
