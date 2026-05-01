import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import optuna
import warnings
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("KAN MODULU - OPTUNA GOD MODE (500 deneme)")
print("=" * 60)
start = time.time()

X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))['target'].values

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Varyans filtresi
variances = X.var()
X = X.loc[:, variances > 0.1]
print(f"Gen sayisi (varyans filtresi sonrasi): {X.shape[1]}")

X_np = X.values
X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.2, random_state=42, stratify=y)
feature_names = X.columns

def objective(trial):
    # Scaler
    scaler_type = trial.suggest_categorical('scaler', ['Standard', 'Robust', 'MinMax'])
    if scaler_type == 'Standard':
        scaler = StandardScaler()
    elif scaler_type == 'Robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()

    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # LASSO Feature Selection
    lasso_c = trial.suggest_float('lasso_c', 0.01, 5.0, log=True)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42, max_iter=1000)
    try:
        lasso.fit(X_tr, y_train)
    except:
        return 0.0

    selector = SelectFromModel(lasso, prefit=True)
    X_tr_sel = selector.transform(X_tr)
    X_te_sel = selector.transform(X_te)

    if X_tr_sel.shape[1] < 2:
        return 0.0

    # SMOTE
    smote_k = trial.suggest_int('smote_k', 3, 10)
    try:
        smote = SMOTE(k_neighbors=smote_k, random_state=42)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_sel, y_train)
    except:
        return 0.0

    # Model
    model_type = trial.suggest_categorical('model', ['LogReg', 'SVM_linear', 'SVM_rbf', 'RF', 'XGB'])

    if model_type == 'LogReg':
        c = trial.suggest_float('lr_c', 0.01, 10.0, log=True)
        pen = trial.suggest_categorical('lr_pen', ['l1', 'l2'])
        solver = 'liblinear' if pen == 'l1' else 'lbfgs'
        model = LogisticRegression(penalty=pen, C=c, solver=solver, max_iter=1000, random_state=42)
    elif model_type == 'SVM_linear':
        c = trial.suggest_float('svm_lin_c', 0.01, 50.0, log=True)
        model = SVC(kernel='linear', C=c, random_state=42)
    elif model_type == 'SVM_rbf':
        c = trial.suggest_float('svm_rbf_c', 0.01, 50.0, log=True)
        model = SVC(kernel='rbf', C=c, random_state=42)
    elif model_type == 'RF':
        n = trial.suggest_int('rf_n', 50, 500)
        d = trial.suggest_int('rf_d', 2, 15)
        model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
    else:
        d = trial.suggest_int('xgb_d', 2, 8)
        lr = trial.suggest_float('xgb_lr', 0.01, 0.3, log=True)
        model = XGBClassifier(max_depth=d, learning_rate=lr, random_state=42, eval_metric='logloss', verbosity=0)

    try:
        model.fit(X_tr_smote, y_tr_smote)
        y_pred = model.predict(X_te_sel)
        return accuracy_score(y_test, y_pred)
    except:
        return 0.0

print("Optuna 500 deneme basliyor... 4080 isiniyor!")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"EN YUKSEK KAN MODULU ACCURACY: %{study.best_value * 100:.2f}")
print(f"{'='*60}")
print(f"Gecen sure: {elapsed:.1f} saniye")
print("\nEfsanevi Parametreler:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
