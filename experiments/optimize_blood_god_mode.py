import pandas as pd
import numpy as np
import os
import optuna
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("🩸 KAN MODELI - GOD MODE OPTUNA (ALL CORES) 🚀")
print("="*60)

print("[1] Veri yukleniyor (23.864 Gen)...")
X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))['target'].values
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

def objective(trial):
    # 1. Feature Selection (LASSO C degeri) - Genleri 23.864'ten asagi dusurur
    lasso_c = trial.suggest_float('lasso_c', 0.01, 1.0, log=True)
    
    # 2. SMOTE K-Neighbors (Max 10 olabilir cunku 13 control hastamiz var)
    smote_k = trial.suggest_int('smote_k', 3, 10)
    
    # 3. Model Secimi
    classifier_name = trial.suggest_categorical("classifier_name", ["SVC", "RandomForest", "LogisticRegression", "GradientBoosting"])
    
    if classifier_name == "SVC":
        svm_c = trial.suggest_float('svm_c', 0.05, 10.0, log=True)
        svm_kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
        model = SVC(kernel=svm_kernel, C=svm_c, random_state=42)
    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 3, 20)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        model = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators, random_state=42)
    elif classifier_name == "LogisticRegression":
        lr_c = trial.suggest_float("lr_c", 0.05, 10.0, log=True)
        model = LogisticRegression(C=lr_c, solver='liblinear', random_state=42)
    elif classifier_name == "GradientBoosting":
        gb_lr = trial.suggest_float("gb_lr", 0.01, 0.2, log=True)
        gb_n_estimators = trial.suggest_int("gb_n_estimators", 50, 200)
        model = GradientBoostingClassifier(learning_rate=gb_lr, n_estimators=gb_n_estimators, random_state=42)
    
    try:
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42))),
            ('smote', SMOTE(k_neighbors=smote_k, random_state=42)),
            ('model', model)
        ])
        
        # 5-Factor dedigin 5-Fold Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        return scores.mean()
    except Exception as e:
        return 0.0

print("\n150 farkli genetik kombinasyon deneniyor, lutfen bekleyin...\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150, n_jobs=-1, show_progress_bar=True)

print("\n" + "="*50)
print("[SUCCESS] OPTIMIZATION TAMAMLANDI!")
print("="*50)
print(f"🏆 EN IYI KAN 5-FOLD CV ACCURACY: % {study.best_value*100:.2f}")
print("En Iyi Parametreler:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*50)
