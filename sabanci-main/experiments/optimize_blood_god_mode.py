import pandas as pd
import numpy as np
import os
import optuna
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost Kontrolu
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("🩸 KAN MODELI - ULTIMATE GOD MODE (ALL CORES) 🚀")
print("="*60)

print("[1] Veri yukleniyor (23.864 Gen)...")
X_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
y_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))['target'].values
X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. TEST VERİSİNİ KASAYA KİLİTLİYORUZ (%20)
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

print(f"Eğitim Verisi: {X_train.shape} | Test Verisi: {X_test.shape}")
print("150 farkli genetik kombinasyon deneniyor, lutfen bekleyin...\n")

def objective(trial):
    # 1. Feature Selection (LASSO C degeri)
    lasso_c = trial.suggest_float('lasso_c', 0.01, 1.0, log=True)
    
    # 2. SMOTE K-Neighbors (13 hasta -> 5-fold = ~10 train hasta. Güvenlik için Max 7!)
    smote_k = trial.suggest_int('smote_k', 3, 7)
    
    # 3. Model Secimi
    models = ["SVC", "RandomForest", "LogisticRegression", "GradientBoosting"]
    if HAS_XGB: models.append("XGBoost")
    
    classifier_name = trial.suggest_categorical("classifier_name", models)
    
    if classifier_name == "SVC":
        svm_c = trial.suggest_float('svm_c', 0.05, 10.0, log=True)
        svm_kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
        model = SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=42)
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
    elif classifier_name == "XGBoost":
        xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
        xgb_n = trial.suggest_int("xgb_n", 50, 200)
        xgb_depth = trial.suggest_int("xgb_depth", 3, 10)
        model = XGBClassifier(learning_rate=xgb_lr, n_estimators=xgb_n, max_depth=xgb_depth, eval_metric='logloss', random_state=42)
    
    try:
        # Hızlı eleme: Eğer Lasso hiç özellik bırakmazsa denemeyi iptal et
        test_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42))
        X_sc_test = RobustScaler().fit_transform(X_train)
        if test_selector.fit_transform(X_sc_test, y_train).shape[1] == 0:
            raise optuna.TrialPruned()

        pipeline = Pipeline([
            # 1. Adım: Ölü genleri at
            ('var_thresh', VarianceThreshold(threshold=0.01)),
            # 2. Adım: HIZLANDIRICI! En iyi 1000 geni istatistiksel olarak seç (Lasso'nun yükünü hafifletir)
            ('kbest', SelectKBest(score_func=f_classif, k=1000)),
            # 3. Adım: Ölçeklendir
            ('scaler', RobustScaler()),
            # 4. Adım: Lasso ile o 1000 gen içinden en kritik olanları nokta atışı seç
            ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42))),
            ('smote', SMOTE(k_neighbors=smote_k, random_state=42)),
            ('model', model)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Accuracy yerine medikal standart ROC_AUC
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        return scores.mean()
    
    except optuna.TrialPruned:
        raise
    except Exception as e:
        return 0.0

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150, n_jobs=1, show_progress_bar=True)

print("\n" + "="*50)
print("[SUCCESS] OPTIMIZATION TAMAMLANDI!")
print("="*50)
print(f"🏆 EN IYI CV ROC_AUC SKORU: % {study.best_value*100:.2f}")
print("En Iyi Parametreler:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*50)

# --- FINAL TEST KISMI ---
print("\n[3] Kazanan Parametrelerle Nihai Model Egitiliyor ve Test Ediliyor...")
best = study.best_params
clf_name = best["classifier_name"]

if clf_name == "SVC":
    final_clf = SVC(kernel=best['svm_kernel'], C=best['svm_c'], probability=True, random_state=42)
elif clf_name == "RandomForest":
    final_clf = RandomForestClassifier(max_depth=best['rf_max_depth'], n_estimators=best['rf_n_estimators'], random_state=42)
elif clf_name == "LogisticRegression":
    final_clf = LogisticRegression(C=best['lr_c'], solver='liblinear', random_state=42)
elif clf_name == "GradientBoosting":
    final_clf = GradientBoostingClassifier(learning_rate=best['gb_lr'], n_estimators=best['gb_n_estimators'], random_state=42)
elif clf_name == "XGBoost":
    final_clf = XGBClassifier(learning_rate=best['xgb_lr'], n_estimators=best['xgb_n'], max_depth=best['xgb_depth'], eval_metric='logloss', random_state=42)

final_pipeline = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.01)),
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=best['lasso_c'], random_state=42))),
    ('smote', SMOTE(k_neighbors=best['smote_k'], random_state=42)),
    ('model', final_clf)
])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)

try:
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    final_roc_auc = roc_auc_score(y_test, y_pred_proba)
except:
    final_roc_auc = "Hesaplanamadı"

print(f"\n[🚀 FINAL TEST SUCCESS] Model: {clf_name}")
print(f"Test Accuracy: %{accuracy_score(y_test, y_pred)*100:.2f}")
if isinstance(final_roc_auc, float):
    print(f"Test ROC AUC:  %{final_roc_auc*100:.2f}")

print("\nFinal Siniflandirma Raporu:")
print(classification_report(y_test, y_pred))