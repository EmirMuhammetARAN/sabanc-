import pandas as pd
import numpy as np
import os
import optuna
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) # Sadece önemli mesajlari goster

# XGBoost Kontrolü
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

print("="*60)
print("🚀 ULTIMATE GOD MODE OPTUNA - NEUROVEIL MRI + MoCA 🧠")
print("="*60)

# --- 1. VERI YUKLEME VE ON ISLEME ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')
df['target'] = (df['group'] == 'PCC').astype(int)

# Saf Biyolojik Model + MoCA Bilisel Skoru
yasakli_sutunlar = [
    'delay_MRI', 'olfaction', 'attention', 'memory', 
    'multitasking', 'word_finding_difficulties', 'fatigue', 
    'gds', 'weimus_p', 'weimus_m', 'weimus_g', 'severity', 
    'disability', 'comorbidities', 'olf_imp_ini', 'olf_imo_late'
]

exclude_cols = ['group', 'target', 'subject_id', 'ID', 'id', 'record_id'] + yasakli_sutunlar
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].copy()
imputer = SimpleImputer(strategy='mean')
X = X.select_dtypes(include=[np.number])
feature_cols_before = X.columns.tolist()

X_imputed = imputer.fit_transform(X)
feature_cols = imputer.get_feature_names_out(feature_cols_before)
X = pd.DataFrame(X_imputed, columns=feature_cols)
y = df['target'].copy()

# --- 2. TRAIN / TEST SPLIT (Optuna sadece Train görecek!) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim Verisi Boyutu: {X_train.shape}")
print(f"Test Verisi Boyutu: {X_test.shape}\n")

# --- 3. OPTUNA OBJECTIVE FONKSIYONU ---
def objective(trial):
    # 1. Feature Selection (LASSO C degeri)
    lasso_c = trial.suggest_float('lasso_c', 0.1, 1.0)
    
    # 2. SMOTE K-Neighbors
    smote_k = trial.suggest_int('smote_k', 3, 10)
    
    # 3. Model Secimi
    models = ["SVC", "RandomForest", "LogisticRegression", "GradientBoosting", "AdaBoost"]
    if HAS_XGB: models.append("XGBoost")
    
    classifier_name = trial.suggest_categorical("classifier_name", models)
    
    if classifier_name == "SVC":
        svm_c = trial.suggest_float('svm_c', 0.01, 20.0, log=True)
        svm_gamma = trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
        model = SVC(kernel='rbf', C=svm_c, gamma=svm_gamma, probability=True, random_state=42)
        
    elif classifier_name == "RandomForest":
        rf_depth = trial.suggest_int("rf_depth", 2, 32)
        rf_n = trial.suggest_int("rf_n", 100, 500)
        model = RandomForestClassifier(max_depth=rf_depth, n_estimators=rf_n, random_state=42)
        
    elif classifier_name == "LogisticRegression":
        lr_c = trial.suggest_float("lr_c", 0.001, 10.0, log=True)
        model = LogisticRegression(C=lr_c, solver='liblinear', random_state=42, max_iter=1000)
        
    elif classifier_name == "GradientBoosting":
        gb_lr = trial.suggest_float("gb_lr", 0.005, 0.3, log=True)
        gb_n = trial.suggest_int("gb_n", 50, 300)
        gb_depth = trial.suggest_int("gb_depth", 3, 10)
        model = GradientBoostingClassifier(learning_rate=gb_lr, n_estimators=gb_n, max_depth=gb_depth, random_state=42)
    
    elif classifier_name == "AdaBoost":
        ada_lr = trial.suggest_float("ada_lr", 0.01, 1.0)
        ada_n = trial.suggest_int("ada_n", 50, 200)
        model = AdaBoostClassifier(learning_rate=ada_lr, n_estimators=ada_n, random_state=42)

    elif classifier_name == "XGBoost":
        xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
        xgb_n = trial.suggest_int("xgb_n", 50, 300)
        xgb_depth = trial.suggest_int("xgb_depth", 3, 10)
        model = XGBClassifier(learning_rate=xgb_lr, n_estimators=xgb_n, max_depth=xgb_depth, use_label_encoder=False, eval_metric='logloss', random_state=42)

    try:
        # Sifir ozellik kalma durumunu hızlıca test et
        scaler_test = RobustScaler()
        X_sc_test = scaler_test.fit_transform(X_train)
        selector_test = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42))
        X_sel_test = selector_test.fit_transform(X_sc_test, y_train)
        
        if X_sel_test.shape[1] == 0: 
            raise optuna.TrialPruned() # Özellik kalmadıysa bu denemeyi es geç

        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42))),
            ('smote', SMOTE(k_neighbors=smote_k, random_state=42)),
            ('model', model)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Accuracy yerine medikal veriye çok daha uygun olan ROC AUC kullanıyoruz
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        return scores.mean()
    except optuna.TrialPruned:
        raise
    except Exception as e:
        return 0.0

# --- 4. OPTIMIZASYONU BASLAT ---
study = optuna.create_study(direction='maximize')
# 200 deneme başlar, n_jobs=-1 ile tüm işlemci çekirdeklerini kullanır
study.optimize(objective, n_trials=200, n_jobs=-1, show_progress_bar=True)

print("\n" + "="*50)
print("🏆 EN IYI SONUC BULUNDU!")
print("="*50)
print(f"En İyi CV Skoru (ROC_AUC): % {study.best_value*100:.2f}")
print("\nEn Iyi Parametreler:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*50)


# --- 5. EN IYI MODELIN TEKRAR KURULMASI VE FINAL TESTI ---
print("\n--- Kazanan Parametrelerle Nihai Model Egitimi Basliyor ---")
best = study.best_params
clf_name = best["classifier_name"]

# Kazanan Modeli Sec
if clf_name == "SVC":
    final_clf = SVC(kernel='rbf', C=best['svm_c'], gamma=best['svm_gamma'], probability=True, random_state=42)
elif clf_name == "RandomForest":
    final_clf = RandomForestClassifier(max_depth=best['rf_depth'], n_estimators=best['rf_n'], random_state=42)
elif clf_name == "LogisticRegression":
    final_clf = LogisticRegression(C=best['lr_c'], solver='liblinear', random_state=42, max_iter=1000)
elif clf_name == "GradientBoosting":
    final_clf = GradientBoostingClassifier(learning_rate=best['gb_lr'], n_estimators=best['gb_n'], max_depth=best['gb_depth'], random_state=42)
elif clf_name == "AdaBoost":
    final_clf = AdaBoostClassifier(learning_rate=best['ada_lr'], n_estimators=best['ada_n'], random_state=42)
elif clf_name == "XGBoost":
    final_clf = XGBClassifier(learning_rate=best['xgb_lr'], n_estimators=best['xgb_n'], max_depth=best['xgb_depth'], use_label_encoder=False, eval_metric='logloss', random_state=42)

# Nihai Pipeline'ı Kur
final_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=best['lasso_c'], random_state=42))),
    ('smote', SMOTE(k_neighbors=best['smote_k'], random_state=42)),
    ('model', final_clf)
])

# Tüm X_train ile Eğit
final_pipeline.fit(X_train, y_train)

# Optuna'nın HİÇ GÖRMEDİĞİ X_test verisi üzerinde test et
y_pred = final_pipeline.predict(X_test)

# ROC-AUC için probability hesapla (SVC probability=True açık, hepsi destekler)
try:
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    final_roc_auc = roc_auc_score(y_test, y_pred_proba)
except:
    final_roc_auc = "Hesaplanamadı"

final_accuracy = accuracy_score(y_test, y_pred)

print(f"\n[🚀 FINAL TEST SUCCESS] Model: {clf_name}")
print(f"Test Accuracy: %{final_accuracy*100:.2f}")
if isinstance(final_roc_auc, float):
    print(f"Test ROC AUC:  %{final_roc_auc*100:.2f}")

print("\nFinal Siniflandirma Raporu:")
print(classification_report(y_test, y_pred))

# --- 6. MODELI KAYDETME ---
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(final_pipeline, os.path.join(models_dir, 'mri_god_mode_pipeline.pkl'))
joblib.dump(feature_cols, os.path.join(models_dir, 'mri_features.pkl'))
print(f"\n✅ Pipeline '{clf_name}' modeliyle 'mri_god_mode_pipeline.pkl' olarak basariyla kaydedildi.")