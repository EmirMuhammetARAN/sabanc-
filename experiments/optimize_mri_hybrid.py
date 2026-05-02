import pandas as pd
import numpy as np
import os
import optuna
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) # Sadece önemli mesajlari goster

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')
df['target'] = (df['group'] == 'PCC').astype(int)

# moca dahil tum anketler YASAKLI! (Saf Biyolojik Model Testi)
yasakli_sutunlar = [
    'delay_MRI', 'moca', 'olfaction', 'attention', 'memory', 
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def objective(trial):
    # 1. Feature Selection (LASSO C degeri)
    lasso_c = trial.suggest_float('lasso_c', 0.05, 1.0, log=True)
    
    # 2. SMOTE K-Neighbors
    smote_k = trial.suggest_int('smote_k', 3, 12)
    
    # 3. Model Secimi ve Kendi Icindeki Parametreleri
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
            ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42, max_iter=1000))),
            ('smote', SMOTE(k_neighbors=smote_k, random_state=42)),
            ('model', model)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        return scores.mean()
    except Exception as e:
        return 0.0

print("="*60)
print("🧠 TUM MODELLER DAHIL GOD MODE OPTUNA 🚀")
print("⚡ ISLEMCI: BUTUN CEKIRDEKLER KULLANILIYOR (n_jobs=-1) ⚡")
print("="*60)
print("150 farkli kombinasyon deneniyor, lutfen bekleyin...\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150, n_jobs=-1, show_progress_bar=True)

print("\n" + "="*50)
print("[SUCCESS] OPTIMIZATION TAMAMLANDI!")
print("="*50)
print(f"🏆 EN IYI 5-FOLD CV ACCURACY: % {study.best_value*100:.2f}")
print("En Iyi Parametreler:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*50)
