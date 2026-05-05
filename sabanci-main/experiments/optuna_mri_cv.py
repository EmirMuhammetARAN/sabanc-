"""
Anti-Covid - MRI Pipeline Optuna CV Optimizer
Amac: Test accuracy yerine 5-Fold CV accuracy'yi maximize etmek
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')
df['target'] = (df['group'] == 'PCC').astype(int)

yasakli_sutunlar = [
    'delay_MRI', 'moca', 'olfaction', 'attention', 'memory', 
    'multitasking', 'word_finding_difficulties', 'fatigue', 
    'gds', 'weimus_p', 'weimus_m', 'weimus_g', 'severity', 
    'disability', 'comorbidities', 'olf_imp_ini', 'olf_imo_late'
]

exclude_cols = ['group', 'target', 'subject_id', 'ID', 'id', 'record_id'] + yasakli_sutunlar
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].select_dtypes(include=[np.number])
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
y = df['target'].values

print(f"Veri: {X.shape[0]} hasta, {X.shape[1]} ozellik")
print(f"Hedef: PASC={sum(y==1)}, Control={sum(y==0)}")
print("=" * 60)
print("Optuna CV Optimizasyonu basliyor (200 trial)...")
print("=" * 60)

def objective(trial):
    lasso_C = trial.suggest_float('lasso_C', 0.001, 1.0, log=True)
    svm_C = trial.suggest_float('svm_C', 0.001, 5.0, log=True)
    smote_k = trial.suggest_int('smote_k', 3, 15)
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=lasso_C, random_state=42)
        )),
        ('smote', SMOTE(k_neighbors=smote_k, random_state=42)),
        ('model', SVC(kernel='linear', C=svm_C, random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Cok yuksek std varsa cezalandir (stabil sonuc istiyoruz)
        penalized = mean_score - (std_score * 0.3)
        return penalized
    except:
        return 0.0

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("\n" + "=" * 60)
print("OPTUNA SONUCLARI")
print("=" * 60)

best = study.best_params
print(f"\nEn iyi parametreler:")
print(f"  LASSO C:    {best['lasso_C']:.6f}")
print(f"  SVM C:      {best['svm_C']:.6f}")
print(f"  SMOTE k:    {best['smote_k']}")

# En iyi parametrelerle son bir CV calistir
best_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=best['lasso_C'], random_state=42)
    )),
    ('smote', SMOTE(k_neighbors=best['smote_k'], random_state=42)),
    ('model', SVC(kernel='linear', C=best['svm_C'], random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_scores = cross_val_score(best_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\n  CV Accuracy:  %{final_scores.mean()*100:.2f} (+/- {final_scores.std()*100:.2f})")
print(f"  Fold Scores:  {[f'%{s*100:.1f}' for s in final_scores]}")

# Eski degerlerle karsilastir
print("\n--- KARSILASTIRMA ---")
old_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.200, random_state=42)
    )),
    ('smote', SMOTE(k_neighbors=8, random_state=42)),
    ('model', SVC(kernel='linear', C=0.336, random_state=42))
])
old_scores = cross_val_score(old_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"  ESKI CV:  %{old_scores.mean()*100:.2f} (+/- {old_scores.std()*100:.2f})")
print(f"  YENI CV:  %{final_scores.mean()*100:.2f} (+/- {final_scores.std()*100:.2f})")
print(f"  FARK:     +%{(final_scores.mean() - old_scores.mean())*100:.2f}")

# Test accuracy'yi de kontrol et
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=best['lasso_C'], random_state=42)
lasso.fit(X_train_s, y_train)
selector = SelectFromModel(lasso, prefit=True)
X_train_sel = selector.transform(X_train_s)
X_test_sel = selector.transform(X_test_s)

n_selected = X_train_sel.shape[1]
print(f"\n  Secilen ozellik sayisi: {n_selected}")

smote = SMOTE(k_neighbors=best['smote_k'], random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train)

model = SVC(kernel='linear', C=best['svm_C'], random_state=42)
model.fit(X_train_sm, y_train_sm)
y_pred = model.predict(X_test_sel)
test_acc = accuracy_score(y_test, y_pred)

print(f"  Test Accuracy: %{test_acc*100:.2f}")
print(f"\n  ESKI Test: %74.29 | YENI Test: %{test_acc*100:.2f}")
print("\nBitti!")
