import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

print("=" * 50)
print("NeuroVeil - Optuna Optimize Edilmis SVM Pipeline")
print("=" * 50)

import os
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

X = df[feature_cols].copy()
imputer = SimpleImputer(strategy='mean')
X = X.select_dtypes(include=[np.number])
feature_cols_before = X.columns.tolist()

X_imputed = imputer.fit_transform(X)
feature_cols = imputer.get_feature_names_out(feature_cols_before)
X = pd.DataFrame(X_imputed, columns=feature_cols)
y = df['target'].copy()

# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- RobustScaler ile Normalizasyon ---")
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print("\n--- Optuna Parametreleriyle Feature Selection (LASSO) ---")
selector_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.20019, random_state=42)
selector_model.fit(X_train_scaled, y_train)

selector = SelectFromModel(selector_model, prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_feature_names = X.columns[selector.get_support()]
print(f"Secilen KESIN ozellik sayisi: {X_train_selected.shape[1]}")

print("\n--- SMOTE (k_neighbors=8) ile Veri Dengeleniyor ---")
smote = SMOTE(k_neighbors=8, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)

# 3. Final Modeli: Linear SVM
final_model = SVC(kernel='linear', C=0.33597, random_state=42, probability=True)
final_model.fit(X_train_smote, y_train_smote)

# 4. Test ve Skor
y_pred = final_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[SUCCESS] OPTUNA PIPELINE ACCURACY: %{accuracy*100:.2f}")
print("\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred))

print("\n--- En Onemli 5 Ozellik (SVM Agirliklari) ---")
importances = pd.Series(np.abs(final_model.coef_[0]), index=selected_feature_names)
top_features = importances.sort_values(ascending=False).head(5)
print(top_features.to_string())

def predict_patient(patient_features: dict) -> dict:
    row = pd.DataFrame([patient_features])[feature_cols]
    row_scaled = pd.DataFrame(scaler.transform(row), columns=feature_cols)
    row_selected = selector.transform(row_scaled)
    pred_class = final_model.predict(row_selected)[0]
    pred_prob = final_model.predict_proba(row_selected)[0][1] * 100 
    
    return {
        'pcc_olasiligi'   : round(pred_prob, 1),
        'tahmin_sinifi'   : "PCC (Long COVID)" if pred_class == 1 else "Saglikli/UPC"
    }

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='flare')
plt.title('SVM En Onemli 5 Ozellik (Mutlak Katsayi)')
plt.xlabel('Agirlik / Onem')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'neuroveil_results.png'), dpi=150, bbox_inches='tight')

import joblib
models_dir = os.path.join(BASE_DIR, 'models')
joblib.dump(scaler, os.path.join(models_dir, 'mri_scaler.pkl'))
joblib.dump(selector, os.path.join(models_dir, 'mri_selector.pkl'))
joblib.dump(final_model, os.path.join(models_dir, 'mri_model.pkl'))
# Sütun isimlerini de kaydedelim
joblib.dump(feature_cols, os.path.join(models_dir, 'mri_features.pkl'))

print("\nGrafik ve Modeller kaydedildi.")
print("Pipeline tamamlandi.")