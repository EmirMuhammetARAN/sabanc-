import pandas as pd
import numpy as np
import os
import warnings
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("=" * 60)
print("Anti-Covid - GOD MODE XGBoost 5-Fold CV Kontrolu")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')
df['target'] = (df['group'] == 'PCC').astype(int)

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

# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Pipeline Kuruluyor (Yeni XGBoost Parametreleriyle) ---")

# Optuna'nın bulduğu en iyi parametreler
best_lasso_c = 0.7255955640005789
best_smote_k = 4
best_xgb_lr = 0.015089610655383503
best_xgb_n = 267
best_xgb_depth = 7

selector_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_lasso_c, random_state=42)
final_clf = XGBClassifier(
    learning_rate=best_xgb_lr, 
    n_estimators=best_xgb_n, 
    max_depth=best_xgb_depth, 
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)

pipeline = ImbPipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(selector_model)),
    ('smote', SMOTE(k_neighbors=best_smote_k, random_state=42)),
    ('model', final_clf)
])

print("\n--- 5-Fold Cross Validation Basliyor ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV Accuracy ve ROC_AUC Skorlari
cv_scores_acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
cv_scores_roc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"Fold Accuracy Skorlari: {np.round(cv_scores_acc * 100, 2)}")
print(f"[CV SUCCESS] Ortalama Yeni CV Accuracy: %{cv_scores_acc.mean() * 100:.2f} (+/- %{cv_scores_acc.std() * 100:.2f})")
print(f"[CV SUCCESS] Ortalama Yeni CV ROC_AUC:  %{cv_scores_roc.mean() * 100:.2f}")

print("\n--- Nihai Model Egitimi ve Test Verisi Kontrolu ---")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

final_accuracy = accuracy_score(y_test, y_pred)
final_roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n[FINAL TEST] Yeni Model Test Accuracy: %{final_accuracy*100:.2f}")
print(f"[FINAL TEST] Yeni Model Test ROC_AUC:  %{final_roc_auc*100:.2f}")

print("\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred))

# ==========================================
# EKLENEN KISIM: FEATURE IMPORTANCE (ÖNEMLİ ÖZELLİKLER)
# ==========================================
print("\n--- En Cok Bakilan Alanlar (Feature Importances) ---")

# 1. Pipeline'dan seçici (selector) ve modeli çekelim
selector_step = pipeline.named_steps['selector']
model_step = pipeline.named_steps['model']

# 2. LASSO'nun hayatta bıraktığı özellikleri bulalım
selected_mask = selector_step.get_support()
selected_feature_names = X_train.columns[selected_mask]

# 3. XGBoost'un bu özelliklere verdiği ağırlıkları alalım
importances = pd.Series(model_step.feature_importances_, index=selected_feature_names)

# 4. Büyükten küçüğe sıralayıp en önemli ilk 10 tanesini alalım
top_features = importances.sort_values(ascending=False).head(10)

print(top_features.to_string())

# 5. Görselleştirme (Grafik Çizimi)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('XGBoost - En Onemli 10 MR Ozelligi', fontsize=14)
plt.xlabel('Onem Derecesi (Agirlik)', fontsize=12)
plt.ylabel('Ozellik (Feature)', fontsize=12)
plt.tight_layout()

# 6. Grafiği kaydetme
results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)
grafik_yolu = os.path.join(results_dir, 'xgboost_feature_importances.png')
plt.savefig(grafik_yolu, dpi=150, bbox_inches='tight')

print(f"\n[SUCCESS] Ozellik onem grafigi basariyla kaydedildi.")

# 7. Modeli ve Özellik Listesini Kaydetme
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(pipeline, os.path.join(models_dir, 'mri_champion_pipeline.pkl'))
joblib.dump(selected_feature_names, os.path.join(models_dir, 'mri_features.pkl'))

print(f"[SUCCESS] MRI Pipeline modeli ve ozellikleri kaydedildi.")
print("\nYeni model degerlendirmesi tamamlandi. Sonuclari base modelle karsilastirabilirsin!")
