import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import warnings
import time

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("NeuroVeil - KAN GEN IFADESI PIPELINE (Optuna Optimized)")
print("=" * 60)
start = time.time()

print("\n[1] Veri yukleniyor...")
X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))['target'].values

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"  Hasta: {X.shape[0]}, Gen: {X.shape[1]}")
print(f"  PASC (Long COVID): {sum(y==1)}, Control: {sum(y==0)}")

variances = X.var()
X = X.loc[:, variances > 0.1]
print(f"  Varyans filtresi sonrasi gen: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[2] RobustScaler ile normalizasyon...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[3] LASSO (C=4.985814683721306) ile ozellik secimi...")
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=4.985814683721306, random_state=42, max_iter=1000)
lasso.fit(X_train_scaled, y_train)

selector = SelectFromModel(lasso, prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_genes = X.columns[selector.get_support()]
print(f"  Secilen kritik gen sayisi: {len(selected_genes)}")

print("\n[4] SMOTE (k=10) ile veri dengeleniyor...")
smote = SMOTE(k_neighbors=10, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)
print(f"  SMOTE sonrasi egitim sayisi: {len(y_train_smote)}")

print("\n[5] SVM Linear (C=4.7066432208096876) modeli egitiliyor...")
model = SVC(kernel='linear', C=4.7066432208096876, random_state=42, probability=True)
model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"[SUCCESS] KAN MODULU ACCURACY (Test): %{acc*100:.2f}")
print(f"{'='*60}")
print(f"\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Control', 'PASC (Long COVID)']))

print(f"\n--- En Onemli 10 Gen (SVM Agirliklari) ---")
importances = pd.Series(np.abs(model.coef_[0]), index=selected_genes)
top_genes = importances.sort_values(ascending=False).head(10)
print(top_genes.to_string())

# 5-Fold CV
print(f"\n--- 5-Fold Cross-Validation ---")
from imblearn.pipeline import Pipeline
cv_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=4.986, max_iter=1000))),
    ('smote', SMOTE(k_neighbors=10, random_state=42)),
    ('model', SVC(kernel='linear', C=4.707, random_state=42))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"  CV Accuracy: %{cv_scores.mean()*100:.2f} (+/- {cv_scores.std()*100:.2f})")

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genes.values[:5], y=top_genes.index[:5], palette='magma')
plt.title('Kan Modulu - En Onemli 5 Gen (Optuna Optimized SVM)')
plt.xlabel('Agirlik / Onem')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'blood_module_results.png'), dpi=150, bbox_inches='tight')

import joblib
models_dir = os.path.join(BASE_DIR, 'models')
joblib.dump(scaler, os.path.join(models_dir, 'blood_scaler.pkl'))
joblib.dump(selector, os.path.join(models_dir, 'blood_selector.pkl'))
joblib.dump(model, os.path.join(models_dir, 'blood_model.pkl'))
# Seçilen genlerin isimlerini kaydedelim
joblib.dump(X.columns, os.path.join(models_dir, 'blood_features.pkl'))

elapsed = time.time() - start
print(f"\nGrafik ve Modeller kaydedildi. Sure: {elapsed:.1f}s")
print("Kan modulu tamamlandi!")
