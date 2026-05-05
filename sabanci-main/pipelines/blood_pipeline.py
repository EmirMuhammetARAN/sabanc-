import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Anti-Covid - KAN GEN İFADESİ (ŞAMPİYON BASE MODEL)")
print("=" * 60)
start = time.time()

print("\n[1] Veri yukleniyor...")
X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))['target'].values

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"  Hasta: {X.shape[0]}, Toplam Gen: {X.shape[1]}")
print(f"  PASC (Long COVID): {sum(y==1)}, Control: {sum(y==0)}")

# 1. ADIM: TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[2] Şampiyon Parametrelerle Pipeline Kuruluyor...")

# KAZANAN ESKİ (BASE) PARAMETRELER
cv_pipeline = ImbPipeline([
    ('var_thresh', VarianceThreshold(threshold=0.1)),
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.43739, random_state=42, max_iter=1000))),
    ('smote', SMOTE(k_neighbors=6, random_state=42)),
    ('model', GradientBoostingClassifier(learning_rate=0.1364, n_estimators=153, random_state=42))
])

print("\n[3] Model Eğitiliyor ve Final Testi Yapılıyor...")
cv_pipeline.fit(X_train, y_train)

y_pred = cv_pipeline.predict(X_test)
y_pred_proba = cv_pipeline.predict_proba(X_test)[:, 1]

final_acc = accuracy_score(y_test, y_pred)
final_roc = roc_auc_score(y_test, y_pred_proba)

print(f"{'='*60}")
print(f"[FINAL TEST] KAN ŞAMPİYON MODEL ACCURACY: %{final_acc*100:.2f}")
print(f"[FINAL TEST] KAN ŞAMPİYON MODEL ROC_AUC:  %{final_roc*100:.2f}")
print(f"{'='*60}")

print(f"\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Control', 'PASC (Long COVID)']))

# ==========================================
# FEATURE IMPORTANCE (EN ÖNEMLİ 10 GEN)
# ==========================================
print("\n[4] En Önemli 10 Gen Hesaplanıyor ve Grafiğe Dökülüyor...")

var_step = cv_pipeline.named_steps['var_thresh']
selector_step = cv_pipeline.named_steps['selector']
model_step = cv_pipeline.named_steps['model']

genes_after_var = X_train.columns[var_step.get_support()]
genes_after_lasso = genes_after_var[selector_step.get_support()]

importances = pd.Series(model_step.feature_importances_, index=genes_after_lasso)
top_genes = importances.sort_values(ascending=False).head(10)

print("\n--- En Kritik 10 Gen ---")
print(top_genes.to_string())

plt.figure(figsize=(12, 7))
sns.barplot(x=top_genes.values, y=top_genes.index, palette='rocket')
plt.title('Long COVID (PASC) Teşhisinde En Önemli 10 Gen İfadesi', fontsize=14, fontweight='bold')
plt.xlabel('Önem Derecesi (Ağırlık)', fontsize=12)
plt.ylabel('Gen (Feature)', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

results_dir = os.path.join(BASE_DIR, 'results')
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

grafik_yolu = os.path.join(results_dir, 'blood_top10_genes.png')
plt.savefig(grafik_yolu, dpi=150, bbox_inches='tight')

# Nihai Pipeline'ı Kaydet
joblib.dump(cv_pipeline, os.path.join(models_dir, 'blood_champion_pipeline.pkl'))
joblib.dump(genes_after_lasso, os.path.join(models_dir, 'blood_features.pkl'))

elapsed = time.time() - start
print(f"\n✅ Şampiyon Pipeline modeli 'blood_champion_pipeline.pkl' olarak kaydedildi.")
print(f"✅ Özellik önem grafiği kaydedildi: {grafik_yolu}")
print(f"İşlem Tamamlandı. Toplam Süre: {elapsed:.1f}s")