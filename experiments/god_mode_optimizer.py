import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
import warnings
import time

warnings.filterwarnings('ignore')

print("=" * 50)
print("BRUTE-FORCE (GOD MODE) OPTIMIZER BASLIYOR")
print("=" * 50)
start_time = time.time()

df = pd.read_excel('Dryad_final.xlsx', sheet_name='Combined')
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
feature_cols_before = X.columns.tolist()
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
feature_cols = imputer.get_feature_names_out(feature_cols_before)
X = pd.DataFrame(X_imputed, columns=feature_cols)
y = df['target'].values

# 1. KORELASYON TEMİZLİĞİ (Birbirini tekrar eden %95 üstü benzer sütunları sil)
print("\nAdim 1: Korelasyon Matrisi Cikariliyor (Birbirinin kopyasi sutunlar silinecek)...")
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)
print(f"  {len(to_drop)} adet cok yuksek korelasyonlu sutun kope atildi.")
print(f"  Kalan Sutun Sayisi: {X.shape[1]}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nAdim 2: RobustScaler ile Normalizasyon...")
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print("\nAdim 3: RFECV (Tek Tek Sütun Kapatma/Deneme) Algoritmasi Calisiyor...")
print("Bu islem binlerce kombinasyon deneyecegi icin uzun surebilir, 4080'in isinmaya baslayabilir :)")

# RFECV (Recursive Feature Elimination with Cross-Validation)
# SVM kullanarak her adimda en onemsiz sutunu atar, skora bakar. Skoru en yuksek yapan sutun kombinasyonunu bulur.
svc = SVC(kernel="linear", C=0.33597)
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
rfecv.fit(X_train_scaled, y_train)

print(f"\n  MUKEMMEL SUTUN KOMBINASYONU BULUNDU!")
print(f"  Optimum Sutun Sayisi: {rfecv.n_features_}")

# Sadece efsanevi sutunlari tut
X_train_selected = rfecv.transform(X_train_scaled)
X_test_selected = rfecv.transform(X_test_scaled)

print("\nAdim 4: SMOTE ile Veri Dengeleniyor...")
smote = SMOTE(k_neighbors=8, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)

print("\nAdim 5: Final Modeli Egitiliyor ve Test Ediliyor...")
final_model = SVC(kernel='linear', C=0.33597, random_state=42, probability=True)
final_model.fit(X_train_smote, y_train_smote)

from sklearn.metrics import accuracy_score
y_pred = final_model.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"[GOD MODE] ACCURACY (TEST SETINDE): %{acc*100:.2f}")
print("=" * 50)
print(f"Gecen Sure: {time.time() - start_time:.1f} saniye")
