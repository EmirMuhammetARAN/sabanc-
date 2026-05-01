import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import warnings
import time

warnings.filterwarnings('ignore')

print("=" * 60)
print("KORELASYON ESIK DEGERI vs DOGRULUK TABLOSU")
print("=" * 60)

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

X_raw = df[feature_cols].select_dtypes(include=[np.number])
feature_cols_before = X_raw.columns.tolist()
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_raw)
feature_cols = imputer.get_feature_names_out(feature_cols_before)
X_full = pd.DataFrame(X_imputed, columns=feature_cols)
y = df['target'].values

thresholds = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
results = []

for thresh in thresholds:
    start = time.time()
    print(f"\n--- Korelasyon Esigi: {thresh} ---")
    
    X = X_full.copy()
    
    # Korelasyon temizligi
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thresh)]
    X = X.drop(columns=to_drop)
    
    remaining = X.shape[1]
    dropped = len(to_drop)
    print(f"  Silinen: {dropped}, Kalan: {remaining}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # RFECV
    svc = SVC(kernel="linear", C=0.33597)
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train_scaled, y_train)
    
    optimal_n = rfecv.n_features_
    print(f"  RFECV Optimal Sutun: {optimal_n}")
    
    # SMOTE + Final Model
    X_train_sel = rfecv.transform(X_train_scaled)
    X_test_sel = rfecv.transform(X_test_scaled)
    
    smote = SMOTE(k_neighbors=8, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)
    
    final_model = SVC(kernel='linear', C=0.33597, random_state=42, probability=True)
    final_model.fit(X_train_smote, y_train_smote)
    
    y_pred = final_model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    
    elapsed = time.time() - start
    print(f"  Accuracy: %{acc*100:.2f} ({elapsed:.1f}s)")
    
    results.append({
        'Korelasyon Esigi': thresh,
        'Silinen Sutun': dropped,
        'Kalan Sutun': remaining,
        'RFECV Secilen': optimal_n,
        'Accuracy (%)': round(acc * 100, 2),
        'Sure (s)': round(elapsed, 1)
    })

print("\n" + "=" * 60)
print("SONUC TABLOSU")
print("=" * 60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("=" * 60)
