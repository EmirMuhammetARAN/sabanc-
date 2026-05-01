import gzip
import pandas as pd
import numpy as np

print("=" * 60)
print("GSE226260 - KOHORT 1 + KOHORT 2 BIRLESTIRME")
print("=" * 60)

# ===== KOHORT 1 =====
print("\n[KOHORT 1] Gen ifadesi okunuyor...")
expr1 = pd.read_csv('GSE226260_combined.csv.gz', compression='gzip', index_col=0, low_memory=False)
print(f"  Shape: {expr1.shape}")

# SOFT'tan label cek
samples = {}
cur = None
with gzip.open('GSE226260_family.soft.gz', 'rt', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if line.startswith('^SAMPLE'):
            cur = line.split('=')[1].strip()
            samples[cur] = {}
        elif cur and 'characteristics_ch1' in line:
            val = line.split('=', 1)[1].strip()
            if ':' in val:
                key, value = val.split(':', 1)
                samples[cur][key.strip()] = value.strip()

meta = pd.DataFrame.from_dict(samples, orient='index')

# GCID -> Label mapping
gcid_to_label = {}
for gsm, row in meta.iterrows():
    gcid = str(row.get('gcid', '')).strip()
    pasc = str(row.get('pasc status', '')).strip().upper()
    if gcid and pasc in ['PASC', 'NOPASC']:
        gcid_to_label[gcid] = 1 if pasc == 'PASC' else 0

# Kohort 1 eslestir
k1_cols = []
k1_labels = []
for col in expr1.columns:
    gcid = col.split('_')[0]
    if gcid in gcid_to_label:
        k1_cols.append(col)
        k1_labels.append(gcid_to_label[gcid])

X1 = expr1[k1_cols].T.reset_index(drop=True)
y1 = np.array(k1_labels)
print(f"  Eslesen: {len(k1_cols)} (PASC={sum(y1==1)}, NOPASC={sum(y1==0)})")

# ===== KOHORT 2 =====
print("\n[KOHORT 2] Gen ifadesi okunuyor...")
expr2 = pd.read_csv('GSE226260_additional.csv.gz', compression='gzip', index_col=0, low_memory=False)
print(f"  Shape: {expr2.shape}")

# Label direkt sutun isminden
k2_cols_pasc = [c for c in expr2.columns if 'PASC' in c]
k2_cols_recov = [c for c in expr2.columns if 'Recovered' in c]

X2_pasc = expr2[k2_cols_pasc].T.reset_index(drop=True)
X2_recov = expr2[k2_cols_recov].T.reset_index(drop=True)
X2 = pd.concat([X2_pasc, X2_recov], axis=0).reset_index(drop=True)
y2 = np.array([1]*len(k2_cols_pasc) + [0]*len(k2_cols_recov))
print(f"  PASC: {len(k2_cols_pasc)}, Recovered: {len(k2_cols_recov)}")

# ===== BIRLESTIR =====
print("\n[BIRLESTIRME]")
# Ortak genleri bul
common_genes = expr1.index.intersection(expr2.index)
print(f"  Ortak gen sayisi: {len(common_genes)}")

# Ortak genlerle tekrar olustur
X1_common = expr1.loc[common_genes, k1_cols].T.reset_index(drop=True)
X2_common = expr2.loc[common_genes, k2_cols_pasc + k2_cols_recov].T.reset_index(drop=True)

X_all = pd.concat([X1_common, X2_common], axis=0).reset_index(drop=True)
y_all = np.concatenate([y1, y2])

print(f"\n{'='*60}")
print(f"FINAL BIRLESMIS VERI SETI:")
print(f"  Toplam Hasta: {X_all.shape[0]}")
print(f"  Toplam Gen: {X_all.shape[1]}")
print(f"  LONG COVID (PASC): {sum(y_all == 1)}")
print(f"  CONTROL (Recovered/NOPASC): {sum(y_all == 0)}")
print(f"{'='*60}")

X_all.to_csv('blood_X.csv', index=False)
pd.DataFrame({'target': y_all}).to_csv('blood_y.csv', index=False)
print("Kaydedildi: blood_X.csv, blood_y.csv")
