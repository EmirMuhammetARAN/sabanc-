import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')

print(f"Toplam sutun: {len(df.columns)}")
print(f"Toplam hasta: {len(df)}")
print("\n=== ILK 117 SUTUN ===\n")

for i, col in enumerate(df.columns[:118]):
    col_lower = col.lower()
    tag = ""
    if any(k in col_lower for k in ['dti', 'dwi', 'diffusion', 'fa', 'md', 'rd', 'ad', 'adc', 'tract']):
        tag = " <<< DIFUZYON/DTI"
    elif any(k in col_lower for k in ['csf', 'ventricl', 'fluid']):
        tag = " <<< CSF/SIVI"
    elif any(k in col_lower for k in ['wm', 'white']):
        tag = " <<< BEYAZ MADDE"
    elif any(k in col_lower for k in ['gm', 'gray', 'grey']):
        tag = " <<< GRI MADDE"
    elif any(k in col_lower for k in ['vol', 'volume']):
        tag = " <<< HACIM"
    elif any(k in col_lower for k in ['thick']):
        tag = " <<< KALINLIK"
    print(f"  [{i:3d}] {col}{tag}")
