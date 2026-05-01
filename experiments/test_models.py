import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

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
X = SimpleImputer(strategy='mean').fit_transform(X)
y = df['target']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogReg (L1 - Lasso)': Pipeline([
        ('s', StandardScaler()), 
        ('m', LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42))
    ]),
    'LogReg (L2 - Ridge)': Pipeline([
        ('s', StandardScaler()), 
        ('m', LogisticRegression(penalty='l2', C=0.01, random_state=42))
    ]),
    'SVM (RBF)': Pipeline([
        ('s', StandardScaler()), 
        ('m', SVC(kernel='rbf', C=1.0, random_state=42))
    ]),
    'SVM (Linear)': Pipeline([
        ('s', StandardScaler()), 
        ('m', SVC(kernel='linear', C=0.01, random_state=42))
    ]),
    'PCA + SVM': Pipeline([
        ('s', StandardScaler()), 
        ('pca', PCA(n_components=20)),
        ('m', SVC(kernel='rbf', C=1.0, random_state=42))
    ]),
    'Lasso Feature Selection + LogReg': Pipeline([
        ('s', StandardScaler()),
        ('sel', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1))),
        ('m', LogisticRegression(random_state=42))
    ])
}

print('--- 5-Fold Cross-Validation Accuracies ---')
for name, model in models.items():
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    print(f'{name}: {acc*100:.2f}%')
