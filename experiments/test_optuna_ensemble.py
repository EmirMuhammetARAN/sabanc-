import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
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

# Base Models for Stacking
estimators = [
    ('l1', LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)),
    ('svm', SVC(kernel='linear', C=0.01, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3, learning_rate=0.05))
]

# Stacking Classifier (Meta-model is Logistic Regression)
stacking_model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

# Pipeline with Scaling, Feature Selection, SMOTE, and Stacking
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1))),
    ('smote', SMOTE(random_state=42)),
    ('stack', stacking_model)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = cross_val_score(final_pipeline, X, y, cv=cv, scoring='accuracy').mean()

print(f"Stacking Ensemble Accuracy: {acc*100:.2f}%")
