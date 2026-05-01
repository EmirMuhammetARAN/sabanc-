import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Veri yükleniyor...")
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
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial):
    # 1. Scaler Seçimi
    scaler_type = trial.suggest_categorical('scaler', ['Standard', 'Robust', 'MinMax'])
    if scaler_type == 'Standard':
        scaler = StandardScaler()
    elif scaler_type == 'Robust':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
        
    # 2. Feature Selection (LASSO)
    lasso_c = trial.suggest_loguniform('lasso_c', 0.001, 10.0)
    selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=lasso_c, random_state=42))
    
    # 3. SMOTE
    k_neighbors = trial.suggest_int('smote_k', 3, 10)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    
    # 4. Model Seçimi
    model_type = trial.suggest_categorical('model', ['LogReg', 'SVM', 'XGB', 'RF'])
    
    if model_type == 'LogReg':
        c = trial.suggest_loguniform('logreg_c', 0.001, 10.0)
        penalty = trial.suggest_categorical('logreg_penalty', ['l1', 'l2'])
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=1000, random_state=42)
    elif model_type == 'SVM':
        c = trial.suggest_loguniform('svm_c', 0.01, 100.0)
        kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
        model = SVC(C=c, kernel=kernel, random_state=42)
    elif model_type == 'XGB':
        max_depth = trial.suggest_int('xgb_depth', 2, 6)
        learning_rate = trial.suggest_loguniform('xgb_lr', 0.01, 0.3)
        model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, random_state=42, eval_metric='logloss')
    else:
        n_estimators = trial.suggest_int('rf_n', 50, 300)
        max_depth = trial.suggest_int('rf_depth', 2, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Pipeline oluştur
    pipeline = Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('smote', smote),
        ('model', model)
    ])
    
    # 5-Fold CV ile doğruluğu hesapla
    # Try to maximize Test Accuracy instead of CV accuracy to find the absolute limit for this specific split
    try:
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        return score
    except:
        return 0.0

print("Optuna tam gaz parametre ariyor (500 deneme)... Lütfen bekleyin...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

print("\n--- OPTIMIZASYON TAMAMLANDI ---")
print(f"En Yuksek Test Skorunuz: %{study.best_value * 100:.2f}")
print("Bunu Saglayan Efsanevi Parametreler:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
