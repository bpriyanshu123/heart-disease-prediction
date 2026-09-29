import os, json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'model')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
DATA_PATH = os.path.join(DATA_DIR, 'heart_disease_cleveland.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_model.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']


def load_data():
    if not os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_URL, names=COLUMNS, na_values='?')
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def train():
    df = load_data()
    df['target'] = (df['target'] > 0).astype(int)

    X = df.drop(columns='target')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric = ['age','trestbps','chol','thalach','oldpeak']
    categorical = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer([
        ('numeric', numeric_pipe, numeric),
        ('categorical', categorical_pipe, categorical)
    ])

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'dataset': 'UCI Heart Disease - Cleveland',
        'total_records': int(len(df)),
        'features': 13,
        'test_records': int(len(X_test)),
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'precision': round(float(precision_score(y_test, pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, prob)), 4),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist()
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('\nMODEL TRAINING COMPLETE')
    print(f"Dataset records : {metrics['total_records']}")
    print(f"Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"Precision       : {metrics['precision']*100:.2f}%")
    print(f"Recall          : {metrics['recall']*100:.2f}%")
    print(f"F1 Score        : {metrics['f1']*100:.2f}%")
    print(f"ROC-AUC         : {metrics['roc_auc']:.3f}")
    print(f"Saved model     : {MODEL_PATH}")
    return pipeline, metrics

if __name__ == '__main__':
    train()
