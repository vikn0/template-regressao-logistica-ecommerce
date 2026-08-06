# ==============================================================================
# TEMPLATE: PREVISÃO DE DEVOLUÇÕES (REGRESSÃO LOGÍSTICA & THRESHOLD TUNING)
# Autor: Felipe Camilo
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÕES DO PROJETO
# ------------------------------------------------------------------------------
DATASET_PATH = 'fashion_returns_master.csv' # Ou o caminho local/URL
TARGET_COL = 'returned'
DROPPED_COLS = ['return_reason', 'order_id']  # Remoção de vazamento de dados
DECISION_THRESHOLD = 0.35                       # Limiar ajustado para maior Recall

# ------------------------------------------------------------------------------
# 2. CARREGAMENTO E TRATAMENTO
# ------------------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)

cols_to_drop = [TARGET_COL] + DROPPED_COLS
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
y = df[TARGET_COL]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ------------------------------------------------------------------------------
# 3. PIPELINE DE M.L.
# ------------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

# ------------------------------------------------------------------------------
# 4. TREINAMENTO
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 5. PREDIÇÃO COM THRESHOLD AJUSTADO
# ------------------------------------------------------------------------------
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred_custom = (y_proba >= DECISION_THRESHOLD).astype(int)

# ------------------------------------------------------------------------------
# 6. AVALIAÇÃO DE DESEMPENHO
# ------------------------------------------------------------------------------
print("=" * 60)
print(f"RELATÓRIO DE DESEMPENHO (Limiar = {DECISION_THRESHOLD*100:.0f}%)")
print("=" * 60)
print(classification_report(y_test, y_pred_custom))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")

# ------------------------------------------------------------------------------
# 7. EXTRAÇÃO E INTERPRETAÇÃO DOS COEFICIENTES (ODDS RATIO)
# ------------------------------------------------------------------------------
model = pipeline.named_steps['classifier']
prep = pipeline.named_steps['preprocessor']

feature_names = prep.get_feature_names_out()

df_coef = pd.DataFrame({
    'Feature': feature_names,
    'Coeficiente': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0])
}).sort_values(by='Coeficiente', ascending=False)

print("\n" + "=" * 60)
print("TOP 5 FATORES QUE AUMENTAM A CHANCE DE DEVOLUÇÃO")
print("=" * 60)
print(df_coef.head(5).to_string(index=False))

print("\n" + "=" * 60)
print("TOP 5 FATORES QUE REDUZEM A CHANCE DE DEVOLUÇÃO")
print("=" * 60)
print(df_coef.tail(5).to_string(index=False))
