import pandas as pd
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv')


print("ИНФОРМАЦИЯ О ДАННЫХ")
print("Первые 5 строк данных \n:", df.head())
print("-"*50)
print("Размер данных \n", df.shape)
print("-"*50)
print("Типы данных \n", df.dtypes)
print("-"*50)
print("Наличие пропусков \n", df.isnull().sum())
print("-"*50)
print("Распределение целевой переменной \n", df['Machine failure'].value_counts())
print("-"*50)

#==================================================================================
print("АНАЛИЗ ДАТАСЕТА")
# Проанализируем столбцы датасета, чтобы понять, с какими следует работать
print("Распределение переменной Product ID \n", df['Product ID'].value_counts())
print("Максимальное количество уникальных значений Product ID: ", max(df['Product ID'].value_counts()))
print("-"*50)
# Product ID не следует использовать, так как каждое значение уникально, пу смыслу близко просто ID,
# никакой полезной нагрузки для обучения не несет
print("Распределение переменной Type \n", df['Type'].value_counts())
# Видим три типа, полезно использовать для обучения
types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
print("Связь TWF, HDF, PWF, OSF, RNF с целевой переменной: \n", (df['Machine failure'] == df[types].any(axis=1)).mean())
# TWF, HDF, PWF, OSF, RNF не следует использовать, так как они связаны с целевой переменной
# (описывают тип неисправности)

#Итог:
# float64 (Air temperature, Process temperature, Rotational speed, Torque, Tool wear)
# object (Type)
# целевая переменная: Machine failure 

#==================================================================================
print("ПРЕДОБРАБОТКА ДАННЫХ")

target_col = 'Machine failure'
numeric_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]
categorical_cols = ['Type']

y = df[target_col]
X = df[numeric_cols + categorical_cols].copy()

# Type содержит всего три типа, следовательно проще всего закодировать ONe hot encoding
X = pd.get_dummies(X, columns=categorical_cols)
print("Преддобработанные признаки: \n", X.head())


#==================================================================================
print("ОБУЧЕНИЕ МОДЕЛЕЙ")

#RandomForestClassifier
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
result_rf = rf.predict_proba(X_val)[:, 1]

print("Random Forest")
print(f"  Precision: {precision_score(y_val, result_rf >= 0.05):.4f}")
print(f"  Recall:    {recall_score(y_val, result_rf >= 0.05):.4f}")
print(f"  F1:        {f1_score(y_val, result_rf >= 0.05):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_val, result_rf):.4f}")

rf_table = pd.DataFrame({
    'sample_id': df.loc[X_val.index, 'UDI'].values,
    'actual': y_val.values,
    'probability': result_rf,
    'alert': result_rf >= 0.05,
})
print("Таблица validation - Random Forest:")
print(rf_table.head(10))

# CatBoost
catboost = CatBoostClassifier(iterations=100, random_state=42, verbose=0, auto_class_weights='Balanced')
catboost.fit(X_train, y_train)
result_cb = catboost.predict_proba(X_val)[:, 1]

print("CatBoost")
print(f"  Precision: {precision_score(y_val, result_cb >= 0.05):.4f}")
print(f"  Recall:    {recall_score(y_val, result_cb >= 0.05):.4f}")
print(f"  F1:        {f1_score(y_val, result_cb >= 0.05):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_val, result_cb):.4f}")

cb_table = pd.DataFrame({
    'sample_id': df.loc[X_val.index, 'UDI'].values,
    'actual': y_val.values,
    'probability': result_cb,
    'alert': result_cb >= 0.05,
})
print("Таблица validation - CatBoost:")
print(cb_table.head(10))

# В итоге Recall 0.94 Catboost, RandomForest 0.89
# Recall в данной задаче -метрика, на которую стоит ориентироваться, так как
# важнее не пропустить отказ машины, чем выявить ложное срабатывание
# следовательно при заданном дисбалансе классов 339-0, 9661-1, модели показывают хорошее качество

