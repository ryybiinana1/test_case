# Предсказание отказа оборудования

Решение тестового задания: предсказание отказа оборудования на датасете (https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv).

## Структура проекта

| Файл | Описание |
|------|----------|
| `ml_solution.py` | EDA, предобработка, обучение **RandomForest** и **CatBoost** |
| `requirements.txt` | Зависимости проекта |

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python ml_solution.py
```

## Пример вывода

```
ИНФОРМАЦИЯ О ДАННЫХ
Первые 5 строк данных 
:    UDI Product ID Type  Air temperature [K]  ...  HDF  PWF  OSF  RNF
0    1     M14860    M                298.1  ...    0    0    0    0
1    2     L47181    L                298.2  ...    0    0    0    0
2    3     L47182    L                298.1  ...    0    0    0    0
3    4     L47183    L                298.2  ...    0    0    0    0
4    5     L47184    L                298.2  ...    0    0    0    0

[5 rows x 14 columns]
--------------------------------------------------
Размер данных 
 (10000, 14)
--------------------------------------------------
Типы данных 
 UDI                          int64
Product ID                  object
Type                        object
Air temperature [K]        float64
Process temperature [K]    float64
Rotational speed [rpm]       int64
Torque [Nm]                float64
Tool wear [min]              int64
Machine failure              int64
TWF                          int64
HDF                          int64
PWF                          int64
OSF                          int64
RNF                          int64
dtype: object
--------------------------------------------------
Наличие пропусков 
 UDI                        0
Product ID                 0
Type                       0
Air temperature [K]        0
Process temperature [K]    0
Rotational speed [rpm]     0
Torque [Nm]                0
Tool wear [min]            0
Machine failure            0
TWF                        0
HDF                        0
PWF                        0
OSF                        0
RNF                        0
dtype: int64
--------------------------------------------------
Распределение целевой переменной 
 Machine failure
0    9661
1     339
Name: count, dtype: int64
--------------------------------------------------
АНАЛИЗ ДАТАСЕТА
Распределение переменной Product ID 
 Product ID
M14860    1
L53850    1
...
Name: count, Length: 10000, dtype: int64
Максимальное количество уникальных значений Product ID:  1
--------------------------------------------------
Распределение переменной Type 
 Type
L    6000
M    2997
H    1003
Name: count, dtype: int64
Связь TWF, HDF, PWF, OSF, RNF с целевой переменной: 
 0.9973
ПРЕДОБРАБОТКА ДАННЫХ
Преддобработанные признаки: 
   Air temperature [K]  Process temperature [K]  ...  Type_L  Type_M
0                298.1                    308.6  ...   False    True
1                298.2                    308.7  ...    True   False
2                298.1                    308.5  ...    True   False
3                298.2                    308.6  ...    True   False
4                298.2                    308.7  ...    True   False

[5 rows x 8 columns]
ОБУЧЕНИЕ МОДЕЛЕЙ
Random Forest
  Precision: 0.2837
  Recall:    0.8971
  F1:        0.4311
  ROC-AUC:   0.9635
Таблица validation - Random Forest:
   sample_id  actual  probability  alert
0       2998       0         0.17   True
1       4872       0         0.00  False
2       3859       0         0.07   True
3        952       0         0.02  False
4       6464       0         0.07   True
5       3265       0         0.00  False
6       4509       0         0.00  False
7       2101       0         0.00  False
8       7886       0         0.00  False
9       2422       0         0.08   True
CatBoost
  Precision: 0.1580
  Recall:    0.9412
  F1:        0.2706
  ROC-AUC:   0.9703
Таблица validation - CatBoost:
   sample_id  actual  probability  alert
0       2998       0     0.716556   True
1       4872       0     0.004241  False
2       3859       0     0.235374   True
3        952       0     0.009936  False
4       6464       0     0.440274   True
5       3265       0     0.011787  False
6       4509       0     0.196158   True
7       2101       0     0.016202  False
8       7886       0     0.003486  False
9       2422       0     0.438347   True
```

## Кратко о результатах

- **Целевая переменная:** `Machine failure` (0 — без отказа, 1 — отказ)
- **Разбиение:** train / validation = 80 / 20
- **Порог alert:** probability ≥ 0.05
- **Основная метрика:** Recall — важнее не пропустить отказ при сильном дисбалансе классов
