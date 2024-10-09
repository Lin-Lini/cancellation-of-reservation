import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import joblib

# === Шаг 1: Загрузка обучающих данных ===
df_train = pd.read_csv('train.csv', sep=';')
print(df_train.head())
print(df_train.columns)

# === Шаг 2: Feature Engineering для обучающих данных ===
# Преобразование дат в формат datetime
df_train['Дата бронирования'] = pd.to_datetime(df_train['Дата бронирования'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)
df_train['Дата отмены'] = pd.to_datetime(df_train['Дата отмены'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)
df_train['Заезд'] = pd.to_datetime(df_train['Заезд'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)
df_train['Выезд'] = pd.to_datetime(df_train['Выезд'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)

# Создание новых признаков на основе дат
df_train['Разница_бронирование_заезд'] = (df_train['Заезд'] - df_train['Дата бронирования']).dt.days
df_train['День_недели_заезда'] = df_train['Заезд'].dt.weekday
df_train['Месяц_заезда'] = df_train['Заезд'].dt.month

# Признак: количество номеров на количество ночей
df_train['Номера_на_ночи'] = df_train['Номеров'] * df_train['Ночей']

# Преобразование столбцов в числовые значения
df_train['Стоимость'] = pd.to_numeric(df_train['Стоимость'], errors='coerce')
df_train['Гостей'] = pd.to_numeric(df_train['Гостей'], errors='coerce')

# Обработка пропущенных значений
df_train['Гостей'].fillna(1, inplace=True)  # Если нет гостей, заменяем на 1
df_train['Стоимость'].fillna(0, inplace=True)  # Заменяем на 0, если стоимость отсутствует

# Признак: стоимость на одного гостя
df_train['Стоимость_на_гостя'] = df_train['Стоимость'] / df_train['Гостей']

# Признак: процент предоплаты
df_train['Процент_предоплаты'] = df_train['Внесена предоплата'] / df_train['Стоимость'].replace(0, np.nan)  # Избегаем деления на 0

# Целевая переменная (если дата отмены заполнена, то отмена была)
df_train['Отмена'] = df_train['Дата отмены'].notna().astype(int)

# Удаление ненужных столбцов в обучающих данных
df_train.drop(columns=['Дата бронирования', 'Дата отмены', 'Заезд', 'Выезд', 'Категория номера'], inplace=True)

# === Шаг 3: Разделение обучающих данных на признаки и целевую переменную ===
X_train = df_train.drop(columns=['Отмена'])
y_train = df_train['Отмена']

# === Шаг 4: Подготовка категориальных и числовых признаков ===
# Числовые и категориальные столбцы
numeric_features = ['Номеров', 'Стоимость', 'Внесена предоплата', 'Ночей', 'Гостей',
                    'Разница_бронирование_заезд', 'Номера_на_ночи', 'Стоимость_на_гостя', 'Процент_предоплаты']
categorical_features = ['Способ оплаты', 'Источник', 'Гостиница', 'День_недели_заезда', 'Месяц_заезда']

# Создание пайплайна для обработки данных
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Комбинируем все в общий трансформер
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# === Шаг 5: Ансамблирование моделей ===
catboost = CatBoostClassifier(verbose=0, random_seed=42)
lightgbm = LGBMClassifier(random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
ada_boost = AdaBoostClassifier(random_state=42)

# Стэкинг моделей с мета-моделью (логистическая регрессия)
stacking_clf = StackingClassifier(
    estimators=[
        ('catboost', catboost),
        ('lightgbm', lightgbm),
        ('random_forest', random_forest),
        ('gradient_boosting', gradient_boosting),
        ('ada_boost', ada_boost)
    ],
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# Полный пайплайн с обработкой данных и обучением моделей
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', stacking_clf)])

# === Шаг 6: Поиск гиперпараметров ===
param_grid = {
    'classifier__final_estimator__C': [0.1, 1, 10],
    'classifier__catboost__depth': [4, 6, 8],
    'classifier__lightgbm__num_leaves': [31, 50, 70],
    'classifier__random_forest__n_estimators': [50, 100, 200],
}

grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f'Лучшие параметры: {grid_search.best_params_}')
print(f'Лучший AUC: {grid_search.best_score_:.4f}')

# Сохранение модели
joblib.dump(grid_search.best_estimator_, 'best_model.joblib')

# === Шаг 7: Загрузка и обработка тестового набора данных ===
df_test = pd.read_csv('test.csv', sep=';')
print(df_test.head())

# Преобразование дат в тестовом наборе данных
df_test['Дата бронирования'] = pd.to_datetime(df_test['Дата бронирования'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)
df_test['Заезд'] = pd.to_datetime(df_test['Заезд'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)
df_test['Выезд'] = pd.to_datetime(df_test['Выезд'], format='%d.%m.%Y %H:%M', errors='coerce', dayfirst=True)

# Создание новых признаков на основе дат
df_test['Разница_бронирование_заезд'] = (df_test['Заезд'] - df_test['Дата бронирования']).dt.days
df_test['День_недели_заезда'] = df_test['Заезд'].dt.weekday
df_test['Месяц_заезда'] = df_test['Заезд'].dt.month

# Признак: количество номеров на количество ночей
df_test['Номера_на_ночи'] = df_test['Номеров'] * df_test['Ночей']

# Преобразование столбцов в числовые значения
df_test['Стоимость'] = pd.to_numeric(df_test['Стоимость'], errors='coerce')
df_test['Гостей'] = pd.to_numeric(df_test['Гостей'], errors='coerce')

# Проверка на наличие NaN
print(df_test.isnull().sum())

# Заполнение NaN, если необходимо
df_test['Гостей'].fillna(1, inplace=True)  # Если нет гостей, заменяем на 1
df_test['Стоимость'].fillna(0, inplace=True)  # Заменяем на 0, если стоимость отсутствует

# Признак: стоимость на одного гостя
df_test['Стоимость_на_гостя'] = df_test['Стоимость'] / df_test['Гостей']

# Признак: процент предоплаты
df_test['Процент_предоплаты'] = df_test['Внесена предоплата'] / df_test['Стоимость'].replace(0, np.nan)  # Избегаем деления на 0

# Удаление ненужных столбцов в тестовом наборе данных
df_test.drop(columns=['Дата бронирования', 'Заезд', 'Выезд', 'Категория номера'], inplace=True)

# === Шаг 8: Предсказания на тестовом наборе данных ===
test_predictions = grid_search.best_estimator_.predict_proba(df_test)[:, 1]

# Сохранение предсказаний в файл
output = pd.DataFrame({'id': df_test.index, 'Отмена': test_predictions})
output.to_csv('test_predictions.csv', index=False)

print("Предсказания сохранены в файл 'test_predictions.csv'")

# === Шаг 9: Визуализация результатов ===
# Построение ROC-кривой
fpr, tpr, thresholds = roc_curve(y_train, grid_search.best_estimator_.predict_proba(X_train)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC-кривая (AUC = {:.2f})'.format(roc_auc_score(y_train, grid_search.best_estimator_.predict_proba(X_train)[:, 1])))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Ложноположительная доля')
plt.ylabel('Истинноположительная доля')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.grid()
plt.show()
