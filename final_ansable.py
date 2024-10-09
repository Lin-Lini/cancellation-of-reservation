import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import joblib

# Загрузка данных
train_data = pd.read_csv('Last_train.csv', sep=';')
test_data = pd.read_csv('Last_test.csv', sep=';')

# Переименование колонок (с учетом новых признаков)
train_data.columns = ['Rooms', 'Cost', 'Prepayment_made', 'Payment_method', 'Booking_date',
                      'Cancellation_date', 'Check_in', 'Nights', 'Check_out', 'Source',
                      'Room_category', 'Guests', 'Hotel',
                      'Min_Temperature', 'Max_Temperature', 'Humidity', 'Precipitation', 'Wind_Speed']

test_data.columns = ['Rooms', 'Cost', 'Prepayment_made', 'Payment_method', 'Booking_date',
                     'Check_in', 'Nights', 'Check_out', 'Source',
                     'Room_category', 'Guests', 'Hotel',
                     'Min_Temperature', 'Max_Temperature', 'Humidity', 'Precipitation', 'Wind_Speed']

# Создание целевой переменной
train_data['Target'] = train_data['Cancellation_date'].notna().astype(int)
train_data.drop(columns=['Cancellation_date'], inplace=True)

# Преобразование 'Cost' в числовой формат
train_data['Cost'] = train_data['Cost'].str.replace(' ', '').str.replace(',', '.')
test_data['Cost'] = test_data['Cost'].str.replace(' ', '').str.replace(',', '.')

train_data['Cost'] = pd.to_numeric(train_data['Cost'], errors='coerce')
test_data['Cost'] = pd.to_numeric(test_data['Cost'], errors='coerce')

# Заполнение пропусков в 'Cost'
train_data['Cost'] = train_data['Cost'].fillna(train_data['Cost'].median())
test_data['Cost'] = test_data['Cost'].fillna(test_data['Cost'].median())

# Преобразование 'Prepayment_made' в строковый формат и очистка данных
train_data['Prepayment_made'] = train_data['Prepayment_made'].astype(str).str.replace(' ', '').str.replace(',', '.')
test_data['Prepayment_made'] = test_data['Prepayment_made'].astype(str).str.replace(' ', '').str.replace(',', '.')

# Преобразование в числовой формат
train_data['Prepayment_made'] = pd.to_numeric(train_data['Prepayment_made'], errors='coerce')
test_data['Prepayment_made'] = pd.to_numeric(test_data['Prepayment_made'], errors='coerce')

# Заполнение пропусков медианой
train_data['Prepayment_made'] = train_data['Prepayment_made'].fillna(train_data['Prepayment_made'].median())
test_data['Prepayment_made'] = test_data['Prepayment_made'].fillna(test_data['Prepayment_made'].median())

# Обработка новых признаков (температура, влажность, осадки, скорость ветра)
for col in ['Min_Temperature', 'Max_Temperature', 'Humidity', 'Precipitation', 'Wind_Speed']:
    # Замена запятых на точки
    train_data[col] = train_data[col].astype(str).str.replace(',', '.')
    test_data[col] = test_data[col].astype(str).str.replace(',', '.')

    # Преобразование в числовой формат
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

    # Заполнение пропусков медианой
    train_data[col] = train_data[col].fillna(train_data[col].median())
    test_data[col] = test_data[col].fillna(test_data[col].median())

# Добавление новых признаков

# Признак: стоимость на одного гостя
train_data['Стоимость_на_гостя'] = train_data['Cost'] / train_data['Guests'].replace(0, np.nan)
test_data['Стоимость_на_гостя'] = test_data['Cost'] / test_data['Guests'].replace(0, np.nan)

# Признак: процент предоплаты
train_data['Процент_предоплаты'] = train_data['Prepayment_made'] / train_data['Cost'].replace(0, np.nan)
test_data['Процент_предоплаты'] = test_data['Prepayment_made'] / test_data['Cost'].replace(0, np.nan)

# Заполнение пропусков медианой для новых признаков
train_data['Стоимость_на_гостя'] = train_data['Стоимость_на_гостя'].fillna(train_data['Стоимость_на_гостя'].median())
test_data['Стоимость_на_гостя'] = test_data['Стоимость_на_гостя'].fillna(test_data['Стоимость_на_гостя'].median())

train_data['Процент_предоплаты'] = train_data['Процент_предоплаты'].fillna(train_data['Процент_предоплаты'].median())
test_data['Процент_предоплаты'] = test_data['Процент_предоплаты'].fillna(test_data['Процент_предоплаты'].median())

# Работа с категориальными признаками
categorical_cols = ['Payment_method', 'Source', 'Room_category', 'Hotel']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Кодирование категориальных признаков
X_train_cat = encoder.fit_transform(train_data[categorical_cols])
X_test_cat = encoder.transform(test_data[categorical_cols])

# Объединение категориальных и числовых признаков
X_train = np.hstack((X_train_cat, train_data[['Rooms', 'Cost', 'Prepayment_made', 'Nights', 'Guests',
                                              'Min_Temperature', 'Max_Temperature', 'Humidity',
                                              'Precipitation', 'Wind_Speed', 'Стоимость_на_гостя',
                                              'Процент_предоплаты']].values))
X_test = np.hstack((X_test_cat, test_data[['Rooms', 'Cost', 'Prepayment_made', 'Nights', 'Guests',
                                           'Min_Temperature', 'Max_Temperature', 'Humidity',
                                           'Precipitation', 'Wind_Speed', 'Стоимость_на_гостя',
                                           'Процент_предоплаты']].values))

# Целевая переменная
y_train = train_data['Target']

# Разделение данных на тренировочную и валидационную выборки
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                              random_state=42)

# Определение базовых моделей для стэкинга
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)
lgbm_model = LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=4, min_child_samples=20)
extra_trees_model = ExtraTreesClassifier(n_estimators=500, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=500, random_state=42)
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4, min_child_weight=1, use_label_encoder=False,
                          eval_metric='logloss')
gb_model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=4)
ada_model = AdaBoostClassifier(n_estimators=500)

# Определение моделей для стэкинга
estimators = [
    ('catboost', catboost_model),
    ('lgbm', lgbm_model),
    ('extra_trees', extra_trees_model),
    ('random_forest', random_forest_model),
    ('xgb', xgb_model),
    ('gb', gb_model),
    ('ada', ada_model)
]

# Использование LogisticRegressionCV как мета-модели
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegressionCV(cv=5))


# Функция для оценки модели на кросс-валидации
def evaluate_model(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return scores


# Оценка обновленного ансамбля
print("Результаты обновленного стэкинга:")
scores = evaluate_model(stacking_model, X_train_split, y_train_split)
print(f'ROC-AUC на кросс-валидации: {scores.mean()}')

# Визуализация результатов кросс-валидации
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')
plt.title('ROC-AUC во время кросс-валидации')
plt.xlabel('Номер фолда')
plt.ylabel('ROC-AUC Score')
plt.xticks(range(1, len(scores) + 1))
plt.grid()
plt.show()

# Финальное обучение модели и предсказание для тестового набора
stacking_model.fit(X_train_split, y_train_split)
y_pred = stacking_model.predict(X_test)

joblib.dump(stacking_model, 'best_model.joblib')
# Сохранение предсказаний
output = pd.DataFrame({'Cancellation_Prediction': y_pred})
output.to_csv('test_predictions_complex_ensemble.csv', index=False)

# Финальная оценка на валидационной выборке
y_val_pred = stacking_model.predict(X_val)
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'Финальный ROC-AUC на валидационной выборке: {roc_auc}')
