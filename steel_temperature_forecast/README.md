# Предсказание температуры стали

[ipynb](https://github.com/cenzukari/yandex_practicum_data_science_projects/blob/main/steel_temperature_forecast/%20steel_temperature_forecast.ipynb)

## Описание проекта

Металлургический комбинат решил уменьшить потребление электроэнергии на этапе обработки стали, чтобы оптимизировать производственные расходы. Задача построить модель, которая предскажет температуру стали. 

## Навыки и инструменты

- **python**
- **pandas**
- **numpy**
- **seaborn**
- **matplotlib**
- sklearn.preprocessing.**StandardScaler**
- sklearn.model_selection.**cross_validate**
- sklearn.model_selection.**GridSearchCV**
- sklearn.metrics.**mean_absolute_error**
- sklearn.metrics.**r2_score**
- sklearn.inspection.**permutation_importance**
- sklearn.linear_model.**LinearRegression**
- sklearn.ensemble.**RandomForestRegressor**
- sklearn.dummy.**DummyRegressor**
- catboost.**CatBoostRegressor**
- lightgbm.**LGBMRegressor**


## 

## Общий вывод

Было проведено исследование и подготовка данных и признаков. Были обучены 4 модели и выбрана лучшая для имеющейся задачи - CatBoostRegressor.
