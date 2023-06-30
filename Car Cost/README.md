# Определение стоимости автомобилей с пробегом

[HTML](https://github.com/cenzukari/Portfolio/blob/main/Car%20Cost/car_cost_determination.html)     [ipynb](https://github.com/cenzukari/Portfolio/blob/main/Car%20Cost/car_cost_determination.ipynb)

## Описание проекта

Нужно построить модель, которая будет определять рыночную стоимость автомобиля в приложении для клиентов сервиса по продаже автомобилей.  В распоряжении данные о технических характеристиках, комплектации и ценах других автомобилей.

## Навыки и инструменты

- **python**
- **pandas**
- **numpy**
- **seaborn**
- **matplotlib**
- sklearn.preprocessing.**OneHotEncoder**
- sklearn.preprocessing.**OrdinalEncoder**
- sklearn.preprocessing.**StandardScaler**
- sklearn.model_selection.**cross_val_score**
- sklearn.model_selection.**GridSearchCV**
- sklearn.metrics.**mean_squared_error**
- sklearn.linear_model.**LinearRegression**
- sklearn.tree.**DecisionTreeRegressor**
- sklearn.dummy.**DummyRegressor**
- catboost.**CatBoostRegressor**
- lightgbm.**LGBMRegressor**


## 

## Общий вывод

Было проведено исследование данных, их кодирование для разных моделей и обучение 4 моделей на этой основе. В итоге была выбрана лучшая моделель по совокупности метрики RMSE и времени обучения - CatBoostRegressor.
