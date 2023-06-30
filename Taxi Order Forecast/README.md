# Прогноз количества заказов для сервиса такси

[HTML](https://github.com/cenzukari/Portfolio/blob/main/Taxi%20Order%20Forecast/taxi_order_forecasting.html)     [ipynb](https://github.com/cenzukari/Portfolio/blob/main/Taxi%20Order%20Forecast/taxi_order_forecasting.ipynb)

## Описание проекта

Нужно построить модель для прогнозирования количество заказов такси на следующий час для привлечения большего количества водителей в период пиковой нагрузки. Для этой задачи компания заказчик предоставила исторические данные о заказах такси в аэропортах.

## Навыки и инструменты

- **python**
- **pandas**
- **numpy**
- **matplotlib**
- statsmodels.tsa.seasonal.**seasonal_decompose**
- sklearn.model_selection.**TimeSeriesSplit**
- sklearn.model_selection.**cross_val_score**
- sklearn.model_selection.**GridSearchCV**
- sklearn.metrics.**mean_squared_error**
- sklearn.metrics.**make_scorer**
- sklearn.linear_model.**LinearRegression**
- sklearn.tree.**DecisionTreeRegressor**
- catboost.**CatBoostRegressor**
- lightgbm.**LGBMRegressor**




## Общий вывод

В ходе работы над проектом проведено исследование временного ряда на предмет трендовых и сезонных закономерностей, случайной составляющей. На подготовленных данных обучено 4 модели и выбрана лучшая - модель CatBoostRegressor.
