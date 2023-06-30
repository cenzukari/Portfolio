# Предсказание температуры стали

**План проекта:**
        
1. Описание задачи 
2. Исследовательский анализ данных
3. Предобработка данных
4. Анализ и подготовка признаков
5. Обучение моделей
6. Итоговые выводы
7. Отчет

## 1 Описание задачи

**Задача:**  
  
Металлургический комбинат ООО «Так закаляем сталь» решил уменьшить потребление электроэнергии на этапе обработки стали, чтобы оптимизировать производственные расходы. Задача построить модель, которая предскажет температуру стали. Показателями успешности модели будет метрика MAE. Дополнительная метрика R2. 

**Цель:**  
  
Достигнуть значения ключевой метрики MAE не выше 6.8 на тестовой выборке.
  
**Вводная информация по этапам обработки стали:**  
  
Сталь обрабатывают в металлическом ковше вместимостью около 100 тонн. Чтобы ковш выдерживал высокие температуры, изнутри его облицовывают огнеупорным кирпичом. Расплавленную сталь заливают в ковш и подогревают до нужной температуры графитовыми электродами. Они установлены в крышке ковша.   
  
Из сплава выводится сера (десульфурация), добавлением примесей корректируется химический состав и отбираются пробы. Сталь легируют — изменяют её состав — подавая куски сплава из бункера для сыпучих материалов или проволоку через специальный трайб-аппарат (англ. tribe, «масса»).  
  
Перед тем как первый раз ввести легирующие добавки, измеряют температуру стали и производят её химический анализ. Потом температуру на несколько минут повышают, добавляют легирующие материалы и продувают сплав инертным газом. Затем его перемешивают и снова проводят измерения. Такой цикл повторяется до достижения целевого химического состава и оптимальной температуры плавки.  
  
Тогда расплавленная сталь отправляется на доводку металла или поступает в машину непрерывной разливки. Оттуда готовый продукт выходит в виде заготовок-слябов (англ. *slab*, «плита»).  
   
**Имеющиеся данные:**  
    
Данные состоят из файлов, полученных из разных источников:  
  
- `data_arc_new.csv` — данные об электродах
- `data_bulk_new.csv` — данные о подаче сыпучих материалов (объём)
- `data_bulk_time_new.csv` *—* данные о подаче сыпучих материалов (время)
- `data_gas_new.csv` — данные о продувке сплава газом
- `data_temp_new.csv` — результаты измерения температуры
- `data_wire_new.csv` — данные о проволочных материалах (объём)
- `data_wire_time_new.csv` — данные о проволочных материалах (время)    
  
Во всех файлах столбец `key` содержит номер партии. В файлах может быть несколько строк с одинаковым значением `key`: они соответствуют разным итерациям обработки.


## 2 Исследовательский анализ данных


```python
# импортируем библиотеки

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import catboost as cb
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from numpy.random import RandomState

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.dummy import DummyRegressor

import time

RANDOM_STATE = 220523

import warnings
warnings.filterwarnings('ignore')
```


```python
# прочитаем датасеты

try:
    data_arc = pd.read_csv(
        '/datasets/data_arc_new.csv')
    data_bulk = pd.read_csv(
        '/datasets/data_bulk_new.csv')
    data_bulk_time = pd.read_csv(
        '/datasets/data_bulk_time_new.csv')
    data_gas = pd.read_csv(
        '/datasets/data_gas_new.csv')
    data_temp = pd.read_csv(
        '/datasets/data_temp_new.csv')
    data_wire = pd.read_csv(
        'datasets/data_wire_new.csv')
    data_wire_time = pd.read_csv(
        '/datasets/data_wire_time_new.csv')
        
except:
    data_arc = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_arc_new.csv')
    data_bulk = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_bulk_new.csv')
    data_bulk_time = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_bulk_time_new.csv')
    data_gas = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_gas_new.csv')
    data_temp = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_temp_new.csv')
    data_wire = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_wire_new.csv')
    data_wire_time = pd.read_csv(
        'https://code.s3.yandex.net//datasets/data_wire_time_new.csv')
```

У нас 7 датасетов и не вся информация из них пригодится в проекте. Посмотрим, какие данные у нас имееются и в каком виде они находятся в таблицах.

### 2.1 Данные об электродах


```python
# посмотрим датасет с данными об электродах data_arc

data_arc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Начало нагрева дугой</th>
      <th>Конец нагрева дугой</th>
      <th>Активная мощность</th>
      <th>Реактивная мощность</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:02:14</td>
      <td>2019-05-03 11:06:02</td>
      <td>0.305130</td>
      <td>0.211253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2019-05-03 11:07:28</td>
      <td>2019-05-03 11:10:33</td>
      <td>0.765658</td>
      <td>0.477438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2019-05-03 11:11:44</td>
      <td>2019-05-03 11:14:36</td>
      <td>0.580313</td>
      <td>0.430460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2019-05-03 11:18:14</td>
      <td>2019-05-03 11:24:19</td>
      <td>0.518496</td>
      <td>0.379979</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2019-05-03 11:26:09</td>
      <td>2019-05-03 11:28:37</td>
      <td>0.867133</td>
      <td>0.643691</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим датасет с данными об электродах data_arc

data_arc.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Начало нагрева дугой</th>
      <th>Конец нагрева дугой</th>
      <th>Активная мощность</th>
      <th>Реактивная мощность</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14871</th>
      <td>3241</td>
      <td>2019-09-06 16:49:05</td>
      <td>2019-09-06 16:51:42</td>
      <td>0.439735</td>
      <td>0.299579</td>
    </tr>
    <tr>
      <th>14872</th>
      <td>3241</td>
      <td>2019-09-06 16:55:11</td>
      <td>2019-09-06 16:58:11</td>
      <td>0.646498</td>
      <td>0.458240</td>
    </tr>
    <tr>
      <th>14873</th>
      <td>3241</td>
      <td>2019-09-06 17:06:48</td>
      <td>2019-09-06 17:09:52</td>
      <td>1.039726</td>
      <td>0.769302</td>
    </tr>
    <tr>
      <th>14874</th>
      <td>3241</td>
      <td>2019-09-06 17:21:58</td>
      <td>2019-09-06 17:22:55</td>
      <td>0.530267</td>
      <td>0.361543</td>
    </tr>
    <tr>
      <th>14875</th>
      <td>3241</td>
      <td>2019-09-06 17:24:54</td>
      <td>2019-09-06 17:26:15</td>
      <td>0.389057</td>
      <td>0.251347</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_arc

data_arc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14876 entries, 0 to 14875
    Data columns (total 5 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   key                   14876 non-null  int64  
     1   Начало нагрева дугой  14876 non-null  object 
     2   Конец нагрева дугой   14876 non-null  object 
     3   Активная мощность     14876 non-null  float64
     4   Реактивная мощность   14876 non-null  float64
    dtypes: float64(2), int64(1), object(2)
    memory usage: 581.2+ KB



```python
# посмотрим  описание датасета data_arc

data_arc.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Активная мощность</th>
      <th>Реактивная мощность</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14876.000000</td>
      <td>14876.000000</td>
      <td>14876.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1615.220422</td>
      <td>0.662752</td>
      <td>0.438986</td>
    </tr>
    <tr>
      <th>std</th>
      <td>934.571502</td>
      <td>0.258885</td>
      <td>5.873485</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.223120</td>
      <td>-715.479924</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>806.000000</td>
      <td>0.467115</td>
      <td>0.337175</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1617.000000</td>
      <td>0.599587</td>
      <td>0.441639</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2429.000000</td>
      <td>0.830070</td>
      <td>0.608201</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
      <td>1.463773</td>
      <td>1.270284</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_arc['key'].unique())
```




    3214




```python
# проверим наличие явных дубликатов

data_arc.duplicated().sum()
```




    0




```python
# посмотрим распределние значений на гистограммах

data_arc.hist(figsize=(15,8));
```


    
![png](output_15_0.png)
    



```python
# посмотрим распределение количества нагревов по каждой партии

data_arc['key'].value_counts().hist(figsize=(9,5));
```


    
![png](output_16_0.png)
    



```python
# посмотрим график ящик с усами для реактивной на предмет выбросов

data_arc.boxplot(column='Реактивная мощность', grid= True, figsize=(9,5));
```


    
![png](output_17_0.png)
    



```python
# посмотрим график ящик с усами для активной мощностей на предмет выбросов

data_arc.boxplot(column='Активная мощность', grid= True, figsize=(9,5));
```


    
![png](output_18_0.png)
    


**Вывод по данным об электродах:**  
   
- в датасете 5 столбцов и 14876 строк
- 3214 уникальных ключей - номеров партий
- нет явных дубликатов
- имеются выбросы: в столбце реактивной мощности одно отрицательное значение, в столбце активная мощность есть несколько значений выше 1.4
- количество нагревов варьируется от 1 до 16
- не верный тип данных у столбца начала и конца нагрева дугой
- наименования столбцов на русском и в верблюжьем регистре   
   
На этапе предобработке данных поправим тип данных, наименования столбцов и удалим строку с отрицательным значением в реактивной мощности.

### 2.2 Данные о подаче сыпучих материалов: объём и время


```python
# посмотрим датасет с данными о подаче сыпучих материалов (объём) data_bulk

data_bulk.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Bulk 1</th>
      <th>Bulk 2</th>
      <th>Bulk 3</th>
      <th>Bulk 4</th>
      <th>Bulk 5</th>
      <th>Bulk 6</th>
      <th>Bulk 7</th>
      <th>Bulk 8</th>
      <th>Bulk 9</th>
      <th>Bulk 10</th>
      <th>Bulk 11</th>
      <th>Bulk 12</th>
      <th>Bulk 13</th>
      <th>Bulk 14</th>
      <th>Bulk 15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>206.0</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>206.0</td>
      <td>NaN</td>
      <td>149.0</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>205.0</td>
      <td>NaN</td>
      <td>152.0</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>81.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>207.0</td>
      <td>NaN</td>
      <td>153.0</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>203.0</td>
      <td>NaN</td>
      <td>151.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим датасет с данными о подаче сыпучих материалов (объём) data_bulk

data_bulk.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Bulk 1</th>
      <th>Bulk 2</th>
      <th>Bulk 3</th>
      <th>Bulk 4</th>
      <th>Bulk 5</th>
      <th>Bulk 6</th>
      <th>Bulk 7</th>
      <th>Bulk 8</th>
      <th>Bulk 9</th>
      <th>Bulk 10</th>
      <th>Bulk 11</th>
      <th>Bulk 12</th>
      <th>Bulk 13</th>
      <th>Bulk 14</th>
      <th>Bulk 15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3124</th>
      <td>3237</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>170.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>252.0</td>
      <td>NaN</td>
      <td>130.0</td>
      <td>206.0</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>3238</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>126.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>254.0</td>
      <td>NaN</td>
      <td>108.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>3239</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>114.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>158.0</td>
      <td>NaN</td>
      <td>270.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>3240</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>192.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>3241</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>180.0</td>
      <td>52.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_bulk

data_bulk.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3129 entries, 0 to 3128
    Data columns (total 16 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   key      3129 non-null   int64  
     1   Bulk 1   252 non-null    float64
     2   Bulk 2   22 non-null     float64
     3   Bulk 3   1298 non-null   float64
     4   Bulk 4   1014 non-null   float64
     5   Bulk 5   77 non-null     float64
     6   Bulk 6   576 non-null    float64
     7   Bulk 7   25 non-null     float64
     8   Bulk 8   1 non-null      float64
     9   Bulk 9   19 non-null     float64
     10  Bulk 10  176 non-null    float64
     11  Bulk 11  177 non-null    float64
     12  Bulk 12  2450 non-null   float64
     13  Bulk 13  18 non-null     float64
     14  Bulk 14  2806 non-null   float64
     15  Bulk 15  2248 non-null   float64
    dtypes: float64(15), int64(1)
    memory usage: 391.2 KB



```python
# посмотрим  описание датасета data_bulk

data_bulk.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Bulk 1</th>
      <th>Bulk 2</th>
      <th>Bulk 3</th>
      <th>Bulk 4</th>
      <th>Bulk 5</th>
      <th>Bulk 6</th>
      <th>Bulk 7</th>
      <th>Bulk 8</th>
      <th>Bulk 9</th>
      <th>Bulk 10</th>
      <th>Bulk 11</th>
      <th>Bulk 12</th>
      <th>Bulk 13</th>
      <th>Bulk 14</th>
      <th>Bulk 15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3129.000000</td>
      <td>252.000000</td>
      <td>22.000000</td>
      <td>1298.000000</td>
      <td>1014.000000</td>
      <td>77.000000</td>
      <td>576.000000</td>
      <td>25.000000</td>
      <td>1.0</td>
      <td>19.000000</td>
      <td>176.000000</td>
      <td>177.000000</td>
      <td>2450.000000</td>
      <td>18.000000</td>
      <td>2806.000000</td>
      <td>2248.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1624.383509</td>
      <td>39.242063</td>
      <td>253.045455</td>
      <td>113.879045</td>
      <td>104.394477</td>
      <td>107.025974</td>
      <td>118.925347</td>
      <td>305.600000</td>
      <td>49.0</td>
      <td>76.315789</td>
      <td>83.284091</td>
      <td>76.819209</td>
      <td>260.471020</td>
      <td>181.111111</td>
      <td>170.284747</td>
      <td>160.513345</td>
    </tr>
    <tr>
      <th>std</th>
      <td>933.337642</td>
      <td>18.277654</td>
      <td>21.180578</td>
      <td>75.483494</td>
      <td>48.184126</td>
      <td>81.790646</td>
      <td>72.057776</td>
      <td>191.022904</td>
      <td>NaN</td>
      <td>21.720581</td>
      <td>26.060347</td>
      <td>59.655365</td>
      <td>120.649269</td>
      <td>46.088009</td>
      <td>65.868652</td>
      <td>51.765319</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>228.000000</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>11.000000</td>
      <td>17.000000</td>
      <td>47.000000</td>
      <td>49.0</td>
      <td>63.000000</td>
      <td>24.000000</td>
      <td>8.000000</td>
      <td>53.000000</td>
      <td>151.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>816.000000</td>
      <td>27.000000</td>
      <td>242.000000</td>
      <td>58.000000</td>
      <td>72.000000</td>
      <td>70.000000</td>
      <td>69.750000</td>
      <td>155.000000</td>
      <td>49.0</td>
      <td>66.000000</td>
      <td>64.000000</td>
      <td>25.000000</td>
      <td>204.000000</td>
      <td>153.250000</td>
      <td>119.000000</td>
      <td>105.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1622.000000</td>
      <td>31.000000</td>
      <td>251.500000</td>
      <td>97.500000</td>
      <td>102.000000</td>
      <td>86.000000</td>
      <td>100.000000</td>
      <td>298.000000</td>
      <td>49.0</td>
      <td>68.000000</td>
      <td>86.500000</td>
      <td>64.000000</td>
      <td>208.000000</td>
      <td>155.500000</td>
      <td>151.000000</td>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2431.000000</td>
      <td>46.000000</td>
      <td>257.750000</td>
      <td>152.000000</td>
      <td>133.000000</td>
      <td>132.000000</td>
      <td>157.000000</td>
      <td>406.000000</td>
      <td>49.0</td>
      <td>70.500000</td>
      <td>102.000000</td>
      <td>106.000000</td>
      <td>316.000000</td>
      <td>203.500000</td>
      <td>205.750000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
      <td>185.000000</td>
      <td>325.000000</td>
      <td>454.000000</td>
      <td>281.000000</td>
      <td>603.000000</td>
      <td>503.000000</td>
      <td>772.000000</td>
      <td>49.0</td>
      <td>147.000000</td>
      <td>159.000000</td>
      <td>313.000000</td>
      <td>1849.000000</td>
      <td>305.000000</td>
      <td>636.000000</td>
      <td>405.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_bulk['key'].unique())
```




    3129




```python
# проверим наличие явных дубликатов

data_bulk.duplicated().sum()
```




    0




```python
# посмотрим распределние значений на гистограммах

data_bulk.hist(figsize=(18,14));
```


    
![png](output_27_0.png)
    


**Вывод по данным об объеме подачи сыпучих материалов:**  
   
- в датасете 15 столбцов и 3129 строк - все они уникальные ключи (номера партий)
- нет явных дубликатов
- много пропусков - это специфика отрасли
- bulk 8 использовался только 1 раз - будем иметь ввиду при дальнейшей работе
- в большинсвте случаев распределение нормальное
  
На этапе предобработки данных нужно заполнить пропуска нулями, привести наименования столбцов к змеиному регистру


```python
# посмотрим датасет с данными о подаче сыпучих материалов (время) data_bulk_time

data_bulk_time.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Bulk 1</th>
      <th>Bulk 2</th>
      <th>Bulk 3</th>
      <th>Bulk 4</th>
      <th>Bulk 5</th>
      <th>Bulk 6</th>
      <th>Bulk 7</th>
      <th>Bulk 8</th>
      <th>Bulk 9</th>
      <th>Bulk 10</th>
      <th>Bulk 11</th>
      <th>Bulk 12</th>
      <th>Bulk 13</th>
      <th>Bulk 14</th>
      <th>Bulk 15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 11:28:48</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 11:24:31</td>
      <td>NaN</td>
      <td>2019-05-03 11:14:50</td>
      <td>2019-05-03 11:10:43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 11:36:50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 11:53:30</td>
      <td>NaN</td>
      <td>2019-05-03 11:48:37</td>
      <td>2019-05-03 11:44:39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 12:32:39</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 12:27:13</td>
      <td>NaN</td>
      <td>2019-05-03 12:21:01</td>
      <td>2019-05-03 12:16:16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 12:43:22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 12:58:00</td>
      <td>NaN</td>
      <td>2019-05-03 12:51:11</td>
      <td>2019-05-03 12:46:36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 13:30:47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-05-03 13:30:47</td>
      <td>NaN</td>
      <td>2019-05-03 13:34:12</td>
      <td>2019-05-03 13:30:47</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим датасет с данными о подаче сыпучих материалов (время) data_bulk_time

data_bulk_time.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Bulk 1</th>
      <th>Bulk 2</th>
      <th>Bulk 3</th>
      <th>Bulk 4</th>
      <th>Bulk 5</th>
      <th>Bulk 6</th>
      <th>Bulk 7</th>
      <th>Bulk 8</th>
      <th>Bulk 9</th>
      <th>Bulk 10</th>
      <th>Bulk 11</th>
      <th>Bulk 12</th>
      <th>Bulk 13</th>
      <th>Bulk 14</th>
      <th>Bulk 15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3124</th>
      <td>3237</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 11:54:15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 11:49:45</td>
      <td>NaN</td>
      <td>2019-09-06 11:45:22</td>
      <td>2019-09-06 11:40:06</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>3238</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 12:26:52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 12:18:35</td>
      <td>NaN</td>
      <td>2019-09-06 12:31:49</td>
      <td>2019-09-06 12:26:52</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>3239</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 15:06:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 15:01:44</td>
      <td>NaN</td>
      <td>2019-09-06 14:58:15</td>
      <td>2019-09-06 14:48:06</td>
    </tr>
    <tr>
      <th>3127</th>
      <td>3240</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 16:24:28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 16:07:29</td>
      <td>2019-09-06 16:01:34</td>
    </tr>
    <tr>
      <th>3128</th>
      <td>3241</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019-09-06 17:26:33</td>
      <td>2019-09-06 17:23:15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_bulk_time

data_bulk_time.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3129 entries, 0 to 3128
    Data columns (total 16 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   key      3129 non-null   int64 
     1   Bulk 1   252 non-null    object
     2   Bulk 2   22 non-null     object
     3   Bulk 3   1298 non-null   object
     4   Bulk 4   1014 non-null   object
     5   Bulk 5   77 non-null     object
     6   Bulk 6   576 non-null    object
     7   Bulk 7   25 non-null     object
     8   Bulk 8   1 non-null      object
     9   Bulk 9   19 non-null     object
     10  Bulk 10  176 non-null    object
     11  Bulk 11  177 non-null    object
     12  Bulk 12  2450 non-null   object
     13  Bulk 13  18 non-null     object
     14  Bulk 14  2806 non-null   object
     15  Bulk 15  2248 non-null   object
    dtypes: int64(1), object(15)
    memory usage: 391.2+ KB



```python
# посмотрим описание датасета data_bulk_time

data_bulk_time.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1624.383509</td>
    </tr>
    <tr>
      <th>std</th>
      <td>933.337642</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>816.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1622.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2431.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_bulk_time['key'].unique())
```




    3129




```python
# проверим наличие явных дубликатов

data_bulk_time.duplicated().sum()
```




    0



**Вывод по данным о времени подачи сыпучих материалов:**  
   
- в датасете 15 столбцов и 3129 строк - все они уникальные ключи (номера партий) - как и в данных об объеме
- нет явных дубликатов
- много пропусков
  
Эти данные служат для нас проверкой верности данных в датасете об объеме подачи сыпучих материалов. Можно сказать, что все в порядке, так как количество столбцов и строк и ключей совпадает. 

### 2.3 Данные о продувке сплава газом


```python
# посмотрим датасет с данными о продувке сплава газом data_gas

data_gas.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Газ 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>29.749986</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>12.555561</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>28.554793</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18.841219</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.413692</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим датасет с данными о продувке сплава газом data_gas

data_gas.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Газ 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3234</th>
      <td>3237</td>
      <td>5.543905</td>
    </tr>
    <tr>
      <th>3235</th>
      <td>3238</td>
      <td>6.745669</td>
    </tr>
    <tr>
      <th>3236</th>
      <td>3239</td>
      <td>16.023518</td>
    </tr>
    <tr>
      <th>3237</th>
      <td>3240</td>
      <td>11.863103</td>
    </tr>
    <tr>
      <th>3238</th>
      <td>3241</td>
      <td>12.680959</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_gas

data_gas.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3239 entries, 0 to 3238
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   key     3239 non-null   int64  
     1   Газ 1   3239 non-null   float64
    dtypes: float64(1), int64(1)
    memory usage: 50.7 KB



```python
# посмотрим  описание датасета data_gas

data_gas.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Газ 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3239.000000</td>
      <td>3239.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1621.861377</td>
      <td>11.002062</td>
    </tr>
    <tr>
      <th>std</th>
      <td>935.386334</td>
      <td>6.220327</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.008399</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>812.500000</td>
      <td>7.043089</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1622.000000</td>
      <td>9.836267</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2431.500000</td>
      <td>13.769915</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
      <td>77.995040</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_gas['key'].unique())
```




    3239




```python
# проверим наличие явных дубликатов

data_gas.duplicated().sum()
```




    0




```python
# посмотрим распределние значений на гистограммах

data_gas.hist(figsize=(15,5));
```


    
![png](output_43_0.png)
    



```python
# посмотрим график ящик с усами для газа 1 на предмет выбросов

data_gas.boxplot(column='Газ 1', grid= True, figsize=(9,5));
```


    
![png](output_44_0.png)
    


**Вывод по данным о продувке сплава газом:**  
   
- в датасете 2 столбца и 3239 строк - все они уникальные ключи (номера партий)
- наименования столбца газ написано верблюжьим регистром и на русском языке
- нет явных дубликатов
- нет пропусков
- распределение нормальное, но есть выбросы
  
На этапе предобработки данных переименуем столбец. И будем иметь ввиду выбросы значений выше 40. 

### 2.4 Данные с результатами измерения температуры


```python
# посмотрим датасет с данными с результатами измерения температуры data_temp

data_temp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Время замера</th>
      <th>Температура</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:02:04</td>
      <td>1571.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2019-05-03 11:07:18</td>
      <td>1604.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2019-05-03 11:11:34</td>
      <td>1618.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2019-05-03 11:18:04</td>
      <td>1601.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2019-05-03 11:25:59</td>
      <td>1606.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим датасет с данными с результатами измерения температуры data_temp

data_temp.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Время замера</th>
      <th>Температура</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18087</th>
      <td>3241</td>
      <td>2019-09-06 16:55:01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18088</th>
      <td>3241</td>
      <td>2019-09-06 17:06:38</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18089</th>
      <td>3241</td>
      <td>2019-09-06 17:21:48</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18090</th>
      <td>3241</td>
      <td>2019-09-06 17:24:44</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18091</th>
      <td>3241</td>
      <td>2019-09-06 17:30:05</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_temp

data_temp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18092 entries, 0 to 18091
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   key           18092 non-null  int64  
     1   Время замера  18092 non-null  object 
     2   Температура   14665 non-null  float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 424.2+ KB



```python
# посмотрим  описание датасета data_temp

data_temp.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Температура</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>18092.000000</td>
      <td>14665.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1616.460977</td>
      <td>1590.722741</td>
    </tr>
    <tr>
      <th>std</th>
      <td>934.641385</td>
      <td>20.394381</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1191.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>807.750000</td>
      <td>1580.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1618.000000</td>
      <td>1590.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2429.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
      <td>1705.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_temp['key'].unique())
```




    3216




```python
# проверим наличие явных дубликатов

data_temp.duplicated().sum()
```




    0




```python
# посмотрим распределние значений на гистограммах

data_temp.hist(figsize=(15,5));
```


    
![png](output_53_0.png)
    



```python
# посмотрим график ящик с усами для газа 1 на предмет выбросов

data_temp.boxplot(column='Температура', grid= True, figsize=(11,10));
```


    
![png](output_54_0.png)
    


По информации выше видно, что много пропусков и в хвосте отсутсвуют финальные замеры. Надо проверить более детально нулевые значения, чтобы сделать выводы.


```python
# проверим количество нулевых значений замеров

data_temp['Температура'].isnull().sum()
```




    3427




```python
# выведем с какого значения начинаются нулевые измерения

data_temp[data_temp['Температура'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Время замера</th>
      <th>Температура</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13927</th>
      <td>2500</td>
      <td>2019-08-10 14:13:11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13928</th>
      <td>2500</td>
      <td>2019-08-10 14:18:12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13929</th>
      <td>2500</td>
      <td>2019-08-10 14:25:53</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13930</th>
      <td>2500</td>
      <td>2019-08-10 14:29:39</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13932</th>
      <td>2501</td>
      <td>2019-08-10 14:49:15</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18087</th>
      <td>3241</td>
      <td>2019-09-06 16:55:01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18088</th>
      <td>3241</td>
      <td>2019-09-06 17:06:38</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18089</th>
      <td>3241</td>
      <td>2019-09-06 17:21:48</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18090</th>
      <td>3241</td>
      <td>2019-09-06 17:24:44</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18091</th>
      <td>3241</td>
      <td>2019-09-06 17:30:05</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3427 rows × 3 columns</p>
</div>




```python
# выведем ключ 2500 и посмотрим все ли значения у него отсутсуют

data_temp[data_temp['key'] == 2500]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Время замера</th>
      <th>Температура</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13926</th>
      <td>2500</td>
      <td>2019-08-10 14:04:39</td>
      <td>1539.0</td>
    </tr>
    <tr>
      <th>13927</th>
      <td>2500</td>
      <td>2019-08-10 14:13:11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13928</th>
      <td>2500</td>
      <td>2019-08-10 14:18:12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13929</th>
      <td>2500</td>
      <td>2019-08-10 14:25:53</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13930</th>
      <td>2500</td>
      <td>2019-08-10 14:29:39</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Вывод по данным о температуре:**  
   
- в датасете 2 столбца и 18092 строк
- 3216 уникальных ключа (номера партий)
- для каждой партии несколько замеров температуры - важные для нас первый и последний (target)
- наименования столбца температуры написано верблюжьим регистром и на русском языке
- нет явных дубликатов
- есть пропуски в данных
- с 2500 партии есть только первый замер температуры
- есть выбросы температуры ниже 1500 и выше 1650
  
На этапе предобработки данных переименуем столбец, удалим выбросы ниже 1500 градусов, посмотрим пристально на пропуски и решим по их обработке, а также удалим все партии с 2500, так как у них нет целевого замера - финальной температуры.

### 2.5 Данные о проволочных материалах: объём и время


```python
# посмотрим датасет с данными о проволочных материалах (объём) data_wire

data_wire.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Wire 1</th>
      <th>Wire 2</th>
      <th>Wire 3</th>
      <th>Wire 4</th>
      <th>Wire 5</th>
      <th>Wire 6</th>
      <th>Wire 7</th>
      <th>Wire 8</th>
      <th>Wire 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60.059998</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>96.052315</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>91.160157</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>89.063515</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>89.238236</td>
      <td>9.11456</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_wire

data_wire.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3081 entries, 0 to 3080
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   key     3081 non-null   int64  
     1   Wire 1  3055 non-null   float64
     2   Wire 2  1079 non-null   float64
     3   Wire 3  63 non-null     float64
     4   Wire 4  14 non-null     float64
     5   Wire 5  1 non-null      float64
     6   Wire 6  73 non-null     float64
     7   Wire 7  11 non-null     float64
     8   Wire 8  19 non-null     float64
     9   Wire 9  29 non-null     float64
    dtypes: float64(9), int64(1)
    memory usage: 240.8 KB



```python
# посмотрим  описание датасета data_wire

data_wire.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Wire 1</th>
      <th>Wire 2</th>
      <th>Wire 3</th>
      <th>Wire 4</th>
      <th>Wire 5</th>
      <th>Wire 6</th>
      <th>Wire 7</th>
      <th>Wire 8</th>
      <th>Wire 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3081.000000</td>
      <td>3055.000000</td>
      <td>1079.000000</td>
      <td>63.000000</td>
      <td>14.000000</td>
      <td>1.000</td>
      <td>73.000000</td>
      <td>11.000000</td>
      <td>19.000000</td>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1623.426485</td>
      <td>100.895853</td>
      <td>50.577323</td>
      <td>189.482681</td>
      <td>57.442841</td>
      <td>15.132</td>
      <td>48.016974</td>
      <td>10.039007</td>
      <td>53.625193</td>
      <td>34.155752</td>
    </tr>
    <tr>
      <th>std</th>
      <td>932.996726</td>
      <td>42.012518</td>
      <td>39.320216</td>
      <td>99.513444</td>
      <td>28.824667</td>
      <td>NaN</td>
      <td>33.919845</td>
      <td>8.610584</td>
      <td>16.881728</td>
      <td>19.931616</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.918800</td>
      <td>0.030160</td>
      <td>0.144144</td>
      <td>24.148801</td>
      <td>15.132</td>
      <td>0.034320</td>
      <td>0.234208</td>
      <td>45.076721</td>
      <td>4.622800</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>823.000000</td>
      <td>72.115684</td>
      <td>20.193680</td>
      <td>95.135044</td>
      <td>40.807002</td>
      <td>15.132</td>
      <td>25.053600</td>
      <td>6.762756</td>
      <td>46.094879</td>
      <td>22.058401</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1619.000000</td>
      <td>100.158234</td>
      <td>40.142956</td>
      <td>235.194977</td>
      <td>45.234282</td>
      <td>15.132</td>
      <td>42.076324</td>
      <td>9.017009</td>
      <td>46.279999</td>
      <td>30.066399</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2434.000000</td>
      <td>126.060483</td>
      <td>70.227558</td>
      <td>276.252014</td>
      <td>76.124619</td>
      <td>15.132</td>
      <td>64.212723</td>
      <td>11.886057</td>
      <td>48.089603</td>
      <td>43.862003</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
      <td>330.314424</td>
      <td>282.780152</td>
      <td>385.008668</td>
      <td>113.231044</td>
      <td>15.132</td>
      <td>180.454575</td>
      <td>32.847674</td>
      <td>102.762401</td>
      <td>90.053604</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_wire['key'].unique())
```




    3081




```python
# проверим наличие явных дубликатов

data_wire.duplicated().sum()
```




    0




```python
# посмотрим распределние значений на гистограммах

data_wire.hist(figsize=(18,14));
```


    
![png](output_66_0.png)
    


**Вывод по данным об объеме проволочных материалов:**  
   
- в датасете 10 столбцов и 3081 строк - все они уникальные ключи (номера партий)
- нет явных дубликатов
- много пропусков - это специфика отрасли
- wire 5 использовался только 1 раз - будем иметь ввиду при дальнейшей работе
- в большинсвте случаев распределение нормальное
  
На этапе предобработки данных нужно заполнить пропуски нулями, привести наименования столбцов к змеиному регистру.


```python
# посмотрим датасет с данными о проволочных материалах (время) data_wire_time

data_wire_time.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>Wire 1</th>
      <th>Wire 2</th>
      <th>Wire 3</th>
      <th>Wire 4</th>
      <th>Wire 5</th>
      <th>Wire 6</th>
      <th>Wire 7</th>
      <th>Wire 8</th>
      <th>Wire 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:06:19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2019-05-03 11:36:50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2019-05-03 12:11:46</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2019-05-03 12:43:22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2019-05-03 13:20:44</td>
      <td>2019-05-03 13:15:34</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим информацию о датасете data_wire_time

data_wire_time.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3081 entries, 0 to 3080
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   key     3081 non-null   int64 
     1   Wire 1  3055 non-null   object
     2   Wire 2  1079 non-null   object
     3   Wire 3  63 non-null     object
     4   Wire 4  14 non-null     object
     5   Wire 5  1 non-null      object
     6   Wire 6  73 non-null     object
     7   Wire 7  11 non-null     object
     8   Wire 8  19 non-null     object
     9   Wire 9  29 non-null     object
    dtypes: int64(1), object(9)
    memory usage: 240.8+ KB



```python
# посмотрим  описание датасета data_wire_time

data_wire_time.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3081.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1623.426485</td>
    </tr>
    <tr>
      <th>std</th>
      <td>932.996726</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>823.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1619.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2434.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3241.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим на количество ункальных ключей

len(data_wire_time['key'].unique())
```




    3081




```python
# проверим наличие явных дубликатов

data_wire_time.duplicated().sum()
```




    0



**Вывод по данным о времени подачи проволочных материалов:**  
   
Эти данные служат для нас проверкой верности данных в датасете об объеме подачи проволочных материалов, как файл data_bulk_time служил для проверки даннных о сыпучих. Можно сказать, что все в порядке, так как количество столбцов 10 и строк 3081 и ключей 3081 совпадает. 

### 2.6 Вывод

Итак, после знакомства с данными мы знаем, что: 

- во всех датасетах разное количество уникальных ключей: data_arc 3214, data_bulk 3129, data_gas 3239, data_temp 3216, data_wire 3081
- датасеты data_bulk_time data_wire_time можно не использовать в дальнейшей работе
- есть неверные типы данных в data_arc
- много пропусков в data_bulk и data_wire
- наименования столбцов везде написаны верблюжьим регистром и русским языком
- нет явных дубликатов в данных
- bulk 8 и wire 5 использовались по 1 разу
- есть выбросы в данных о температуре, подаче газа и реактивной мощности
- нет замеров температуры кроме начальной у партий с 2500 номера

На предорбработке данных поправим наименования, заполним пропуски, удалим выброс в реактивной мощностиб а также удалим все партии с 2500, так как у них нет целевого замера - финальной температуры, и затем объединим данные в единую таблицу.   
  
Для объединения в таблицу нужно будет обработать значения в таблице data_temp, чтобы остались только начальная температура и конечная для каждого уникального ключа. А в таблице data_arc найти значение полной мощности и посмотреть при анализе признаков стоит ли оставить только его.   
  
Поскольку ключей у нас везде разное количество, то будем действовать по принципу внутреннего объединения исходя из наличия значений температуры нагрева. То есть присоединять будем к таблице data_temp.

## 3 Предобработка данных

Сначала обработаем данные для каждого датасета, а затем объединим их в единую таблицу.

### 3.1 Данные об электродах


```python
# удалим значения key начиная с 2500 партии

data_arc = data_arc[data_arc['key']<2500]
```


```python
# переименуем столбцы

data_arc.columns = ['key', 'start_heat', 'end_heat', 'active_power', 'reactive_power']
```


```python
# проверим преобразобвания

data_arc.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>start_heat</th>
      <th>end_heat</th>
      <th>active_power</th>
      <th>reactive_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11444</th>
      <td>2498</td>
      <td>2019-08-10 13:20:18</td>
      <td>2019-08-10 13:20:59</td>
      <td>0.508786</td>
      <td>0.363916</td>
    </tr>
    <tr>
      <th>11445</th>
      <td>2499</td>
      <td>2019-08-10 13:33:31</td>
      <td>2019-08-10 13:38:43</td>
      <td>0.700677</td>
      <td>0.534767</td>
    </tr>
    <tr>
      <th>11446</th>
      <td>2499</td>
      <td>2019-08-10 13:41:44</td>
      <td>2019-08-10 13:45:09</td>
      <td>0.333776</td>
      <td>0.269695</td>
    </tr>
    <tr>
      <th>11447</th>
      <td>2499</td>
      <td>2019-08-10 13:46:38</td>
      <td>2019-08-10 13:51:33</td>
      <td>0.406252</td>
      <td>0.263303</td>
    </tr>
    <tr>
      <th>11448</th>
      <td>2499</td>
      <td>2019-08-10 13:55:06</td>
      <td>2019-08-10 13:56:17</td>
      <td>0.296379</td>
      <td>0.229071</td>
    </tr>
  </tbody>
</table>
</div>




```python
# приведем начало и конец нагрева к типу данных datetime

data_arc[['start_heat', 'end_heat']] = data_arc[['start_heat', 'end_heat']].apply(pd.to_datetime)
```


```python
# проверим типы данных

data_arc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11449 entries, 0 to 11448
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   key             11449 non-null  int64         
     1   start_heat      11449 non-null  datetime64[ns]
     2   end_heat        11449 non-null  datetime64[ns]
     3   active_power    11449 non-null  float64       
     4   reactive_power  11449 non-null  float64       
    dtypes: datetime64[ns](2), float64(2), int64(1)
    memory usage: 536.7 KB



```python
# удалим отрицательное значение реактивной мощности - выброс

data_arc = data_arc[data_arc['reactive_power']>0]
```


```python
# проверим изменение

data_arc.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>active_power</th>
      <th>reactive_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11448.000000</td>
      <td>11448.000000</td>
      <td>11448.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1242.061233</td>
      <td>0.665347</td>
      <td>0.489740</td>
    </tr>
    <tr>
      <th>std</th>
      <td>718.930527</td>
      <td>0.260229</td>
      <td>0.198992</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.223895</td>
      <td>0.153777</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>623.000000</td>
      <td>0.468179</td>
      <td>0.339256</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1245.000000</td>
      <td>0.601417</td>
      <td>0.442781</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1870.000000</td>
      <td>0.833455</td>
      <td>0.611949</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2499.000000</td>
      <td>1.463773</td>
      <td>1.270284</td>
    </tr>
  </tbody>
</table>
</div>



Мы исправили все ошибки и теперь посчитаем:  
  
- продолжительность нагрева - разница конца и начала времени нагрева
- полную мощность - квадратный корень из суммы квадратов активной и реактивной мощностей



```python
# найдем продолжительность нагрева

data_arc['heating_time'] = data_arc['end_heat'] - data_arc['start_heat']
data_arc['heating_time'] = data_arc['heating_time'].dt.seconds
```


```python
# посчитаем полную мощность

data_arc['full_power'] = np.sqrt(data_arc['active_power']**2 + data_arc['reactive_power']**2)
```


```python
# проверим добавились ли расчеты в таблицу

data_arc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>start_heat</th>
      <th>end_heat</th>
      <th>active_power</th>
      <th>reactive_power</th>
      <th>heating_time</th>
      <th>full_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:02:14</td>
      <td>2019-05-03 11:06:02</td>
      <td>0.305130</td>
      <td>0.211253</td>
      <td>228</td>
      <td>0.371123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2019-05-03 11:07:28</td>
      <td>2019-05-03 11:10:33</td>
      <td>0.765658</td>
      <td>0.477438</td>
      <td>185</td>
      <td>0.902319</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2019-05-03 11:11:44</td>
      <td>2019-05-03 11:14:36</td>
      <td>0.580313</td>
      <td>0.430460</td>
      <td>172</td>
      <td>0.722536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2019-05-03 11:18:14</td>
      <td>2019-05-03 11:24:19</td>
      <td>0.518496</td>
      <td>0.379979</td>
      <td>365</td>
      <td>0.642824</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2019-05-03 11:26:09</td>
      <td>2019-05-03 11:28:37</td>
      <td>0.867133</td>
      <td>0.643691</td>
      <td>148</td>
      <td>1.079934</td>
    </tr>
  </tbody>
</table>
</div>



Поскольку замеры у нас имеются для каждого ключа несколько раз, то нам нужно сделать таблицу, где для каждого ключа будут суммарные исчисления. При этом можно убрать столюцы с началом и концом нагрева дугой.


```python
# сгруппируем данные по key и агрегируем их по суммам

data_arc_sum = data_arc.groupby('key').agg(heating_time = ('heating_time', 'sum'), 
                                           active_power = ('active_power', 'sum'),
                                           reactive_power = ('reactive_power', 'sum'),
                                           full_power = ('full_power', 'sum'))
```


```python
# посмотрим на получившуюся таблицу

data_arc_sum.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heating_time</th>
      <th>active_power</th>
      <th>reactive_power</th>
      <th>full_power</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1098</td>
      <td>3.036730</td>
      <td>2.142821</td>
      <td>3.718736</td>
    </tr>
    <tr>
      <th>2</th>
      <td>811</td>
      <td>2.139408</td>
      <td>1.453357</td>
      <td>2.588349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>655</td>
      <td>4.063641</td>
      <td>2.937457</td>
      <td>5.019223</td>
    </tr>
    <tr>
      <th>4</th>
      <td>741</td>
      <td>2.706489</td>
      <td>2.056992</td>
      <td>3.400038</td>
    </tr>
    <tr>
      <th>5</th>
      <td>869</td>
      <td>2.252950</td>
      <td>1.687991</td>
      <td>2.816980</td>
    </tr>
    <tr>
      <th>6</th>
      <td>952</td>
      <td>2.725274</td>
      <td>1.881313</td>
      <td>3.313074</td>
    </tr>
    <tr>
      <th>7</th>
      <td>673</td>
      <td>2.626877</td>
      <td>1.960419</td>
      <td>3.283913</td>
    </tr>
    <tr>
      <th>8</th>
      <td>913</td>
      <td>2.678958</td>
      <td>2.096952</td>
      <td>3.405956</td>
    </tr>
    <tr>
      <th>9</th>
      <td>625</td>
      <td>3.520820</td>
      <td>2.527365</td>
      <td>4.335261</td>
    </tr>
    <tr>
      <th>10</th>
      <td>825</td>
      <td>3.118778</td>
      <td>2.154941</td>
      <td>3.791005</td>
    </tr>
  </tbody>
</table>
</div>



**Вывод:**  
  
Мы исправили наименования столбцов, удалили выброс, убрали данные для партий с 2500, так как в них не было замеров, и сформировали новый датасет с информацией о длительности нагрева, активной и реактивной мощностях, полной мощности.

### 3.2 Данные о подаче сыпучих материалов


```python
# удалим значения key начиная с 2500 партии

data_bulk = data_bulk[data_bulk['key']<2500]
```


```python
# переименуем столбцы

data_bulk.columns = ['key', 'bulk_1', 'bulk_2', 'bulk_3', 'bulk_4', 'bulk_5', 
                     'bulk_6', 'bulk_7', 'bulk_8', 'bulk_9', 'bulk_10', 'bulk_11', 
                     'bulk_12', 'bulk_13', 'bulk_14', 'bulk_15']
```


```python
# заполним пропуски нулями в соответсвии с указаниями заказчика

data_bulk.fillna(0, inplace=True)
```


```python
# проверим преобразования

data_bulk.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>bulk_1</th>
      <th>bulk_2</th>
      <th>bulk_3</th>
      <th>bulk_4</th>
      <th>bulk_5</th>
      <th>bulk_6</th>
      <th>bulk_7</th>
      <th>bulk_8</th>
      <th>bulk_9</th>
      <th>bulk_10</th>
      <th>bulk_11</th>
      <th>bulk_12</th>
      <th>bulk_13</th>
      <th>bulk_14</th>
      <th>bulk_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2403</th>
      <td>2495</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90.0</td>
      <td>0.0</td>
      <td>256.0</td>
      <td>0.0</td>
      <td>129.0</td>
      <td>223.0</td>
    </tr>
    <tr>
      <th>2404</th>
      <td>2496</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>63.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>122.0</td>
      <td>0.0</td>
      <td>256.0</td>
      <td>0.0</td>
      <td>129.0</td>
      <td>226.0</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>2497</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>85.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>0.0</td>
      <td>124.0</td>
      <td>226.0</td>
    </tr>
    <tr>
      <th>2406</th>
      <td>2498</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>0.0</td>
      <td>206.0</td>
      <td>0.0</td>
      <td>129.0</td>
      <td>207.0</td>
    </tr>
    <tr>
      <th>2407</th>
      <td>2499</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>233.0</td>
      <td>0.0</td>
      <td>126.0</td>
      <td>227.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# проверим остались ли пропуски

data_bulk.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2408 entries, 0 to 2407
    Data columns (total 16 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   key      2408 non-null   int64  
     1   bulk_1   2408 non-null   float64
     2   bulk_2   2408 non-null   float64
     3   bulk_3   2408 non-null   float64
     4   bulk_4   2408 non-null   float64
     5   bulk_5   2408 non-null   float64
     6   bulk_6   2408 non-null   float64
     7   bulk_7   2408 non-null   float64
     8   bulk_8   2408 non-null   float64
     9   bulk_9   2408 non-null   float64
     10  bulk_10  2408 non-null   float64
     11  bulk_11  2408 non-null   float64
     12  bulk_12  2408 non-null   float64
     13  bulk_13  2408 non-null   float64
     14  bulk_14  2408 non-null   float64
     15  bulk_15  2408 non-null   float64
    dtypes: float64(15), int64(1)
    memory usage: 319.8 KB


Мы удалили данные о партиях начиная с 2500, переименовали столбцы и заполнили пропуски нулями.

### 3.3 Данные о продувке сплава газом


```python
# удалим значения key начиная с 2500 партии

data_gas = data_gas[data_gas['key']<2500]
```


```python
# переименуем столбцы

data_gas.columns = ['key', 'gas_1']
```


```python
# проверим изменения

data_gas.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>gas_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2492</th>
      <td>2495</td>
      <td>7.125735</td>
    </tr>
    <tr>
      <th>2493</th>
      <td>2496</td>
      <td>9.412616</td>
    </tr>
    <tr>
      <th>2494</th>
      <td>2497</td>
      <td>6.271699</td>
    </tr>
    <tr>
      <th>2495</th>
      <td>2498</td>
      <td>14.953657</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>2499</td>
      <td>11.336151</td>
    </tr>
  </tbody>
</table>
</div>



**Вывод:**     
  
Мы переименовали столбец с информацией о газе, удалили партии с 2500.

### 3.4 Данные с результатами измерения температуры


```python
# удалим значения key начиная с 2500 партии

data_temp = data_temp[data_temp['key']<2500]
```


```python
# переименуем столбцы

data_temp.columns = ['key', 'measurement_time', 'temperature']
```


```python
# проверим датасет

data_temp.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>measurement_time</th>
      <th>temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13921</th>
      <td>2499</td>
      <td>2019-08-10 13:33:21</td>
      <td>1569.0</td>
    </tr>
    <tr>
      <th>13922</th>
      <td>2499</td>
      <td>2019-08-10 13:41:34</td>
      <td>1604.0</td>
    </tr>
    <tr>
      <th>13923</th>
      <td>2499</td>
      <td>2019-08-10 13:46:28</td>
      <td>1593.0</td>
    </tr>
    <tr>
      <th>13924</th>
      <td>2499</td>
      <td>2019-08-10 13:54:56</td>
      <td>1588.0</td>
    </tr>
    <tr>
      <th>13925</th>
      <td>2499</td>
      <td>2019-08-10 13:58:58</td>
      <td>1603.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# приведем начало и конец нагрева к типу данных datetime

data_temp['measurement_time'] = data_temp['measurement_time'].apply(pd.to_datetime)
```


```python
# проверим типы данных

data_temp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 13926 entries, 0 to 13925
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   key               13926 non-null  int64         
     1   measurement_time  13926 non-null  datetime64[ns]
     2   temperature       13926 non-null  float64       
    dtypes: datetime64[ns](1), float64(1), int64(1)
    memory usage: 435.2 KB



```python
# оставим в таблице только первый и последний замер для каждого ключа

data_temp_edges = data_temp.groupby(by = 'key', as_index = False).agg(['first', 'last'])
```


```python
# проверим получившуюся таблицу

data_temp_edges.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">measurement_time</th>
      <th colspan="2" halign="left">temperature</th>
    </tr>
    <tr>
      <th></th>
      <th>first</th>
      <th>last</th>
      <th>first</th>
      <th>last</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2019-05-03 11:02:04</td>
      <td>2019-05-03 11:30:38</td>
      <td>1571.0</td>
      <td>1613.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-05-03 11:34:04</td>
      <td>2019-05-03 11:55:09</td>
      <td>1581.0</td>
      <td>1602.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-05-03 12:06:44</td>
      <td>2019-05-03 12:35:57</td>
      <td>1596.0</td>
      <td>1599.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05-03 12:39:27</td>
      <td>2019-05-03 12:59:47</td>
      <td>1601.0</td>
      <td>1625.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-05-03 13:11:03</td>
      <td>2019-05-03 13:36:39</td>
      <td>1576.0</td>
      <td>1602.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# переименуем столбцы, чтобы избавиться от второй линии в названиях 

data_temp_edges.columns = ['time_first', 'time_last', 'temp_first', 'temp_last']
```


```python
# проверим получившуюся таблицу

data_temp_edges.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_first</th>
      <th>time_last</th>
      <th>temp_first</th>
      <th>temp_last</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2019-05-03 11:02:04</td>
      <td>2019-05-03 11:30:38</td>
      <td>1571.0</td>
      <td>1613.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-05-03 11:34:04</td>
      <td>2019-05-03 11:55:09</td>
      <td>1581.0</td>
      <td>1602.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-05-03 12:06:44</td>
      <td>2019-05-03 12:35:57</td>
      <td>1596.0</td>
      <td>1599.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05-03 12:39:27</td>
      <td>2019-05-03 12:59:47</td>
      <td>1601.0</td>
      <td>1625.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-05-03 13:11:03</td>
      <td>2019-05-03 13:36:39</td>
      <td>1576.0</td>
      <td>1602.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим остались ли выбросы в сведенных данных

data_temp_edges.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp_first</th>
      <th>temp_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2477.000000</td>
      <td>2477.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1588.405733</td>
      <td>1595.334275</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29.232904</td>
      <td>16.019339</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1191.000000</td>
      <td>1541.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1572.000000</td>
      <td>1587.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1588.000000</td>
      <td>1593.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1605.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1679.000000</td>
      <td>1700.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# посмотрим график ящик с усами для начальных замеров на предмет выбросов

data_temp_edges.boxplot(column='temp_first', grid= True, figsize=(11,10));
```


    
![png](output_117_0.png)
    



```python
# посмотрим график ящик с усами для конечных замеров на предмет выбросов

data_temp_edges.boxplot(column='temp_last', grid= True, figsize=(11,10));
```


    
![png](output_118_0.png)
    


Выбросы есть в минамальных замерах меньше 1500 градусов, по условию такие партии можно удалить, так как при этой температура металл не плавится. 


```python
# удалим выбросы в замерах меньше 1500 градусов

data_temp_edges = data_temp_edges[data_temp_edges['temp_first']>1500]
```


```python
# посмотрим на данные после удаления выбросов

data_temp_edges.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp_first</th>
      <th>temp_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2473.000000</td>
      <td>2473.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1589.016175</td>
      <td>1595.338051</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24.996127</td>
      <td>16.031388</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1519.000000</td>
      <td>1541.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1572.000000</td>
      <td>1587.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1588.000000</td>
      <td>1593.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1605.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1679.000000</td>
      <td>1700.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Вывод:**  
  
Мы удалили все партии с 2500, оставили только первые и последние замеры темературы, избавились от выбросов.  Данные с измерением температуры готовы. 

### 3.5 Данные о проволочных материалах


```python
# удалим значения key начиная с 2500 партии

data_wire = data_wire[data_wire['key']<2500]
```


```python
# переименуем столбцы

data_wire.columns = ['key', 'wire_1', 'wire_2', 'wire_3', 'wire_4', 
                     'wire_5', 'wire_6', 'wire_7', 'wire_8', 'wire_9']
```


```python
# заполним пропуски нулями в соответсвии с указаниями заказчика

data_wire.fillna(0, inplace=True)
```


```python
# проверим преобразования

data_wire.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>wire_1</th>
      <th>wire_2</th>
      <th>wire_3</th>
      <th>wire_4</th>
      <th>wire_5</th>
      <th>wire_6</th>
      <th>wire_7</th>
      <th>wire_8</th>
      <th>wire_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2368</th>
      <td>2495</td>
      <td>89.150879</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2369</th>
      <td>2496</td>
      <td>114.179527</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2370</th>
      <td>2497</td>
      <td>94.086723</td>
      <td>9.04800</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2371</th>
      <td>2498</td>
      <td>118.110717</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2372</th>
      <td>2499</td>
      <td>110.160958</td>
      <td>50.00528</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# проверим остались ли пропуски

data_wire.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2373 entries, 0 to 2372
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   key     2373 non-null   int64  
     1   wire_1  2373 non-null   float64
     2   wire_2  2373 non-null   float64
     3   wire_3  2373 non-null   float64
     4   wire_4  2373 non-null   float64
     5   wire_5  2373 non-null   float64
     6   wire_6  2373 non-null   float64
     7   wire_7  2373 non-null   float64
     8   wire_8  2373 non-null   float64
     9   wire_9  2373 non-null   float64
    dtypes: float64(9), int64(1)
    memory usage: 203.9 KB


**Вывод:**  
  
Мы удалили информацию о партиях с 2500, заполнили пропуски нулями и привели наименования столбцов к змеиному регистру.

### 3.6 Объединение таблиц

Данные во всех датасетах готовы и теперь их можно объединить в одну таблицу для дальнейшей работы с признаками. Объединять будем по принципу внутреннего объединения и присоединение будет к таблицу о замерах температуру по номерам партий.


```python
# для таблиц о замерах температура и эдектродах key с индекса на столбец для удобства объединения

data_arc_sum = data_arc_sum.rename_axis('key').reset_index()
data_temp_edges = data_temp_edges.rename_axis('key').reset_index()
```


```python
# проверим результат

data_arc_sum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>heating_time</th>
      <th>active_power</th>
      <th>reactive_power</th>
      <th>full_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1098</td>
      <td>3.036730</td>
      <td>2.142821</td>
      <td>3.718736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>811</td>
      <td>2.139408</td>
      <td>1.453357</td>
      <td>2.588349</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>655</td>
      <td>4.063641</td>
      <td>2.937457</td>
      <td>5.019223</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>741</td>
      <td>2.706489</td>
      <td>2.056992</td>
      <td>3.400038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>869</td>
      <td>2.252950</td>
      <td>1.687991</td>
      <td>2.816980</td>
    </tr>
  </tbody>
</table>
</div>




```python
# проверим результат

data_temp_edges.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>time_first</th>
      <th>time_last</th>
      <th>temp_first</th>
      <th>temp_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:02:04</td>
      <td>2019-05-03 11:30:38</td>
      <td>1571.0</td>
      <td>1613.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2019-05-03 11:34:04</td>
      <td>2019-05-03 11:55:09</td>
      <td>1581.0</td>
      <td>1602.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2019-05-03 12:06:44</td>
      <td>2019-05-03 12:35:57</td>
      <td>1596.0</td>
      <td>1599.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2019-05-03 12:39:27</td>
      <td>2019-05-03 12:59:47</td>
      <td>1601.0</td>
      <td>1625.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2019-05-03 13:11:03</td>
      <td>2019-05-03 13:36:39</td>
      <td>1576.0</td>
      <td>1602.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# присоединяем к датасету о температурных замерах данные об электродах

data = data_temp_edges.merge(data_arc_sum, on='key', how='inner')
data = data.merge(data_bulk, on='key', how='inner')
data = data.merge(data_gas, on='key', how='inner')
data = data.merge(data_wire, on='key', how='inner')
```


```python
# посмотрим на получившуюся таблицу

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>time_first</th>
      <th>time_last</th>
      <th>temp_first</th>
      <th>temp_last</th>
      <th>heating_time</th>
      <th>active_power</th>
      <th>reactive_power</th>
      <th>full_power</th>
      <th>bulk_1</th>
      <th>...</th>
      <th>gas_1</th>
      <th>wire_1</th>
      <th>wire_2</th>
      <th>wire_3</th>
      <th>wire_4</th>
      <th>wire_5</th>
      <th>wire_6</th>
      <th>wire_7</th>
      <th>wire_8</th>
      <th>wire_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:02:04</td>
      <td>2019-05-03 11:30:38</td>
      <td>1571.0</td>
      <td>1613.0</td>
      <td>1098</td>
      <td>3.036730</td>
      <td>2.142821</td>
      <td>3.718736</td>
      <td>0.0</td>
      <td>...</td>
      <td>29.749986</td>
      <td>60.059998</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2019-05-03 11:34:04</td>
      <td>2019-05-03 11:55:09</td>
      <td>1581.0</td>
      <td>1602.0</td>
      <td>811</td>
      <td>2.139408</td>
      <td>1.453357</td>
      <td>2.588349</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.555561</td>
      <td>96.052315</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2019-05-03 12:06:44</td>
      <td>2019-05-03 12:35:57</td>
      <td>1596.0</td>
      <td>1599.0</td>
      <td>655</td>
      <td>4.063641</td>
      <td>2.937457</td>
      <td>5.019223</td>
      <td>0.0</td>
      <td>...</td>
      <td>28.554793</td>
      <td>91.160157</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2019-05-03 12:39:27</td>
      <td>2019-05-03 12:59:47</td>
      <td>1601.0</td>
      <td>1625.0</td>
      <td>741</td>
      <td>2.706489</td>
      <td>2.056992</td>
      <td>3.400038</td>
      <td>0.0</td>
      <td>...</td>
      <td>18.841219</td>
      <td>89.063515</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2019-05-03 13:11:03</td>
      <td>2019-05-03 13:36:39</td>
      <td>1576.0</td>
      <td>1602.0</td>
      <td>869</td>
      <td>2.252950</td>
      <td>1.687991</td>
      <td>2.816980</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.413692</td>
      <td>89.238236</td>
      <td>9.11456</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
# посмотрим на размеры

data.shape
```




    (2325, 34)




```python
# посомтрим на данные и описание

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2325 entries, 0 to 2324
    Data columns (total 34 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   key             2325 non-null   int64         
     1   time_first      2325 non-null   datetime64[ns]
     2   time_last       2325 non-null   datetime64[ns]
     3   temp_first      2325 non-null   float64       
     4   temp_last       2325 non-null   float64       
     5   heating_time    2325 non-null   int64         
     6   active_power    2325 non-null   float64       
     7   reactive_power  2325 non-null   float64       
     8   full_power      2325 non-null   float64       
     9   bulk_1          2325 non-null   float64       
     10  bulk_2          2325 non-null   float64       
     11  bulk_3          2325 non-null   float64       
     12  bulk_4          2325 non-null   float64       
     13  bulk_5          2325 non-null   float64       
     14  bulk_6          2325 non-null   float64       
     15  bulk_7          2325 non-null   float64       
     16  bulk_8          2325 non-null   float64       
     17  bulk_9          2325 non-null   float64       
     18  bulk_10         2325 non-null   float64       
     19  bulk_11         2325 non-null   float64       
     20  bulk_12         2325 non-null   float64       
     21  bulk_13         2325 non-null   float64       
     22  bulk_14         2325 non-null   float64       
     23  bulk_15         2325 non-null   float64       
     24  gas_1           2325 non-null   float64       
     25  wire_1          2325 non-null   float64       
     26  wire_2          2325 non-null   float64       
     27  wire_3          2325 non-null   float64       
     28  wire_4          2325 non-null   float64       
     29  wire_5          2325 non-null   float64       
     30  wire_6          2325 non-null   float64       
     31  wire_7          2325 non-null   float64       
     32  wire_8          2325 non-null   float64       
     33  wire_9          2325 non-null   float64       
    dtypes: datetime64[ns](2), float64(30), int64(2)
    memory usage: 635.7 KB


**Вывод:**  
  
Общая таблица готова. У нас 34 столбца и 2325 строк. Наименования и типы данных верные. Пропусков нет. Лишние данные удалены. 

### 3.7 Вывод

На этом этапе мы обработали данные всех имеющихся таблиц:  
  
- в data_arc: исправили наименования столбцов, удалили отрицательное значение (выброс) в реактивной мощности, убрали данные для партий с 2500, так как в них не было замеров, сформировали новый датасет с информацией о длительности нагрева, активной и реактивной мощностях, полной мощности
- в data_bulk: удалили данные о партиях начиная с 2500, переименовали столбцы и заполнили пропуски нулями
- в data_gas: переименовали столбец с информацией о газе, удалили партии с 2500
- в data_temp: удалили все партии с 2500, оставили только первые и последние замеры темературы, избавились от выбросов
- в data_wire: удалили информацию о партиях с 2500, заполнили пропуски нулями и привели наименования столбцов к змеиному регистру  
   
После подготовки данных мы объединили их в единую таблицу для дальнейшей работы с признаками. Переходим к анализу признаков и подготовке выборок для обучения моделей.

## 4 Анализ и подготовка признаков

На этом шага будет работать с признаками: посмотрим распределние признаков на гистограмме, проверим корреляцию, удалим не коррелериющие признаки и поделим датасет на признаки и целевой признак, а так же отделим обучающую и тестовую выборки.

### 4.1 Анализ распределения и корреляции признаков


```python
# посмотрим на сводную таблицу

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>time_first</th>
      <th>time_last</th>
      <th>temp_first</th>
      <th>temp_last</th>
      <th>heating_time</th>
      <th>active_power</th>
      <th>reactive_power</th>
      <th>full_power</th>
      <th>bulk_1</th>
      <th>...</th>
      <th>gas_1</th>
      <th>wire_1</th>
      <th>wire_2</th>
      <th>wire_3</th>
      <th>wire_4</th>
      <th>wire_5</th>
      <th>wire_6</th>
      <th>wire_7</th>
      <th>wire_8</th>
      <th>wire_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2019-05-03 11:02:04</td>
      <td>2019-05-03 11:30:38</td>
      <td>1571.0</td>
      <td>1613.0</td>
      <td>1098</td>
      <td>3.036730</td>
      <td>2.142821</td>
      <td>3.718736</td>
      <td>0.0</td>
      <td>...</td>
      <td>29.749986</td>
      <td>60.059998</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2019-05-03 11:34:04</td>
      <td>2019-05-03 11:55:09</td>
      <td>1581.0</td>
      <td>1602.0</td>
      <td>811</td>
      <td>2.139408</td>
      <td>1.453357</td>
      <td>2.588349</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.555561</td>
      <td>96.052315</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2019-05-03 12:06:44</td>
      <td>2019-05-03 12:35:57</td>
      <td>1596.0</td>
      <td>1599.0</td>
      <td>655</td>
      <td>4.063641</td>
      <td>2.937457</td>
      <td>5.019223</td>
      <td>0.0</td>
      <td>...</td>
      <td>28.554793</td>
      <td>91.160157</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2019-05-03 12:39:27</td>
      <td>2019-05-03 12:59:47</td>
      <td>1601.0</td>
      <td>1625.0</td>
      <td>741</td>
      <td>2.706489</td>
      <td>2.056992</td>
      <td>3.400038</td>
      <td>0.0</td>
      <td>...</td>
      <td>18.841219</td>
      <td>89.063515</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2019-05-03 13:11:03</td>
      <td>2019-05-03 13:36:39</td>
      <td>1576.0</td>
      <td>1602.0</td>
      <td>869</td>
      <td>2.252950</td>
      <td>1.687991</td>
      <td>2.816980</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.413692</td>
      <td>89.238236</td>
      <td>9.11456</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
# посмотрим распределние значений на гистограммах

data.hist(figsize=(50,45), bins=20);
```


    
![png](output_146_0.png)
    


По графикам видно, что в болбьшинстве случаев распределние нормальное, но так же есть сыпучие смеси и проволочные материалы, которые добавлялись один или малое количество раз. Посмотрим, как они коррелируют с другими признаками и с целевым признаком.


```python
# посмотрим тепловую карту корреляций признаков 

f, ax = plt.subplots(figsize=(30, 25))

corr = data.corr()
heatmap = sns.heatmap(corr,
                      xticklabels=True, 
                      yticklabels=True, 
                      annot=True, 
                      cmap='Blues_r', 
                      square=True);
plt.title('Тепловая карта корреляции признаков', fontsize=19)
plt.show();
```


    
![png](output_148_0.png)
    


Из тепловой карты корреляции видим, что:  
  
- wire_5 полностью пустой, нет корреляции ни с чем, этот признак можно удалить
- у целевого признака temp_last нет сильной корреляции с другими признаками, с некоторыми даже есть отрицательная
- между признаками активной и реактивной мощности и полной мощность большая корреляция из-за того, как вычисляется мощность, поэтому стоит оставить только full_power
- на тепловой карте  время первого и последнего замера не обозначились, но они так же не имеют сильного значения и их можно удалить
  
Удалим лишние признаки и построем таблицу корреляции с целевым признаком для оставшихся



```python
# удалим лишние признаки

data = data.drop(['key', 'active_power', 'reactive_power', 
                  'wire_5', 'time_first', 
                  'time_last'], axis = 1)
```


```python
# проверим результат

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp_first</th>
      <th>temp_last</th>
      <th>heating_time</th>
      <th>full_power</th>
      <th>bulk_1</th>
      <th>bulk_2</th>
      <th>bulk_3</th>
      <th>bulk_4</th>
      <th>bulk_5</th>
      <th>bulk_6</th>
      <th>...</th>
      <th>bulk_15</th>
      <th>gas_1</th>
      <th>wire_1</th>
      <th>wire_2</th>
      <th>wire_3</th>
      <th>wire_4</th>
      <th>wire_6</th>
      <th>wire_7</th>
      <th>wire_8</th>
      <th>wire_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1571.0</td>
      <td>1613.0</td>
      <td>1098</td>
      <td>3.718736</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>154.0</td>
      <td>29.749986</td>
      <td>60.059998</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1581.0</td>
      <td>1602.0</td>
      <td>811</td>
      <td>2.588349</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>73.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>154.0</td>
      <td>12.555561</td>
      <td>96.052315</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1596.0</td>
      <td>1599.0</td>
      <td>655</td>
      <td>5.019223</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>153.0</td>
      <td>28.554793</td>
      <td>91.160157</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1601.0</td>
      <td>1625.0</td>
      <td>741</td>
      <td>3.400038</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>154.0</td>
      <td>18.841219</td>
      <td>89.063515</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1576.0</td>
      <td>1602.0</td>
      <td>869</td>
      <td>2.816980</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>152.0</td>
      <td>5.413692</td>
      <td>89.238236</td>
      <td>9.11456</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# посмотрим тепловую карту корреляций признаков после удаления лишнего

f, ax = plt.subplots(figsize=(30, 25))
corr = data.corr()
heatmap = sns.heatmap(corr,
                      xticklabels=True, 
                      yticklabels=True, 
                      annot=True, 
                      cmap='Blues_r', 
                      square=True);
plt.title('Тепловая карта корреляции признаков', fontsize=19)
plt.show();
```


    
![png](output_152_0.png)
    


Теперь по графику стало лучше видно зависимости:  
  
- bulk_9 и wire_8 имеют почти полную зависимость
- большая связь между bulk2 и bulk_7, wire_4, wire_6, wire_7; bulk_7 и wire_4, wire_6; wire_4 и wire_6 
- с целевым признаком temp_last связь у остальных маленькая  
  
Выведем корреляцию с последним измерением температуры отдельно


```python
# выведем тепловую карту корреляции всех признаков с последним замером температуры

f,ax = plt.subplots(figsize=(20,2))

num = data.select_dtypes(exclude='object')
numcorr = num.corr()

sns.heatmap(numcorr.sort_values(by=['temp_last'], ascending=False).head(1), 
            cmap='Blues_r', 
            annot=True)
plt.title('Корреляция признаков с последним измерением температуры', fontsize=12)
plt.show();
```


    
![png](output_154_0.png)
    


Корреляция не большая, но все-таки прослеживается между последним замером температуры и первым замером, полной мощностью, сыпучими смесями 4, 12, 14, 15, проволочной 1.  
  
С признаками wire_8 и bulk_9 последний замер коррелирует в отрицательной и очень слабой зависимости. Выведем описание и решим, какой удалить для дальнейшей работы в линейных моделях. 


```python
data[['bulk_9', 'wire_8']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bulk_9</th>
      <th>wire_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2325.000000</td>
      <td>2325.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.445591</td>
      <td>0.311847</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.225965</td>
      <td>4.406475</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>147.000000</td>
      <td>102.762401</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[['bulk_9', 'wire_8']].hist(figsize=(16,6));
```


    
![png](output_157_0.png)
    


Удалим wire_8


```python
# удалим проволочный материал 8

data = data.drop(['wire_8'], axis=1)
```


```python
# проверим удаление 

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2325 entries, 0 to 2324
    Data columns (total 27 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   temp_first    2325 non-null   float64
     1   temp_last     2325 non-null   float64
     2   heating_time  2325 non-null   int64  
     3   full_power    2325 non-null   float64
     4   bulk_1        2325 non-null   float64
     5   bulk_2        2325 non-null   float64
     6   bulk_3        2325 non-null   float64
     7   bulk_4        2325 non-null   float64
     8   bulk_5        2325 non-null   float64
     9   bulk_6        2325 non-null   float64
     10  bulk_7        2325 non-null   float64
     11  bulk_8        2325 non-null   float64
     12  bulk_9        2325 non-null   float64
     13  bulk_10       2325 non-null   float64
     14  bulk_11       2325 non-null   float64
     15  bulk_12       2325 non-null   float64
     16  bulk_13       2325 non-null   float64
     17  bulk_14       2325 non-null   float64
     18  bulk_15       2325 non-null   float64
     19  gas_1         2325 non-null   float64
     20  wire_1        2325 non-null   float64
     21  wire_2        2325 non-null   float64
     22  wire_3        2325 non-null   float64
     23  wire_4        2325 non-null   float64
     24  wire_6        2325 non-null   float64
     25  wire_7        2325 non-null   float64
     26  wire_9        2325 non-null   float64
    dtypes: float64(26), int64(1)
    memory usage: 508.6 KB


**Вывод:**    
  
На этом шаге мы построили тепловую карту корреляции признаков и:  
  
- удалили столбцы key, heating_time, active_power, reactive_power, wire_5, time_first, time_last, так как у них либо не было корреляции либо она была очень сильная между собой и full_power
- обнаружили, что большая связь между bulk2 и bulk_7, wire_4, wire_6, wire_7; bulk_7 и wire_4, wire_6; wire_4 и wire_6
- увидели, что bulk_9 и wire_8 имеют почти полную зависимость и проверив их оба, удалили wire_8,  чтобы их свзяь не мешала работе линейных моделей
- узнали, что корреляция не большая, но все-таки прослеживается между последним замером температуры и первым замером, полной мощностью, сыпучими смесями 4, 12, 14, 15, проволочной 1.  
  
Теперь можно переходить к разделению признаков и созданию выборок.

### 4.2 Разделение на признаки и целевой признак


```python
# выделим признаки и целевой признак

features = data.drop(['temp_last'], axis=1)
target = data['temp_last']
```


```python
# разобьем на обучающую и тестовую выборки

features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target,
                                                                            test_size=.25,
                                                                            random_state=RANDOM_STATE)
```


```python
# проверим размеры выборок

for data in [features_train, features_test, target_train, target_test]:
    print(data.shape)
```

    (1743, 26)
    (582, 26)
    (1743,)
    (582,)


Данные разбиты на признаки и целевой признак, поделены на обучающую и тестовую выборку. Поскольку мы имеем числовые данные, то их следует стандартизировать. 


```python
# стандартизируем численные признаки в обучающей выборке

scaler = StandardScaler()

scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)
```

### 4.3 Вывод

На этом этом этапе мы проанализировали общую таблицу, создали тепловые карты и разделили данные на признаки и выборки. В процессе работы выводы были сделаны следующие:    
  
- большая связь между bulk2 и bulk_7, wire_4, wire_6, wire_7; bulk_7 и wire_4, wire_6; wire_4 и wire_6
- корреляция не большая, но все-таки прослеживается между последним замером температуры и первым замером, полной мощностью, сыпучими смесями 4, 12, 14, 15, проволочной 1.  
  
Данные готовы к работе с моделями машинного обучения.

## 5 Обучение моделей

На этом этапе мы обучим модели и выберем среди них лучшую. 

### 5.1 Линейная регрессия


```python
%%time

# обучим модель линейной регрессии с кросс-валидацией

model_lin_reg = LinearRegression()

scores_lin_reg = cross_validate(model_lin_reg,
                                features_train, 
                                target_train, 
                                scoring=('neg_mean_absolute_error', 'r2'),
                                cv=7)

mae_lin_reg = np.mean(abs(scores_lin_reg['test_neg_mean_absolute_error']))
r2_lin_reg = np.mean(scores_lin_reg['test_r2'])

print()
print('MAE:', round(mae_lin_reg, 4))
print('R2:', round(r2_lin_reg,4))
print()
```

    
    MAE: 6.3347
    R2: 0.3932
    
    CPU times: user 121 ms, sys: 181 ms, total: 303 ms
    Wall time: 228 ms


Для линейной регрессии мы получили MAE = 6.3347, метрика R2 = 0.3932. Это означает, что модель в принципе справиться с новыми неизвестными данными, но средне. 

### 5.2 Модель случайный лес в регрессии


```python
%%time

# обучим модель случайный лес

model_forest = RandomForestRegressor(random_state=RANDOM_STATE)

parameteres = {'n_estimators': [100, 300, 500], 
               'max_depth': [2, 4, 6], 
               'min_samples_split': [2, 5, 8]}

scoring=['neg_mean_absolute_error', 'r2']

grid_forest = GridSearchCV(model_forest, 
                           parameteres,                     
                           scoring=scoring, 
                           refit='neg_mean_absolute_error',
                           cv=7)

grid_forest.fit(features_train, target_train)

mae_forest = np.mean(abs(grid_forest.best_score_))
r2_forest = np.mean(abs(grid_forest.cv_results_['mean_test_r2']))

print()
print('MAE:', round(mae_forest, 4))
print('R2:', round(r2_forest, 4))
print('Параметры лучшей модели:', grid_forest.best_params_)
print()
```

    
    MAE: 6.3114
    R2: 0.3226
    Параметры лучшей модели: {'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 100}
    
    CPU times: user 3min 29s, sys: 1.13 s, total: 3min 30s
    Wall time: 3min 30s


Для модели случайного леса в регрессии мы получили MAE = 6.3114 и R2 = 0.3226

### 5.3 CatBoost


```python
%%time

# обучим модель CatBoostRegressor

model_cat = CatBoostRegressor(random_state=RANDOM_STATE)

parameteres = ({'learning_rate': [0.1, 0.3, 0.4],
                'verbose': [200],
                'n_estimators': [100]})

scoring=['neg_mean_absolute_error', 'r2']

grid = GridSearchCV(model_cat, 
                    parameteres,
                    scoring=scoring, 
                    refit='neg_mean_absolute_error',
                    cv=7)
    
grid.fit(features_train, target_train)

mae_cat = np.mean(abs(grid.best_score_))
r2_cat = np.mean(abs(grid.cv_results_['mean_test_r2']))

print()
print('MAE:', round(mae_cat, 4))
print('R2:', round(r2_cat, 4))
print('Параметры лучшей модели:', grid.best_params_)
print()
```

    0:	learn: 10.9803094	total: 50.8ms	remaining: 5.03s
    99:	learn: 6.6503941	total: 344ms	remaining: 0us
    0:	learn: 11.0533544	total: 4.2ms	remaining: 416ms
    99:	learn: 6.6532851	total: 329ms	remaining: 0us
    0:	learn: 10.6976602	total: 3.58ms	remaining: 354ms
    99:	learn: 6.6071027	total: 289ms	remaining: 0us
    0:	learn: 10.9765775	total: 3.42ms	remaining: 339ms
    99:	learn: 6.7252128	total: 303ms	remaining: 0us
    0:	learn: 10.9269815	total: 3.17ms	remaining: 314ms
    99:	learn: 6.7069740	total: 311ms	remaining: 0us
    0:	learn: 10.9290143	total: 3.27ms	remaining: 324ms
    99:	learn: 6.7150886	total: 287ms	remaining: 0us
    0:	learn: 10.8698329	total: 4.38ms	remaining: 434ms
    99:	learn: 6.5604914	total: 350ms	remaining: 0us
    0:	learn: 10.4600401	total: 3.52ms	remaining: 348ms
    99:	learn: 4.5402210	total: 305ms	remaining: 0us
    0:	learn: 10.5892739	total: 3.32ms	remaining: 329ms
    99:	learn: 4.5735653	total: 294ms	remaining: 0us
    0:	learn: 10.1809910	total: 3.13ms	remaining: 310ms
    99:	learn: 4.4023195	total: 302ms	remaining: 0us
    0:	learn: 10.4657084	total: 3.3ms	remaining: 327ms
    99:	learn: 4.4627628	total: 302ms	remaining: 0us
    0:	learn: 10.4214240	total: 3.36ms	remaining: 332ms
    99:	learn: 4.4663572	total: 308ms	remaining: 0us
    0:	learn: 10.3774798	total: 3.5ms	remaining: 347ms
    99:	learn: 4.6709650	total: 294ms	remaining: 0us
    0:	learn: 10.3041684	total: 3.4ms	remaining: 337ms
    99:	learn: 4.4393964	total: 314ms	remaining: 0us
    0:	learn: 10.2353151	total: 3.35ms	remaining: 332ms
    99:	learn: 3.8914031	total: 305ms	remaining: 0us
    0:	learn: 10.3889719	total: 3.55ms	remaining: 351ms
    99:	learn: 3.9246286	total: 319ms	remaining: 0us
    0:	learn: 9.9569699	total: 3.33ms	remaining: 329ms
    99:	learn: 3.8757927	total: 324ms	remaining: 0us
    0:	learn: 10.2462791	total: 3.28ms	remaining: 325ms
    99:	learn: 3.7032061	total: 305ms	remaining: 0us
    0:	learn: 10.2025592	total: 3.47ms	remaining: 344ms
    99:	learn: 3.9393368	total: 367ms	remaining: 0us
    0:	learn: 10.1373765	total: 5.23ms	remaining: 517ms
    99:	learn: 3.9926156	total: 314ms	remaining: 0us
    0:	learn: 10.0594510	total: 3.53ms	remaining: 349ms
    99:	learn: 3.7982587	total: 317ms	remaining: 0us
    0:	learn: 10.9249661	total: 3.58ms	remaining: 355ms
    99:	learn: 6.8164977	total: 329ms	remaining: 0us
    
    MAE: 6.1499
    R2: 0.4151
    Параметры лучшей модели: {'learning_rate': 0.1, 'n_estimators': 100, 'verbose': 200}
    
    CPU times: user 7.47 s, sys: 274 ms, total: 7.74 s
    Wall time: 52.8 s


Для модели CatBoostRegressor мы получили MAE = 6.1499 и R2 = 0.4151  

### 5.4 Константная Dummy модель


```python
%%time 

# проверим адекватность на константной модели 

model_dummy = DummyRegressor()
model_dummy.fit(features_train, target_train)
prediction_dummy = model_dummy.predict(features_test)

mae_dummy = mean_absolute_error(target_test, prediction_dummy)
r2_dummy = r2_score(target_test, prediction_dummy)

print()
print('MAE:', round(mae_dummy, 4))
print('R2:', round(r2_dummy, 4))
print()
```

    
    MAE: 8.1894
    R2: -0.0009
    
    CPU times: user 4.04 ms, sys: 1e+03 ns, total: 4.04 ms
    Wall time: 2.7 ms


Для Dummy модели мы получили MAE = 8.1894.  
Сразу проверили коэффециент детерминации, как дополнительную оценку для модели: R2 = -0.0009, что показывает - константная модель будет плохо работать с неизвестным набором данных.

### 5.5 Анализ моделей


```python
# сделаем сводную таблицу результатов

index = ['Logistic Regression',
         'Random Forest Regressor',
         'CatBoost Regressor',
         'Dummy Regressor']

data = {'MAE':[round(mae_lin_reg, 4),
              round(mae_forest, 4),
              round(mae_cat, 4),
              round(mae_dummy, 4)],
        
        'R2':[round(r2_lin_reg, 4),
              round(r2_forest, 4),
              round(r2_cat, 4),
              round(r2_dummy, 4)],}

kpi_data = pd.DataFrame(data=data, index=index)
kpi_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAE</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic Regression</th>
      <td>6.3347</td>
      <td>0.3932</td>
    </tr>
    <tr>
      <th>Random Forest Regressor</th>
      <td>6.3114</td>
      <td>0.3226</td>
    </tr>
    <tr>
      <th>CatBoost Regressor</th>
      <td>6.1499</td>
      <td>0.4151</td>
    </tr>
    <tr>
      <th>Dummy Regressor</th>
      <td>8.1894</td>
      <td>-0.0009</td>
    </tr>
  </tbody>
</table>
</div>



По таблице видно, что наилучший резлутат показывает модель CatBoostRegeressor с MAE = 6.1499 и с параметрами n_estimators = 100, learning_rate=0.1. R2 = 0.4151, что самый выскоий коэффициент детерменации среди обученных моделей. Это хороший результат в сравнении с константной моделью. Проверим модель на тестовых данных.

### 5.6 Тестирование лучшей модели


```python
%%time

# проверим выбранную модель на тестовой выборке

model_cat = CatBoostRegressor(n_estimators = 100,  
                              learning_rate=0.1,
                              random_state=RANDOM_STATE)

model_cat.fit(features_train, target_train, verbose=50)
prediction_cat = model_cat.predict(features_test)

mae_cat_test = mean_absolute_error(target_test, prediction_cat)
r2_cat_test = r2_score(target_test, prediction_cat)

print()
print('MAE:', round(mae_cat_test, 4))
print('R2:', round(r2_cat_test, 4))
print()
```

    0:	learn: 10.9249661	total: 5.65ms	remaining: 559ms
    50:	learn: 7.5389147	total: 175ms	remaining: 168ms
    99:	learn: 6.8164977	total: 331ms	remaining: 0us
    
    MAE: 6.1772
    R2: 0.4208
    
    CPU times: user 343 ms, sys: 20.1 ms, total: 363 ms
    Wall time: 1.77 s


На тестовых данных модель показала:  
  
MAE = 6.1772  
R2 = 0.4208  
  
Это очень хороший результат для метрики средней абсолютной ошибки. И приемлемый для коэффициента детерминации. Он показывает, что модель на новых неизвестных данных будут предсказывать хорошо. 

### 5.7 Анализ значимости признаков


```python
# построим график значимости признаков для модели CatBoostRegressor

feature_importance = model_cat.feature_importances_
sorted_indicies = np.argsort(feature_importance)

fig = plt.figure(figsize=(17, 13))
plt.barh(range(len(sorted_indicies)), feature_importance[sorted_indicies], align='center')
plt.yticks(range(len(sorted_indicies)), np.array(features.columns)[sorted_indicies])
plt.title('График значимости признаков');
```


    
![png](output_191_0.png)
    


Гррафик значимости признаков модели показывает:  
  
- наиболее значимые признаки: длительность нагрева, первый замер температуры, wire 1
- средне значимые признаки: wire 2, bulk 6, bulk 15, gas 1, полная мощность, bulk 14
- мало значимые признаки, но при этом явно использованные в модели: bulk 3, bulk 12, bulk 4, bulk 1, wire 6, bulk 11, wire 3, bulk 5, wire 4, bulk 10
- не значимые признаки: bulk 13, bulk 2, bulk 7, bulk 8, bulk 9, wire 7, wire 9 

  
Получается, что не все добавки в равной степени влияют на работу модели, а вот общая продолжительность нагрева, первый замер температуры и проволочная добавка wire 1 - влияют сильно. Можно так же отметить, что на тепловой карте корреляция признаков, ставших наиболее значимыми для модели, к финальному замеру температуру была самая высокая: heating time = 0.28, temp first = 0.3 и wire 1 = 0.32.

### 5.8 Вывод

Мы обучили 3 модели и получили следующие результаты:  
   
- Logistic Regression: MAE = 6.3347, R2 = 0.3932
- Random Forest Regressor: MAE = 6.3114, R2 = 0.3226
- CatBoost Regressor: MAE = 6.1499, R2 = 0.4208
- Dummy Regressor: MAE = 8.1894, R2 = -0.0009
  
Наилучший результат на обучение показала модель CatBoostRegressor с MAE = 6.1499, R2 = 0.4208 с параметрами n_estimators = 100, learning_rate=0.1.  
На тестовых данных модель показала MAE = 6.1772 и R2 = 0.4208, это хороший результат. С условием модель справилась.   
  
Анализ признаков значимости модели показал, что:  
  
- наиболее значимые признаки: длительность нагрева, первый замер температуры, wire 1
- средне значимые признаки: wire 2, bulk 6, bulk 15, gas 1, полная мощность, bulk 14
- мало значимые признаки, но при этом явно использованные в модели: bulk 3, bulk 12, bulk 4, bulk 1, wire 6, bulk 11, wire 3, bulk 5, wire 4, bulk 10
- не значимые признаки: bulk 13, bulk 2, bulk 7, bulk 8, bulk 9, wire 7, wire 9 
  
Из этого мы сделали вывод, что наилучшая модель CatBoostRegressor и наиболее значимые для нее признаки общая продолжительность нагрева, первый замер температуры и проволочная добавка wire_1.

## 6 Итоговый вывод

Нам нужно было построить модель, которая предскажет температуру стали для металлургического комбината ООО «Так закаляем сталь». Температура стали напрямую влияет на потребление электроэнергии на этапе обработки стали и вычисления необходимы, чтобы оптимизировать производственные расходы. Показателями успешности модели по задаче у нас была метрика MAE не ниже 6.8 на тестовых данных. Дополнительная метрика R2.  
  
В ходе работы мы:  
  
1. **Провели исследовательский анализ данных:**  

   На данном этапе мы рассмотрели все предоставленные датасеты и сделали по ним выводы, что:  
     
   - на предорбработке данных нужно будет поправить наименования, заполнить пропуски, удалить выбросы в реактивной мощности, а также удалить все партии с 2500, так как у них нет целевого замера - финальной температуры, и затем объединить данные в единую таблицу  
   - для объединения в таблицу нужно будет обработать значения в таблице data_temp, чтобы остались только начальная температура и конечная для каждого уникального ключа, а в таблице data_arc найти значение полной мощности и посмотреть при анализе признаков стоит ли оставить только его   
   - ключей везде разное количество, то нужно будет действовать по принципу внутреннего объединения исходя из наличия значений температуры нагрева, то есть присоединять будем к таблице data_temp
    

2. **Предобработали данные:**  
    
   На этом этапе мы сделали необходимые корректировки:  
     
   - в data_arc: исправили наименования столбцов, удалили отрицательное значение (выброс) в реактивной мощности, убрали данные для партий с 2500, так как в них не было замеров, сформировали новый датасет с информацией о длительности нагрева, активной и реактивной мощностях, полной мощности
   - в data_bulk: удалили данные о партиях начиная с 2500, переименовали столбцы и заполнили пропуски нулями
   - в data_gas: переименовали столбец с информацией о газе, удалили партии с 2500
   - в data_temp: удалили все партии с 2500, оставили только первые и последние замеры темературы, избавились от выбросов
   - в data_wire: удалили информацию о партиях с 2500, заполнили пропуски нулями и привели наименования столбцов к змеиному регистру 
   - объединили данные в единую таблицу для дальнейшей работы с признаками  
     

3. **Проанализировали признаки и подготовили их:**  
  
   На этом этом этапе мы проанализировали общую таблицу, создали тепловые карты и разделили данные на признаки и выборки. В процессе работы выводы были сделаны следующие:    
  
   - большая связь между bulk2 и bulk_7, wire_4, wire_6, wire_7; bulk_7 и wire_4, wire_6; wire_4 и wire_6
   - корреляция не большая, но все-таки прослеживается между последним замером температуры и первым замером, полной мощностью, сыпучими смесями 4, 12, 14, 15, проволочной 1.  
    
  
4. **Обучили модели:**  
  
   На этом шаге мы обучили 3 модели и получили следующие результаты:  
   
   - Logistic Regression: MAE = 6.3347, R2 = 0.3932
   - Random Forest Regressor: MAE = 6.3114, R2 = 0.3226
   - CatBoost Regressor: MAE = 6.1499, R2 = 0.4208
   - Dummy Regressor: MAE = 8.1894, R2 = -0.0009
  
   Наилучший результат на обучение показала модель CatBoostRegressor с MAE = 6.1499, R2 = 0.4208 с параметрами n_estimators = 100, learning_rate=0.1.  
   На тестовых данных модель показала MAE = 6.1772 и R2 = 0.4208, это хороший результат. С условием модель справилась.   
  
   Анализ признаков значимости модели показал, что:  
  
   - наиболее значимые признаки: длительность нагрева, первый замер температуры, wire 1 (можно так же отметить, что на тепловой карте корреляция признаков, ставших наиболее значимыми для модели, к финальному замеру температуру была самая высокая: heating time = 0.28, temp first = 0.3 и wire 1 = 0.32)
   - средне значимые признаки: wire 2, bulk 6, bulk 15, gas 1, полная мощность, bulk 14
   - мало значимые признаки, но при этом явно использованные в модели: bulk 3, bulk 12, bulk 4, bulk 1, wire 6, bulk 11, wire 3, bulk 5, wire 4, bulk 10
   - не значимые признаки: bulk 13, bulk 2, bulk 7, bulk 8, bulk 9, wire 7, wire 9 
  
   Из этого мы сделали вывод, что наилучшая модель CatBoostRegressor и наиболее значимые для нее признаки общая продолжительность нагрева, первый замер температуры и  проволочная добавка wire 1.
     
**ВЫВОД:** для предсказания температуры стали лучше всего использовать модель `CatBoostReegressor` с параметрами n_estimators = 100, learning_rate=0.1, обученную на предоставленных данных.

## 7 Отчет


Перед нами стояла задача построить модель, которая предскажет температуру стали для металлургического комбината ООО «Так закаляем сталь». Эти вычисления были необходимы предприятию, чтобы оптимизировать производственные расходы, а именно уменьшить расход потребления электроэнергии. Показателями успешности модели по задаче у нас была метрика MAE не ниже 6.8 на тестовых данных.   
  
Поставленная задача была выполнена. В ходе проекта мы ислледовали данные, предобработали их, подготовили признаки и обучили 3 модели. Затем выбрали самую удачную и проанализировали ее на тестовых данных, получив хороший результат MAE = 6.1772  
  
Подведем итог проделанной работе. 

### 7.1 Сравнение решения и плана работы над проектом

Первоначально план проекта была такой:  
  
1. Описание задачи 
2. Исследовательский анализ данных
3. Предобработка данных
4. Анализ и подготовка признаков
5. Обучение моделей
6. Итоговые выводы
7. Отчет  
  
В ходе работы мы выполнили все пункты, ничего не было пропущено, так же, как и добавлять к ним было нечего. Шаги были изначально сформированы верно.

### 7.2 Трудности проекта

Основными трудностями проекта стало большое количество пропусков в данных, отсутствие замеров после партии номер 2500 и низкая корреляция признаков с целевым признаком.  
  
Эти места мы решили следующим образом:  
   
- удалили данные после 2500 партий, что сократило выборку, но оставило нам достаточно материала для дальнейшей работы
- заполнили пропуски нулями, проконсультировавшись с заказчиком и получив утверждение на такой шаг
- низкая корреляция повлияла на метрику оценки модели - мы взяли дополнительную метрику R2, чтобы посмотреть насколько она влияет на модель, при этом стало понятно, что изначально выбор метрики MAE для оценки работы модели был верным и более точным в наших условиях.

### 7.3 Ключевые шаги решения

Вся проделанная работа была значимой и важной, при этом нужно отметить, что ключевыми шагами стали:  
  
- обработка данных из таблицы об электродах: здесь мы получили полное время нагревания смеси, которое в итоге сыграло важную роль для машинного обучение
- обработка таблицы замера температуры: здесь мы получили структуру под общую таблицу, удалив партии с пустыми замерами и оставив только первый и последний результат температур, что так же сыграло важную роль для последующей работы

### 7.4 Признаки и их обработка

После того, как мы обработали и объединили все данные в один датасет, нам было необходимо изучить признаки. Мы проанализировали общую таблицу, создали тепловые карты и разделили данные на признаки и выборки. В процессе работы выводы были сделаны следующие:

- большая связь между bulk2 и bulk_7, wire_4, wire_6, wire_7; bulk_7 и wire_4, wire_6; wire_4 и wire_6
- корреляция не большая, но все-таки прослеживается между последним замером температуры и первым замером, полной мощностью, сыпучими смесями 4, 12, 14, 15, проволочной 1
  
После выбора лучшей модели, мы ее протестировали и посмотрели на графике значимость признаков. Вот что вышло:  
  
   - наиболее значимые признаки: длительность нагрева, первый замер температуры, wire 1 (можно так же отметить, что на тепловой карте корреляция признаков, ставших наиболее значимыми для модели, к финальному замеру температуру была самая высокая: heating time = 0.28, temp first = 0.3 и wire 1 = 0.32)
   - средне значимые признаки: wire 2, bulk 6, bulk 15, gas 1, полная мощность, bulk 14
   - мало значимые признаки, но при этом явно использованные в модели: bulk 3, bulk 12, bulk 4, bulk 1, wire 6, bulk 11, wire 3, bulk 5, wire 4, bulk 10
   - не значимые признаки: bulk 13, bulk 2, bulk 7, bulk 8, bulk 9, wire 7, wire 9   
     
Большинство из признаков были изначально в исходных данных, но некоторые были получены в ходе предобработки и затем подготовки признаков:  
  
- heating_time - длительность нагрева - эти данные были получены из таблицы data_arc через расчет разница конца и начала времени нагрева и перевода получившегося времени в секунды
- temp_first - первый замер температуры для партии - были получены из таблицы data_temp путем вывода только первых и последних замеров температуры для каждой партии
- temp_last - последний замер температуры - был получен так же как и temp_first и стал ключевым признаком для обучения моделей
- full_power - полная мощность - эти данные мы получили из аблицы data_arc через расчет квадратного корня из суммы квадратов активной и реактивной мощностей


### 7.5 Итоговая модель

Итоговой моделью для предсказания температуры стали была выбрана модель `CatBoostReegressor` с параметрами n_estimators = 100, learning_rate = 0.1, random_state = 220523 и метрикой MAE = 6.1772, что полностью соответсвует поставленной задаче.
