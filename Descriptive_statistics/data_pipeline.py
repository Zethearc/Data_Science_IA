# Pipeline de procesamiento de datos numericos

'Estadistica es la ingesta de datos'

# Escalamiento lineal

'''

- Por que usarlos?
    Modelos de machine learning eficientes en el rango [-1,1]
- Hay diferentes tipos?
    max-min, Clipping, Z-score, Winsorizing, etc
- Cuando usarlos
    Data simetrica o uniformemente distribuida

'''

## Escalamiento min-max 

'''

x -> x_s (x_s = 2x-min-max / max-min)

'''

## Clipping

'''

Cortar los datos entre dos valores limite

X_s -> dentro del rango sigue ahi
    -> < colapsa el dato al del limite
    -> > colapsa el dato al del limite

'''

## Z-score

'''

{x_1, X_2, ..., X_n} -> u, o

X_s = X-u / o       

'''

# Transformacion no lineal

'''

- Por que usarlos?
    Datos fuertemente sesgados, no simetricos.
- Hay diferentes tipos?
    Logaritmos, sigmoides, polinomiales, etc.
- Cuando usarlos?
    Antes de escalamientos lineales!

'''

## Tanh(x)
## sqrt(x)

# Procesamiento de datos numericos

import timeit
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

X, y = datasets.load_diabetes(return_X_y=True)
raw = X[:, None, 2]

# Detalles del dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

# escalamiento max-min
max_raw = max(raw)
min_raw = min(raw)
scaled = (2*raw - max_raw -min_raw)/(max_raw - min_raw)

# normalización Z-score

avg = np.average(raw)
std = np.std(raw)
z_scaled = (raw - avg)/std

fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)

axs[0].hist(raw)
axs[1].hist(scaled)
axs[2].hist(z_scaled)

# modelos para entrenamiento
def train_raw():
    linear_model.LinearRegression().fit(raw, y)

def train_scaled():
    linear_model.LinearRegression().fit(scaled, y)

def train_z_scaled():
    linear_model.LinearRegression().fit(z_scaled, y)
    
# entrenamiento de modelo
raw_time = timeit.timeit(train_raw, number = 100)
scaled_time = timeit.timeit(train_scaled, number = 100)
z_scaled_time = timeit.timeit(train_z_scaled, number = 100)
print('trainning time for raw data : {} '.format(raw_time))
print('trainning time for scaled data : {}'.format(scaled_time))
print('trainning time for z_scaled data : {}'.format(z_scaled_time))

# * max-min scaling: mejor para datos uniformemente distribuidos
# * z-score scaling: mejor para datos distribuidos "normalmente" (forma de campana de gauss)\

# Transformaciones no lineales

df = pd.read_csv('cars.csv')
df.price_usd.hist()

# Transformacion con tanh(x)

p = 10000
df.price_usd.apply(lambda x: np.tanh(x/p)).hist()

# * mapear datos a una distribucion gaussiana: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html#sphx-glr-auto-examples-preprocessing-plot-map-data-to-normal-py

# Pipelines de procesamiento de datos categoricos

'Estadistica es la ingesta de datos'

# Mapeos numericos

'''

- Dummy
    Representacion compacta
    Mejor para inputs linealmente independientes
- One-hot
    Permite describir categorias no incluidas inicialmente

'''

# Procesamiento de datos categoricos

'La pregunta, podemos tratar variables numericas como categoricas?'

'''

Categoria   Dummy   One-Hot
ingles      [0,0]   [1,0,0]   
español     [0,1]   [0,1,0]
frances     [1,0]   [0,0,1]
"nan"         ?     [0,0,0]

'''

import pandas as pd 

df = pd.read_csv('cars.csv')

# Pandas dummies: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

pd.get_dummies(df['engine_type'])

# One-hot con Scikit: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features 

import sklearn.preprocessing as preprocessing
encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

encoder.fit(df[['engine_type']].values)

encoder.transform([['gasoline'],['diesel'],['aceite']]).toarray()

# Variables numericas discretas pueden ser codificadas como categoricas

encoder.fit(df[['year_produced']].values)

encoder.transform([[2016],[2009],[190]]).toarray()