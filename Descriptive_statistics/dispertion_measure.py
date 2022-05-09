# Medidas de dispersion

'''

Rango es el intervalo entre el valor máximo y el valor mínimo.

Rango intercuartil.

Desviacion estandar. (Poblacion o muestra) (u - 3o) (99.72%)


'''

# Medidas de dispersion

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cars.csv')

# Desviación estandar
df['price_usd'].std()

# Rango = valor max - valor min
rango = df['price_usd'].max() - df['price_usd'].min()

# Quartiles
median = df['price_usd'].median()
Q1 = df['price_usd'].quantile(q=0.25)
Q3 = df['price_usd'].quantile(q=0.75)
min_val = df['price_usd'].quantile(q=0)
max_val = df['price_usd'].quantile(q=1.0)
print(min_val, Q1, median, Q3, max_val)

iqr = Q3 - Q1
iqr

'''Limites para deteccion de outliers
Datos entre Q1-1.5 x IQR y Q3 + 1.5 IQR
'''
minlimit = Q1 - 1.5*iqr
maxlimit = Q3 + 1.5*iqr
print('rango para detección de outliers: {}, {}'.format(minlimit, maxlimit))

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.6, .4)})
sns.histplot(df['price_usd'], ax=ax_hist)
sns.boxplot(df['price_usd'], ax=ax_box)
ax_hist.set(xlabel='')

'''Es posible calcular varios box-plot separando por una cierta variable categorica'''

sns.boxplot(x = 'engine_fuel', y = 'price_usd', data = df)