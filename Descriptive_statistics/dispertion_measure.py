# Medidas de dispersion

'''

Rango es el intervalo entre el valor máximo y el valor mínimo.

Rango intercuartil.

Desviacion estandar. (Poblacion o muestra) (u - 3o) (99.72%)


'''

# Medidas de dispersion

import pandas as pd 
import seaborn as sns

df = pd.read_csv('Dataset/cars.csv')

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

'''Limites para deteccion de outliers
Datos entre Q1-1.5 x IQR y Q3 + 1.5 IQR
'''
minlimit = Q1 - 1.5*iqr
maxlimit = Q3 + 1.5*iqr
print('rango para detección de outliers: {}, {}'.format(minlimit, maxlimit))

sns.histplot(df['price_usd'])
sns.boxplot(df['price_usd'])

'''Es posible calcular varios box-plot separando por una cierta variable categorica'''

sns.boxplot(x = 'engine_fuel', y = 'price_usd', data = df)

# Exploracion visual de datos

'''Una imagen vale mas que mil palabras, pero una buena imagen'''

# https://datavizproject.com/

# Diagramas de dispersion

iris = sns.load_dataset('iris')
iris.head()

# scatterplot 
sns.scatterplot(data=iris, x = 'sepal_length', y = 'petal_length', hue = 'species')

# joint plot
sns.jointplot(data=iris, x = 'sepal_length', y = 'petal_length', hue = 'species')

# box plot 
sns.boxplot(x = 'species', y = 'sepal_length', data = iris)

#Barplot
sns.barplot(x = 'species', y = 'sepal_length', data = iris)