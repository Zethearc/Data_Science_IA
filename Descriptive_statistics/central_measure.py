# Medidas de tendencia central

'''

Media (Promedio)
Mediana (Dato Central)
Moda (Dato que mas se repite)

Cuando usar cual?

La media es suceptible a valores atipicos

La moda no aplica para datos numericos continuos

'''

import pandas as pd 

df = pd.read_csv('Dataset/cars.csv')
print(df.head())

# Inspeccionemos el atributo price_usd (Variable numerica continua) de los autos listados en el dataset:
    
df['price_usd'].mean()
df['price_usd'].median()
df['price_usd'].plot.hist(bins=20)

#Resulta mas interesante analizar los precios por marcas

import seaborn as sns

sns.displot(df, x = 'price_usd', hue = 'manufacturer_name')
sns.displot(df, x="price_usd", hue="engine_type")

#el histograma anterior es muy dificil de analizar, ¿donde están los autos eléctricos?

sns.displot(df, x='price_usd', hue = 'engine_type', multiple='stack')
df.groupby('engine_type').count()

# Reto inspeccionar precios de una marca y modelo particular
Q7_df = df[(df['manufacturer_name']=='Audi') & (df['model_name']=='Q7')]
sns.histplot(Q7_df, x='price_usd', hue = 'year_produced')