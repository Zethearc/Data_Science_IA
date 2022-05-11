# Correlaciones

'''

¿Qué es la correlación?
    La correlación es una medida estadística que expresa hasta qué punto dos 
    variables están relacionadas linealmente (esto es, cambian conjuntamente a 
    una tasa constante).
¿Qué es la covarianza?
    Es un valor que indica el grado de variación conjunta de dos variables 
    aleatorias respecto a sus medias.
¿Qué es el coeficiente de correlación?
    El coeficiente de correlación es la medida específica que cuantifica la 
    intensidad de la relación lineal entre dos variables en un análisis de 
    correlación.

1 Perfect Positive Correlation
0.9 High positive correlation
0.5 Low positive correlation
0 No correlation
-0.5 Low negative correlation
-0.9 High negative correlation
-1 Perfect negative correlation

ro = cov / std(x)std(y)

'''

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset('iris')

sns.pairplot(iris, hue = 'species')

scaler = StandardScaler()
scaled = scaler.fit_transform(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)
scaled.T

covariance_matrix = np.cov(scaled.T)
covariance_matrix

plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
hm = sns.heatmap(covariance_matrix,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 yticklabels=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                 xticklabels=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# PCA: Analisis de componentes principales

'''

* Proyeccion de un vector

'''

# Reduccion de dimensionalidad con PCA

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


iris = sns.load_dataset('iris')

scaler = StandardScaler()
scaled = scaler.fit_transform(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    )

covariance_matrix = np.cov(scaled.T)
covariance_matrix

sns.jointplot(x= iris['petal_length'], y=iris['petal_width'])
sns.jointplot(x = scaled[:, 2], y = scaled[:,3])

# Decomposicion en vectores y valores propios

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values
eigen_vectors

variance_explained = []
for i in eigen_values:
    variance_explained.append((i/sum(eigen_values))*100)
print(variance_explained)

# PCA con skikit

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.fit(scaled)

pca.explained_variance_ratio_

reduced_scaled = pca.transform(scaled)

iris['pca_1'] = scaled[:,0]
iris['pca_2'] = scaled[:,1]
sns.jointplot(iris['pca_1'], iris['pca_2'], hue = iris['species'])