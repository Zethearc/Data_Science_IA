# Funciones

'''

Vamos a entender como funciona el taximetro de un taxi.

    El precio puede depender del trayecto recorrido.
    
    El precio puede depender del modelo del auto
    
    El precio puede depender del numero de personas

Una funcion la podemos tomar como una maquina

Entra un elemento X y sale un elemento Y. En caso de una
variable,

y = f(x)

# Funcion.

Es una regla donde a cada elemento de un conjunto A se le 
asigna un elemnto del conjunto B.

    Verbalmente
    Numericamente
    Visualmente
    Algebraicamente
    
# Tipos de variables
    Cualitativas
        Nominales
            Son a las que les asignamos una cualidad. LEDS
        Ordinales
            Como su nombre lo dice, representan un orden. Altura
        Binarias
            Solo toman dos valores, usualmente usadas para
            representar estados.  
    Cuantitativas
        Discretas
            Son finitas y toman ciertos valores, como los numeros
            en una tabla por ejemplo.
        Continuas
            Sus valores pueden verse como infinitos al tomar cualquier
            valor dentro de los numeros reales en un rango establecido

'''

# https://colab.research.google.com/drive/1El3FgiMicjVXqt5OlmTkYPT42svORnJg#scrollTo=n3vMv_IGHGpy
# https://colab.research.google.com/drive/1usIrJhOqONXOP1lavukVLHnjdzhSSrQQ
# https://colab.research.google.com/drive/13cqAJKL2iBSRQINlRgB_u6O5LzsC6PB1

# Percetron

'''
                x1 ->
                x2 ->
                x3 ->
Señales de      x4 ->       Union       ->      Funcion de  ->    y
entrada         x5 ->       Sumadora            activacion      Salida
                x6 ->
                x7 ->
                x8 ->
            Pesos Sinapticos

'''

# Funciones de activacion

# https://colab.research.google.com/drive/1-n9TecHaaotdLLRSASpZ72wON8-mg6Od
'''

Función lineal: mantener valores a lo largo de un proceso, predecir venta

Función escalón o de Heaviside: clasificaciones categóricas, ej. binarias

Función sigmoide: regresión logística, para probabilidades

Función tangente hiperbólica: para escalamiento

Función ReLU: simulación de “neuronas muertas”

'''

# Modelar funciones

import numpy as np
import matplotlib.pyplot as plt

x= np.array([1.2,2,3.2,2.5,5,6,4,8])
y= np.array([2,3,3.4,3.1,4,4.7,3.8,7])

fig, ax = plt.subplots()
plt.scatter(x,y, color = 'lightcoral', alpha=0.6)
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
plt.show()

# Como se calcula el error?

'''error = ^y - y'''