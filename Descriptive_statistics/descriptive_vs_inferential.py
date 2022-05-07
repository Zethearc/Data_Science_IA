# Estadistica descriptiva vs inferencial.

'''

Estadisticas de un jugador

    Descriptiva: resumir historial deportivo    
    Inferencial: predecir desempeño futuro del jugador

Puedes mentir con estadistica?
    Dependera de la definicion de quien es el mejor jugador de futbol.
    No hay una definicion objetiva.
    Los diferentes estadisticos descriptivos dan nociones diferentes 
    sobre los mismo datos.
    
¿Por qué aprender estadística?
    Resumir grandes cantidades de información para tomar mejores decisiones.
    Responder preguntas con relevancia social.
    Reconocer patrones en los datos.
    Descubrir a quienes usan estas herramientas con fines nefastos.

"Con frecuencia construimos un caso estadisticos con datos imperfectos,
como resultado hay numerosas razones por las cuales individuos intelectuales
intelectuales respetables pueden no estar de acuerdo sobre los resultados estadisticos."
'El gran problema de la estadistica descriptiva' (Naked Statistics, Charles Wheelan)

'''

# Flujo de trabajo en Data Science

'''

                                                Data                        Research
Business                                        Analysis                    Scientist

                                                                            ML
Engineering                     Data                Data                    Engineer            Developer
                                Engineer            Scientis

Data            Data            Data            Model       Model       Model       Model       End user
ingestion       Visualization   Preparation     Training    Evaluation  Validation  Serving     Interface

Campos donde se usa estadistica descriptiva

Ingesta de datos, validacion (Tipos de datos, pipeline de procesamiento)

Preparacion, entrenamiento modelo (Analisis exploratorio, estadistica descriptiva, correlaciones, reducciones de datos)

Evaluacion modelo (Probabilidad e inferencia)

Modelo en produccion, Interaccion usuario final (Test de hipotesis)

'''

# Tipos de datos en estadística inferencial 

'''
Aquí tenemos en cuenta los diferentes tipos de datos o variables estructuradas en cualquier problema de ciencia de datos.

* datos categóricos: ordinales y nominales
* datos numéricos: discretos y continuos

a continuación exploraremos un dataset que contiene todos estos tipos de datos: https://www.kaggle.com/lepchenkov/usedcarscatalog
'''

import pandas as pd

df = pd.read_csv('Dataset/cars.csv')

'''

Los tipos de datos de cada columna en el dataset se pueden obtener directamente con dtypes

Aquí vemos que los tipos de datos se identifican de la siguiente manera: 

* Categoricos: `object`, `bool`
* Numéricos: `int64` (discreto), `float64` (contínuo)

Ahora, con la libreria pandas podemos generar un conjunto completo de estadisticos descriptivos 
del dataset usando `pandas.DataFrame.describe()`

'''