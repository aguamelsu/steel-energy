# Proyecto final 
## Steel Industry Energy Consumption
## Este proyecto pretende generar un dashboard en el que se visualizaran la predicción de uso de energía de la industria del acero. 

# Dataset obtenido de:
## V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C.  

# Integrantes:  
- Agustín Arturo Melian Su
- Addán Isaí Cruz Cruz

# Problema a resolver:
## Este reporte se enfoca en estudiar el dataset “Steel Industry Energy Consumption” El trabajo citado es una colección de datos de DAEWOO Steel Co. Ltd in Gwangyang, South Korea, empresa que fabrica distintos productos de acero. La importancia de realizar un análisis a estos datos es que podremos predecir cuál podrá ser el consumo energético que es un pilar y un elemento esencial para el desarrollo de ciudades inteligentes y la eficiencia del uso de energía eléctrica. Buscamos generar predicciones en el tiempo en una empresa en la que se produzcan piezas de acero, concretamente en el uso kwh de energía, esto puede ayudar a tener mejor control sobre la previsión de consumo de energía ayudando en la reducción de costos y mejor administración de energía.

# ¿Por qué el dataset elegido?
- Consideramos que este dataset tiene una buena cantidad de predictores
- Es de una fuente confiable DAEWOO Steel Co. Ltd in Gwangyang, South Korea
- Tiene buena cantida de registros (35040)
- Tiene un paper de referencia: https://www.semanticscholar.org/paper/Efficient-energy-consumption-prediction-model-for-a-SathishkumarV-Shin/a4e10d9c93ed4b2fd89ad34e15a37eb1dc251168

# Diccionario de datos:
|date|Other|Date| | |no|
|:----|:----|:----|:----|:----|:----|
|Usage_kWh|Feature|Continuous|Industry Energy Consumption|kWh|no| Uso en kwh
|Lagging_Current_Reactive.Power_kVarh|Feature|Continuous| |kVarh|no| Potencia reactiva que se da cuando la carga es inductiva
|Leading_Current_Reactive_Power_kVarh|Feature|Continuous| |kVarh|no| Potencia reactiva con corriente en adelanto.
|CO2(tCO2)|Feature|Continuous| |ppm|no| Emisiones de CO2
|Lagging_Current_Power_Factor|Feature|Continuous| |%|no| Factor de potencia reactiva que se da cuando la carga es inductiva
|Leading_Current_Power_Factor|Feature|Continuous| |%|no| Factor de potencia reactiva con corriente en adelanto.
|NSM|Feature|Integer| |s|no| Número de segundos a partir de medianoche
|WeekStatus|Feature|Categorical|Weekend (0) or a Weekday(1)| |no| Es fin de semana o no
|Day_of_week|Feature|Categorical|Sunday, Monday, ..., Saturday| |no| Día de la semana


# Procesamiento y Feature Engineering
## Limpieza de datos

Conversión en el nombre de columnas.
Conversión de fechas a formato datetime.
Codificación de variables categóricas (load_type y weekstatus)

## Data de entrenamiento y de testeo

Dividimos los datos en una parte de entrenamiento que compone la mayor parte del año y la parte de testeo que componen las últimas dos semanas del año 2018. 


# Modelo ocupado:

Se entrenaron y evaluaron diferentes modelos usando métricas de regresión como **MSE**, **MAE** y **R²**.

Los modelos entrenados fueron: 
- Linear Regression
- Ridge
- Decision Tree
- Random Forest
- Histogram Gradient Boosting

Además se realizó una búsqueda de hiperparametros que pudo mejorar las métricas de cada modelo.

Dado el resultado que se observa en el jupyter notebook decidimos ocupar Random Forest como modelo predictor ya que es el que obtiene un resultado muy acertado con un error muy bajo.


# Conclusiones

- El consumo energético en la industria del acero tiene patrones complejos y no lineales.  
- **Random Forest** es el mejor modelo para este proyecto gracias a su robustez y precisión.
