# Clasificacion de Especies de Iris

Proyecto Final - Mineria de Datos
Universidad de la Costa - Departamento de Ciencias de la Computacion y Electronica
Profesor: Jose Escorcia-Gutierrez, Ph.D.

---

## Integrantes del grupo

- Nombre 1
- Nombre 2

---

## Descripcion del proyecto

Este proyecto implementa un pipeline completo de mineria de datos para clasificar
flores Iris en tres especies: setosa, versicolor y virginica.

Se utilizaron cuatro variables numericas:
- Longitud del Sepalo
- Ancho del Sepalo
- Longitud del Petalo
- Ancho del Petalo

Los resultados se presentan en un dashboard interactivo desarrollado con Streamlit.

---

## Flujo de trabajo

1. Comprension de los datos
   Carga del dataset, revision con head(), info(), describe() y analisis de valores nulos.

2. Preprocesamiento
   Division del dataset en 80% entrenamiento y 20% prueba con muestreo estratificado.
   Normalizacion de variables con StandardScaler.

3. Modelado
   Entrenamiento de un clasificador Random Forest con 100 arboles de decision.
   Se eligio este algoritmo por su robustez, resistencia al sobreajuste y su capacidad
   de calcular la importancia de las variables de forma nativa.

4. Evaluacion
   Calculo de Exactitud, Precision, Sensibilidad y F1 Score.
   Visualizacion de la matriz de confusion.

5. Dashboard
   Comunicacion de todos los resultados a traves de una aplicacion Streamlit interactiva.

---

## Contenido del dashboard

Pestana 1 - Comprension de los Datos
   Muestra head(), info(), describe(), valores nulos y distribucion de clases.

Pestana 2 - Visualizacion de Datos
   Histogramas, scatter plots, heatmap de correlacion y matriz de dispersion.

Pestana 3 - Metricas del Modelo
   Exactitud, Precision, Sensibilidad, F1 Score y matriz de confusion.

Pestana 4 - Prediccion en Tiempo Real
   Sliders para ingresar medidas de la flor, prediccion automatica de la especie,
   probabilidades por clase y grafico 3D con la posicion del nuevo punto.

Pestana 5 - Importancia de Variables
   Grafica de barras con la contribucion de cada variable al modelo.

---

## Como ejecutar el proyecto

Paso 1 - Clonar el repositorio o descargar los archivos.

Paso 2 - Instalar las dependencias:

   pip install -r requirements.txt

Paso 3 - Ejecutar la aplicacion:

   streamlit run Proyect.py

Paso 4 - Abrir el navegador en la URL que aparece en la terminal.
   Normalmente es: http://localhost:8501

---

## Estructura del repositorio

Proyect.py           Aplicacion principal de Streamlit
requirements.txt     Dependencias de Python
README.md            Documentacion del proyecto

---

## Librerias utilizadas

- streamlit    Dashboard interactivo
- pandas       Manejo de datos
- numpy        Operaciones numericas
- scikit-learn Modelo de clasificacion y preprocesamiento
- plotly       Graficos interactivos 3D y 2D
- matplotlib   Graficos estaticos
- seaborn      Heatmap de correlacion
