import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACION DE PAGINA
# ============================================================
st.set_page_config(
    page_title="Clasificacion de Especies de Iris",
    layout="wide"
)


# ============================================================
# CARGA DEL DATASET Y ENTRENAMIENTO DEL MODELO
# ============================================================
@st.cache_resource
def cargar_y_entrenar():
    # Paso 3: Cargar el dataset Iris
    iris = load_iris()

    nombres_columnas = [
        "Longitud Sepalo (cm)",
        "Ancho Sepalo (cm)",
        "Longitud Petalo (cm)",
        "Ancho Petalo (cm)"
    ]

    X = pd.DataFrame(iris.data, columns=nombres_columnas)
    y = iris.target
    nombres_especies = list(iris.target_names)

    df = X.copy()
    df["Especie"] = [nombres_especies[i] for i in y]

    # Paso 7: Dividir datos 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Paso 6: Normalizar los datos con StandardScaler
    escalador = StandardScaler()
    X_train_sc = escalador.fit_transform(X_train)
    X_test_sc  = escalador.transform(X_test)

    # Paso 8: Entrenar el modelo Random Forest
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_sc, y_train)

    # Paso 9: Predicciones
    y_pred = modelo.predict(X_test_sc)

    # Paso 10: Metricas de evaluacion
    metricas = {
        "Exactitud (Accuracy)":  round(accuracy_score(y_test, y_pred), 4),
        "Precision":             round(precision_score(y_test, y_pred, average="weighted"), 4),
        "Sensibilidad (Recall)": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "F1 Score":              round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    mc = confusion_matrix(y_test, y_pred)

    importancias = pd.Series(
        modelo.feature_importances_, index=nombres_columnas
    ).sort_values(ascending=False)

    return modelo, escalador, metricas, mc, df, nombres_especies, importancias, nombres_columnas


modelo, escalador, metricas, mc, df, nombres_especies, importancias, columnas = cargar_y_entrenar()

COLORES = {
    "setosa":     "#4C72B0",
    "versicolor": "#55A868",
    "virginica":  "#C44E52",
}


# ============================================================
# ENCABEZADO
# ============================================================
st.title("Clasificacion de Especies de Iris")
st.markdown(
    "Proyecto Final - Mineria de Datos  |  "
    "Universidad de la Costa  |  "
    "Profesor: Jose Escorcia-Gutierrez, Ph.D."
)
st.markdown("---")


# ============================================================
# PESTANAS PRINCIPALES
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Comprension de los Datos",
    "Visualizacion de Datos",
    "Metricas del Modelo",
    "Prediccion en Tiempo Real",
    "Importancia de Variables"
])


# ============================================================
# PESTANA 1 - COMPRENSION DE LOS DATOS (Data Understanding)
# ============================================================
with tab1:
    st.subheader("Comprension del Dataset Iris")
    st.markdown(
        "El dataset Iris contiene 150 muestras de flores clasificadas en tres especies. "
        "Cada muestra tiene cuatro variables numericas: longitud y ancho del sepalo, "
        "y longitud y ancho del petalo."
    )

    st.markdown("### Primeras filas del dataset (head)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Informacion general del dataset (info)")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total de filas", df.shape[0])
    col_info2.metric("Total de columnas", df.shape[1])
    col_info3.metric("Valores nulos", int(df.isnull().sum().sum()))

    st.markdown("#### Tipos de datos por columna")
    tipos = pd.DataFrame({
        "Columna": df.columns,
        "Tipo de dato": [str(df[c].dtype) for c in df.columns]
    })
    st.dataframe(tipos, use_container_width=True)

    st.markdown("### Estadisticas descriptivas (describe)")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### Distribucion de clases")
    conteo = df["Especie"].value_counts().reset_index()
    conteo.columns = ["Especie", "Cantidad"]
    fig_conteo = px.bar(
        conteo, x="Especie", y="Cantidad",
        color="Especie",
        color_discrete_map=COLORES,
        title="Cantidad de muestras por especie",
        text="Cantidad"
    )
    fig_conteo.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_conteo, use_container_width=True)


# ============================================================
# PESTANA 2 - VISUALIZACION DE DATOS
# ============================================================
with tab2:
    st.subheader("Visualizacion de Datos")

    st.markdown("### Histogramas por variable")
    variable_hist = st.selectbox("Selecciona una variable", columnas)
    fig_hist = px.histogram(
        df, x=variable_hist, color="Especie",
        color_discrete_map=COLORES,
        barmode="overlay",
        nbins=25,
        opacity=0.75,
        title=f"Distribucion de {variable_hist}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Scatter Plot entre dos variables")
    col_s1, col_s2 = st.columns(2)
    var_x = col_s1.selectbox("Variable eje X", columnas, index=0)
    var_y = col_s2.selectbox("Variable eje Y", columnas, index=2)
    fig_scatter = px.scatter(
        df, x=var_x, y=var_y, color="Especie",
        color_discrete_map=COLORES,
        title=f"{var_x} vs {var_y}",
        symbol="Especie",
        opacity=0.8
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Heatmap de Correlacion")
    fig_heat, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        df[columnas].corr(),
        annot=True, fmt=".2f",
        cmap="Blues", ax=ax,
        linewidths=0.5
    )
    ax.set_title("Correlacion entre variables")
    st.pyplot(fig_heat)
    plt.close()

    st.markdown("### Matriz de Dispersion (Scatter Matrix)")
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=columnas,
        color="Especie",
        color_discrete_map=COLORES,
        title="Matriz de Dispersion - Todas las variables"
    )
    fig_matrix.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.7))
    fig_matrix.update_layout(height=580)
    st.plotly_chart(fig_matrix, use_container_width=True)


# ============================================================
# PESTANA 3 - METRICAS DEL MODELO
# ============================================================
with tab3:
    st.subheader("Metricas del Modelo")
    st.markdown(
        "Se entreno un modelo **Random Forest** con 100 arboles de decision. "
        "Los datos fueron divididos en 80% entrenamiento y 20% prueba, "
        "y las variables fueron normalizadas con StandardScaler."
    )

    col1, col2, col3, col4 = st.columns(4)
    for col, (nombre, valor) in zip([col1, col2, col3, col4], metricas.items()):
        col.metric(nombre, f"{valor:.2%}")

    st.markdown("### Matriz de Confusion")
    fig_mc = px.imshow(
        mc,
        labels=dict(x="Prediccion", y="Valor Real", color="Cantidad"),
        x=[n.capitalize() for n in nombres_especies],
        y=[n.capitalize() for n in nombres_especies],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Confusion"
    )
    fig_mc.update_layout(height=400)
    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown(
        "Una matriz de confusion perfecta tiene todos los valores en la diagonal principal, "
        "lo que significa que el modelo clasifico correctamente cada muestra."
    )


# ============================================================
# PESTANA 4 - PREDICCION EN TIEMPO REAL
# ============================================================
with tab4:
    st.subheader("Prediccion en Tiempo Real")
    st.markdown("Ajusta los valores y el modelo predice la especie automaticamente.")

    col_a, col_b = st.columns(2)

    with col_a:
        long_sepalo = st.slider("Longitud del Sepalo (cm)", 4.0, 8.0, 5.8, 0.1)
        ancho_sepalo = st.slider("Ancho del Sepalo (cm)", 2.0, 4.5, 3.0, 0.1)

    with col_b:
        long_petalo = st.slider("Longitud del Petalo (cm)", 1.0, 7.0, 4.0, 0.1)
        ancho_petalo = st.slider("Ancho del Petalo (cm)", 0.1, 2.5, 1.2, 0.1)

    # Prediccion automatica al mover cualquier slider
    muestra = np.array([[long_sepalo, ancho_sepalo, long_petalo, ancho_petalo]])
    muestra_sc = escalador.transform(muestra)
    idx_pred = modelo.predict(muestra_sc)[0]
    prob_pred = modelo.predict_proba(muestra_sc)[0]
    especie_pred = nombres_especies[idx_pred]

    st.markdown("---")
    st.success(f"Especie predicha: **{especie_pred.capitalize()}**")

    df_prob = pd.DataFrame({
        "Especie":      [n.capitalize() for n in nombres_especies],
        "Probabilidad": prob_pred
    })
    fig_prob = px.bar(
        df_prob, x="Especie", y="Probabilidad",
        color="Especie",
        color_discrete_map={n.capitalize(): c for n, c in COLORES.items()},
        text_auto=".1%",
        title="Probabilidad de prediccion por clase"
    )
    fig_prob.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Grafico 3D - Posicion del nuevo punto en el dataset")
    fig_3d = go.Figure()

    for sp in nombres_especies:
        subset = df[df["Especie"] == sp]
        fig_3d.add_trace(go.Scatter3d(
            x=subset["Longitud Sepalo (cm)"],
            y=subset["Ancho Sepalo (cm)"],
            z=subset["Longitud Petalo (cm)"],
            mode="markers",
            name=sp.capitalize(),
            marker=dict(size=5, color=COLORES[sp], opacity=0.7)
        ))

    fig_3d.add_trace(go.Scatter3d(
        x=[long_sepalo], y=[ancho_sepalo], z=[long_petalo],
        mode="markers",
        name="Nueva Muestra",
        marker=dict(size=11, color="gold", symbol="diamond",
                    line=dict(width=2, color="black"))
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Long. Sepalo",
            yaxis_title="Ancho Sepalo",
            zaxis_title="Long. Petalo"
        ),
        height=540,
        legend=dict(itemsizing="constant")
    )
    st.plotly_chart(fig_3d, use_container_width=True)


# ============================================================
# PESTANA 5 - IMPORTANCIA DE VARIABLES
# ============================================================
with tab5:
    st.subheader("Importancia de las Variables")
    st.markdown(
        "El modelo Random Forest calcula automaticamente cuanto contribuye "
        "cada variable a la toma de decisiones. "
        "Las variables con mayor importancia son las que mas ayudan a separar las especies."
    )

    fig_imp = px.bar(
        x=importancias.values,
        y=importancias.index,
        orientation="h",
        labels={"x": "Importancia", "y": "Variable"},
        title="Importancia de Variables - Random Forest",
        color=importancias.values,
        color_continuous_scale="Teal"
    )
    fig_imp.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=370
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        "La longitud y el ancho del petalo son las variables mas importantes. "
        "Esto tiene sentido porque visualmente son las que mejor separan las tres especies, "
        "especialmente a Iris Setosa de las demas."
    )


# ============================================================
# PIE DE PAGINA
# ============================================================
st.markdown("---")
st.markdown(
    "Universidad de la Costa - Mineria de Datos  |  "
    "Profesor: Jose Escorcia-Gutierrez, Ph.D."
)
