import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data

def load_and_process():
    df_raw = pd.read_csv('data/raw/Steel_industry_data.csv')

    df_baking = df_raw.copy()

    df_baking.columns = df_baking.columns.str.lower()

    columns = [
        'date',
        'usage_kwh',
        'lagging_kvarh',
        'leading_kvarh',
        'co2',
        'lagging_factor',
        'leading_factor',
        'nsm',
        'weekstatus',
        'day_of_week',
        'load_type'
    ]

    df_baking.columns = columns

    cat_cols = ['weekstatus']
    df_baking[cat_cols] = df_baking[cat_cols].astype('category')

    df_baking['date'] = pd.to_datetime(df_baking['date'], format='%d/%m/%Y %H:%M')

    df_baking['hour'] = df_baking['date'].dt.hour

    # df_baking['day_of_week'] = df_baking['day_of_week']

    df_baking['load_type'] = df_baking['load_type'].replace(
        ['Light_Load', 'Medium_Load', 'Maximum_Load'], [0, 1, 2])
    
    df_baking = df_baking.drop(columns=['nsm'])

    df = df_baking.copy()

    return df


    #Metricas de los modelos
def metrics(y_test, y_hat):
    mse = round(mean_squared_error(y_test, y_hat), 2)
    r2 = round(r2_score(y_test, y_hat), 2)
    mae = round(mean_absolute_error(y_test, y_hat), 2)

    return mse, r2, mae


    #Gráfico de metricas por modelo
def display_metrics(metrics_df, name):
    fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("MSE", "R2", "MAE"))

    fig.add_trace(
        go.Scatter(x=metrics_df.index, y=metrics_df['MSE']),
    row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_df.index, y=metrics_df['R2']),
        row=1, col=2
    )
        
    fig.add_trace(
        go.Scatter(x=metrics_df.index, y=metrics_df['MAE']),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(name, fontsize=16)

#Timeseries comparison
def plot_time_series(y_test, y_hat, model_name):
    fig_series = go.Figure()

    # Serie real
    fig_series.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test,
        mode='lines',
        name='Real',
        line=dict(color='red')
    ))

    # Serie predicha
    fig_series.add_trace(go.Scatter(
        x=y_test.index,
        y=y_hat,
        mode='lines',
        name='Prediction',
        line=dict(color='green')
    ))

    # Títulos y layout
    fig_series.update_layout(
        title=f'Timeseries comparison for {model_name}',
        xaxis_title='Time',
        yaxis_title='Usage kwh',
        width=900,
        height=450,
        template='plotly_white'
    )

    st.plotly_chart(fig_series, config = {'scrollZoom': False})


df = load_and_process()

st.set_page_config(
    page_title="Steel Industry Energy Dashboard",
    layout="wide"
)

st.title("Consumo de energía en la industria del acero")
st.markdown("""
Este dashboard muestra el análisis y la predicción del **consumo de energía (kWh)** 
en la industria del acero, utilizando datos de **DAEWOO Steel Co. Ltd (Gwangyang, South Korea)** 
y una lista de diferentes modelos para realizar predicciones.
""")

st.sidebar.header("Filtros")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

mask = (
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1])
)

df_filtered = df[mask].copy()

st.subheader("KPIs del consumo")

if df_filtered.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta el rango de fechas o filtros.")
else:
    total_kwh = df_filtered["usage_kwh"].sum()
    mean_kwh = df_filtered["usage_kwh"].mean()
    max_kwh = df_filtered["usage_kwh"].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Consumo total (kWh)", f"{total_kwh:,.2f}")
    col2.metric("Consumo medio (kWh)", f"{mean_kwh:,.2f}")
    col3.metric("Consumo máximo (kWh)", f"{max_kwh:,.2f}")

    total_co2 = df_filtered["co2"].sum()
    st.caption(f"Emisiones totales de CO2 en el periodo filtrado: {total_co2:,.2f} tCO2")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Serie de tiempo",
    "Patrones operativos",
    "Modelo",
    "Predicciones",
    "CO2 y sostenibilidad"
])


# Serie de tiempo
with tab1:
    st.subheader("Evolución del consumo de energía en el tiempo")

    if df_filtered.empty:
        st.info("No hay datos para mostrar.")
    else:
        df_daily = df_filtered.set_index("date").resample("D")["usage_kwh"].sum().reset_index()

        fig_ts = px.line(
            df_daily,
            x="date",
            y="usage_kwh",
            title="Consumo diario de energía (kWh)"
        )
        fig_ts.update_layout(xaxis_title="Fecha", yaxis_title="Usage_kWh")
        st.plotly_chart(fig_ts, use_container_width=True)

# Patrones Operativos
with tab2:
    st.subheader("Patrones por día y hora")

    if df_filtered.empty:
        st.info("No hay datos para mostrar.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig_box_dow = px.box(
                df_filtered,
                x="day_of_week",
                y="usage_kwh",
                title="Distribución de consumo por día de la semana"
            )
            fig_box_dow.update_layout(xaxis_title="Día de la Semana", yaxis_title="Usage_kWh")
            st.plotly_chart(fig_box_dow, use_container_width=True)

        with col2:
            fig_box_ws = px.box(
                df_filtered,
                x="weekstatus",
                y="usage_kwh",
                title="Distribución de consumo: Fin de semana vs Entre semana"
            )
            fig_box_ws.update_layout(xaxis_title="WeekStatus", yaxis_title="Usage_kWh")
            st.plotly_chart(fig_box_ws, use_container_width=True)

        st.markdown("---")
        st.subheader("Heatmap hora vs día")

        fig_heat = px.density_heatmap(
            df_filtered,
            x="hour",
            y="day_of_week",
            z="usage_kwh",
            histfunc="avg",
            color_continuous_scale="Viridis",
            title="Promedio de consumo por hora y día"
        )
        fig_heat.update_layout(xaxis_title="Hora del día", yaxis_title="Día de la semana")
        st.plotly_chart(fig_heat, use_container_width=True)


# Modelo
with tab3:
    st.subheader('Modelo de Machine Learning')

    df_2 = df.drop(columns=['co2', 'hour'])

    #Machine Learning
    df_train = df_2[:33600]
    df_test = df_2[33600:]

    X_train = df_train.drop(columns=['usage_kwh'])
    X_test = df_test.drop(columns=['usage_kwh'])

    y_train = df_train['usage_kwh']
    y_test = df_test['usage_kwh']

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    num_cols = X_train.select_dtypes('number').columns
    cat_cols = X_train.select_dtypes('category').columns

    cat_proc = Pipeline(steps=[
        ('cat_proc', OneHotEncoder())
    ])

    num_proc = Pipeline(steps=[
        ('num_proc', StandardScaler())
    ])

    processor = ColumnTransformer(transformers=[
        ('num', num_proc, num_cols),
        ('cat', cat_proc, cat_cols)
    ])

    lr = Pipeline(steps=[
        ('proc', processor),
        ('lr', LinearRegression())
    ])

    rdg = Pipeline(steps=[
        ('proc', processor),
        ('rdg', Ridge())
    ])

    dt = Pipeline([
        ('proc', processor),
        ('tree', DecisionTreeRegressor(random_state=2025))
    ])

    knn = Pipeline([
        ('proc', processor),
        ('knn', KNeighborsRegressor())
    ])

    rf = Pipeline([
        ('proc', processor),
        ('rf', RandomForestRegressor(random_state=2025))
    ])

    hb = Pipeline([
        ('proc', processor),
        ('hb', HistGradientBoostingRegressor(random_state=2025))
    ])


    models = [
        (lr, 'Linear Regression', 'lr'),
        (rdg, 'Ridge', 'rdg'),
        (dt, 'Decision Tree', 'tree'),
        (knn, 'K Nearest', 'knn'),
        (rf, 'Random Forest', 'rf'),
        (hb, 'Histogram G Boosting', 'hb')
    ]

    performance = {}
    st.markdown('Comparación de modelos')

    for est, name, sname in models:
        # Training
        # st.markdown(name)
        est.fit(X_train, y_train)

        # Prediction
        y_hat = est.predict(X_test)

        mse, r2, mae = metrics(y_test, y_hat)

        performance[name] = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae
        }

        plot_time_series(y_test, y_hat, name)

    metrics_df = pd.DataFrame(performance).T

    fig_plotly = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MSE", "R2", "MAE"))

    fig_plotly.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['MSE']),
        row=1, col=1
        )

    fig_plotly.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['R2']),
            row=1, col=2
        )
            
    fig_plotly.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['MAE']),
            row=1, col=3
        )
    
    fig_plotly.update_yaxes(title_text="Error", row=1, col=1)
    fig_plotly.update_yaxes(title_text="Score", row=1, col=1)
    fig_plotly.update_yaxes(title_text="Error", row=1, col=1)
    fig_plotly.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig_plotly.update_layout(title_text="Comparacón de métricas", height=700)

    st.plotly_chart(fig_plotly, config = {'scrollZoom': False})


# Predicción
with tab4:
    st.subheader('Predictor interactivo')
    
    choice = st.radio('Choose model', [model[1] for model in models])
    model = [m[0] for m in models if m[1] == choice][0]

    with st.form("my_form"):
        st.write("Predicción para uso de energía")
        lagging_kvarh = st.number_input("Lagging_kvarh", min_value=0.0, max_value=100.0)
        leading_kvarh = st.number_input("Leading_kvarh", min_value=0.0, max_value=30.0)
        lagging_factor = st.number_input("Lagging factor", min_value=0.0, max_value=100.0)
        leading_factor = st.number_input("Leading factor", min_value=0.0, max_value=100.0)
        weekstatus = st.selectbox("Selecciona weekend o weekday", ("Weekend", "Weekday"))
        load_type = st.selectbox("Load type", (0,1,2))


        submitted = st.form_submit_button("Submit")

        if submitted:
            # Crear un dataframe con los datos ingresados
            input_data = pd.DataFrame({
                "lagging_kvarh": [lagging_kvarh],
                "leading_kvarh": [leading_kvarh],
                "lagging_factor": [lagging_factor],
                "leading_factor": [leading_factor],
                "weekstatus": [weekstatus],
                "load_type": [load_type]
            })

            prediction = rf.predict(input_data)[0]

            st.write(f"Prediction: {prediction}")


# CO2
with tab5:
    st.subheader("Emisiones de CO₂ asociadas al consumo energético")

    df_co2 = df_filtered.copy()
    df_co2["date_only"] = df_co2["date"].dt.date
    df_co2_daily = df_co2.groupby("date_only")[["usage_kwh", "co2"]].sum().reset_index()
    col1, col2 = st.columns(2)

    with col1:
        fig_co2_line = px.line(
            df_co2_daily,
            x="date_only",
            y="co2",
            title="Emisiones diarias de CO₂"
        )
        fig_co2_line.update_layout(xaxis_title="Fecha", yaxis_title="tCO₂")
        st.plotly_chart(fig_co2_line, use_container_width=True)

    with col2:
        fig_co2_scatter = px.scatter(
            df_co2_daily,
            x="usage_kwh",
            y="co2",
            trendline="ols",
            title="Relación entre consumo de energía y CO2"
        )
        fig_co2_scatter.update_layout(xaxis_title="Usage_kWh", yaxis_title="tCO₂")
        st.plotly_chart(fig_co2_scatter, use_container_width=True)