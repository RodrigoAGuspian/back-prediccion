from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.linear_model import LinearRegression


# Carga de datos
DATA_URL = "https://www.datos.gov.co/resource/mcec-87by.csv?$limit=10000"
LOOKBACK = 5

app = FastAPI(title="TRM Forecast API")


modelo = joblib.load("modelo.pkl")
lags = joblib.load("lags.pkl")


def obtener_datos():
    df = pd.read_csv(DATA_URL)
    df["VIGENCIADESDE"] = pd.to_datetime(df["vigenciadesde"])
    df["VALOR"] = df["valor"].astype(float)
    df = df.sort_values("VIGENCIADESDE")
    return df


def crear_lags(serie, lookback=LOOKBACK):
    X, y = [], []

    for i in range(len(serie) - lookback):
        X.append(serie.iloc[i:i+lookback].values)
        y.append(serie.iloc[i+lookback])

    return np.array(X), np.array(y)

def entrenar_modelo(df):
    serie = df["VALOR"]

    X, y = crear_lags(serie)
    corte = int(len(X) * 0.8)

    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    return modelo, X_test

@app.get("/predict/7days")
def predict_week():

    ventana = lags.copy()
    predicciones = []

    for _ in range(7):
        y_pred = modelo.predict(ventana.reshape(1, -1))[0]

        predicciones.append(float(round(y_pred, 2)))

        ventana = np.roll(ventana, -1)
        ventana[-1] = y_pred

    return {
        "forecast_7d": predicciones
    }


@app.post("/retrain")
def retrain():
    global modelo, lags

    df = obtener_datos()

    modelo, X_test = entrenar_modelo(df)

    lags = X_test[-1]

    joblib.dump(modelo, "modelo.pkl")
    joblib.dump(lags, "lags.pkl")

    return {
        "status": "ok",
        "records": len(df),
        "lags_size": len(lags),
        "message": "Modelo y ventana temporal actualizados correctamente"
    }



@app.get("/")
def home():
    return {
        "status": "online",
        "endpoints": [
            "/predict/7days",
            "/retrain"
        ]
    }
