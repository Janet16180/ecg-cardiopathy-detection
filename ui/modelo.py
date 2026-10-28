"""Modelos que usa "Registrar nuevo ECG".

Los dos salen del experimento 067 (notebook 44): gradient boosting sobre las 111 características del Avance 2,
entrenados con todas las etiquetas del entrenamiento del dataset balanceado v3 (notebook 43).
- Binario: solo el riesgo de cardiopatía.
- Multietiqueta: el mismo riesgo y la probabilidad de cada cardiopatía, con un modelo por cardiopatía. Muestra
  las que el modelo distinguió en validación (límite inferior del intervalo de 95% del AUROC por encima de 0.5).
Se elige cuál usar al registrar un ECG, y también el umbral del riesgo: el punto de 99% de sensibilidad en
validación (predeterminado) o el de 95% (PUNTOS). Cada modelo es un objeto con:

    nombre: str
    umbral: float                                  probabilidad a partir de la cual se marca riesgo de cardiopatía
    resultado(senal: np.ndarray) -> dict           senal de forma (5000, 12): 10 s a 500 Hz, en mV, derivaciones
                                                   I, II, III, aVR, aVL, aVF, V1-V6, con filtro pasa-altas de 0.5 Hz
                                                   (el formato del dataset balanceado; registro.py ya lo aplica);
                                                   devuelve {"probabilidad": de 0 a 1} y, si las tiene, "etiquetas"
    riesgo(x: pd.DataFrame) -> np.ndarray          probabilidad de cardiopatía de cada fila de variables ya
                                                   preprocesadas (la usa areas.py para las áreas de interés)

Con MODELOS vacío la UI dice que no hay modelo conectado y nunca muestra una predicción inventada.
"""
import warnings

import joblib
import numpy as np
import pandas as pd

import datos
from ecg_experiment import balanceado_v2 as v2
from ecg_experiment import caracteristicas_avance2 as ca2

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # avisos de pandas al derivar columnas

FS = 500
MUESTRAS = 10 * FS
EXPERIMENTO_067 = datos.RAIZ / "outputs/experiment067_multietiqueta_v3"


def entrada(guardado: dict, senal: np.ndarray) -> pd.DataFrame:
    """Las 111 características preprocesadas de un ECG, como en los notebooks 32 y 44."""
    return guardado["preprocesador"].transform(ca2.features_one(senal[: v2.CROP], guardado["columnas_primarias"]))


class GradientBoostingBinario:
    """Solo si hay cardiopatía: el gradient boosting binario del experimento 067."""

    nombre = "Gradient boosting binario"

    def __init__(self, guardado: dict) -> None:
        self.guardado = guardado
        self.umbral = guardado["umbral"]

    def riesgo(self, x: pd.DataFrame) -> np.ndarray:
        return self.guardado["binario"].predict_proba(x)[:, 1]

    def resultado(self, senal: np.ndarray) -> dict:
        return {"probabilidad": float(self.riesgo(entrada(self.guardado, senal))[0])}


class GradientBoostingMultietiqueta(GradientBoostingBinario):
    """El riesgo del binario y, además, la probabilidad de cada cardiopatía (experimento 067)."""

    nombre = "Gradient boosting multietiqueta"

    def __init__(self, guardado: dict) -> None:
        super().__init__(guardado)
        validacion = pd.DataFrame(guardado["cardiopatias_val"]).set_index("cardiopatía")
        self.cardiopatias = [c for c in guardado["cardiopatias"] if bool(validacion.loc[c, "distingue mejor que el azar"])]

    def resultado(self, senal: np.ndarray) -> dict:
        x = entrada(self.guardado, senal)
        etiquetas = []
        for c in self.cardiopatias:
            p, u = float(self.guardado["modelos"][c].predict_proba(x)[0, 1]), float(self.guardado["umbrales"][c])
            etiquetas.append({"etiqueta": c, "nivel": "cardiopatía", "probabilidad": p, "umbral": u, "presente": p >= u})
        return {"probabilidad": float(self.riesgo(x)[0]), "etiquetas": etiquetas}


GUARDADO = joblib.load(EXPERIMENTO_067 / "modelo_ui_cardiopatias.joblib")
# Umbrales del riesgo medidos en la validación del v3 (notebook 44): "99" y "95" son el % de sensibilidad
PUNTOS = {"99": GUARDADO["puntos"]["99% de sensibilidad"], "95": GUARDADO["puntos"]["95% de sensibilidad"]}
PUNTO_PREDETERMINADO = "99"
MODELOS = {"binario": GradientBoostingBinario(GUARDADO), "multietiqueta": GradientBoostingMultietiqueta(GUARDADO)}
NOMBRES_ANTERIORES = {"Gradient boosting (Avance 2)": "Gradient boosting binario",   # en análisis guardados antes del 2026-10-27
                      "Gradient boosting binario (Avance 2)": "Gradient boosting binario",
                      "Gradient boosting multietiqueta (Avance 2)": "Gradient boosting multietiqueta"}
PREDETERMINADO = "binario"


def predecir(senal: np.ndarray, cual: str, punto: str = PUNTO_PREDETERMINADO) -> dict:
    """Resultado del modelo elegido con el umbral elegido, o {"conectado": False} si no hay modelos."""
    if not MODELOS:
        return {"conectado": False}
    m = MODELOS[cual]
    r = m.resultado(senal)
    umbral = float(PUNTOS[punto]["umbral"])
    return {"conectado": True, "id": cual, "nombre": m.nombre, "punto": punto, "umbral": umbral, "riesgo": r["probabilidad"] >= umbral, **r}


def lista() -> list[dict]:
    return [{"id": i, "nombre": m.nombre} for i, m in MODELOS.items()]


def puntos() -> list[dict]:
    """Los umbrales que se pueden elegir, con la sensibilidad y la especificidad que dieron en validación."""
    return [{"id": i, "nombre": f"{i}% de sensibilidad", "sensibilidad": p["sensibilidad"], "especificidad": p["especificidad"]}
            for i, p in PUNTOS.items()]
