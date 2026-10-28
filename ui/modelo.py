"""Modelos que usa "Registrar nuevo ECG" y que describe "Acerca de > Modelo entrenado".

MODELOS tiene los dos gradient boosting sobre las 111 características del Avance 2, entrenados con todas las
etiquetas del entrenamiento del dataset balanceado: el binario (notebook 38), que solo dice si hay cardiopatía,
y el multietiqueta (notebook 39, experimento 063), que además da la probabilidad de cada categoría y diagnóstico.
La página del modelo deja elegir cuál usar. Cada modelo es un objeto con:

    nombre, resumen: str
    umbral: float                                  probabilidad a partir de la cual se marca riesgo de cardiopatía
    resultado(senal: np.ndarray) -> dict           senal de forma (5000, 12): 10 s a 500 Hz, en mV, derivaciones
                                                   I, II, III, aVR, aVL, aVF, V1-V6, con filtro pasa-altas de 0.5 Hz
                                                   (el formato del dataset balanceado; registro.py ya lo aplica);
                                                   devuelve {"probabilidad": de 0 a 1} y, si las tiene, "etiquetas"
    riesgo(x: pd.DataFrame) -> np.ndarray          probabilidad de cardiopatía de cada fila de variables ya
                                                   preprocesadas (la usa areas.py para las áreas de interés)
    descripcion() -> list[dict]                    bloques de datos.bloque() para la página "Modelo entrenado"

Con MODELOS vacío la UI dice que no hay modelo conectado y nunca muestra una predicción inventada.
"""
import warnings

import joblib
import numpy as np
import pandas as pd

import datos
from datos import tx
from ecg_experiment import balanceado_v2 as v2
from ecg_experiment import caracteristicas_avance2 as ca2

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # avisos de pandas al derivar columnas

FS = 500
MUESTRAS = 10 * FS
VARIABLES = datos.RAIZ / "outputs/avance2_fusion/variables_finales.csv"
EXPERIMENTO_063 = datos.RAIZ / "outputs/experiment063_multietiqueta_v2"


def entrada(guardado: dict, senal: np.ndarray) -> pd.DataFrame:
    """Las 111 características preprocesadas de un ECG, como en los notebooks 32, 38 y 39."""
    return guardado["preprocesador"].transform(ca2.features_one(senal[: v2.CROP], guardado["columnas_primarias"]))


def partes_comunes(guardado: dict) -> dict:
    return {
        "entrada": "12 derivaciones, 10 s a 500 Hz en mV con filtro pasa-altas de 0.5 Hz; el modelo usa los primeros 9.6 s",
        "características": "111 variables de la señal: intervalos, amplitudes y ejes del QRS, ST-T, onda P, ritmo y espectro",
        "preprocesamiento": "imputación por mediana, recorte de extremos, Yeo-Johnson y estandarización, ajustados con el entrenamiento",
        "entrenamiento": tx("los {n} ECGs del entrenamiento del dataset balanceado, con todas sus etiquetas",
                            n=guardado["ecgs"]["entrenamiento"]),
    }


def bloques_deteccion(guardado: dict) -> list[dict]:
    """Métricas de "tiene alguna cardiopatía" y sus puntos de operación, en validación."""
    metricas = pd.Series({
        "AUROC": guardado["metricas_val"]["AUROC"],
        "AUPRC": guardado["metricas_val"]["AUPRC"],
        "especificidad con 95% de sensibilidad": guardado["metricas_val"]["esp_sens95"],
        "especificidad con 99% de sensibilidad": guardado["metricas_val"]["esp_sens99"],
    }, name="valor").rename_axis("métrica").round(3)
    puntos = pd.DataFrame(guardado["puntos"]).T.rename_axis("punto de operación").round(3)
    punto = guardado["puntos"][guardado["punto"]]
    return [
        datos.bloque("tabla", "Métricas en validación", metricas, ancho="medio",
                     nota=tx("{n} ECGs de validación del dataset balanceado. La prueba todavía no se usa.",
                             n=guardado["ecgs"]["validación"])),
        datos.bloque("tabla", "Puntos de operación en validación", puntos, ancho="medio",
                     nota=tx("La interfaz usa el de 99% de sensibilidad: en validación marca como riesgo al {fa}% de los ECGs normales.",
                             fa=round(100 * (1 - punto["especificidad"])))),
    ]


class GradientBoostingBinario:
    """Solo si hay cardiopatía: un gradient boosting binario (notebooks 32 y 38)."""

    nombre = "Gradient boosting binario"
    resumen = "Solo dice si hay cardiopatía"

    def __init__(self) -> None:
        self.guardado = joblib.load(datos.RAIZ / "outputs/experiment058_svm_v2/modelo_ui_gradient_boosting.joblib")
        self.umbral = self.guardado["umbral"]

    def riesgo(self, x: pd.DataFrame) -> np.ndarray:
        return self.guardado["modelo"].predict_proba(x)[:, 1]

    def resultado(self, senal: np.ndarray) -> dict:
        return {"probabilidad": float(self.riesgo(entrada(self.guardado, senal))[0])}

    def descripcion(self) -> list[dict]:
        partes = pd.Series({
            **partes_comunes(self.guardado),
            "modelo": "HistGradientBoostingClassifier de scikit-learn con sus valores por defecto",
            "umbral": "el punto con 99% de sensibilidad en validación (tabla de puntos de operación)",
            "notebooks": "32 (comparación con la SVM de referencia) y 38 (este modelo)",
        }, name="descripción").rename_axis("parte")
        familias = pd.read_csv(VARIABLES)["familia"].value_counts().rename("variables").rename_axis("familia")
        return [datos.bloque("tabla", "Cómo funciona el modelo", partes), *bloques_deteccion(self.guardado),
                datos.bloque("tabla", "Variables de entrada por familia", familias, ancho="medio")]


class GradientBoostingMultietiqueta:
    """Además qué cardiopatía: un gradient boosting por categoría y por diagnóstico (notebook 39)."""

    nombre = "Gradient boosting multietiqueta"
    resumen = "Además dice qué cardiopatía"

    def __init__(self) -> None:
        self.guardado = joblib.load(EXPERIMENTO_063 / "modelo_ui_multietiqueta.joblib")
        self.umbral = self.guardado["umbral"]
        self.etiquetas = pd.DataFrame(self.guardado["etiquetas"]).set_index("etiqueta")

    def riesgo(self, x: pd.DataFrame) -> np.ndarray:
        """Probabilidad de tener alguna de las categorías: 1 − Π(1 − p)."""
        return 1 - np.prod([1 - self.guardado["modelos"][c].predict_proba(x)[:, 1] for c in self.guardado["categorias"]], axis=0)

    def resultado(self, senal: np.ndarray) -> dict:
        x = entrada(self.guardado, senal)
        p = {e: float(m.predict_proba(x)[0, 1]) for e, m in self.guardado["modelos"].items()}
        etiquetas = [{"etiqueta": e, "nivel": f["nivel"], "probabilidad": p[e], "umbral": float(f["umbral"]),
                      "presente": bool(p[e] >= f["umbral"])} for e, f in self.etiquetas.iterrows()]
        return {"probabilidad": float(self.riesgo(x)[0]), "etiquetas": etiquetas}

    def descripcion(self) -> list[dict]:
        g = self.guardado
        partes = pd.Series({
            **partes_comunes(g),
            "modelo": tx("un HistGradientBoostingClassifier por etiqueta: {c} categorías y {d} diagnósticos con al menos 20 ECGs en validación",
                         c=len(g["categorias"]), d=len(g["diagnosticos"])),
            "riesgo": "probabilidad de tener alguna de las categorías: 1 − Π(1 − p)",
            "umbral": "riesgo: el punto con 99% de sensibilidad en validación; cada categoría y diagnóstico: el punto de máximo F1 en validación",
            "notebooks": "39 (este modelo; protocolo del experimento 063)",
        }, name="descripción").rename_axis("parte")
        columnas = {"prevalencia": "prevalencia en validación", "AUROC": "AUROC", "AUPRC": "AUPRC", "umbral": "umbral",
                    "sensibilidad": "sensibilidad", "VPP": "VPP"}
        tabla = self.etiquetas[["nivel", *columnas]].rename(columns=columnas).sort_values("AUROC", ascending=False)
        por_nivel = {n: t.drop(columns="nivel").rename_axis(n).round(3) for n, t in tabla.groupby("nivel")}
        aciertos = pd.DataFrame({"modelo": g["aciertos"], "referencia: siempre las más frecuentes": g["aciertos_referencia"]}
                                ).rename_axis("entre los ECGs con cardiopatía").round(3)
        nota_etiquetas = "Validación del dataset balanceado, con todas las etiquetas. Sensibilidad y VPP en el umbral de máximo F1."
        return [
            datos.bloque("tabla", "Cómo funciona el modelo", partes),
            *bloques_deteccion(g),
            datos.bloque("tabla", "¿Acierta qué cardiopatía?", aciertos, ancho="medio",
                         nota="Fracción de los ECGs con cardiopatía de validación."),
            datos.bloque("tabla", "Categorías: métricas en validación", por_nivel["categoría"], nota=nota_etiquetas),
            datos.bloque("tabla", "Diagnósticos: métricas en validación", por_nivel["diagnóstico"], nota=nota_etiquetas),
        ]


MODELOS = {"binario": GradientBoostingBinario(), "multietiqueta": GradientBoostingMultietiqueta()}
NOMBRES_ANTERIORES = {"Gradient boosting (Avance 2)": "Gradient boosting binario",   # en análisis guardados antes del 2026-10-27
                      "Gradient boosting binario (Avance 2)": "Gradient boosting binario",
                      "Gradient boosting multietiqueta (Avance 2)": "Gradient boosting multietiqueta"}
PREDETERMINADO = "binario"


def predecir(senal: np.ndarray, cual: str) -> dict:
    """Resultado del modelo elegido para un ECG ya en el formato de entrada, o {"conectado": False} si no hay modelos."""
    if not MODELOS:
        return {"conectado": False}
    m = MODELOS[cual]
    r = m.resultado(senal)
    return {"conectado": True, "id": cual, "nombre": m.nombre, "umbral": float(m.umbral), "riesgo": r["probabilidad"] >= m.umbral, **r}


def lista() -> list[dict]:
    return [{"id": i, "nombre": m.nombre, "resumen": m.resumen} for i, m in MODELOS.items()]


def descripcion(cual: str) -> dict:
    if not MODELOS:
        return {"conectado": False, "bloques": []}
    return {"conectado": True, "nombre": MODELOS[cual].nombre, "bloques": MODELOS[cual].descripcion()}


def comparacion() -> list[dict]:
    """Los dos modelos lado a lado, con todas las etiquetas y con las 2,000 de la cohorte (notebook 39)."""
    deteccion = pd.read_csv(EXPERIMENTO_063 / "deteccion_val.csv", dtype={"presupuesto": str})
    acierto = pd.read_csv(EXPERIMENTO_063 / "aciertos_val.csv", dtype={"presupuesto": str})
    media = {p: deteccion[deteccion["presupuesto"] == p].mean(numeric_only=True) for p in ("todo", "2000")}
    hits = {p: acierto[(acierto["modelo"] == "multietiqueta") & (acierto["presupuesto"] == p)].mean(numeric_only=True) for p in ("todo", "2000")}
    binario, multi = MODELOS["binario"].resumen, MODELOS["multietiqueta"].resumen
    filas = {}
    for p, sufijo in (("todo", "todas las etiquetas"), ("2000", "2,000 etiquetas")):
        filas[f"AUROC para detectar cardiopatía, {sufijo}"] = (media[p]["AUROC binario"], media[p]["AUROC multietiqueta"])
        filas[f"especificidad con 99% de sensibilidad, {sufijo}"] = (media[p]["esp. con 99% de sens., binario"],
                                                                     media[p]["esp. con 99% de sens., multietiqueta"])
    for p, sufijo in (("todo", "todas las etiquetas"), ("2000", "2,000 etiquetas")):
        filas[f"la categoría más probable es real, {sufijo}"] = (np.nan, hits[p]["categoría más probable es real"])
        filas[f"uno de los 3 diagnósticos más probables es real, {sufijo}"] = (np.nan, hits[p]["uno de los 3 diagnósticos más probables es real"])
    tabla = pd.DataFrame(filas, index=[binario, multi]).T.rename_axis("validación").round(3)
    no_peor = int((deteccion.loc[deteccion["presupuesto"] == "2000", "IC bajo"] > -0.01).sum())
    referencia = MODELOS["multietiqueta"].guardado["aciertos_referencia"]
    return [datos.bloque("tabla", "Comparación de los dos modelos", tabla, nota=tx(
        "Validación del dataset balanceado; con 2,000 etiquetas, media de 5 semillas. Con 2,000 etiquetas, el modelo "
        "multietiqueta {cumple} la no inferioridad del protocolo para detectar cardiopatía: el límite inferior de la "
        "diferencia de AUROC quedó por encima de −0.01 en {n} de 5 semillas. Adivinando siempre las más frecuentes, la "
        "categoría acierta en {a}% de los ECGs con cardiopatía y los 3 diagnósticos en {b}%.",
        cumple="cumple" if no_peor >= 4 else "no cumple", n=no_peor, a=round(100 * referencia["categoría más probable es real"]),
        b=round(100 * referencia["uno de los 3 diagnósticos más probables es real"])))]
