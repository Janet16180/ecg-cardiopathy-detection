"""Páginas de "Modelos": la mejor variante de cada tipo de modelo probado, explicada, y sus resultados.

Las métricas, los ejemplos y las matrices se leen de las salidas guardadas de los notebooks 31 a 39 (validación
del dataset balanceado) o se calculan aquí a partir de ellas. Las descripciones copian la configuración de cada
notebook y módulo; los conteos que no están en una salida guardada se anotan abajo con su origen. La prueba no
se ha usado.
"""
import json
from decimal import Decimal
from functools import cache

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_curve

import datos
from datos import bloque, tx
from ecg_experiment import balanceado_v2 as v2
from ecg_experiment import caracteristicas_avance2 as ca2

SALIDAS = datos.RAIZ / "outputs"
COMPARACION = SALIDAS / "experiment057_particion_v2"
EN_USO = SALIDAS / "experiment067_multietiqueta_v3"   # modelos de la interfaz (notebook 44)
CARPETA = {"058 SVM y referencias": "experiment058_svm_v2", "059 xECG": "experiment059_xecg_v2", "060 SimDINOv2": "experiment060_simdino_v2"}
GB = ("058 SVM y referencias", "Gradient boosting")
GB_AJUSTADO = ("058 SVM y referencias", "Gradient boosting ajustado")
RL = ("058 SVM y referencias", "Regresión logística (Avance 3)")
SVM = ("058 SVM y referencias", "SVM Avance 4.8")
SVM_AJUSTADA = ("058 SVM y referencias", "SVM ajustada")
EN_INTERFAZ = {GB}   # el método de los modelos de la interfaz, reentrenados con el dataset v3 (experimento 067)
VISIBLE = {"SVM Avance 4.8": "SVM de referencia", "Regresión logística (Avance 3)": "Regresión logística"}   # nombres en las salidas
FUENTE = {"058 SVM y referencias": "Notebook 32", "059 xECG": "Notebook 33 (experimento 059)", "060 SimDINOv2": "Notebook 34 (experimento 060)"}
METRICAS = ["AUROC", "AUPRC", "esp_sens95", "esp_sens99"]
PRESUPUESTOS = {"250": "250", "500": "500", "1000": "1,000", "2000": "2,000", "todo": "todas"}

# Conteos que no están en una salida guardada (medidos el 2026-10-28)
XECG_PARAMETROS = 549_200          # encoder de outputs/experiment043_echonext_xecg_clusters/xecg_teacher.pt
ECHONEXT_PREENTRENAMIENTO = 72_475  # ECGs de entrenamiento de EchoNext (docs/experiment-043-echonext-xecg-clusters.md)
SIMDINO_PARAMETROS = 2_725_824     # pp.ECGEncoder con la configuración del notebook 34
SIN_ETIQUETAR = 77_610             # outputs/experiment060_simdino_v2/sin_etiquetar_100hz.npy (notebook 34), que no va en Git

# Qué sale de cada modelo en sus salidas guardadas: log-odds (se muestra como probabilidad), probabilidad o distancia
TIPO = {GB: "logit", GB_AJUSTADO: "logit", RL: "logit", SVM: "svm", SVM_AJUSTADA: "svm",
        **dict.fromkeys([("059 xECG", m) for m in ("xECG congelado", "xECG afinado", "xECG cabeza y luego todo", "xECG sin preentrenar")], "logit"),
        **dict.fromkeys([("060 SimDINOv2", m) for m in ("SimDINOv2 congelado", "SimDINOv2 congelado + intervalos", "SimDINOv2 047 congelado (contaminado)")], "logit"),
        **dict.fromkeys([("060 SimDINOv2", "SimDINOv2 afinado + intervalos"), ("060 SimDINOv2", "SimDINOv2 cabeza y luego todo")], "prob")}


# Datos

@cache
def tabla_v2() -> pd.DataFrame:
    ecgs = v2.load_table(datos.RAIZ)
    return ecgs.assign(split=pd.read_csv(datos.RAIZ / v2.DATASET / "particion.csv", index_col="fila")["split"])


def validacion() -> pd.DataFrame:
    return tabla_v2()[tabla_v2()["split"] == "val"]


@cache
def senales() -> np.ndarray:
    return v2.load(datos.RAIZ)[1]


@cache
def puntuaciones(experimento: str) -> pd.DataFrame:
    return pd.read_csv(SALIDAS / CARPETA[experimento] / "puntuaciones_val.csv", index_col="fila").loc[validacion().index]


def puntuacion(clave: tuple[str, str], presupuesto: str, semilla: str) -> np.ndarray:
    return puntuaciones(clave[0])[f"{clave[1]} | {presupuesto} | {semilla}"].to_numpy()


def salida(clave: tuple[str, str], presupuesto: str, semilla: str) -> np.ndarray:
    """La salida del modelo, con los log-odds pasados a probabilidad."""
    x = puntuacion(clave, presupuesto, semilla)
    return expit(x) if TIPO[clave] == "logit" else x


@cache
def metricas() -> pd.DataFrame:
    """Una fila por modelo, presupuesto y semilla (notebook 37)."""
    return pd.read_csv(COMPARACION / "comparacion_metricas_val.csv", dtype={"presupuesto": str, "semilla": str})


def resumen(presupuesto: str) -> pd.DataFrame:
    """Media entre semillas de cada modelo con un presupuesto."""
    t = metricas()[metricas()["presupuesto"] == presupuesto]
    return t.groupby(["experimento", "modelo"])[METRICAS].mean()


def fijo(valor: float, decimales: int = 3) -> Decimal:
    """Número con decimales fijos para los textos (0.860, no 0.86); traducciones.py le pone el separador de cada idioma."""
    return Decimal(f"{valor:.{decimales}f}")


def media(presupuesto: str, clave: tuple[str, str], metrica: str = "AUROC") -> Decimal:
    return fijo(resumen(presupuesto).loc[clave, metrica])


@cache
def por_dataset(clave: tuple[str, str]) -> pd.Series:
    """AUROC dentro de cada dataset de origen con 2,000 etiquetas, media de las 5 semillas."""
    val = validacion()
    return pd.concat([v2.auroc_by_dataset(val["y"].to_numpy(), puntuacion(clave, "2000", str(s)), val["dataset"].to_numpy())
                      for s in v2.SEEDS], axis=1).mean(axis=1)


def principales() -> list[tuple[str, str]]:
    """Los modelos de Modelos probados: la mejor variante de cada tipo."""
    return list(seleccion().values())


def nombre(clave: tuple[str, str]) -> str:
    """El nombre del tipo para las variantes que muestra la página; para las demás, su nombre en las salidas."""
    return {c: tipo for tipo, c in seleccion().items()}.get(clave, VISIBLE.get(clave[1], clave[1]))


def umbral(y: np.ndarray, s: np.ndarray, sensibilidad: float) -> float:
    """El umbral más alto que detecta esa fracción de las cardiopatías (como el notebook 38)."""
    _, tpr, cortes = roc_curve(y, s)
    return float(cortes[np.argmax(tpr >= sensibilidad)])


# Modelos probados: tabla ordenada y detalle de cada modelo

@cache
def ejemplos() -> pd.DataFrame:
    """Los dos ECGs de los ejemplos de salida: el primero normal y el primero con cardiopatía de PTB-XL en validación."""
    ptbxl = validacion()[validacion()["dataset"] == "PTB-XL"].sort_index()
    return pd.concat([ptbxl[ptbxl["y"] == 0].head(1), ptbxl[ptbxl["y"] == 1].head(1)])


def ecg(fila: int) -> str:
    return f"{tabla_v2().loc[fila, 'dataset']} {tabla_v2().loc[fila, 'registro']}"


def ejemplo(clave: tuple[str, str]) -> dict:
    presupuesto = "todo" if "todo" in set(metricas().loc[(metricas()["experimento"] == clave[0]) & (metricas()["modelo"] == clave[1]), "presupuesto"]) else "2000"
    s, y = salida(clave, presupuesto, "0"), validacion()["y"].to_numpy()
    u = umbral(y, s, 0.99)
    filas = ejemplos()
    valor = s[validacion().index.get_indexer(filas.index)]
    tabla = pd.DataFrame({
        "clase real": filas["clase"].to_numpy(),
        "cardiopatías": [" · ".join(c) if c else "—" for c in filas["cardiopatias"]],
        "salida": valor.round(3),
        "umbral (99% de sens.)": round(u, 3),
        "decisión": np.where(valor >= u, "con riesgo", "sin riesgo"),
    }, index=[ecg(f) for f in filas.index]).T.rename_axis("ECG de validación")
    nota = ("Modelo entrenado con todas las etiquetas; el umbral da 99% de sensibilidad en validación." if presupuesto == "todo" else
            "Modelo entrenado con 2,000 etiquetas (semilla 0); el umbral da 99% de sensibilidad en validación.")
    distancia = {"svm": "La salida es la distancia del ECG a la frontera, no una probabilidad.",
                 "coseno": "La salida es la distancia de sus fragmentos más raros a lo normal, no una probabilidad."}
    if TIPO[clave] in distancia:
        nota = tx("{a} {b}", a=nota, b=distancia[TIPO[clave]])
    return bloque("tabla", "Ejemplo de salida", tabla, nota=nota)


MEDIDAS = "Calcula las 111 medidas del ECG: intervalos, amplitudes del QRS, segmento ST, onda T, ejes, ritmo y espectro."
AVANCE4 = datos.RAIZ / "outputs/avance4_modelos_alternativos/comparativa_modelos_ajustados.csv"   # comparación de modelos con PTB-XL
TIPOS = {   # en el orden que eligió el equipo para presentarlos
    "Gradient boosting": ("111 medidas del ECG", [GB, GB_AJUSTADO]),
    "SVM": ("111 medidas del ECG", [SVM, SVM_AJUSTADA]),
    "SimDINOv2": ("Encoder transformer propio", [("060 SimDINOv2", m) for m in (   # sin el 047, que vio ECGs de la validación
        "SimDINOv2 congelado", "SimDINOv2 congelado + intervalos", "SimDINOv2 afinado + intervalos", "SimDINOv2 cabeza y luego todo")]),
    "xECG": ("Encoder xLSTM preentrenado", [("059 xECG", m) for m in (
        "xECG congelado", "xECG afinado", "xECG cabeza y luego todo", "xECG sin preentrenar")]),
    "Regresión logística": ("111 medidas del ECG", [RL]),
}


def gradient_boosting(clave: tuple[str, str]) -> dict:
    ptbxl = pd.read_csv(AVANCE4, index_col=0).loc["Gradient boosting (ensamble, solo referencia)", "AUROC"]
    return {
        "pasos": [
            MEDIDAS,
            "Entrena 100 árboles de hasta 31 hojas con tasa de aprendizaje de 0.1 (HistGradientBoostingClassifier de scikit-learn, con sus valores por defecto).",
            "Suma la salida de los árboles y la convierte en probabilidad con una sigmoide.",
        ],
        "por_que": [
            tx("Desde que elegimos el modelo base del proyecto lo usamos como referencia, y con PTB-XL fue el que mejor clasificó: AUROC de {a} en validación cruzada. En la comparación de modelos individuales quedó fuera solo porque es un ensamble.",
               a=fijo(ptbxl)),
            "Varias medidas del ECG se leen con umbrales, como un QRS de 120 ms o más en un bloqueo de rama, y un árbol decide justamente con umbrales.",
            "Da una probabilidad directamente, que es lo que la interfaz necesita para fijar el umbral de riesgo.",
        ],
    }


def svm(clave: tuple[str, str]) -> dict:
    return {
        "pasos": [
            MEDIDAS,
            "Usa un kernel RBF, que permite una frontera curva entre las dos clases.",
            "Prueba 36 combinaciones de C y gamma y se queda con la mejor por validación cruzada de 5 pliegues, agrupada por paciente.",
            "Con 2,000 etiquetas eligió C de 1 o 3 y gamma entre 0.004 y 0.016, según la semilla; con todas, C = 3 y gamma = 0.008.",
        ],
        "por_que": [
            "Ganó la comparación de siete modelos individuales que hicimos con PTB-XL, así que es la referencia contra la que medimos a los demás.",
            "En esa comparación rindió mejor que la red neuronal con cantidades de etiquetas cercanas a las 2,000 que tendrá la cohorte.",
            "Su configuración por defecto ya estaba cerca de la mejor, y con los mismos datos siempre da el mismo modelo, lo que facilita revisarlo.",
        ],
    }


def simdino(clave: tuple[str, str]) -> dict:
    horas = pd.read_csv(SALIDAS / CARPETA["060 SimDINOv2"] / "historial_preentrenamiento.csv")["seconds"].sum() / 3600
    return {
        "pasos": [
            "Corta el ECG en 48 fragmentos de 0.2 s con las 12 derivaciones.",
            tx("Un encoder transformer tipo ViT, con 6 bloques de atención, convierte los fragmentos en 384 números que describen el ECG. Tiene {p} parámetros.",
               p=SIMDINO_PARAMETROS),
            tx("El encoder se preentrenó con SimDINOv2 en {n} ECGs sin etiquetas, sin pacientes de la validación ni de la prueba. Tardó {h} h en el Mac.",
               n=SIN_ETIQUETAR, h=fijo(horas, 1)),
            "A esos números se agregan 10 medidas de ritmo e intervalos: frecuencia, variabilidad de los RR, prematuridad, pausas, PR, onda P, ancho del QRS y número de latidos.",
            "Con las etiquetas solo se entrena la regresión logística; el encoder no cambia.",
        ],
        "por_que": [
            "La idea viene del artículo de imágenes “No Time to Train!” (2025). Ahí un DINOv2 congelado describe cada parte de una imagen y, con unos pocos ejemplos etiquetados, reconoce objetos sin volver a entrenar.",
            "Quisimos hacer lo mismo con señales: que un encoder aprenda a describir ECGs sin etiquetas y que después basten pocas etiquetas para decidir. Para preentrenarlo usamos SimDINOv2 (2025), una forma simplificada de entrenar DINOv2.",
            "Encaja con la cohorte real, que tendrá unos 20,000 ECGs y solo unos 2,000 con etiqueta.",
            "El encoder es nuestro y se preentrenó sin ECGs de validación ni de prueba, así que la comparación con los demás es limpia.",
        ],
    }


def xecg(clave: tuple[str, str]) -> dict:
    return {
        "pasos": [
            "Corta el ECG en 48 fragmentos de 0.2 s con las 12 derivaciones y convierte cada uno en 96 números.",
            tx("Cuatro bloques xLSTM leen los fragmentos en orden, alternando la dirección de lectura. Tiene {p} parámetros.", p=XECG_PARAMETROS),
            tx("Se preentrenó con SimDINOv2 en {n} ECGs de EchoNext, sin etiquetas. EchoNext no forma parte de nuestro dataset.", n=ECHONEXT_PREENTRENAMIENTO),
            "Una capa final recibe el promedio de los 48 fragmentos y da la probabilidad.",
            "Toda la red se ajusta 20 épocas con nuestras etiquetas; el encoder aprende 10 veces más lento que la capa final.",
        ],
        "por_que": [
            "xECG (Lunelli et al.) es un modelo base para ECG: una red xLSTM preentrenada con más de 2.3 millones de ECGs sin etiquetas, pensada para ajustarse después a cada tarea.",
            "Quisimos saber si partir de una red ya preentrenada ayuda cuando hay pocas etiquetas, en lugar de entrenar desde cero.",
            "Usamos una versión compacta que preentrenamos con EchoNext, que no comparte ECGs con nuestro dataset, para que la comparación fuera limpia.",
        ],
    }


def regresion_logistica(clave: tuple[str, str]) -> dict:
    return {
        "pasos": [MEDIDAS, "Ajusta un peso por medida con regularización L2 (C = 0.1), que evita pesos demasiado grandes."],
        "por_que": [
            "Fue el modelo base del proyecto. Cada ECG ya era una fila de 111 medidas numéricas, el formato para el que está pensado un modelo lineal.",
            "Sus pesos se leen directamente: cada uno dice cuánto sube el riesgo con una medida. Eso importa al presentarle el modelo a un cardiólogo.",
            "Es el punto de partida: un modelo más complejo solo vale la pena si lo supera.",
        ],
    }


DETALLE = {"Gradient boosting": gradient_boosting, "SVM": svm,
           "SimDINOv2": simdino, "xECG": xecg, "Regresión logística": regresion_logistica}


@cache
def seleccion() -> dict[str, tuple[str, str]]:
    """De cada tipo, la variante con mejor AUROC con 2,000 etiquetas (a tres decimales); un empate lo decide el AUROC con todas."""
    r2, rt = resumen("2000")["AUROC"], resumen("todo")["AUROC"]
    return {tipo: max(variantes, key=lambda c: (round(r2[c], 3), rt.get(c, 0))) for tipo, (_, variantes) in TIPOS.items()}


def modelos_probados() -> dict:
    """La mejor variante de cada tipo de modelo, con sus métricas con 2,000 etiquetas (media de 5 semillas) y con todas, y su detalle."""
    lista = []
    for tipo, clave in seleccion().items():
        t = metricas()[(metricas()["experimento"] == clave[0]) & (metricas()["modelo"] == clave[1])]
        valores = {p: {m: float(g[m].mean()) for m in METRICAS} for p, g in t.groupby("presupuesto") if p in ("2000", "todo")}
        detalle = {**DETALLE[tipo](clave), "ejemplo": ejemplo(clave), "salida_interfaz": None,
                   "fuente": "Notebooks 32 y 38" if clave == GB else FUENTE[clave[0]]}
        lista.append({"id": " | ".join(clave), "nombre": tipo, "familia": TIPOS[tipo][0], "en_interfaz": clave in EN_INTERFAZ,
                      "valores": valores, "detalle": detalle})
    n = int((tabla_v2()["split"] == "train").sum())
    return {"presupuestos": [{"id": "2000", "nombre": "2,000 etiquetas"}, {"id": "todo", "nombre": tx("Todas las etiquetas ({n})", n=n)}],
            "nota": tx("Validación del dataset balanceado: {v} ECGs. Con 2,000 etiquetas, cada valor es la media de 5 semillas. "
                       "De cada tipo de modelo se muestra la variante con mejor AUROC con 2,000 etiquetas.", v=len(validacion())),
            "modelos": lista}


# Resultados

def matriz(clave: tuple[str, str], sensibilidad: float) -> dict:
    """Matriz de confusión en validación del modelo con todas las etiquetas, con el umbral de esa sensibilidad."""
    y = validacion()["y"].to_numpy()
    s = salida(clave, "todo", "0")
    marcado = s >= umbral(y, s, sensibilidad)
    conteo = lambda clase, riesgo: int(((y == clase) & (marcado == riesgo)).sum())
    tabla = pd.DataFrame([[conteo(1, True), conteo(1, False)], [conteo(0, True), conteo(0, False)]],
                         index=["con cardiopatía", "normal"], columns=["marcado con riesgo", "marcado sin riesgo"]).rename_axis("Realidad")
    return bloque("matriz", tx("Matriz de confusión: {m}, umbral de {s}% de sensibilidad", m=nombre(clave), s=round(100 * sensibilidad)), tabla, ancho="medio")


def con_filas(b: dict, filas: list) -> dict:
    """Pone nombres de fila que se traducen como plantillas (datos.tx), que pandas no guarda en un índice."""
    b["datos"]["index"] = filas
    return b


def modelos_interfaz() -> list[dict]:
    """Los modelos de la interfaz (experimento 067): riesgo, cuántas cardiopatías distingue y el AUROC de cada una."""
    tabla = pd.read_csv(EN_USO / "cardiopatias_todas_las_etiquetas.csv", index_col="cardiopatía")
    deteccion = pd.read_csv(EN_USO / "deteccion_val.csv", dtype={"presupuesto": str})
    acierto = pd.read_csv(EN_USO / "aciertos_val.csv", dtype={"presupuesto": str})
    todo = deteccion[deteccion["presupuesto"] == "todo"].iloc[0]
    top3 = float(acierto.loc[acierto["presupuesto"] == "todo", "una de las 3 más probables es real"].iloc[0])
    distingue = tabla["distingue mejor que el azar"].astype(bool)
    kpis = [{"etiqueta": "Cardiopatías que el modelo distingue", "valor": int(distingue.sum()), "detalle": tx("de {n}", n=len(tabla))},
            {"etiqueta": "AUROC del riesgo de cardiopatía", "valor": float(todo["AUROC"]), "formato": 3},
            {"etiqueta": "Especificidad con 99% de sensibilidad", "valor": float(todo["esp_sens99"]), "formato": 3},
            {"etiqueta": "Una de las 3 cardiopatías más probables es real", "valor": round(100 * top3, 1), "formato": "pct"}]
    cardiopatias = pd.DataFrame({
        "casos en entrenamiento": tabla["entrenamiento"], "casos en validación": tabla["validación"],
        "AUROC": tabla["AUROC"].round(3), "IC 95% bajo": tabla["IC bajo"].round(3), "IC 95% alto": tabla["IC alto"].round(3),
        "¿la distingue?": np.where(distingue, "sí", "no"),
    }).sort_values("casos en validación", ascending=False).rename_axis("cardiopatía")
    return [
        bloque("titulo", "Modelos de la interfaz (experimento 067)"),
        bloque("kpis", datos=kpis),
        bloque("tabla", "AUROC de cada cardiopatía (todas las etiquetas)", cardiopatias, visibles=20),
    ]


def variante_curva(tipo: str, clave: tuple[str, str]) -> tuple[str, str]:
    """La variante del tipo en la curva de aprendizaje: la de Modelos probados si se entrenó con todos los presupuestos;
    si no, la de mejor AUROC con 2,000 etiquetas entre las que sí."""
    m = metricas()
    completa = lambda c: set(PRESUPUESTOS) <= set(m.loc[(m["experimento"] == c[0]) & (m["modelo"] == c[1]), "presupuesto"])
    if completa(clave):
        return clave
    r2 = resumen("2000")["AUROC"]
    return max((c for c in TIPOS[tipo][1] if completa(c)), key=lambda c: r2[c])


def resultados() -> list[dict]:
    m = metricas()
    curva = {}
    for tipo, clave in seleccion().items():
        c = variante_curva(tipo, clave)
        curva[tipo if c == clave else f"{tipo} · {VISIBLE.get(c[1], c[1])}"] =m[(m["experimento"] == c[0]) & (m["modelo"] == c[1])].groupby("presupuesto")["AUROC"].mean()
    curva = pd.DataFrame(curva).reindex(list(PRESUPUESTOS)).rename(index=PRESUPUESTOS).rename_axis("ECGs etiquetados").round(3)

    conteo = validacion().groupby("dataset").size()
    origen = pd.DataFrame({nombre(c): por_dataset(c) for c in principales()}).round(3)
    filas_origen = [tx("{d} ({n} ECGs)", d=d, n=int(conteo[d])) for d in origen.index]

    especificidad = pd.DataFrame({
        f"{sens} de sensibilidad, {etiquetas}": {nombre(c): float(media(p, c, k)) for c in principales()}
        for k, sens in (("esp_sens95", "95%"), ("esp_sens99", "99%")) for p, etiquetas in (("2000", "2,000 etiquetas"), ("todo", "todas las etiquetas"))
    })
    especificidad = especificidad[sorted(especificidad.columns, key=lambda c: ("2,000" not in c, c))]

    return [
        bloque("lineas", "Curva de aprendizaje: AUROC según las etiquetas", curva, formato=3, eje="AUROC", colores=["s1", "s2", "s3", "s4", "gris"]),
        con_filas(bloque("calor", "AUROC dentro de cada dataset de origen (2,000 etiquetas)", origen, formato=3, minimo=0.5, maximo=1, intenso=True),
                  filas_origen),
        bloque("calor", "Especificidad con sensibilidad fija", especificidad, formato=3, maximo=1, intenso=True, nota=
               "Fracción de ECGs normales que no se marcan como riesgo cuando se detecta el 95% o el 99% de las cardiopatías."),
        matriz(GB, 0.99), matriz(GB, 0.95),
        *modelos_interfaz(),
    ]


# Características de la señal

SELECCION = SALIDAS / "avance2_fusion"   # selección del notebook Avance2.8, con el entrenamiento de PTB-XL
# Comprobación del notebook 32 (sección 2): las medidas de ecg_experiment/caracteristicas_avance2.py contra las del
# notebook Avance2.8 en los ECGs de PTB-XL que están en los dos. Sus tablas por ECG (44 MB) no van en Git.
REPRODUCCION = {"ECGs": 7_123, "medidas": 130, "correlación mediana": 0.993}
# Qué mide cada una, según las celdas 19, 20 y 31 del notebook Avance2.8 y ecg_experiment/caracteristicas_avance2.py
POR_DERIVACION = {
    "r_amp_": "Altura de la onda R en {d} (mV)",
    "s_depth_": "Profundidad de la onda S en {d} (mV)",
    "q_depth_": "Profundidad de la onda Q en {d} (mV)",
    "qrs_area_": "Área neta del QRS en {d} (mV·ms)",
    "st_": "Nivel del ST en {d}, de 60 a 80 ms tras el QRS (mV)",
    "t_amp_": "Amplitud de la onda T en {d} (mV)",
    "std_": "Desviación estándar de la señal en {d} (mV)",
    "kurt_": "Curtosis de la señal en {d}",
}
QUE_MIDE = {
    "hr_bpm": "Frecuencia cardiaca (lpm)",
    "hr_class": "Frecuencia en tres clases: bradicardia (< 60 lpm), normal y taquicardia (≥ 100 lpm)",
    "beat_similarity": "Correlación media de cada latido con el latido mediano",
    "qrs_dur_ms": "Duración del QRS (ms)",
    "qrs_class": "Duración del QRS en tres clases: estrecho (< 110 ms), intermedio (110 a 119 ms) y ancho (≥ 120 ms)",
    "pr_ms": "Intervalo PR: del inicio de la onda P al inicio del QRS (ms)",
    "pq_peak_ms": "Del punto de mayor amplitud antes del QRS al inicio del QRS (ms)",
    "qtc_bazett_ms": "QT corregido por la frecuencia con Bazett: QT / √RR (ms)",
    "qtc_class": "QTc en tres clases: normal (< 450 ms), limítrofe (450 a 479 ms) y prolongado (≥ 480 ms)",
    "flag_no_p_wave": "No se encontró una onda P medible",
    "sokolow_lyon_mv": "Índice de Sokolow-Lyon: S en V1 + la mayor R entre V5 y V6 (mV)",
    "cornell_product_mvms": "Producto de Cornell: (R en aVL + S en V3) × duración del QRS (mV·ms)",
    "qrs_axis_sin": "Eje eléctrico del QRS: seno del ángulo",
    "qrs_axis_cos": "Eje eléctrico del QRS: coseno del ángulo",
    "axis_class_izquierda": "Eje del QRS desviado a la izquierda (de -90° a -30°)",
    "axis_class_derecha": "Eje del QRS desviado a la derecha (más de 90°)",
    "axis_class_extrema": "Eje del QRS extremo (menos de -90°)",
    "q_patho_class": "Derivaciones con onda Q patológica (al menos 25% de la R y 0.1 mV): 0, 1, 2 o 3 o más",
    "r_frac_V1": "R / (R + S) en V1",
    "r_prog_slope_mv": "Pendiente de la altura de la R de V1 a V4 (mV por derivación)",
    "r_transition_class_temprana": "Transición temprana de la R: la R supera a la S desde V1 o V2",
    "r_transition_class_tardia": "Transición tardía de la R: la R supera a la S en V5, en V6 o en ninguna",
    "st_min_mv": "Mayor descenso del ST en todo el ECG, sin aVR (mV)",
    "st_max_mv": "Mayor elevación del ST en todo el ECG, sin aVR (mV)",
    "st_dep_anterior_mv": "Mayor descenso del ST en la cara anterior: V1 a V4 (mV)",
    "st_dep_inferior_mv": "Mayor descenso del ST en la cara inferior: II, III y aVF (mV)",
    "st_dep_lateral_mv": "Mayor descenso del ST en la cara lateral: I, aVL, V5 y V6 (mV)",
    "st_elev_inferior_mv": "Mayor elevación del ST en la cara inferior: II, III y aVF (mV)",
    "t_peak_rms_mv": "Amplitud de la onda T en su pico, en la media cuadrática de I, II y V1 a V6 (mV)",
    "t_axis_sin": "Eje de la onda T: seno del ángulo",
    "t_axis_cos": "Eje de la onda T: coseno del ángulo",
    "qrs_t_angle_deg": "Ángulo entre los ejes del QRS y de la onda T (0 a 180°)",
    "t_inverted_any": "Onda T invertida (menor de -0.1 mV) en al menos una de I, II y V3 a V6",
    "sp_low_rel": "Fracción de la potencia de 0.5 a 40 Hz que está entre 0.5 y 5 Hz",
    "sp_mid_rel": "Fracción de la potencia de 0.5 a 40 Hz que está entre 5 y 15 Hz",
    "sp_entropy": "Entropía del espectro de 0.5 a 40 Hz (de 0 a 1)",
    "noise_hf_frac": "Fracción de la potencia por encima de 40 Hz",
}


def que_mide(variable: str) -> str | dict:
    if variable in QUE_MIDE:
        return QUE_MIDE[variable]
    prefijo, derivacion = next((p, variable.removeprefix(p)) for p in POR_DERIVACION
                               if variable.startswith(p) and variable.removeprefix(p) in ca2.LEADS)
    return tx(POR_DERIVACION[prefijo], d=derivacion)


def caracteristicas() -> list[dict]:
    variables = pd.read_csv(SELECCION / "variables_finales.csv")
    seleccion = json.loads((SELECCION / "seleccion.json").read_text())
    umbrales = seleccion["umbrales"]
    quitadas = {"Varianza": len(seleccion["eliminadas_varianza"]), "Relevancia": len(seleccion["eliminadas_relevancia"]),
                "Redundancia": len(seleccion["eliminadas_correlacion"])}
    assert seleccion["seleccionadas"] == variables["variable"].tolist()
    candidatas = len(variables) + sum(quitadas.values())

    kpis = [{"etiqueta": "Características de entrada", "valor": len(variables), "detalle": tx("de {n} medidas candidatas", n=candidatas)},
            {"etiqueta": "Familias", "valor": variables["familia"].nunique()},
            {"etiqueta": "Objetivos de la selección", "valor": 5, "detalle": "cardiopatía y las superclases MI, STTC, CD e HYP de PTB-XL"},
            {"etiqueta": "Coincidencia con el notebook (correlación mediana)", "valor": REPRODUCCION["correlación mediana"], "formato": 3,
             "detalle": tx("Spearman, {m} medidas en {n} ECGs de PTB-XL", m=REPRODUCCION["medidas"], n=REPRODUCCION["ECGs"])}]
    quedan = np.cumsum([candidatas, *(-np.array(list(quitadas.values())))])
    pasos = pd.DataFrame({
        "criterio": ["Medidas de la señal elegidas por su uso clínico",
                     tx("Casi constantes: varianza menor de {v} con cada medida llevada a [0, 1]", v=umbrales["varianza"]),
                     tx("Sin efecto de al menos {e} (η² o V² de Cramér, q < 0.05) frente a ninguno de los 5 objetivos", e=umbrales["efecto"]),
                     tx("Correlación de {r} o más con otra de mayor efecto, que se conserva", r=umbrales["correlacion"])],
        "eliminadas": [0, *quitadas.values()], "quedan": quedan,
    }, index=["Construcción", *quitadas]).rename_axis("paso")
    tabla = pd.DataFrame({
        "qué mide": [que_mide(v) for v in variables["variable"]], "familia": variables["familia"].to_numpy(),
        "prueba estadística": variables["prueba"].to_numpy(), "efecto máximo": variables["efecto_max"].round(3).to_numpy(),
        "objetivo del efecto máximo": variables["objetivo_max"].to_numpy(),
    }, index=variables["variable"]).rename_axis("característica")
    familias = variables["familia"].value_counts().rename("características").rename_axis("familia")
    return [
        bloque("aviso", nota="Se diseñaron y seleccionaron en el notebook Avance2.8 con los ECGs de entrenamiento de PTB-XL. "
                             "La interfaz las calcula con el mismo código, en ecg_experiment/caracteristicas_avance2.py."),
        bloque("kpis", datos=kpis),
        bloque("tabla", "De las medidas candidatas a las características de entrada", pasos),
        bloque("barras", "Características por familia", familias, formato="entero", eje="características", ancho="medio"),
        bloque("tabla", "Las características, de mayor a menor efecto", tabla, visibles=20),
    ]


SECCIONES = {"resultados": cache(resultados), "caracteristicas": cache(caracteristicas)}
modelos_probados = cache(modelos_probados)
