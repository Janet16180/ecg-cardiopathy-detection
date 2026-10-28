"""Páginas de "Acerca de > Modelos": qué se probó, los resultados y por qué la UI usa los dos gradient boosting.

Todos los números se leen de las salidas guardadas de los notebooks 31 a 39 (validación del dataset balanceado)
o se calculan aquí a partir de ellas; ninguno está escrito a mano. La prueba no se ha usado.
"""
from decimal import Decimal
from functools import cache

import numpy as np
import pandas as pd

import datos
import modelo
from datos import bloque, tx
from ecg_experiment import balanceado_v2 as v2

SALIDAS = datos.RAIZ / "outputs"
COMPARACION = SALIDAS / "experiment057_particion_v2"
MULTI = SALIDAS / "experiment063_multietiqueta_v2"
CARPETA = {"058 SVM y referencias": "experiment058_svm_v2", "059 xECG": "experiment059_xecg_v2", "060 SimDINOv2": "experiment060_simdino_v2",
           "061 Anomalías": "experiment061_anomalias_v2", "062 Grafo con memoria": "experiment062_grafo_memoria_v2"}
MULTIETIQUETA = ("063 Multietiqueta", "Gradient boosting multietiqueta")
SVM = ("058 SVM y referencias", "SVM Avance 4.8")
VISIBLE = {"SVM Avance 4.8": "SVM de referencia", "Regresión logística (Avance 3)": "Regresión logística"}   # nombres en las salidas
GB = ("058 SVM y referencias", "Gradient boosting")
FAMILIA = {"058 SVM y referencias": "Medidas de la señal", "059 xECG": "xECG preentrenado", "060 SimDINOv2": "SimDINOv2 propio",
           "061 Anomalías": "Anomalías estilo AnomalyDINO", "062 Grafo con memoria": "Grafo con memoria", "063 Multietiqueta": "Multietiqueta"}
QUE_ES = {
    "058 SVM y referencias": ("Las 111 medidas de la señal con un clasificador clásico: SVM RBF, gradient boosting o regresión logística.", "32"),
    "059 xECG": ("Codificador xECG preentrenado sin etiquetas con EchoNext (experimento 043): congelado, con ajuste fino, primero la cabeza y sin preentrenar como control.", "33"),
    "060 SimDINOv2": ("Codificador propio preentrenado sin etiquetas con SimDINOv2, sin ECGs de validación ni de prueba: congelado, con 10 medidas de intervalos y con ajuste fino.", "34"),
    "061 Anomalías": ("Banco de tokens de 0.2 s de ECGs normales (codificador SimDINOv2 propio); cada ECG se puntúa por la distancia de sus tokens al normal más cercano.", "35"),
    "062 Grafo con memoria": ("Las 12 derivaciones como nodos de un grafo con conexiones aprendidas, convoluciones temporales, memoria de prototipos y salida variacional; inspirado en Li et al. (2026).", "36"),
    "063 Multietiqueta": ("Un gradient boosting por categoría y por diagnóstico sobre las mismas 111 medidas; el riesgo es la probabilidad de tener alguna categoría.", "39"),
}
METRICAS = {"AUROC": "AUROC", "AUPRC": "AUPRC", "esp_sens95": "esp. con 95% de sens.", "esp_sens99": "esp. con 99% de sens."}
PRESUPUESTOS = {"250": "250", "500": "500", "1000": "1,000", "2000": "2,000", "todo": "todas"}
CATEGORIA_GENERAL = pd.read_csv(datos.CLASIFICACION).drop_duplicates("diagnostico_general").set_index("diagnostico_general")["categoria"]   # como el notebook 39
SIN_ETIQUETAR = 77_610   # ECGs de outputs/experiment060_simdino_v2/sin_etiquetar_100hz.npy (notebook 34), que no va en Git
FUENTE = "Validación del dataset balanceado (2,000 ECGs), leída de las salidas de los notebooks 31 a 39. La prueba no se ha usado."


@cache
def validacion() -> pd.DataFrame:
    ecgs = v2.load_table(datos.RAIZ)
    particion = pd.read_csv(datos.RAIZ / v2.DATASET / "particion.csv", index_col="fila")
    return ecgs.loc[particion.index[particion["split"] == "val"]]


def alguna(probas: pd.DataFrame) -> np.ndarray:
    """Probabilidad de tener alguna categoría: 1 − Π(1 − p_k), como en el notebook 39."""
    return 1 - np.prod(1 - probas[modelo.MODELOS["multietiqueta"].guardado["categorias"]].to_numpy(), axis=1)


@cache
def probabilidades_multi() -> dict[tuple[str, str], pd.DataFrame]:
    tabla = pd.read_csv(MULTI / "probabilidades_val.csv", index_col=[0, 1])
    return {tuple(clave.split(" | ")): p.droplevel(0).loc[validacion().index] for clave, p in tabla.groupby(level=0)}


@cache
def metricas() -> pd.DataFrame:
    """Una fila por modelo, presupuesto y semilla: las del notebook 37 más el multietiqueta del notebook 39."""
    tabla = pd.read_csv(COMPARACION / "comparacion_metricas_val.csv", dtype={"presupuesto": str, "semilla": str})
    y = validacion()["y"].to_numpy()
    multi = [{"experimento": MULTIETIQUETA[0], "modelo": MULTIETIQUETA[1], "presupuesto": p, "semilla": s, **v2.summary(y, alguna(probas))}
             for (p, s), probas in probabilidades_multi().items()]
    return pd.concat([tabla, pd.DataFrame(multi)], ignore_index=True)


def resumen(presupuesto: str) -> pd.DataFrame:
    """Media y desviación entre semillas de cada modelo con un presupuesto."""
    t = metricas()[metricas()["presupuesto"] == presupuesto]
    return t.groupby(["experimento", "modelo"])[list(METRICAS)].agg(["mean", "std"])


def puesto(presupuesto: str, clave: tuple[str, str], metrica: str = "AUROC") -> tuple[int, int]:
    """Lugar del modelo (1 = el mejor) entre todos los probados con ese presupuesto, y cuántos son."""
    medias = resumen(presupuesto)[(metrica, "mean")]
    return int((medias > medias[clave]).sum()) + 1, len(medias)


def fijo(valor: float, decimales: int = 3) -> Decimal:
    """Número con decimales fijos para los textos (0.860, no 0.86); traducciones.py le pone el separador de cada idioma."""
    return Decimal(f"{valor:.{decimales}f}")


def media(presupuesto: str, clave: tuple[str, str], metrica: str = "AUROC") -> Decimal:
    return fijo(resumen(presupuesto).loc[clave, (metrica, "mean")])


@cache
def regla() -> pd.DataFrame:
    return pd.read_csv(COMPARACION / "comparacion_regla_decision.csv").set_index(["experimento", "modelo"])


def puntuacion(clave: tuple[str, str], presupuesto: str, semilla: str) -> np.ndarray:
    if clave == MULTIETIQUETA:
        return alguna(probabilidades_multi()[(presupuesto, semilla)])
    tabla = pd.read_csv(SALIDAS / CARPETA[clave[0]] / "puntuaciones_val.csv", index_col="fila")
    return tabla.loc[validacion().index, f"{clave[1]} | {presupuesto} | {semilla}"].to_numpy()


def por_dataset(clave: tuple[str, str]) -> pd.Series:
    """AUROC dentro de cada dataset de origen, con 2,000 etiquetas y la semilla 0 (como el notebook 37)."""
    val = validacion()
    return v2.auroc_by_dataset(val["y"].to_numpy(), puntuacion(clave, "2000", "0"), val["dataset"].to_numpy())


def principales() -> list[tuple[str, str]]:
    """Los dos de la UI, la SVM de referencia y la mejor variante (AUROC con 2,000 etiquetas) de las demás familias."""
    medias = resumen("2000")[("AUROC", "mean")]
    otras = [medias.xs(e, level=0, drop_level=False).idxmax() for e in ("062 Grafo con memoria", "060 SimDINOv2", "059 xECG", "061 Anomalías")]
    return [GB, MULTIETIQUETA, SVM, *otras]


@cache
def etiquetas_multi() -> pd.DataFrame:
    return pd.read_csv(MULTI / "metricas_etiquetas_val.csv", dtype={"presupuesto": str})


def nombre(clave: tuple[str, str]) -> str:
    return VISIBLE.get(clave[1], clave[1])


# Páginas

def modelos() -> list[dict]:
    m = metricas()
    familias = pd.DataFrame({
        "qué es": {FAMILIA[e]: QUE_ES[e][0] for e in FAMILIA},
        "variantes": {FAMILIA[e]: " · ".join(sorted(VISIBLE.get(v, v) for v in m.loc[m["experimento"] == e, "modelo"].unique())) for e in FAMILIA},
        "notebook": {FAMILIA[e]: QUE_ES[e][1] for e in FAMILIA},
    }).rename_axis("familia")
    tablas = []
    for presupuesto, titulo in (("2000", "Resultados con 2,000 etiquetas (media de 5 semillas)"), ("todo", "Resultados con todas las etiquetas (una semilla)")):
        r = resumen(presupuesto)
        t = pd.DataFrame({METRICAS[k]: r[(k, "mean")] for k in METRICAS})
        if presupuesto == "2000":
            t.insert(1, "desv. del AUROC", r[("AUROC", "std")])
            t["semillas mejor que la SVM"] = regla()["semillas_mejor"]
            t["semillas peor que la SVM"] = regla()["semillas_peor"]
        t.insert(0, "familia", [FAMILIA[e] for e in t.index.get_level_values(0)])
        t = t.droplevel(0).rename(index=VISIBLE).rename_axis("modelo").sort_values("AUROC", ascending=False).round(3)
        tablas.append(bloque("tabla", titulo, t, nota=FUENTE if presupuesto == "todo" else
                             "La regla de decisión pide que el intervalo pareado de AUROC frente a la SVM de referencia quede por encima de 0 "
                             "en al menos 4 de 5 semillas. El multietiqueta se comparó con el binario, no con la SVM (ver Modelo entrenado)."))
    atajos = pd.read_csv(COMPARACION / "referencias_atajos.csv").set_index(["referencia", "etiquetas"])["AUROC val (media)"]
    atajos = atajos.unstack().rename(columns={"2000": "2,000 etiquetas", "todo": "todas las etiquetas"}).rename_axis("referencia").round(3)
    return [bloque("aviso", nota=FUENTE),
            bloque("tabla", "Familias de modelos probadas", familias),
            *tablas,
            bloque("tabla", "Referencias de atajos", atajos, ancho="medio",
                   nota="AUROC de una regresión logística que solo ve el dataset de origen o el sexo: el piso que cualquier modelo debe superar (notebook 31).")]


def resultados() -> list[dict]:
    m = metricas()
    curva_modelos = [SVM, GB, ("060 SimDINOv2", "SimDINOv2 congelado + intervalos"), ("059 xECG", "xECG congelado")]
    curva = pd.DataFrame({nombre(c): m[(m["experimento"] == c[0]) & (m["modelo"] == c[1])].groupby("presupuesto")["AUROC"].mean()
                          for c in curva_modelos}).reindex(list(PRESUPUESTOS)).rename(index=PRESUPUESTOS).rename_axis("ECGs etiquetados").round(3)

    val = validacion()
    conteo = val.groupby("dataset").size()
    origen = pd.DataFrame({nombre(c): por_dataset(c) for c in principales()}).round(3)
    origen.index = [f"{d} (n = {conteo[d]})" for d in origen.index]

    especificidad = pd.DataFrame({
        f"{sens} de sensibilidad, {etiquetas}": {nombre(c): float(media(p, c, k)) for c in principales()}
        for k, sens in (("esp_sens95", "95%"), ("esp_sens99", "99%")) for p, etiquetas in (("2000", "2,000 etiquetas"), ("todo", "todas las etiquetas"))
    })
    especificidad = especificidad[sorted(especificidad.columns, key=lambda c: ("2,000" not in c, c))]

    e = etiquetas_multi()
    categorias = modelo.MODELOS["multietiqueta"].guardado["categorias"]
    e["nivel"] = np.where(e["etiqueta"].isin(categorias), "categorías", "diagnósticos")
    macro = e.groupby(["nivel", "presupuesto", "semilla"])["AUROC"].mean().groupby(["nivel", "presupuesto"]).agg(["mean", "std"])
    cuantas = e[e["presupuesto"] == "todo"].groupby("nivel").size()
    tabla_macro = pd.DataFrame({
        "etiquetas": cuantas,
        "2,000 etiquetas (media de 5 semillas)": macro.xs("2000", level="presupuesto")["mean"],
        "desv. entre semillas": macro.xs("2000", level="presupuesto")["std"],
        "todas las etiquetas": macro.xs("todo", level="presupuesto")["mean"],
    }).rename_axis("macro-AUROC").round(3)
    por_etiqueta = e.groupby(["nivel", "etiqueta", "presupuesto"])["AUROC"].mean().unstack("presupuesto").round(3)
    por_etiqueta = por_etiqueta.sort_values(["nivel", "todo"], ascending=[True, False]).droplevel(0)
    por_etiqueta = por_etiqueta.rename(columns={"2000": "2,000 etiquetas (media de 5 semillas)", "todo": "todas las etiquetas"}).rename_axis("etiqueta")

    return [
        bloque("aviso", nota=FUENTE),
        bloque("lineas", "Curva de aprendizaje: AUROC según las etiquetas", curva, formato=3, eje="AUROC",
               colores=["s1", "s2", "s3", "s4"],
               nota="Media de 5 semillas; con todas las etiquetas, una semilla. Solo los modelos que se entrenaron con los cinco presupuestos."),
        bloque("calor", "AUROC dentro de cada dataset de origen", origen, formato=3, minimo=0.5, maximo=1,
               nota="2,000 etiquetas, semilla 0. n son los ECGs de validación de cada dataset: con pocos, el AUROC es muy inestable. "
                    "CPSC 2018 extra no aparece porque en validación todos sus ECGs tienen cardiopatía."),
        bloque("calor", "Especificidad con sensibilidad fija", especificidad, formato=3, maximo=1,
               nota="Fracción de ECGs normales que no se marcan como riesgo cuando se detecta el 95% o el 99% de las cardiopatías. Con 2,000 etiquetas, media de 5 semillas."),
        bloque("tabla", "Macro-AUROC del modelo multietiqueta", tabla_macro, ancho="medio",
               nota="Promedio simple del AUROC de cada etiqueta. Los modelos binarios tienen una sola clase positiva, así que su macro-AUROC es su AUROC."),
        bloque("calor", "AUROC de cada categoría y diagnóstico (multietiqueta)", por_etiqueta, formato=3, minimo=0.5, maximo=1, ancho="medio",
               nota="Primero las 6 categorías y después los diagnósticos, cada grupo de mayor a menor AUROC con todas las etiquetas."),
    ]


def eleccion() -> list[dict]:
    r, e = regla(), etiquetas_multi()
    y = validacion()["y"].to_numpy()
    deteccion = pd.read_csv(MULTI / "deteccion_val.csv", dtype={"presupuesto": str})
    aciertos = pd.read_csv(MULTI / "aciertos_val.csv", dtype={"presupuesto": str})
    guardado = modelo.MODELOS["multietiqueta"].guardado

    def lugar(presupuesto: str, clave: tuple[str, str], metrica: str = "AUROC") -> dict:
        p, n = puesto(presupuesto, clave, metrica)
        return {"r": p, "n": n}

    def frente_svm(clave: tuple[str, str]) -> dict:
        return tx("Frente a la SVM de referencia (regla de decisión): mejor en {k} y peor en {p} de 5 semillas; la regla pide 4.",
                  k=int(r.loc[clave, "semillas_mejor"]), p=int(r.loc[clave, "semillas_peor"]))

    def peor_dataset(clave: tuple[str, str]) -> dict:
        d = por_dataset(clave)
        return tx("AUROC más bajo dentro de un dataset de origen: {v} en {ds} (n = {n}).",
                  v=fijo(d.min()), ds=d.idxmin(), n=int((validacion()["dataset"] == d.idxmin()).sum()))

    def tarjeta(titulo: str, nota: str, fortalezas: list, debilidades: list) -> dict:
        return bloque("lista", titulo, {"fortalezas": fortalezas, "debilidades": debilidades}, nota=nota)

    # Modelo multietiqueta
    multi = {p: aciertos[(aciertos["modelo"] == "multietiqueta") & (aciertos["presupuesto"] == p)].mean(numeric_only=True) for p in ("todo", "2000")}
    referencia = aciertos[aciertos["modelo"] != "multietiqueta"].iloc[0]
    dos_mil = deteccion[deteccion["presupuesto"] == "2000"]
    no_peor = int((dos_mil["IC bajo"] > -0.01).sum())
    todo = deteccion[deteccion["presupuesto"] == "todo"].iloc[0]
    categorias = guardado["categorias"]
    macro = e[e["presupuesto"] == "todo"].assign(categoria=lambda t: t["etiqueta"].isin(categorias)).groupby("categoria")["AUROC"].mean()
    diagnosticos = e[(e["presupuesto"] == "todo") & ~e["etiqueta"].isin(categorias)].set_index("etiqueta")["AUPRC"]
    largo = validacion().loc[:, "cardiopatias_generales"].explode().dropna()
    varias = (largo.map(CATEGORIA_GENERAL).groupby(level=0).nunique() >= 2).sum() / (y == 1).sum()

    pct = lambda v: fijo(100 * v, 1)
    gb_fa = round(100 * (1 - float(media("2000", GB, "esp_sens99"))))
    return [
        bloque("aviso", nota="Los dos modelos que usa la UI van primero. " + FUENTE),
        tarjeta("Gradient boosting binario · en uso", "Notebooks 32 y 38", [
            tx("AUROC de {a} con 2,000 etiquetas: el {r}.º de {n} variantes probadas.", a=media("2000", GB), **lugar("2000", GB)),
            tx("Con todas las etiquetas, AUROC de {a} ({r}.º de {n}).", a=media("todo", GB), **lugar("todo", GB)),
            "Usa las 111 medidas de la señal: cada variable es una medida del ECG con nombre (intervalos, amplitudes, ejes, ST-T).",
        ], [
            frente_svm(GB),
            tx("Con 250 etiquetas, AUROC de {a} contra {b} de la SVM de referencia.", a=media("250", GB), b=media("250", SVM)),
            tx("Con 99% de sensibilidad y 2,000 etiquetas, especificidad de {e}: marca como riesgo al {fa}% de los ECGs normales.",
               e=media("2000", GB, "esp_sens99"), fa=gb_fa),
            peor_dataset(GB),
        ]),
        tarjeta("Gradient boosting multietiqueta · en uso", "Notebook 39 (experimento 063)", [
            tx("Dice qué cardiopatía: la categoría más probable es real en {a}% de los ECGs con cardiopatía ({b}% con 2,000 etiquetas), contra {c}% adivinando siempre la más frecuente.",
               a=pct(multi["todo"]["categoría más probable es real"]), b=pct(multi["2000"]["categoría más probable es real"]),
               c=pct(referencia["categoría más probable es real"])),
            tx("Uno de sus 3 diagnósticos más probables es real en {a}% ({c}% adivinando los más frecuentes).",
               a=pct(multi["todo"]["uno de los 3 diagnósticos más probables es real"]), c=pct(referencia["uno de los 3 diagnósticos más probables es real"])),
            tx("Macro-AUROC de {c} en las categorías y de {d} en los diagnósticos, con todas las etiquetas.",
               c=fijo(macro[True]), d=fijo(macro[False])),
            tx("Con todas las etiquetas, AUROC de {a} para detectar cardiopatía, contra {b} del binario.",
               a=fijo(todo["AUROC multietiqueta"]), b=fijo(todo["AUROC binario"])),
        ], [
            tx("Con 2,000 etiquetas, su detección {cumple} la no inferioridad frente al binario: {k} de 5 semillas, y la regla pide 4.",
               cumple="cumple" if no_peor >= 4 else "no cumple", k=no_peor),
            tx("El AUPRC de los diagnósticos va de {a} ({da}) a {b} ({db}).", a=fijo(diagnosticos.min()), da=diagnosticos.idxmin(),
               b=fijo(diagnosticos.max()), db=diagnosticos.idxmax()),
            tx("Su riesgo combina las categorías como si fueran independientes, pero el {p}% de los ECGs con cardiopatía de validación tiene dos o más.",
               p=pct(varias)),
        ]),
        tarjeta("SVM · referencia", "Notebook 32", [
            tx("Con 250 etiquetas, AUROC de {a}: el {r}.º de {n} variantes entrenadas con ese presupuesto.", a=media("250", SVM), **lugar("250", SVM)),
            tx("Con todas las etiquetas y 99% de sensibilidad, especificidad de {e} ({r}.º de {n}).",
               e=media("todo", SVM, "esp_sens99"), **lugar("todo", SVM, "esp_sens99")),
        ], [
            tx("Con 2,000 etiquetas, AUROC de {a} ({r}.º de {n}).", a=media("2000", SVM), **lugar("2000", SVM)),
            peor_dataset(SVM),
        ]),
        *[tarjeta(t, n, f, d) for t, n, f, d in otras_tarjetas(frente_svm, peor_dataset, lugar)],
    ]


def otras_tarjetas(frente_svm, peor_dataset, lugar) -> list[tuple]:
    grafo = ("062 Grafo con memoria", "Grafo con memoria")
    simdino = ("060 SimDINOv2", "SimDINOv2 congelado + intervalos")
    contaminado = ("060 SimDINOv2", "SimDINOv2 047 congelado (contaminado)")
    limpio = ("060 SimDINOv2", "SimDINOv2 congelado")
    xecg, crudo = ("059 xECG", "xECG afinado"), ("059 xECG", "xECG sin preentrenar")
    anomalia = ("061 Anomalías", "Anomalía: máximo (1% de AnomalyDINO)")
    historial = pd.read_csv(SALIDAS / CARPETA[grafo[0]] / "historial.csv", dtype={"presupuesto": str})
    minutos = historial[historial["presupuesto"] == "2000"].groupby("semilla")["seconds"].sum().mean() / 60
    horas = pd.read_csv(SALIDAS / CARPETA[simdino[0]] / "historial_preentrenamiento.csv")["seconds"].sum() / 3600
    return [
        ("Grafo con memoria · no elegido", "Notebook 36 (experimento 062)", [
            tx("Con todas las etiquetas, AUROC de {a}: el {r}.º de {n}.", a=media("todo", grafo), **lugar("todo", grafo)),
            tx("Con 2,000 etiquetas, AUROC de {a} ({r}.º de {n}).", a=media("2000", grafo), **lugar("2000", grafo)),
        ], [
            frente_svm(grafo),
            tx("Con 99% de sensibilidad y 2,000 etiquetas, especificidad de {e}, contra {s} de la SVM.",
               e=media("2000", grafo, "esp_sens99"), s=media("2000", SVM, "esp_sens99")),
            peor_dataset(grafo),
            tx("Entrenar una semilla con 2,000 etiquetas toma {m} min en el Mac (suma de sus épocas).", m=fijo(minutos, 1)),
            "Se programó a partir del resumen de Li et al. (2026); los tamaños y los pesos de las pérdidas son propios.",
        ]),
        ("SimDINOv2 propio · no elegido", "Notebook 34 (experimento 060)", [
            tx("Aprende sin etiquetas: se preentrenó con {n} ECGs sin etiquetar.", n=SIN_ETIQUETAR),
            tx("Con las 10 medidas de intervalos, AUROC de {a} con 2,000 etiquetas ({r}.º de {n}).", a=media("2000", simdino), **lugar("2000", simdino)),
            tx("El codificador limpio llega a {a} congelado, contra {b} del codificador 047 que vio PTB-XL completo.",
               a=media("2000", limpio), b=media("2000", contaminado)),
        ], [
            frente_svm(simdino),
            tx("El preentrenamiento tomó {h} h en el Mac.", h=fijo(horas, 1)),
        ]),
        ("xECG preentrenado · no elegido", "Notebook 33 (experimento 059)", [
            "Pesos ya preentrenados con EchoNext, que no tiene ECGs del dataset balanceado.",
        ], [
            tx("Su mejor forma, el ajuste fino completo, llega a {a} con 2,000 etiquetas ({r}.º de {n}).", a=media("2000", xecg), **lugar("2000", xecg)),
            frente_svm(xecg),
            tx("El preentrenamiento aporta poco: {a} contra {b} de la misma red sin preentrenar.", a=media("2000", xecg), b=media("2000", crudo)),
        ]),
        ("Anomalías estilo AnomalyDINO · no elegido", "Notebook 35 (experimento 061)", [
            "Solo necesita ECGs normales etiquetados.",
        ], [
            tx("AUROC de {a} con 2,000 etiquetas ({r}.º de {n}).", a=media("2000", anomalia), **lugar("2000", anomalia)),
            tx("Más normales en el banco casi no ayudan: {a} con 250 etiquetas y {b} con 2,000.", a=media("250", anomalia), b=media("2000", anomalia)),
            tx("Con 99% de sensibilidad y 2,000 etiquetas, especificidad de {e}.", e=media("2000", anomalia, "esp_sens99")),
        ]),
    ]


SECCIONES = {"modelos": cache(modelos), "resultados": cache(resultados), "eleccion": cache(eleccion)}
