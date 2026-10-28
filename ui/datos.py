"""Datos y cálculos de la UI.

Junta la tabla del notebook 28 (un ECG por fila, sin los "por definir") con el dataset balanceado del
notebook 29, que entra como un dataset más ("Balanceado"). Cada sección del EDA es una función que
recibe los datasets seleccionados y devuelve una lista de bloques (tablas y gráficas) para el navegador.
"""
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly
from scipy.stats import chi2_contingency

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOKS = RAIZ / "notebooks"   # las rutas de la tabla del notebook 28 son relativas a esta carpeta
TABLA_28 = RAIZ / "outputs/28-ecgs_info.pkl"
CLASIFICACION = NOTEBOOKS / "28-clasificacion-etiquetas.csv"
BALANCEADO = RAIZ / "data/processed/balanceado_jovenes_v2"
CACHE = RAIZ / "outputs/ui"   # lo escribe ui/preparar_senales.py

NUEVO = "Balanceado"
CLASES = ["con cardiopatía", "sin cardiopatía, con otros hallazgos", "normal"]
DOS_CLASES = ["normal", "con cardiopatía"]
DERIVACIONES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_VISOR = 250   # Hz con los que el visor manda cada ECG al navegador


def cargar_tabla() -> pd.DataFrame:
    """Tabla del notebook 28 más los ECGs del dataset balanceado como dataset "Balanceado"."""
    originales = pd.read_pickle(TABLA_28)
    originales["registro"] = originales["registro"].astype(str)
    originales["origen"] = originales["dataset"]

    balanceado = pd.read_csv(BALANCEADO / "ecgs.csv", usecols=["fila", "registro", "dataset"], dtype={"registro": str})
    nuevo = balanceado.merge(originales.drop(columns="origen"), on=["dataset", "registro"], how="left")
    nuevo["origen"] = nuevo["dataset"]
    nuevo["dataset"] = NUEVO

    tabla = pd.concat([originales, nuevo], ignore_index=True)
    # Copias idénticas: entre los datasets originales, y aparte dentro del balanceado
    tabla["universo"] = np.where(tabla["dataset"] == NUEVO, NUEVO, "originales")
    tabla["copia"] = tabla["huella"].notna() & tabla.duplicated(["universo", "huella"], keep=False)
    return tabla


TABLA = cargar_tabla()
ORIGINALES = [d for d in TABLA["dataset"].unique() if d != NUEVO]
DATASETS = ORIGINALES + [NUEVO]

clasificacion = pd.read_csv(CLASIFICACION).drop_duplicates("diagnostico")
CATEGORIA_DE = clasificacion.set_index("diagnostico")["categoria"].to_dict()
GENERAL_DE = clasificacion.set_index("diagnostico")["diagnostico_general"].to_dict()

MEDIDAS = pd.read_pickle(CACHE / "medidas.pkl").merge(
    TABLA[["dataset", "origen", "registro", "clase", "edad", "sexo"]], on=["dataset", "origen", "registro"])
FORMATO = pd.read_csv(CACHE / "formato.csv", index_col="dataset")
# Fila i = fila i de ecgs.csv. No va en Git (2.2 GB): lo crea scripts/data/crear_balanceado.py; sin él, la UI arranca igual
SENALES = np.load(BALANCEADO / "senales.npy", mmap_mode="r") if (BALANCEADO / "senales.npy").exists() else None


def tx(plantilla: str, **valores) -> dict:
    """Texto con números o nombres dentro; se traduce la plantilla y luego se llenan los valores (traducciones.py)."""
    return {"plantilla": plantilla, "valores": valores}


def bloque(tipo: str, titulo: str | dict = "", datos: pd.DataFrame | pd.Series | list | None = None, nota: str | dict = "", **opciones) -> dict:
    """Un elemento de la página; el navegador lo dibuja según su tipo."""
    if isinstance(datos, pd.Series):
        datos = datos.to_frame()
    if isinstance(datos, pd.DataFrame):
        nombre_indice = datos.index.name
        datos = json.loads(datos.to_json(orient="split", force_ascii=False))
        datos["nombre_indice"] = nombre_indice
    return {"tipo": tipo, "titulo": titulo, "nota": nota, "datos": datos, "opciones": opciones}


def filtrar(seleccion: tuple[str, ...]) -> pd.DataFrame:
    return TABLA[TABLA["dataset"].isin(seleccion)]


def aviso_doble_conteo(seleccion: tuple[str, ...]) -> list[dict]:
    if NUEVO in seleccion and len(seleccion) > 1:
        return [bloque("aviso", nota="El dataset balanceado reutiliza ECGs de los originales: al seleccionarlo junto con ellos, esos ECGs cuentan dos veces en los totales.")]
    return []


def cardiopatias_largo(t: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cardiopatía de cada ECG, con su categoría y su diagnóstico general."""
    cardio = t[["dataset", "cardiopatias"]].explode("cardiopatias").dropna().rename_axis("ecg").reset_index()
    cardio["categoria"] = cardio["cardiopatias"].map(CATEGORIA_DE)
    cardio["general"] = cardio["cardiopatias"].map(GENERAL_DE)
    return cardio


def histograma(valores: pd.Series, bordes: np.ndarray) -> np.ndarray:
    return np.histogram(valores.dropna(), bordes)[0]


def cajas(t: pd.DataFrame, columna: str, por: str, orden: list[str]) -> pd.DataFrame:
    """Cuartiles y bigotes (1.5 veces el rango intercuartílico, como matplotlib) de una medida por grupo."""
    filas = {}
    for nombre, grupo in t.groupby(por):
        valores = grupo[columna].dropna()
        q1, mediana, q3 = valores.quantile([0.25, 0.5, 0.75])
        dentro = valores[valores.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))]
        filas[nombre] = {"bigote inferior": dentro.min(), "Q1": q1, "mediana": mediana, "Q3": q3,
                         "bigote superior": dentro.max(), "ECGs": len(valores)}
    return pd.DataFrame.from_dict(filas, orient="index").reindex([o for o in orden if o in filas])


def v_de_cramer(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér entre dos variables categóricas, de 0 (nada) a 1 (una determina a la otra)."""
    tabla = pd.crosstab(x, y)
    if min(tabla.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tabla)[0]
    return float(np.sqrt(chi2 / (tabla.to_numpy().sum() * (min(tabla.shape) - 1))))


def describir(agrupado) -> pd.DataFrame:
    """describe() de pandas con los nombres de las columnas en español."""
    return agrupado.describe().rename(columns={
        "count": "ECGs con dato", "mean": "media", "std": "desv. estándar", "min": "mín",
        "50%": "mediana", "max": "máx"})


def rango_fc(fc: pd.Series) -> pd.Series:
    rango = pd.cut(fc, [0, 60, 100, 1000], right=False, labels=["menos de 60", "60 a 99", "100 o más"])
    return rango.cat.add_categories("sin estimar").fillna("sin estimar")


@lru_cache(maxsize=64)
def resumen(seleccion: tuple[str, ...]) -> list[dict]:
    t = filtrar(seleccion)
    por = t.groupby("dataset", sort=False)
    clase = pd.crosstab(t["dataset"], t["clase"], normalize="index").mul(100)
    planas = MEDIDAS[MEDIDAS["dataset"].isin(seleccion)].groupby("dataset")["derivaciones planas"].apply(lambda s: 100 * (s > 0).mean())
    tabla = pd.DataFrame({
        "ECGs": por.size(),
        "pacientes": por["paciente"].nunique(),
        "% con ID de paciente": por["paciente"].apply(lambda s: 100 * s.notna().mean()),
        "edad (mediana)": por["edad"].median(),
        "% mujeres": por["sexo"].apply(lambda s: 100 * s.dropna().eq("F").mean()),
        "cardiopatías distintas": por["cardiopatias"].apply(lambda s: s.dropna().explode().nunique()),
        "cardiopatías por ECG": por["n_cardiopatias"].mean().astype(float),
        "% con cardiopatía": clase.get("con cardiopatía"),
        "% normal": clase.get("normal"),
        "% con copia idéntica": 100 * por["copia"].mean(),
        "% con derivación plana (muestra)": planas,
    }).reindex(list(seleccion)).round(1)
    tabla["pacientes"] = tabla["pacientes"].astype(object).where(tabla["pacientes"] > 0, "sin ID")

    kpis = [{"etiqueta": "ECGs", "valor": len(t)},
            {"etiqueta": "Datasets", "valor": len(seleccion)},
            {"etiqueta": "Cardiopatías distintas", "valor": t["cardiopatias"].dropna().explode().nunique()},
            {"etiqueta": "Con cardiopatía", "valor": round(100 * t["clase"].dropna().eq("con cardiopatía").mean(), 1), "formato": "pct"}]
    return [
        *aviso_doble_conteo(seleccion),
        bloque("kpis", datos=kpis),
        bloque("tabla", "Resumen comparativo", tabla,
               nota="% con cardiopatía y % normal sobre los ECGs con diagnóstico. La derivación plana se mide en la muestra de señales (1,000 ECGs por dataset original, 100 de PTB Diagnostic; todos los del balanceado)."),
        bloque("barras", "ECGs por dataset", por.size().reindex(list(seleccion)).rename("ECGs"), formato="entero", eje="ECGs"),
        bloque("barras", "% de ECGs con cardiopatía", tabla["% con cardiopatía"], formato="pct", eje="% de ECGs con diagnóstico", ancho="medio"),
        bloque("barras", "Edad mediana", tabla["edad (mediana)"], formato=0, eje="años", ancho="medio"),
    ]


@lru_cache(maxsize=64)
def pacientes(seleccion: tuple[str, ...]) -> list[dict]:
    t = filtrar(seleccion)
    sel = list(seleccion)
    por = t.groupby("dataset", sort=False)

    edad = describir(por["edad"]).reindex(sel).round(1)
    bordes = np.arange(0, 106, 5)
    hist_edad = pd.DataFrame({d: histograma(g["edad"], bordes) for d, g in por}, index=bordes[:-1]).reindex(columns=sel)

    sexo = (pd.crosstab(t["dataset"], t["sexo"].fillna("sin dato"), normalize="index").mul(100)
            .reindex(index=sel).reindex(columns=["F", "M", "sin dato"], fill_value=0)
            .rename(columns={"F": "mujer", "M": "hombre"}).round(1))

    vacios = (t[["paciente", "edad", "sexo", "diagnosticos"]].isna().groupby(t["dataset"], sort=False).mean()
              .mul(100).reindex(sel).round(1).rename(columns={"diagnosticos": "diagnósticos"}))

    con_id = t.dropna(subset=["paciente"])
    ids = pd.DataFrame({
        "ECGs": por.size(),
        "ECGs con ID": con_id.groupby("dataset").size(),
        "pacientes": con_id.groupby("dataset")["paciente"].nunique(),
    }).reindex(sel).fillna(0).astype(int)
    ids["ECGs por paciente"] = (ids["ECGs con ID"] / ids["pacientes"].replace(0, np.nan)).round(2)

    return [
        *aviso_doble_conteo(seleccion),
        bloque("titulo", "Edad"),
        bloque("tabla", "Edad por dataset (años)", edad,
               nota=tx("Edad ya corregida en el notebook 28: 300 de PTB-XL pasa a 90 y -1 de CPSC queda vacío. "
                       "La edad 0 de Ningbo ({n} ECGs) se trata como desconocida y no entra al balanceado (notebooks 29 y 30).",
                       n=int((TABLA["dataset"].eq("Ningbo") & TABLA["edad"].eq(0)).sum()))),
        bloque("histogramas", "Distribución de la edad", hist_edad, ancho_bin=5, eje="Edad (años)", unidad="ECGs"),
        bloque("titulo", "Sexo"),
        bloque("apiladas", "Sexo por dataset", sexo, colores=["s3", "s2", "gris"]),
        bloque("titulo", "Datos vacíos y pacientes"),
        bloque("calor", "% de ECGs sin el dato", vacios, formato=1, maximo=100),
        bloque("tabla", "ECGs por paciente", ids,
               nota="Los datasets del Challenge (Ningbo, Chapman, Georgia, CPSC) no traen ID de paciente."),
    ]


@lru_cache(maxsize=64)
def diagnosticos(seleccion: tuple[str, ...]) -> list[dict]:
    t = filtrar(seleccion)
    sel = list(seleccion)
    cardio = cardiopatias_largo(t)
    con_dx = t.dropna(subset=["cardiopatias"]).groupby("dataset").size()

    clases = (pd.crosstab(t["dataset"], t["clase"], normalize="index").mul(100)
              .reindex(index=sel, columns=CLASES).fillna(0).round(1))
    por_ecg = (pd.crosstab(t["dataset"], t["n_cardiopatias"].clip(upper=4), normalize="index").mul(100)
               .reindex(sel).fillna(0).round(1))
    por_ecg.columns = [("4 o más" if c == 4 else str(c)) for c in por_ecg.columns]

    distintas = (cardio.drop_duplicates(["dataset", "cardiopatias"])
                 .pivot_table(index="dataset", columns="categoria", values="cardiopatias", aggfunc="count", fill_value=0)
                 .reindex(sel).fillna(0).astype(int))
    distintas["total"] = distintas.sum(axis=1)

    prevalencia = (cardio.drop_duplicates(["ecg", "categoria"])
                   .pivot_table(index="dataset", columns="categoria", values="ecg", aggfunc="count", fill_value=0)
                   .div(con_dx, axis=0).mul(100).reindex(sel).fillna(0).round(1))

    # % de ECGs de cada dataset con cada cardiopatía; las 20 con mayor % promedio entre los datasets
    pct = (cardio.drop_duplicates(["ecg", "cardiopatias"]).groupby(["cardiopatias", "dataset"]).size()
           .unstack().div(con_dx).mul(100))
    frecuentes = pct.loc[pct.fillna(0).mean(axis=1).nlargest(20).index].reindex(columns=sel).round(1).rename_axis("cardiopatía")

    por_edad = (t.pivot_table(index="dataset", columns="grupo_edad", values="con_cardiopatia", aggfunc="mean", observed=False)
                .mul(100).reindex(sel).round(1))
    por_sexo = (t.pivot_table(index="dataset", columns="sexo", values="con_cardiopatia", aggfunc="mean")
                .mul(100).reindex(sel).rename(columns={"F": "mujer", "M": "hombre"}).round(1))

    pares = (t["cardiopatias"].dropna().apply(lambda l: [" + ".join(par) for par in combinations(l, 2)])
             .explode().dropna().value_counts().head(15).rename_axis("par de cardiopatías").to_frame("ECGs"))

    bloques = [
        *aviso_doble_conteo(seleccion),
        bloque("titulo", "Clase de cada ECG"),
        bloque("apiladas", "Clase de cada ECG", clases, colores=["s1", "s3", "s4"],
               nota="Clasificación provisional de las etiquetas (notebook 28, sección 5), pendiente de revisar con el cardiólogo."),
        bloque("apiladas", "Cardiopatías distintas por ECG", por_ecg, colores=["o1", "o2", "o3", "o4", "o5"]),
        bloque("titulo", "Cardiopatías reportadas"),
        bloque("barras", "Cardiopatías distintas por dataset", distintas["total"], formato="entero", eje="cardiopatías distintas"),
        bloque("tabla", "Cardiopatías distintas por categoría", distintas),
        bloque("calor", "% de ECGs con al menos una cardiopatía de cada categoría", prevalencia, formato=1, maximo=100),
        bloque("calor", "Las 20 cardiopatías más frecuentes (% de los ECGs con diagnóstico)", frecuentes, formato=1,
               maximo=float(np.nanmax(frecuentes.to_numpy())) if frecuentes.size else 100,
               nota="Orden: % promedio entre los datasets seleccionados. Celda vacía: el dataset no reporta esa cardiopatía."),
    ]
    if len(sel) > 1:
        conjuntos = cardio.groupby("dataset")["cardiopatias"].agg(set)
        generales = cardio.groupby("dataset")["general"].agg(set)
        con = [d for d in sel if d in conjuntos.index]
        comunes = pd.DataFrame({a: {b: len(conjuntos[a] & conjuntos[b]) for b in con} for a in con}).loc[con, con]
        comunes_g = pd.DataFrame({a: {b: len(generales[a] & generales[b]) for b in con} for a in con}).loc[con, con]
        bloques += [
            bloque("calor", "Cardiopatías en común entre datasets (diagonal: total del dataset)", comunes, formato=0,
                   maximo=int(comunes.to_numpy().max())),
            bloque("calor", "Diagnósticos generales en común (junta subtipos de una misma enfermedad)", comunes_g, formato=0,
                   maximo=int(comunes_g.to_numpy().max())),
        ]
    bloques += [
        bloque("titulo", "Cardiopatía según edad y sexo"),
        bloque("calor", "% de ECGs con cardiopatía por grupo de edad", por_edad, formato=1, maximo=100),
        bloque("calor", "% de ECGs con cardiopatía por sexo", por_sexo, formato=1, maximo=100, ancho="medio"),
        bloque("tabla", "Pares de cardiopatías más frecuentes", pares, ancho="medio",
               nota="Sobre todos los ECGs seleccionados juntos."),
    ]
    return bloques


@lru_cache(maxsize=64)
def clases(seleccion: tuple[str, ...]) -> list[dict]:
    t = filtrar(seleccion)
    t = t[t["clase"].isin(DOS_CLASES)]
    colores = ["s4", "s1"]   # normal, con cardiopatía (como en el notebook 29)

    origenes = [d for d in ORIGINALES if d in set(t["origen"])]
    origen = pd.crosstab(t["origen"], t["clase"]).reindex(index=origenes, columns=DOS_CLASES, fill_value=0)
    pct_origen = origen.div(origen.sum(axis=1), axis=0).mul(100).round(1)
    origen["total"] = origen.sum(axis=1)

    edad = describir(t.groupby("clase")["edad"]).reindex(DOS_CLASES).round(1)
    bordes = np.arange(0, 106, 5)
    hist_edad = pd.DataFrame({c: histograma(t.loc[t["clase"] == c, "edad"], bordes) for c in DOS_CLASES}, index=bordes[:-1])

    sexo = (pd.crosstab(t["clase"], t["sexo"].fillna("sin dato"), normalize="index").mul(100)
            .reindex(index=DOS_CLASES).reindex(columns=["F", "M", "sin dato"], fill_value=0)
            .rename(columns={"F": "mujer", "M": "hombre"}).round(1))
    mujeres = (t.assign(mujer=t["sexo"].eq("F"))
               .pivot_table(index="grupo_edad", columns="clase", values="mujer", aggfunc="mean", observed=False)
               .reindex(columns=DOS_CLASES).mul(100).round(1).rename_axis("grupo de edad"))

    asociacion = pd.Series({
        "dataset de origen": v_de_cramer(t["origen"], t["clase"]),
        "sexo": v_de_cramer(t["sexo"], t["clase"]),
        "grupo de edad": v_de_cramer(t["grupo_edad"], t["clase"]),
    }, name="V de Cramér").dropna().round(3)

    enfermos = t[t["clase"] == "con cardiopatía"]
    generales = enfermos["cardiopatias"].apply(lambda l: sorted({GENERAL_DE[c] for c in l}))
    joven = enfermos["edad"] < 30
    jovenes = pd.DataFrame({
        "menores de 30 (ECGs)": generales[joven].explode().value_counts(),
        "30 o más (ECGs)": generales[~joven].explode().value_counts(),
    }).fillna(0).astype(int)
    jovenes["% en menores de 30"] = (100 * jovenes["menores de 30 (ECGs)"] / max(joven.sum(), 1)).round(1)
    jovenes["% en 30 o más"] = (100 * jovenes["30 o más (ECGs)"] / max((~joven).sum(), 1)).round(1)
    jovenes = jovenes.sort_values("menores de 30 (ECGs)", ascending=False).head(15).rename_axis("diagnóstico general")

    bloques = [
        *aviso_doble_conteo(seleccion),
        bloque("aviso", nota=tx("Solo ECGs normales y con cardiopatía ({n}), analizados como un solo conjunto. "
                                "Con el dataset balanceado reproduce el notebook 29.", n=len(t))),
        bloque("titulo", "Clase y dataset de origen"),
        bloque("apiladas", "Clase según el dataset de origen", pct_origen, colores=colores),
        bloque("tabla", "ECGs por dataset de origen y clase", origen, ancho="medio"),
        bloque("barras", "Asociación con la clase (V de Cramér: 0 = ninguna, 1 = la determina)", asociacion,
               formato=3, eje="V de Cramér", ancho="medio"),
        bloque("titulo", "Edad y sexo por clase"),
        bloque("tabla", "Edad por clase (años)", edad),
        bloque("histogramas", "Edad por clase", hist_edad, ancho_bin=5, eje="Edad (años)", unidad="ECGs", colores=colores),
        bloque("apiladas", "Sexo por clase", sexo, colores=["s3", "s2", "gris"], ancho="medio"),
        bloque("tabla", "% de mujeres por grupo de edad y clase", mujeres, ancho="medio"),
        bloque("titulo", "Cardiopatías en jóvenes y mayores"),
        bloque("tabla", "Diagnósticos generales más frecuentes en menores de 30", jovenes,
               nota=tx("ECGs con cardiopatía: {jovenes} de menores de 30 y {mayores} de 30 o más.",
                       jovenes=int(joven.sum()), mayores=int((~joven).sum()))),
    ]

    m = MEDIDAS[MEDIDAS["dataset"].isin(seleccion) & MEDIDAS["clase"].isin(DOS_CLASES)]
    amplitud = m.groupby("clase")[DERIVACIONES].median().reindex(DOS_CLASES).T.round(2).rename_axis("derivación").dropna(how="all")
    fc = pd.crosstab(m["clase"], rango_fc(m["frecuencia cardiaca"]), normalize="index").mul(100).reindex(DOS_CLASES).round(1)
    bloques += [
        bloque("titulo", "Señales por clase"),
        bloque("aviso", nota=tx("Medidas sobre los {n} ECGs con señal medida: la muestra de 1,000 por dataset original "
                                "(100 de PTB Diagnostic) y todos los del balanceado.", n=len(m))),
        bloque("agrupadas", "Amplitud pico a pico mediana por derivación (mV)", amplitud, colores=colores, formato=2, eje="mV"),
        bloque("apiladas", "Frecuencia cardiaca por clase (latidos por minuto, derivación II)", fc,
               colores=["o2", "o4", "o5", "gris"], ancho="medio"),
        bloque("tabla", "Frecuencia cardiaca por clase", describir(m.groupby("clase")["frecuencia cardiaca"]).reindex(DOS_CLASES).round(1),
               ancho="medio", nota="Estimada con una versión simplificada de Pan-Tompkins (notebook 29)."),
    ]
    return bloques


@lru_cache(maxsize=64)
def senales(seleccion: tuple[str, ...]) -> list[dict]:
    sel = list(seleccion)
    m = MEDIDAS[MEDIDAS["dataset"].isin(seleccion)]

    calidad = m.groupby("dataset").agg(
        ecgs=("registro", "size"),
        planas=("derivaciones planas", lambda s: 100 * (s > 0).mean()),
        vacias=("muestras vacías", lambda s: 100 * (s > 0).mean()),
        amplitud=("II", "median"),
        deriva=("deriva de línea base", "median"),
        ruido=("ruido", "median"),
        fc=("frecuencia cardiaca", "median"),
    ).reindex(sel).round({"planas": 2, "vacias": 2, "amplitud": 2, "deriva": 2, "ruido": 3, "fc": 1})
    calidad.columns = ["ECGs medidos", "% con derivación plana", "% con muestras vacías", "amplitud II (mediana)",
                       "deriva de línea base (mediana)", "ruido (mediana)", "frecuencia cardiaca (mediana)"]

    amplitud = m.groupby("dataset")[DERIVACIONES].median().reindex(sel).round(2)
    fc = pd.crosstab(m["dataset"], rango_fc(m["frecuencia cardiaca"]), normalize="index").mul(100).reindex(sel).round(1)
    extremos = m.assign(extremo=m[DERIVACIONES].max(axis=1) > 10).groupby(["dataset", "origen"]).agg(ECGs=("extremo", "sum"), medidos=("extremo", "size"))
    extremos = extremos[extremos["ECGs"] > 0].reindex(sel, level="dataset")
    extremos["% de los medidos"] = (100 * extremos["ECGs"] / extremos["medidos"]).round(1)
    extremos.index = [d if d == o else f"{d} · {o}" for d, o in extremos.index]

    bloques = [
        bloque("titulo", "Formato"),
        bloque("tabla", "Derivaciones, frecuencia de muestreo y duración", FORMATO.reindex(sel)),
        bloque("titulo", "Calidad"),
        bloque("aviso", nota="Medidas en los primeros 10 s de las 12 derivaciones, sobre la muestra de 1,000 ECGs por dataset "
                             "original (100 de PTB Diagnostic, semilla 42, la misma del notebook 28) y los 20,000 del balanceado. "
                             "Las señales del balanceado llevan un filtro pasa-altas de 0.5 Hz (notebook 29)."),
        bloque("tabla", "Calidad de las señales (mV)", calidad, nota="Amplitud, deriva y ruido en mV."),
        bloque("barras", "% de ECGs con al menos una derivación plana", calidad["% con derivación plana"], formato="pct",
               eje="% de ECGs medidos", ancho="medio"),
        bloque("barras", "% de ECGs con muestras vacías", calidad["% con muestras vacías"], formato="pct",
               eje="% de ECGs medidos", ancho="medio"),
        bloque("cajas", "Amplitud pico a pico de la derivación II", cajas(m, "II", "dataset", sel), eje="mV", ancho="medio"),
        bloque("cajas", "Deriva de línea base", cajas(m, "deriva de línea base", "dataset", sel), eje="mV", ancho="medio"),
        bloque("cajas", "Ruido (cambios de menos de 10 ms)", cajas(m, "ruido", "dataset", sel), eje="mV", ancho="medio"),
        bloque("cajas", "Frecuencia cardiaca (derivación II)", cajas(m, "frecuencia cardiaca", "dataset", sel), eje="latidos por minuto", ancho="medio"),
        bloque("titulo", "Amplitud y frecuencia cardiaca"),
        bloque("calor", "Amplitud pico a pico mediana por derivación (mV)", amplitud, formato=2,
               maximo=float(np.nanmax(amplitud.to_numpy())) if amplitud.size else 1),
        bloque("apiladas", "Frecuencia cardiaca estimada (latidos por minuto)", fc, colores=["o2", "o4", "o5", "gris"]),
    ]
    if len(extremos):
        bloques.append(bloque("tabla", "ECGs con alguna derivación de más de 10 mV pico a pico", extremos.rename_axis("dataset")))
    return bloques


@lru_cache(maxsize=64)
def duplicados(seleccion: tuple[str, ...]) -> list[dict]:
    t = filtrar(seleccion)
    sel = list(seleccion)
    copias = t[t["copia"]]
    por = pd.DataFrame({
        "ECGs": t.groupby("dataset").size(),
        "ECGs con una copia idéntica": copias.groupby("dataset").size(),
    }).reindex(sel).fillna(0).astype(int)
    por["% con copia"] = (100 * por["ECGs con una copia idéntica"] / por["ECGs"]).round(1)

    # Grupos de ECGs idénticos entre los datasets originales que tocan algún dataset seleccionado
    originales = TABLA[(TABLA["universo"] == "originales") & TABLA["copia"]]
    grupos = originales.groupby("huella")["dataset"].agg(lambda d: frozenset(d))
    grupos = grupos[grupos.apply(lambda d: bool(d & set(seleccion)))]
    combinaciones = (grupos.apply(lambda d: " + ".join(x for x in ORIGINALES if x in d)).value_counts()
                     .rename_axis("datasets donde aparece cada copia").to_frame("grupos"))

    sin_archivo = t.loc[t["huella"].isna() & (t["dataset"] != NUEVO), ["dataset", "registro"]].set_index("dataset")

    bloques = [
        bloque("aviso", nota="Dos ECGs son copias idénticas si su archivo de señal tiene la misma huella MD5 (notebook 28, sección 15). "
                             "Se buscan entre los datasets originales y, aparte, dentro del balanceado."),
        bloque("barras", "% de ECGs con una copia idéntica", por["% con copia"], formato="pct", eje="% de ECGs", ancho="medio"),
        bloque("tabla", "Copias idénticas por dataset", por, ancho="medio"),
    ]
    if len(combinaciones):
        bloques.append(bloque("tabla", tx("Grupos de ECGs idénticos ({n} grupos)", n=len(grupos)), combinaciones))
    if len(sin_archivo):
        bloques.append(bloque("tabla", "ECGs con cabecera pero sin archivo de señal", sin_archivo))
    return bloques


SECCIONES = {"resumen": resumen, "pacientes": pacientes, "diagnosticos": diagnosticos,
             "clases": clases, "senales": senales, "duplicados": duplicados}


def vacio_a_none(valor):
    if isinstance(valor, list):
        return valor
    return None if pd.isna(valor) else valor


def ecg_al_azar(dataset: str, clase: str | None) -> dict:
    """Un ECG al azar del dataset (y de la clase, si se pide): sus datos y los primeros 10 s de sus 12 derivaciones."""
    t = TABLA[TABLA["dataset"] == dataset]
    if clase:
        t = t[t["clase"] == clase]
    if t.empty:
        raise LookupError(tx("{dataset} no tiene ECGs de clase '{clase}'.", dataset=dataset, clase=clase))
    fila = t.sample(1).iloc[0]
    if dataset == NUEVO:
        if SENALES is None:
            raise LookupError(tx("Las señales del dataset balanceado no están en este equipo: créalas con scripts/data/crear_balanceado.py."))
        senal, fs = SENALES[int(fila["fila"])].astype(np.float32), 500
    else:
        ruta = NOTEBOOKS / fila["ruta"]
        if not ruta.with_name(ruta.name + ".hea").exists():
            raise LookupError(tx("Los archivos de {dataset} no están en data/raw: descárgalos de PhysioNet para ver sus ECGs.", dataset=dataset))
        senal, campos = wfdb.rdsamp(str(ruta))
        fs = int(campos["fs"])
    senal = resample_poly(senal[: 10 * fs, :12], FS_VISOR, fs, axis=0)
    return {
        "dataset": dataset,
        "origen": fila["origen"],
        "registro": fila["registro"],
        "edad": vacio_a_none(fila["edad"]),
        "sexo": vacio_a_none(fila["sexo"]),
        "clase": vacio_a_none(fila["clase"]),
        "cardiopatias": vacio_a_none(fila["cardiopatias"]),
        "diagnosticos": vacio_a_none(fila["diagnosticos"]),
        "ecgs_disponibles": len(t),
        "fs": FS_VISOR,
        "derivaciones": DERIVACIONES,
        "senal": json.loads(pd.DataFrame(senal.T).round(3).to_json(orient="values")),
    }
