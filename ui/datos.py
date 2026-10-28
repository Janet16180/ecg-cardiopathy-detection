"""Datos y cálculos de la UI.

Junta la tabla del notebook 28 (un ECG por fila, sin los "por definir") con el dataset balanceado del
notebook 43 (v3), que entra como un dataset más ("Balanceado"). Cada sección del EDA es una función que
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

from ecg_experiment import balanceado_v3 as v3

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOKS = RAIZ / "notebooks"   # las rutas de la tabla del notebook 28 son relativas a esta carpeta
TABLA_28 = RAIZ / "outputs/28-ecgs_info.pkl"
BALANCEADO = RAIZ / "data/processed/balanceado_jovenes_v3"
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

    balanceado = v3.load_table(RAIZ).reset_index()[["fila", "registro", "dataset", "grupo"]]
    nuevo = balanceado.merge(originales.drop(columns=["origen", "paciente"]), on=["dataset", "registro"], how="left")
    nuevo = nuevo.rename(columns={"grupo": "paciente"})   # ID del dataset o, si no trae, el propio registro (notebook 31)
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

MEDIDAS = pd.read_pickle(CACHE / "medidas.pkl").merge(
    TABLA[["dataset", "origen", "registro", "clase", "edad", "sexo"]], on=["dataset", "origen", "registro"])
FORMATO = pd.read_csv(CACHE / "formato.csv", index_col="dataset")
# Fila i = fila i de ecgs.csv. No va en Git (2.2 GB): lo crea scripts/data/crear_balanceado.py; sin él, la UI arranca igual
SENALES = np.load(BALANCEADO / "senales.npy", mmap_mode="r") if (BALANCEADO / "senales.npy").exists() else None
# Entrenamiento, validación o prueba de cada ECG del balanceado (notebook 43)
PARTICION = pd.read_csv(BALANCEADO / "particion.csv", index_col="fila")["split"]
# Código original de cada etiqueta (SNOMED CT o SCP), en el orden de "diagnosticos"; lo escribe ui/preparar_etiquetas.py
CODIGOS = pd.read_csv(CACHE / "codigos_etiquetas.csv", dtype=str).set_index(["dataset", "registro"])["codigos"].str.split(",").to_dict()


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


def cardiopatias_largo(t: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cardiopatía de cada ECG."""
    return t[["dataset", "cardiopatias"]].explode("cardiopatias").dropna().rename_axis("ecg").reset_index()


def histograma(valores: pd.Series, bordes: np.ndarray) -> np.ndarray:
    return np.histogram(valores.dropna(), bordes)[0]


def describir(agrupado) -> pd.DataFrame:
    """describe() de pandas con los nombres de las columnas en español."""
    return agrupado.describe().rename(columns={
        "count": "ECGs con dato", "mean": "media", "std": "desv. estándar", "min": "mín",
        "50%": "mediana", "max": "máx"})


def rango_fc(fc: pd.Series) -> pd.Series:
    rango = pd.cut(fc, [0, 60, 100, 1000], right=False, labels=["menos de 60", "60 a 99", "100 o más"])
    return rango.cat.add_categories("sin estimar").fillna("sin estimar")


def particion_balanceado() -> pd.DataFrame:
    """ECGs, clases y pacientes de cada parte del dataset balanceado."""
    b = TABLA[TABLA["dataset"] == NUEVO]
    parte = PARTICION.reindex(b["fila"]).map({"train": "entrenamiento", "val": "validación", "test": "prueba"}).to_numpy()
    por = b.groupby(parte)
    tabla = pd.DataFrame({
        "ECGs": por.size(),
        "% del dataset": (100 * por.size() / len(b)).round(1),
        "normales": por["clase"].apply(lambda s: int(s.eq("normal").sum())),
        "con cardiopatía": por["clase"].apply(lambda s: int(s.eq("con cardiopatía").sum())),
        "pacientes": por["paciente"].nunique(),
    }).reindex(["entrenamiento", "validación", "prueba"])
    tabla.loc["total"] = [len(b), 100.0, int(b["clase"].eq("normal").sum()), int(b["clase"].eq("con cardiopatía").sum()), b["paciente"].nunique()]
    return tabla.rename_axis("parte")


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
        bloque("kpis", datos=kpis),
        bloque("tabla", "Resumen comparativo", tabla),
        *([bloque("tabla", "Partición del dataset balanceado", particion_balanceado())] if NUEVO in seleccion else []),
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
        bloque("titulo", "Edad"),
        bloque("tabla", "Edad por dataset (años)", edad),
        bloque("histogramas", "Distribución de la edad", hist_edad, ancho_bin=5, eje="Edad (años)", unidad="ECGs"),
        bloque("titulo", "Sexo"),
        bloque("apiladas", "Sexo por dataset", sexo, colores=["s3", "s2", "gris"]),
        bloque("titulo", "Datos vacíos y pacientes"),
        bloque("calor", "% de ECGs sin el dato", vacios, formato=1, maximo=100),
        bloque("tabla", "ECGs por paciente", ids),
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

    distintas = cardio.drop_duplicates(["dataset", "cardiopatias"]).groupby("dataset").size().reindex(sel, fill_value=0)

    # ECGs y % de ECGs de cada dataset con cada cardiopatía, ordenadas por su % promedio entre los datasets.
    # Bajo 1%, dos cifras significativas: con un decimal, las que tienen pocos ECGs se verían como 0.
    n = (cardio.drop_duplicates(["ecg", "cardiopatias"]).groupby(["cardiopatias", "dataset"]).size()
         .unstack(fill_value=0).reindex(columns=sel, fill_value=0))
    pct = n.div(con_dx.reindex(sel), axis=1).mul(100)
    orden = pct.mean(axis=1).sort_values(ascending=False).index
    todas = pd.DataFrame(index=orden).rename_axis("cardiopatía")
    for d in sel:
        todas[f"{d} · ECGs"] = n.loc[orden, d]
        todas[f"{d} · %"] = pct.loc[orden, d].map(lambda v: round(v, 1) if v >= 1 else float(f"{v:.2g}"))

    por_edad = (t.pivot_table(index="dataset", columns="grupo_edad", values="con_cardiopatia", aggfunc="mean", observed=False)
                .mul(100).reindex(sel).round(1))
    por_sexo = (t.pivot_table(index="dataset", columns="sexo", values="con_cardiopatia", aggfunc="mean")
                .mul(100).reindex(sel).rename(columns={"F": "mujer", "M": "hombre"}).round(1))

    pares = (t["cardiopatias"].dropna().apply(lambda l: [" + ".join(par) for par in combinations(l, 2)])
             .explode().dropna().value_counts().head(15).rename_axis("par de cardiopatías").to_frame("ECGs"))

    bloques = [
        bloque("titulo", "Clase de cada ECG"),
        bloque("apiladas", "Clase de cada ECG", clases, colores=["s1", "s3", "s4"]),
        bloque("apiladas", "Cardiopatías distintas por ECG", por_ecg, colores=["o1", "o2", "o3", "o4", "o5"]),
        bloque("titulo", "Cardiopatías reportadas"),
        bloque("barras", "Cardiopatías distintas por dataset", distintas.rename("cardiopatías distintas"), formato="entero",
               eje="cardiopatías distintas"),
        bloque("tabla", "Cardiopatías, de la más a la menos frecuente (ECGs y % de los ECGs con diagnóstico)", todas, visibles=20),
    ]
    if len(sel) > 1:
        conjuntos = cardio.groupby("dataset")["cardiopatias"].agg(set)
        con = [d for d in sel if d in conjuntos.index]
        comunes = pd.DataFrame({a: {b: len(conjuntos[a] & conjuntos[b]) for b in con} for a in con}).loc[con, con]
        bloques += [
            bloque("calor", "Cardiopatías en común entre datasets (diagonal: total del dataset)", comunes, formato=0,
                   maximo=int(comunes.to_numpy().max())),
        ]
    bloques += [
        bloque("titulo", "Cardiopatía según edad y sexo"),
        bloque("calor", "% de ECGs con cardiopatía por grupo de edad", por_edad, formato=1, maximo=100),
        bloque("calor", "% de ECGs con cardiopatía por sexo", por_sexo, formato=1, maximo=100, ancho="medio"),
        bloque("tabla", "Pares de cardiopatías más frecuentes", pares, ancho="medio"),
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

    enfermos = t[t["clase"] == "con cardiopatía"]
    joven = enfermos["edad"] < 30
    jovenes = pd.DataFrame({
        "menores de 30 (ECGs)": enfermos.loc[joven, "cardiopatias"].explode().value_counts(),
        "30 o más (ECGs)": enfermos.loc[~joven, "cardiopatias"].explode().value_counts(),
    }).fillna(0).astype(int)
    jovenes["% en menores de 30"] = (100 * jovenes["menores de 30 (ECGs)"] / max(joven.sum(), 1)).round(1)
    jovenes["% en 30 o más"] = (100 * jovenes["30 o más (ECGs)"] / max((~joven).sum(), 1)).round(1)
    jovenes = jovenes.sort_values("menores de 30 (ECGs)", ascending=False).head(15).rename_axis("cardiopatía")

    bloques = [
        bloque("titulo", "Clase y dataset de origen"),
        bloque("apiladas", "Clase según el dataset de origen", pct_origen, colores=colores, ancho="medio"),
        bloque("tabla", "ECGs por dataset de origen y clase", origen, ancho="medio"),
        bloque("titulo", "Edad y sexo por clase"),
        bloque("tabla", "Edad por clase (años)", edad),
        bloque("histogramas", "Edad por clase", hist_edad, ancho_bin=5, eje="Edad (años)", unidad="ECGs", colores=colores),
        bloque("apiladas", "Sexo por clase", sexo, colores=["s3", "s2", "gris"], ancho="medio"),
        bloque("tabla", "% de mujeres por grupo de edad y clase", mujeres, ancho="medio"),
        bloque("titulo", "Cardiopatías en jóvenes y mayores"),
        bloque("tabla", "Cardiopatías más frecuentes en menores de 30", jovenes),
    ]

    m = MEDIDAS[MEDIDAS["dataset"].isin(seleccion) & MEDIDAS["clase"].isin(DOS_CLASES)]
    amplitud = m.groupby("clase")[DERIVACIONES].median().reindex(DOS_CLASES).T.round(2).rename_axis("derivación").dropna(how="all")
    fc = pd.crosstab(m["clase"], rango_fc(m["frecuencia cardiaca"]), normalize="index").mul(100).reindex(DOS_CLASES).round(1)
    bloques += [
        bloque("titulo", "Señales por clase"),
        bloque("agrupadas", "Amplitud pico a pico mediana por derivación (mV)", amplitud, colores=colores, formato=2, eje="mV"),
        bloque("apiladas", "Frecuencia cardiaca por clase (latidos por minuto, derivación II)", fc,
               colores=["o2", "o4", "o5", "gris"], ancho="medio"),
        bloque("tabla", "Frecuencia cardiaca por clase", describir(m.groupby("clase")["frecuencia cardiaca"]).reindex(DOS_CLASES).round(1),
               ancho="medio", nota="Estimada con una versión simplificada de Pan-Tompkins."),
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

    return [
        bloque("titulo", "Formato"),
        bloque("tabla", "Derivaciones, frecuencia de muestreo y duración", FORMATO.reindex(sel)),
        bloque("titulo", "Calidad"),
        bloque("tabla", "Calidad de las señales (mV)", calidad, nota="Amplitud, deriva y ruido en mV."),
        bloque("barras", "% de ECGs con al menos una derivación plana", calidad["% con derivación plana"], formato="pct",
               eje="% de ECGs medidos", ancho="medio"),
        bloque("barras", "% de ECGs con muestras vacías", calidad["% con muestras vacías"], formato="pct",
               eje="% de ECGs medidos", ancho="medio"),
    ]


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
        bloque("aviso", nota="Dos ECGs son copias idénticas si su archivo de señal tiene la misma huella MD5."),
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
    nombres = vacio_a_none(fila["diagnosticos"]) or []
    codigos = CODIGOS.get((fila["origen"], fila["registro"]), [None] * len(nombres))   # PTB Diagnostic no usa códigos
    return {
        "dataset": dataset,
        "origen": fila["origen"],
        "registro": fila["registro"],
        "edad": vacio_a_none(fila["edad"]),
        "sexo": vacio_a_none(fila["sexo"]),
        "clase": vacio_a_none(fila["clase"]),
        "cardiopatias": vacio_a_none(fila["cardiopatias"]),
        "etiquetas": [{"codigo": c, "nombre": n} for c, n in zip(codigos, nombres)],
        "ecgs_disponibles": len(t),
        "fs": FS_VISOR,
        "derivaciones": DERIVACIONES,
        "senal": json.loads(pd.DataFrame(senal.T).round(3).to_json(orient="values")),
    }
