"""Lectura y análisis de un ECG nuevo ("Registrar nuevo ECG" en Inicio).

Los archivos llegan en memoria. Los de WFDB se escriben un momento en una carpeta temporal para que wfdb los
lea, y la carpeta se borra al terminar. Acepta WFDB (.hea con su .dat o .mat), CSV y NumPy. Lo que se conserva
(archivos y resultado) lo guarda historial.py en la base SQLite local.
"""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

import areas
import datos
import modelo
from datos import tx
from medidas_senal import DERIVACIONES, medidas, pasa_altas, picos_r

NOMBRES = {d.lower(): d for d in DERIVACIONES}             # "avr" -> "aVR"
A_MV = {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "μv": 1e-3}    # factor para pasar cada unidad a mV
COLUMNAS_DE_TIEMPO = {"time", "tiempo", "t", "s", "ms", "seconds", "segundos", "sample", "muestra", "index"}


class ECGIlegible(Exception):
    """El archivo no se puede leer como ECG de 12 derivaciones; el mensaje es una plantilla traducible."""


def ordenar(senal: np.ndarray, nombres: list[str], avisos: list) -> np.ndarray:
    """Deja las 12 derivaciones en el orden I, II, III, aVR, aVL, aVF, V1-V6."""
    claves = [NOMBRES.get(str(n).strip().lower()) for n in nombres]
    if all(d in claves for d in DERIVACIONES):
        return senal[:, [claves.index(d) for d in DERIVACIONES]]
    if senal.shape[1] >= 12:
        avisos.append(tx("No se reconocieron los nombres de las derivaciones; se asumió el orden estándar (I, II, III, aVR, aVL, aVF, V1 a V6)."))
        return senal[:, :12]
    raise ECGIlegible(tx("Se necesitan 12 derivaciones y el archivo tiene {n}.", n=senal.shape[1]))


def leer_wfdb(archivos: dict[str, bytes], avisos: list) -> tuple[np.ndarray, float, dict]:
    cabecera = next(n for n in archivos if n.lower().endswith(".hea"))
    with tempfile.TemporaryDirectory() as carpeta:
        for nombre, contenido in archivos.items():
            (Path(carpeta) / nombre).write_bytes(contenido)
        senal, campos = wfdb.rdsamp(str(Path(carpeta) / Path(cabecera).stem))
    factores = []
    for unidad in campos["units"]:
        if unidad.lower() not in A_MV:
            avisos.append(tx("La unidad {unidad} no se reconoce; se asumió mV.", unidad=unidad))
        factores.append(A_MV.get(unidad.lower(), 1.0))
    senal = ordenar(senal * np.array(factores), campos["sig_name"], avisos)
    ficha = {"archivo": cabecera, "formato": "WFDB", "derivaciones_archivo": len(campos["sig_name"]),
             "unidades": ", ".join(sorted(set(campos["units"])))}
    return senal, float(campos["fs"]), ficha


def leer_tabla(nombre: str, contenido: bytes, avisos: list) -> tuple[np.ndarray, dict]:
    """CSV o NumPy: una columna por derivación (o una fila por derivación en NumPy), en mV."""
    if nombre.lower().endswith(".npy"):
        arreglo = np.squeeze(np.load(io.BytesIO(contenido), allow_pickle=False)).astype(float)   # sin pickle: no ejecuta código
        if arreglo.ndim != 2:
            raise ECGIlegible(tx("El arreglo debe tener 2 dimensiones (muestras × derivaciones)."))
        if arreglo.shape[0] < arreglo.shape[1]:
            arreglo = arreglo.T
        columnas = arreglo.shape[1]
        senal, formato = ordenar(arreglo, [""] * columnas, avisos), "NumPy"
    else:
        tabla = pd.read_csv(io.BytesIO(contenido))
        if all(pd.to_numeric(pd.Series(tabla.columns), errors="coerce").notna()):   # sin encabezado
            tabla = pd.read_csv(io.BytesIO(contenido), header=None)
        tabla = tabla.drop(columns=[c for c in tabla.columns if str(c).strip().lower() in COLUMNAS_DE_TIEMPO])
        tabla = tabla.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        columnas = tabla.shape[1]
        senal, formato = ordenar(tabla.to_numpy(float), list(tabla.columns), avisos), "CSV"
    avisos.append(tx("Se asumió que los valores están en mV."))
    return senal, {"archivo": nombre, "formato": formato, "derivaciones_archivo": columnas, "unidades": "mV"}


def leer(archivos: dict[str, bytes], fs_indicada: float) -> tuple[np.ndarray, float, dict, list]:
    """Señal (muestras × 12, en mV), frecuencia de muestreo, ficha del archivo y avisos."""
    avisos = []
    nombres = [n.lower() for n in archivos]
    try:
        if any(n.endswith(".hea") for n in nombres):
            senal, fs, ficha = leer_wfdb(archivos, avisos)
        elif len(archivos) == 1 and nombres[0].endswith((".csv", ".npy")):
            nombre, contenido = next(iter(archivos.items()))
            senal, ficha = leer_tabla(nombre, contenido, avisos)
            fs = fs_indicada
            avisos.append(tx("Se usó la frecuencia de muestreo indicada al subir el archivo: {fs} Hz.", fs=fs_indicada))
        else:
            raise ECGIlegible(tx("Formato no reconocido: sube un .hea con su .dat o .mat (WFDB), un .csv o un .npy."))
    except ECGIlegible:
        raise
    except FileNotFoundError as error:   # la cabecera nombra un archivo de señal que no se subió
        raise ECGIlegible(tx("Falta el archivo {archivo} que indica la cabecera.", archivo=Path(error.filename).name))
    except Exception as error:   # archivo dañado o incompleto: se avisa en vez de fallar
        raise ECGIlegible(tx("No se pudo leer el archivo: {detalle}", detalle=str(error)))
    return senal, fs, {**ficha, "fs": fs, "duracion": round(len(senal) / fs, 2)}, avisos


def analizar(senal: np.ndarray, fs: float, ficha: dict, avisos: list, cual: str) -> dict:
    """Medidas de la señal, comparación con el dataset balanceado, ECG para dibujar, resultado del modelo elegido y
    sus áreas de interés (None si el modelo no corrió)."""
    fs_entera = int(round(fs))
    duracion = len(senal) / fs
    primeros = senal[: int(10 * fs)]
    m = medidas(primeros, fs_entera)
    filtrada = pasa_altas(np.nan_to_num(primeros), fs)   # como las señales del dataset balanceado v2
    m_filtrada = medidas(filtrada, fs_entera)
    picos = picos_r(np.nan_to_num(primeros[:, 1]), fs)

    if duracion < 10:
        avisos.append(tx("El ECG dura {s} s, menos de los 10 s que usan las medidas y el modelo.", s=round(duracion, 1)))
    if m["derivaciones planas"]:
        avisos.append(tx("Derivaciones planas (señal constante): {n}.", n=m["derivaciones planas"]))
    if m["muestras vacías"]:
        avisos.append(tx("Hay {n} muestras vacías (NaN); se trataron como 0.", n=m["muestras vacías"]))
    if np.isnan(m["frecuencia cardiaca"]):
        avisos.append(tx("No se pudo estimar la frecuencia cardiaca en la derivación II."))
    if max(m[d] for d in DERIVACIONES) > 10:
        avisos.append(tx("Alguna derivación pasa de 10 mV pico a pico; revisa las unidades."))

    senal_medidas = pd.Series({
        "latidos detectados (derivación II)": len(picos),
        "frecuencia cardiaca (latidos por minuto)": m["frecuencia cardiaca"],
        "intervalo RR medio (ms)": 1000 * np.mean(np.diff(picos)) / fs if len(picos) > 1 else np.nan,
        "derivaciones planas": m["derivaciones planas"],
        "muestras vacías": m["muestras vacías"],
    }, name="valor").rename_axis("medida").round(1)

    # Referencia: los 20,000 ECGs del dataset balanceado, por clase
    referencia = datos.MEDIDAS[datos.MEDIDAS["dataset"] == datos.NUEVO]
    referencia = referencia.assign(**{"amplitud máxima": referencia[DERIVACIONES].max(axis=1)})
    normales = referencia[referencia["clase"] == "normal"]
    enfermos = referencia[referencia["clase"] == "con cardiopatía"]
    este = {**m_filtrada, "amplitud máxima": max(m_filtrada[d] for d in DERIVACIONES)}
    filas = {"frecuencia cardiaca (latidos por minuto)": "frecuencia cardiaca", "amplitud de la derivación II (mV)": "II",
             "amplitud máxima entre derivaciones (mV)": "amplitud máxima", "deriva de línea base (mV)": "deriva de línea base",
             "ruido (mV)": "ruido"}
    comparacion = pd.DataFrame({
        "este ECG": {f: este[c] for f, c in filas.items()},
        "mediana de los normales": {f: normales[c].median() for f, c in filas.items()},
        "mediana de los ECGs con cardiopatía": {f: enfermos[c].median() for f, c in filas.items()},
        "percentil entre los normales": {f: np.nan if np.isnan(este[c]) else 100 * (normales[c].dropna() < este[c]).mean()
                                         for f, c in filas.items()},
    }).rename_axis("medida").round({"este ECG": 3, "mediana de los normales": 3, "mediana de los ECGs con cardiopatía": 3,
                                    "percentil entre los normales": 0})

    interes = None
    if duracion >= 10:
        entrada = resample_poly(filtrada, modelo.FS, fs_entera, axis=0)[: modelo.MUESTRAS].astype(np.float32)
        resultado = modelo.predecir(entrada, cual)
        if modelo.MODELOS:
            interes = areas.calcular(entrada, modelo.MODELOS[cual])
    else:
        resultado = {"conectado": bool(modelo.MODELOS),
                     "motivo": tx("El modelo necesita 10 s de señal y este ECG dura {s} s.", s=round(duracion, 1))}

    para_dibujar = resample_poly(primeros, datos.FS_VISOR, fs_entera, axis=0)
    return {
        "ficha": ficha,
        "avisos": avisos,
        "modelo": resultado,
        "areas": interes,
        "resumen": {"frecuencia_cardiaca": None if np.isnan(m["frecuencia cardiaca"]) else round(float(m["frecuencia cardiaca"]), 1),
                    "latidos": len(picos)},   # columnas de la tabla del historial
        "ecg": {"fs": datos.FS_VISOR, "derivaciones": DERIVACIONES,
                "senal": json.loads(pd.DataFrame(para_dibujar.T).round(3).to_json(orient="values"))},
        "bloques": [
            datos.bloque("tabla", titulo_medidas(duracion), senal_medidas, ancho="medio"),
            datos.bloque("tabla", "Comparación con el dataset balanceado", comparacion, ancho="medio"),
        ],
    }


def titulo_medidas(duracion: float) -> str:
    """Las medidas usan los primeros 10 s; solo se aclara cuando el ECG dura más."""
    return "Medidas de la señal (primeros 10 s)" if duracion > 10 else "Medidas de la señal"


def al_dia(resultado: dict) -> dict:
    """Un análisis guardado, con los bloques y nombres de hoy: los guardados antes del 2026-10-27 traen la gráfica de
    amplitudes, la nota de la comparación, "(primeros 10 s)" aunque el ECG dure 10 s y "(Avance 2)" en el nombre del
    modelo. El resultado del modelo no cambia."""
    titulo = titulo_medidas(resultado["ficha"]["duracion"])
    bloques = [{**b, "nota": "", "titulo": titulo if b["titulo"].startswith("Medidas de la señal") else b["titulo"]}
               for b in resultado["bloques"] if b["tipo"] == "tabla"]
    m = resultado["modelo"]
    return {**resultado, "bloques": bloques, "modelo": {**m, "nombre": modelo.NOMBRES_ANTERIORES.get(m.get("nombre"), m.get("nombre"))}}
