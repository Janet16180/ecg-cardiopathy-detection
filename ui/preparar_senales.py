"""Calcula una sola vez las medidas de señal que muestra la UI y las guarda en outputs/ui/.

Uso, desde la raíz del repo:
    .venv-mps/bin/python ui/preparar_senales.py

Lee las cabeceras de todos los ECGs (formato de cada dataset) y las señales de la misma muestra
del notebook 28 (1,000 ECGs por dataset, 100 de PTB Diagnostic, semilla 42), más los 20,000 ECGs
del dataset balanceado (notebook 29).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from medidas_senal import DERIVACIONES, medidas

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data/raw"
BALANCEADO = RAIZ / "data/processed/balanceado_jovenes_v2"
SALIDA = RAIZ / "outputs/ui"

N_MUESTRA = 1000
SEMILLA = 42
NUEVO = "Balanceado"

PTBXL = RAW / "ptb-xl/1.0.3"
CON_CABECERAS = {
    "Ningbo": RAW / "challenge-2021/1.0.3/training/ningbo",
    "Chapman-Shaoxing": RAW / "challenge-2021/1.0.3/training/chapman_shaoxing",
    "Georgia": RAW / "challenge-2020/1.0.2/training/georgia",
    "CPSC 2018": RAW / "challenge-2020/1.0.2/training/cpsc_2018",
    "CPSC 2018 extra": RAW / "challenge-2020/1.0.2/training/cpsc_2018_extra",
    "PTB Diagnostic": RAW / "ptb-diagnostic-ecg-database-1.0.0",
}


def leer_cabeceras(carpeta: Path) -> pd.DataFrame:
    """Una fila por ECG con su ruta, frecuencia, número de muestras y derivaciones (mismo orden que el notebook 28)."""
    filas = []
    for hea in sorted(carpeta.glob("*/*.hea")):
        cabecera = wfdb.rdheader(str(hea.with_suffix("")))
        filas.append({
            "registro": cabecera.record_name,
            "ruta": str(hea.with_suffix("")),
            "fs": cabecera.fs,
            "n_muestras": cabecera.sig_len,
            "n_derivaciones": cabecera.n_sig,
            "nombres_derivaciones": ",".join(cabecera.sig_name),
        })
    return pd.DataFrame(filas).set_index("registro")


def formato(cabeceras: pd.DataFrame, fuente: str) -> dict:
    """Derivaciones, frecuencia y duración de un dataset a partir de sus cabeceras."""
    duracion = cabeceras["n_muestras"] / cabeceras["fs"]
    return {
        "ECGs (archivos)": len(cabeceras),
        "derivaciones": ", ".join(map(str, sorted(cabeceras["n_derivaciones"].unique()))),
        "nombres de derivaciones": " | ".join(cabeceras["nombres_derivaciones"].unique()),
        "fs (Hz)": ", ".join(f"{fs:g}" for fs in sorted(cabeceras["fs"].unique())),
        "duración mín (s)": round(float(duracion.min()), 1),
        "duración máx (s)": round(float(duracion.max()), 1),
        "fuente": fuente,
    }


SALIDA.mkdir(parents=True, exist_ok=True)
inicio = time.time()
formatos = {}
filas = []

# PTB-XL: la muestra sale de su CSV y el formato de las cabeceras de records500, como en el notebook 28
ptbxl = pd.read_csv(PTBXL / "ptbxl_database.csv", index_col="ecg_id")
formatos["PTB-XL"] = formato(leer_cabeceras(PTBXL / "records500"), "cabeceras .hea")
for registro, archivo in ptbxl["filename_hr"].sample(N_MUESTRA, random_state=SEMILLA).items():
    senal, campos = wfdb.rdsamp(str(PTBXL / archivo))
    filas.append({"dataset": "PTB-XL", "origen": "PTB-XL", "registro": str(registro), **medidas(senal, int(campos["fs"]))})
print(f"PTB-XL listo ({time.time() - inicio:.0f} s)", flush=True)

for nombre, carpeta in CON_CABECERAS.items():
    cabeceras = leer_cabeceras(carpeta)
    formatos[nombre] = formato(cabeceras, "cabeceras .hea")
    n = 100 if nombre == "PTB Diagnostic" else N_MUESTRA   # registros de hasta 2 min a 1000 Hz
    for registro, ruta in cabeceras["ruta"].sample(n, random_state=SEMILLA).items():
        senal, campos = wfdb.rdsamp(ruta)
        filas.append({"dataset": nombre, "origen": nombre, "registro": registro, **medidas(senal, int(campos["fs"]))})
    print(f"{nombre} listo ({time.time() - inicio:.0f} s)", flush=True)

# Dataset balanceado: todos sus ECGs, ya a 500 Hz y 10 s (notebook 29)
balanceado = pd.read_csv(BALANCEADO / "ecgs.csv", index_col="fila", dtype={"registro": str})
senales = np.load(BALANCEADO / "senales.npy", mmap_mode="r")
for fila, (registro, origen) in enumerate(zip(balanceado["registro"], balanceado["dataset"])):
    filas.append({"dataset": NUEVO, "origen": origen, "registro": registro, **medidas(senales[fila].astype(np.float32), 500)})
formatos[NUEVO] = {
    "ECGs (archivos)": senales.shape[0],
    "derivaciones": str(senales.shape[2]),
    "nombres de derivaciones": ",".join(DERIVACIONES),
    "fs (Hz)": "500",
    "duración mín (s)": senales.shape[1] / 500,
    "duración máx (s)": senales.shape[1] / 500,
    "fuente": "forma de senales.npy; fs del notebook 29",
}
print(f"{NUEVO} listo ({time.time() - inicio:.0f} s)", flush=True)

pd.DataFrame(filas).to_pickle(SALIDA / "medidas.pkl")
pd.DataFrame.from_dict(formatos, orient="index").rename_axis("dataset").to_csv(SALIDA / "formato.csv")
print("Guardado en", SALIDA, f"({len(filas):,} ECGs medidos, {time.time() - inicio:.0f} s)")
