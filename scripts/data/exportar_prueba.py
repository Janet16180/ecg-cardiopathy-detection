"""Copia los ECGs de prueba del dataset balanceado v3 a una carpeta, para cargarlos uno por uno en la app.

Uso, desde la raíz del repo:
    .venv-mps/bin/python scripts/data/exportar_prueba.py

Crea data/processed/balanceado_jovenes_v3/Test/ con una carpeta por clase ("Normal" y "Con cardiopatia") y, dentro,
una carpeta por ECG ("<dataset> - <registro>") con sus archivos originales de PhysioNet sin cambios: el .hea y los
archivos de señal que nombra (.dat, .mat y, en PTB Diagnostic, .xyz). Test/ecgs.csv dice qué hay en cada carpeta:
dataset, registro, edad, sexo, clase, cardiopatías y etiquetas originales. Los archivos salen de data/raw; si faltan,
scripts/data/crear_balanceado.py --descargar los baja.
"""
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd
import wfdb

RAIZ = Path(__file__).resolve().parents[2]
DATASET = RAIZ / "data/processed/balanceado_jovenes_v3"
SALIDA = DATASET / "Test"
NOTEBOOKS = RAIZ / "notebooks"   # las rutas de ecgs.csv son relativas a esta carpeta
CARPETA = {"normal": "Normal", "con cardiopatía": "Con cardiopatia"}
EXTENSION = {"PTB-XL": ".dat", "PTB Diagnostic": ".dat"}   # archivo de señal con la huella de origen; el Challenge usa .mat

if SALIDA.exists():
    raise SystemExit(f"{SALIDA} ya existe; bórrala para volver a crearla.")

ecgs = pd.read_csv(DATASET / "ecgs.csv", index_col="fila", dtype={"registro": str})
split = pd.read_csv(DATASET / "particion.csv", index_col="fila")["split"]
huellas = pd.read_csv(DATASET / "huellas.csv", index_col="fila")["huella_origen"]
tabla_28 = pd.read_pickle(RAIZ / "outputs/28-ecgs_info.pkl")
tabla_28["registro"] = tabla_28["registro"].astype(str)
etiquetas = tabla_28.set_index(["dataset", "registro"])["diagnosticos"]

prueba = ecgs[split == "test"]
filas = []
for fila, e in prueba.iterrows():
    ruta = Path(os.path.normpath(NOTEBOOKS / e["ruta"]))
    destino = SALIDA / CARPETA[e["clase"]] / f"{e['dataset']} - {e['registro']}"
    destino.mkdir(parents=True)
    for nombre in sorted({ruta.name + ".hea", *wfdb.rdheader(str(ruta)).file_name}):
        shutil.copy2(ruta.parent / nombre, destino / nombre)
    senal = destino / (ruta.name + EXTENSION.get(e["dataset"], ".mat"))
    assert hashlib.md5(senal.read_bytes()).hexdigest() == huellas[fila], f"{senal} no es el archivo que se usó"
    filas.append({"carpeta": destino.relative_to(SALIDA).as_posix(), "dataset": e["dataset"], "registro": e["registro"],
                  "edad": e["edad"], "sexo": e["sexo"], "clase": e["clase"], "cardiopatias": e["cardiopatias"],
                  "etiquetas_originales": "; ".join(etiquetas[(e["dataset"], e["registro"])])})

indice = pd.DataFrame(filas)
indice.to_csv(SALIDA / "ecgs.csv", index=False)
archivos = sum(1 for a in SALIDA.rglob("*") if a.is_file()) - 1
print(SALIDA, f"| {len(indice):,} ECGs ({(indice['clase'] == 'normal').sum():,} normales y "
              f"{(indice['clase'] == 'con cardiopatía').sum():,} con cardiopatía) | {archivos:,} archivos copiados, iguales a los usados")
