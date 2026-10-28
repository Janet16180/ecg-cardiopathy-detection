"""Recrea data/processed/balanceado_jovenes_v3/senales.npy, idéntico al original, a partir de los datos de PhysioNet.

Los notebooks 29 y 43 eligieron los 20,000 ECGs y guardaron la lista en ecgs.csv, que va en Git. Este script lee esos mismos
registros de data/raw con la misma lectura del notebook 29 (primeros 10 s de las 12 derivaciones a 500 Hz, en mV,
con el filtro pasa-altas de 0.5 Hz) y comprueba con huellas.csv, ECG por ECG, que cada archivo de PhysioNet y cada
señal guardada son idénticos a los originales.

Uso, desde la raíz del repo:
    python scripts/data/crear_balanceado.py               # con los datasets ya descargados en data/raw
    python scripts/data/crear_balanceado.py --descargar   # baja de PhysioNet solo los registros que falten (40,177 archivos, 3.2 GB)

Necesita las versiones de numpy, scipy y wfdb fijadas en pyproject.toml; con otras, la señal puede no salir idéntica.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data/raw"
DATASET = RAIZ / "data/processed/balanceado_jovenes_v3"
NOTEBOOKS = RAIZ / "notebooks"   # las rutas de ecgs.csv son relativas a esta carpeta (notebook 29)

FS = 500
FS_ORIGEN = {"PTB Diagnostic": 1000}   # los demás datasets están a 500 Hz
PASA_ALTAS = butter(2, 0.5, btype="highpass", fs=FS, output="sos")
EXTENSION = {"PTB-XL": ".dat", "PTB Diagnostic": ".dat"}   # los datasets del Challenge usan .mat (notebook 28)
PHYSIONET = {   # carpeta dentro de data/raw → archivos del proyecto en PhysioNet (acceso abierto)
    "ptb-xl/1.0.3": "https://physionet.org/files/ptb-xl/1.0.3",
    "challenge-2020/1.0.2": "https://physionet.org/files/challenge-2020/1.0.2",
    "challenge-2021/1.0.3": "https://physionet.org/files/challenge-2021/1.0.3",
    "ptb-diagnostic-ecg-database-1.0.0": "https://physionet.org/files/ptbdb/1.0.0",
}


# Lectura del notebook 29, sección 2, sin cambios
def tramo_plano(senal: np.ndarray) -> bool:
    """True si alguna derivación no cambia durante 1 s o más."""
    sin_cambio = (np.diff(senal, axis=0) == 0).astype(np.float32)
    return bool((uniform_filter1d(sin_cambio, size=FS, axis=0, mode="constant") > 0.999).any())


def error_einthoven(senal: np.ndarray) -> float:
    """Qué tanto se alejan III, aVR, aVL y aVF de lo que dan I y II por la ley de Einthoven (0 = coherentes)."""
    I, II, III, aVR, aVL, aVF = senal[:, :6].T
    esperadas = [(III, II - I), (aVR, -(I + II) / 2), (aVL, (I - III) / 2), (aVF, (II + III) / 2)]
    return max(np.std(medida - esperada) / np.std(medida) for medida, esperada in esperadas)


def leer_10s(ruta: str, fs: int) -> tuple[np.ndarray | None, str]:
    """Primeros 10 s de las 12 derivaciones estándar a 500 Hz, en mV y con filtro pasa-altas; si no sirve, None y el motivo."""
    senal = wfdb.rdsamp(ruta)[0][:, :12]
    if fs == 1000:
        senal = resample_poly(senal, 1, 2, axis=0)
    senal = senal[:5000]
    if len(senal) < 5000:
        return None, "dura menos de 10 s"
    if np.isnan(senal).any():
        return None, "muestras vacías"
    amplitud = np.ptp(senal, axis=0)
    if (amplitud == 0).any():
        return None, "derivación plana"
    if (amplitud < 0.1).any():
        return None, "derivación de menos de 0.1 mV"
    if (amplitud > 10).any():
        return None, "derivación de más de 10 mV"
    if tramo_plano(senal):
        return None, "tramo plano de 1 s o más"
    if error_einthoven(senal) > 0.1:
        return None, "derivaciones de las extremidades incoherentes"
    filtrada = sosfiltfilt(PASA_ALTAS, senal, axis=0)
    amplitud = np.ptp(filtrada, axis=0)
    if (amplitud < 0.1).any() or (amplitud > 10).any():
        return None, "derivación fuera de 0.1 a 10 mV después del filtro"
    return filtrada, "ok"


def huella_origen(ruta: Path, dataset: str) -> str | None:
    """MD5 del archivo de señal de PhysioNet, como la huella del notebook 28; None si no existe."""
    archivo = ruta.with_name(ruta.name + EXTENSION.get(dataset, ".mat"))
    return hashlib.md5(archivo.read_bytes()).hexdigest() if archivo.exists() else None


def huella_senal(senal: np.ndarray) -> str:
    """SHA-256 de una fila de senales.npy (float16)."""
    return hashlib.sha256(np.ascontiguousarray(senal).tobytes()).hexdigest()


def falta(ruta: Path) -> bool:
    """True si falta la cabecera .hea o alguno de los archivos de señal que nombra."""
    if not ruta.with_name(ruta.name + ".hea").exists():
        return True
    return any(not (ruta.parent / nombre).exists() for nombre in set(wfdb.rdheader(str(ruta)).file_name))


def bajar(url: str, destino: Path) -> None:
    """Un archivo de PhysioNet; se escribe aparte y se renombra al final, así un corte no deja archivos a medias."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".parcial")
    for intento in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as respuesta:
                temporal.write_bytes(respuesta.read())
            temporal.replace(destino)
            return
        except OSError:
            if intento == 2:
                raise
            time.sleep(5)


def descargar(ruta: Path) -> None:
    """La cabecera de un registro y los archivos de señal que nombra (PTB Diagnostic trae .dat y .xyz)."""
    relativa = ruta.relative_to(RAW).as_posix()
    carpeta = next(c for c in PHYSIONET if relativa.startswith(c + "/"))
    url = PHYSIONET[carpeta] + relativa[len(carpeta):]
    cabecera = ruta.with_name(ruta.name + ".hea")
    if not cabecera.exists():
        bajar(url + ".hea", cabecera)
    for nombre in set(wfdb.rdheader(str(ruta)).file_name):
        if not (ruta.parent / nombre).exists():
            bajar(url.rsplit("/", 1)[0] + "/" + nombre, ruta.parent / nombre)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--descargar", action="store_true", help="baja de PhysioNet solo los registros que falten en data/raw")
    parser.add_argument("--referencia", action="store_true",
                        help="escribe huellas.csv a partir del senales.npy actual (solo después de rehacer el dataset con los notebooks 29 y 43)")
    args = parser.parse_args()

    ecgs = pd.read_csv(DATASET / "ecgs.csv", index_col="fila", dtype={"registro": str})
    rutas = [Path(os.path.normpath(NOTEBOOKS / ruta)) for ruta in ecgs["ruta"]]   # sin resolve: data/raw puede ser un enlace
    destino = DATASET / "senales.npy"

    if args.referencia:
        senales = np.load(destino, mmap_mode="r")
        huellas = pd.DataFrame({"registro": ecgs["registro"],
                                "huella_origen": [huella_origen(r, d) for r, d in zip(rutas, ecgs["dataset"])],
                                "huella_senal": [huella_senal(s) for s in senales]}, index=ecgs.index)
        huellas.to_csv(DATASET / "huellas.csv")
        print(DATASET / "huellas.csv", huellas.shape)
        return

    huellas = pd.read_csv(DATASET / "huellas.csv", index_col="fila", dtype={"registro": str})
    assert huellas.index.equals(ecgs.index) and huellas["registro"].equals(ecgs["registro"]), "huellas.csv no corresponde a ecgs.csv"

    if destino.exists():
        senales = np.load(destino, mmap_mode="r")
        if senales.shape == (len(ecgs), 5000, 12) and all(huella_senal(s) == h for s, h in zip(senales, huellas["huella_senal"])):
            print(f"{destino} ya existe y es idéntico al original. No hay nada que hacer.")
            return

    # 1. Archivos de PhysioNet
    faltan = [ruta for ruta in rutas if falta(ruta)]
    if faltan and not args.descargar:
        print(f"Faltan {len(faltan):,} de {len(rutas):,} registros en data/raw:")
        print(pd.Series([r.relative_to(RAW).parts[0] for r in faltan]).value_counts().to_string())
        print("Descarga esos datasets de PhysioNet en data/raw, o corre de nuevo con --descargar para bajar solo estos registros.")
        sys.exit(1)
    if faltan:
        print(f"Descargando {len(faltan):,} registros de PhysioNet...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as hilos:
            for n, _ in enumerate(hilos.map(descargar, faltan), start=1):
                if n % 1000 == 0 or n == len(faltan):
                    print(f"  {n:,} de {len(faltan):,}", flush=True)

    # 2. Misma versión de los datos crudos
    distintos = [(d, r.name) for r, d, h in zip(rutas, ecgs["dataset"], huellas["huella_origen"]) if huella_origen(r, d) != h]
    if distintos:
        print(f"{len(distintos):,} archivos de data/raw no son los mismos que se usaron (otra versión del dataset). Primeros:")
        print("\n".join(f"  {d} {r}" for d, r in distintos[:10]))
        sys.exit(1)
    print(f"Los {len(rutas):,} archivos de PhysioNet son los mismos que se usaron.", flush=True)

    # 3. Señales, con la lectura del notebook 29; la fila i de senales.npy es la fila i de ecgs.csv
    temporal = DATASET / "senales.npy.parcial"
    senales = np.lib.format.open_memmap(temporal, mode="w+", dtype=np.float16, shape=(len(ecgs), 5000, 12))
    for i, (ruta, dataset) in enumerate(zip(rutas, ecgs["dataset"])):
        senal, motivo = leer_10s(str(ruta), FS_ORIGEN.get(dataset, FS))
        if senal is None:
            sys.exit(f"El ECG {dataset} {ruta.name} (fila {i}) no pasó la lectura: {motivo}.")
        senales[i] = senal
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,} de {len(ecgs):,} señales", flush=True)
    senales.flush()
    del senales

    # 4. Comparación con las señales originales
    senales = np.load(temporal, mmap_mode="r")
    iguales = sum(huella_senal(s) == h for s, h in zip(senales, huellas["huella_senal"]))
    del senales
    if iguales < len(ecgs):
        print(f"Solo {iguales:,} de {len(ecgs):,} señales salieron idénticas; revisa las versiones de numpy, scipy y wfdb.")
        print(f"No se reemplazó {destino.name}; lo creado quedó en {temporal}.")
        sys.exit(1)
    temporal.replace(destino)
    print(f"Listo: {destino}, {len(ecgs):,} señales idénticas a las originales.")


if __name__ == "__main__":
    main()
