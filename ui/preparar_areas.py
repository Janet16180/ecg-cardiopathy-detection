"""Calcula una sola vez las referencias de las áreas de interés y las guarda en outputs/ui/areas_referencia.json.

Uso, desde la raíz del repo:
    .venv-mps/bin/python ui/preparar_areas.py

Solo usa los ECGs normales del entrenamiento del dataset balanceado v3 (ni validación ni prueba):
- mediana y escala robusta (1.4826 × MAD) de cada medida primaria del Avance 2, de la tabla del experimento 067;
- umbrales de los latidos atípicos: correlación con el latido mediano, RR prematuro y pausa, elegidos para que
  cada regla no marque ningún latido en el 98% de esos ECGs.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt

import areas
from ecg_experiment import balanceado_v2 as v2
from ecg_experiment import balanceado_v3 as v3
from ecg_experiment import caracteristicas_avance2 as ca2

SIN_MARCAS = 0.98   # fracción de ECGs normales del entrenamiento en los que cada regla no marca ningún latido


def latidos_de(senal: np.ndarray) -> tuple[float, float, float]:
    """Correlación mínima con el latido mediano, RR relativo mínimo y máximo de un ECG; NaN si no hay latidos."""
    limpia = sosfiltfilt(ca2.SOS_CLEAN, senal.astype(np.float64), axis=0)
    tabla = areas.latidos(limpia, ca2.detect_r_peaks(limpia))
    if tabla is None:
        return np.nan, np.nan, np.nan
    return tabla["correlacion"].min(), tabla["rr"].min(), tabla["rr"].max()


ecgs, senales = v3.load(areas.RAIZ)
particion = pd.read_csv(areas.RAIZ / v3.DATASET / "particion.csv", index_col="fila")
normales = particion.index[(particion["split"] == "train") & (ecgs["y"] == 0)]

primarias = pd.read_csv(areas.RAIZ / "outputs/experiment067_multietiqueta_v3/caracteristicas_primarias.csv", index_col="fila").loc[normales]
primarias = primarias[primarias["status"] == "ok"].drop(columns=["status", "implausible", "flat_leads"])
mediana = primarias.median()
escala = 1.4826 * (primarias - mediana).abs().median()
escala = escala.where(escala > 0, primarias.std()).replace(0, 1)

inicio = time.perf_counter()
por_ecg = pd.DataFrame([latidos_de(senales[i, : v2.CROP]) for i in normales], index=normales,
                       columns=["correlacion", "rr_minimo", "rr_maximo"]).dropna()
print(f"Latidos de {len(por_ecg):,} ECGs normales en {time.perf_counter() - inicio:.0f} s")

referencia = {
    "ecgs_normales": int(len(normales)),
    "mediana": mediana.round(6).to_dict(),
    "escala": escala.round(6).to_dict(),
    "latidos": {"correlacion": float(por_ecg["correlacion"].quantile(1 - SIN_MARCAS)),
                "prematuro": float(por_ecg["rr_minimo"].quantile(1 - SIN_MARCAS)),
                "pausa": float(por_ecg["rr_maximo"].quantile(SIN_MARCAS)),
                "ecgs": int(len(por_ecg)), "sin_marcas": SIN_MARCAS},
}
areas.REFERENCIA.write_text(json.dumps(referencia, indent=1, ensure_ascii=False))
print("Umbrales de latidos:", referencia["latidos"])
print("Guardado en", areas.REFERENCIA)
