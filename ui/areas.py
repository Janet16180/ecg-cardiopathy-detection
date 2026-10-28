"""Áreas de interés de un ECG, para el botón "Mostrar áreas de interés" del resultado.

Se calculan sobre la misma entrada que el modelo (500 Hz, primeros 9.6 s, filtro pasa-altas de 0.5 Hz) y tienen
dos capas:

- Del modelo: cada grupo de medidas del Avance 2 (onda P, QRS, ST, T, QT, ritmo, espectro) se reemplaza por su
  mediana en los ECGs normales del entrenamiento y se mide cuánto baja la probabilidad del modelo elegido. Los
  grupos que más la bajan se sombrean sobre su onda en los latidos de forma típica. En los grupos medidos por
  derivación se marcan las derivaciones que más se alejan de los normales.
- Latidos atípicos, por regla y sin modelo: latidos prematuros, pausas y latidos de forma distinta al latido
  mediano del mismo ECG. Cada regla tiene un umbral que no marca nada en el 98% de los ECGs normales del
  entrenamiento.

Las referencias las escribe preparar_areas.py en outputs/ui/areas_referencia.json.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt

from ecg_experiment import balanceado_v2 as v2
from ecg_experiment import caracteristicas_avance2 as ca2

RAIZ = Path(__file__).resolve().parent.parent
REFERENCIA = RAIZ / "outputs/ui/areas_referencia.json"
FS, PRE, POST = ca2.FS, ca2.PRE, ca2.POST
NUCLEO = slice(PRE - 50, PRE + 200)   # de 0.1 s antes a 0.4 s después del R, como ca2.median_beat
MAXIMO_GRUPOS = 3
MINIMO_EFECTO = 0.2                   # baja mínima del logit de la probabilidad para mostrar un grupo
DESVIO = 2.0                          # desviaciones robustas desde la mediana de los normales para marcar una derivación
MAXIMO_DERIVACIONES = 4
ANTES, DESPUES = 0.2, 0.4             # segundos que se sombrean alrededor del R de un latido atípico
DESFASE = int(0.04 * FS)              # corrimiento máximo al alinear un latido con el latido mediano
AMPLITUD_MINIMA = 0.5                 # fracción de la amplitud mediana en 5-15 Hz para contar una detección como latido


def grupo(zona: str | None, globales: tuple = (), por_derivacion: tuple = ()) -> dict:
    variables = [*globales, *(f"{p}_{d}" for p in por_derivacion for d in ca2.LEADS)]
    return {"zona": zona, "por_derivacion": list(por_derivacion), "variables": variables}


# Cada medida primaria del Avance 2 está en un solo grupo; zona None: no tiene un lugar en el trazo
GRUPOS = {
    "Onda P e intervalo PR": grupo("P", ("p_rms_mv", "pr_ms", "pq_peak_ms", "p_amp_II")),
    "Duración del QRS": grupo("QRS", ("qrs_dur_ms",)),
    "Voltaje y forma del QRS": grupo("QRS", por_derivacion=("r_amp", "s_depth", "qrs_area", "std", "ptp", "kurt")),
    "Ondas Q": grupo("Q", por_derivacion=("q_depth",)),
    "Segmento ST": grupo("ST", por_derivacion=("st",)),
    "Onda T": grupo("T", ("t_peak_rms_mv",), ("t_amp",)),
    "Intervalo QT": grupo("QT", ("qt_ms",)),
    "Ritmo y frecuencia": grupo(None, ("n_beats", "rr_mean_ms", "rr_sdnn_ms", "rr_rmssd_ms", "rr_range_rel", "hr_bpm",
                                       "beat_similarity")),
    "Ruido y espectro de la señal": grupo(None, ("noise_drift_frac", "noise_hf_frac", "noise_powerline_frac", "sp_low_rel",
                                                 "sp_mid_rel", "sp_high_rel", "sp_entropy", "sp_centroid_hz")),
}
SIN_ONDAS = {"derivacion_plana": "Hay una derivación plana: el modelo no puede medir las ondas.",
             "sin_latidos": "No se detectaron latidos suficientes para medir las ondas."}


@lru_cache
def referencia() -> dict:
    return json.loads(REFERENCIA.read_text())


def latidos(limpia: np.ndarray, picos: np.ndarray) -> pd.DataFrame | None:
    """
    Cada latido comparado con el latido mediano del mismo ECG.

    Descarta las detecciones con menos de AMPLITUD_MINIMA veces la amplitud mediana en la banda de 5 a 15 Hz
    (ondas T altas o ruido contados como latido) y alinea cada latido con el latido mediano corriéndolo hasta
    ±40 ms antes de correlacionar, porque el detector a veces centra un latido en la S y otro en la R. Los
    latidos de los bordes, donde el filtro los deforma, no se correlacionan.

    Returns
    -------
    pd.DataFrame | None
        ``pico`` (muestra, ya alineada), ``correlacion`` (NaN si no cabe la ventana) y ``rr`` (intervalo
        previo entre la mediana de los intervalos; NaN en el primero); None si quedan menos de 3 latidos.
    """
    medio = int(0.06 * FS)
    banda = np.sqrt(np.sum(sosfiltfilt(ca2.SOS_QRS, limpia[:, ca2.INDEPENDENT], axis=0) ** 2, axis=1))   # 5 a 15 Hz
    amplitud = np.array([banda[max(p - medio, 0): p + medio].max() for p in picos])
    picos = picos[amplitud >= AMPLITUD_MINIMA * np.median(amplitud)]
    plantilla, _ = ca2.median_beat(limpia, picos)
    if len(picos) < 3 or np.isnan(plantilla).any():
        return None
    tipico = plantilla[NUCLEO][:, ca2.INDEPENDENT].ravel()
    a, b = NUCLEO.start - PRE, NUCLEO.stop - PRE
    filas = []
    for p in picos:
        mejor = (np.nan, p)
        if p - PRE - DESFASE >= 0 and p + POST + DESFASE <= len(limpia):   # en los bordes el filtro deforma el latido
            for q in range(p - DESFASE, p + DESFASE + 1, 2):
                r = np.corrcoef(limpia[q + a: q + b, ca2.INDEPENDENT].ravel(), tipico)[0, 1]
                if np.isnan(mejor[0]) or r > mejor[0]:
                    mejor = (r, q)
        filas.append({"pico": mejor[1], "correlacion": mejor[0]})
    tabla = pd.DataFrame(filas)
    rr = np.diff(picos)
    tabla["rr"] = np.r_[np.nan, rr / np.median(rr)]
    return tabla


def latidos_atipicos(tabla: pd.DataFrame, muestras: int) -> list[dict]:
    """
    Latidos prematuros o de forma distinta, y pausas, según los umbrales de los normales del entrenamiento.

    La pausa que sigue a un latido marcado (la pausa compensatoria de una extrasístole) va como motivo de ese
    latido; las demás pausas se sombrean en el hueco entre los dos latidos.
    """
    u = referencia()["latidos"]
    picos, rr = tabla["pico"].to_numpy() / FS, tabla["rr"].to_numpy()
    pausa_despues = np.r_[rr[1:] > u["pausa"], False]
    salida = []
    for i, (p, r) in enumerate(zip(picos, tabla["correlacion"])):
        motivos = (["llega antes de tiempo"] if rr[i] < u["prematuro"] else []) + \
                  (["forma distinta a la del latido típico"] if r < u["correlacion"] else [])
        if motivos:
            salida.append({"tipo": "latido", "tiempo": round(p, 3), "inicio": round(max(p - ANTES, 0), 3),
                           "fin": round(min(p + DESPUES, muestras / FS), 3),
                           "motivos": motivos + (["seguido de una pausa larga"] if pausa_despues[i] else [])})
        elif pausa_despues[i]:
            inicio, fin = p + DESPUES, picos[i + 1] - ANTES
            if fin - inicio < 0.1:
                inicio, fin = p, picos[i + 1]
            salida.append({"tipo": "pausa", "tiempo": round(p, 3), "inicio": round(inicio, 3), "fin": round(fin, 3),
                           "motivos": ["pausa larga entre latidos"]})
    return salida


def probabilidades(fila: pd.DataFrame, m) -> pd.Series:
    """Probabilidad del modelo para el ECG tal cual ("original") y con cada grupo en la mediana de los normales."""
    mediana = referencia()["mediana"]
    variantes = {"original": fila, **{nombre: fila.assign(**{v: mediana[v] for v in g["variables"]}) for nombre, g in GRUPOS.items()}}
    tabla = pd.concat(variantes.values(), ignore_index=True)
    return pd.Series(m.riesgo(m.guardado["preprocesador"].transform(ca2.derive(tabla))), index=list(variantes))


def desvios(fila: pd.Series, prefijos: list[str]) -> pd.Series:
    """Por derivación, la mayor distancia de sus medidas a la mediana de los normales, en escalas robustas."""
    ref = referencia()
    return pd.Series({d: np.nanmax([abs(fila[f"{p}_{d}"] - ref["mediana"][f"{p}_{d}"]) / ref["escala"][f"{p}_{d}"]
                                    for p in prefijos] + [0.0]) for d in ca2.LEADS})


def ventanas(m: dict) -> dict[str, tuple[int, int]]:
    """Inicio y fin de cada zona en el latido mediano (muestras; el R está en PRE)."""
    fin_t = float(m["t_end"] if np.isfinite(m["t_end"]) else m["t_stop"])
    inicio_p = float(m["p_onset"] if np.isfinite(m["p_onset"]) else m["p_start"])
    return {"P": (inicio_p, m["onset"]), "Q": (m["onset"], max(PRE, m["onset"] + 10)), "QRS": (m["onset"], m["offset"]),
            "ST": (m["offset"], m["t_start"]), "T": (m["t_start"], fin_t), "QT": (m["t_peak"], fin_t)}


def calcular(senal: np.ndarray, m, umbral: float) -> dict:
    """
    Áreas de interés de un ECG para un modelo de modelo.MODELOS.

    Parameters
    ----------
    senal : np.ndarray
        ``(5000, 12)``, la misma entrada que recibe el modelo.
    m : object
        Modelo con ``guardado`` (preprocesador y columnas primarias) y ``riesgo(X)``.
    umbral : float
        Probabilidad a partir de la cual se marca riesgo (el punto de operación elegido).

    Returns
    -------
    dict
        ``disponible``; si es True, ``riesgo``, ``grupos`` (si el modelo marcó riesgo, los grupos que más suben
        la probabilidad, con sus derivaciones e intervalos en segundos) y ``latidos`` (latidos atípicos y
        pausas); si no, ``motivo``.
    """
    entrada = np.asarray(senal[: v2.CROP], dtype=np.float64)
    fila = pd.DataFrame([ca2.extract(entrada)]).reindex(columns=m.guardado["columnas_primarias"])
    if fila.at[0, "status"] != "ok":
        return {"disponible": False, "motivo": SIN_ONDAS[fila.at[0, "status"]]}
    fila["implausible"] = fila["implausible"].fillna("")

    p = probabilidades(fila, m)
    logit = np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6)))
    efecto = (logit["original"] - logit.drop("original")).sort_values(ascending=False)
    riesgo = bool(p["original"] >= umbral)
    elegidos = efecto[efecto >= MINIMO_EFECTO].index[:MAXIMO_GRUPOS] if riesgo else []

    limpia = sosfiltfilt(ca2.SOS_CLEAN, entrada, axis=0)
    picos = ca2.detect_r_peaks(limpia)
    plantilla, _ = ca2.median_beat(limpia, picos)
    zonas = ventanas(ca2.delineate(plantilla, np.diff(picos).mean() / FS * 1000))   # las del modelo
    tabla = latidos(limpia, picos)
    atipicos = latidos_atipicos(tabla, len(entrada)) if tabla is not None else []
    marcados = {round(a["tiempo"] * FS) for a in atipicos if a["tipo"] == "latido"}
    tipicos = [p for p in (tabla["pico"] if tabla is not None else picos) if p not in marcados]

    grupos = []
    for nombre in elegidos:
        g = GRUPOS[nombre]
        intervalos = []
        if g["zona"]:
            a, b = zonas[g["zona"]]
            intervalos = [[round(float(max((p - PRE + a) / FS, 0)), 3), round(float(min((p - PRE + b) / FS, len(entrada) / FS)), 3)]
                          for p in tipicos if p - PRE + b > 0 and p - PRE + a < len(entrada)]
        derivaciones, desvio = list(ca2.LEADS), {}
        if g["por_derivacion"]:
            z = desvios(fila.iloc[0], g["por_derivacion"]).sort_values(ascending=False)
            fuera = z[z >= DESVIO]
            derivaciones = list(fuera.index[:MAXIMO_DERIVACIONES]) if len(fuera) else list(z.index[:2])
            desvio = {d: round(float(z[d]), 1) for d in derivaciones}
        grupos.append({"nombre": nombre, "zona": g["zona"], "efecto": round(float(efecto[nombre]), 3),
                       "probabilidad_sin": float(p[nombre]), "derivaciones": derivaciones, "desvios": desvio,
                       "intervalos": intervalos})
    return {"disponible": True, "riesgo": riesgo, "grupos": grupos, "latidos": atipicos}
