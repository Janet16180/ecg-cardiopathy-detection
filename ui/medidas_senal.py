"""Medidas de un ECG de 12 derivaciones (notebooks 28 y 29); las usan preparar_senales.py y registro.py."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt

DERIVACIONES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def picos_r(derivacion: np.ndarray, fs: float) -> np.ndarray:
    """Posiciones de los picos R (versión simplificada de Pan-Tompkins, como en el notebook 29)."""
    energia = np.diff(filtfilt(*butter(2, [5, 15], btype="band", fs=fs), derivacion)) ** 2
    energia = uniform_filter1d(energia, size=int(0.15 * fs))
    return find_peaks(energia, height=0.3 * energia.max(), distance=int(0.25 * fs))[0]


def pasa_altas(senal: np.ndarray, fs: float) -> np.ndarray:
    """Filtro pasa-altas de 0.5 Hz, el mismo que llevan las señales del dataset balanceado v2 (notebook 29)."""
    return sosfiltfilt(butter(2, 0.5, btype="highpass", fs=fs, output="sos"), senal, axis=0)


def frecuencia_cardiaca(derivacion: np.ndarray, fs: float) -> float:
    """Latidos por minuto a partir de la mediana de los intervalos RR."""
    picos = picos_r(derivacion, fs)
    return 60 * fs / np.median(np.diff(picos)) if len(picos) > 1 else np.nan


def medidas(senal: np.ndarray, fs: int) -> dict:
    """Medidas de los primeros 10 s de las 12 derivaciones estándar (notebooks 28 y 29)."""
    senal = senal[: 10 * fs, :12]
    limpia = np.nan_to_num(senal)
    linea_base = uniform_filter1d(limpia, size=fs, axis=0)                  # promedio móvil de 1 s
    ruido = limpia - uniform_filter1d(limpia, size=fs // 100, axis=0)       # cambios de menos de 10 ms
    amplitud = np.ptp(limpia, axis=0)
    return {
        "derivaciones planas": int((amplitud == 0).sum()),
        "muestras vacías": int(np.isnan(senal).sum()),
        **dict(zip(DERIVACIONES, amplitud.astype(float))),
        "deriva de línea base": float(np.median(np.ptp(linea_base, axis=0))),
        "ruido": float(np.median(np.std(ruido, axis=0))),
        "frecuencia cardiaca": frecuencia_cardiaca(limpia[:, 1], fs),   # derivación II
    }
