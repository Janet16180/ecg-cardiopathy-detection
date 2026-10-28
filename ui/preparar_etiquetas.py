"""Guarda en outputs/ui/ el código original de cada etiqueta de cada ECG, para mostrarlo en el visor.

Uso, desde la raíz del repo:
    .venv-mps/bin/python ui/preparar_etiquetas.py

Los datasets del Challenge usan códigos SNOMED CT (línea "# Dx:" del .hea) y PTB-XL códigos SCP (columna scp_codes
de su CSV). Los nombres ya están en la tabla del notebook 28, en el mismo orden que los códigos; el script comprueba,
ECG por ECG, que cada código da ese nombre. PTB Diagnostic no usa códigos: su etiqueta es el motivo de ingreso.
"""
import ast
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data/raw"
TABLA_28 = RAIZ / "outputs/28-ecgs_info.pkl"
SALIDA = RAIZ / "outputs/ui/codigos_etiquetas.csv"

PTBXL = RAW / "ptb-xl/1.0.3"
SNOMED = RAW / "challenge-2021/snomed"
CHALLENGE = {
    "Ningbo": RAW / "challenge-2021/1.0.3/training/ningbo",
    "Chapman-Shaoxing": RAW / "challenge-2021/1.0.3/training/chapman_shaoxing",
    "Georgia": RAW / "challenge-2020/1.0.2/training/georgia",
    "CPSC 2018": RAW / "challenge-2020/1.0.2/training/cpsc_2018",
    "CPSC 2018 extra": RAW / "challenge-2020/1.0.2/training/cpsc_2018_extra",
}


def codigos_dx(carpeta: Path) -> pd.Series:
    """Códigos SNOMED CT de la línea "# Dx:" de cada .hea, en el orden del archivo."""
    codigos = {}
    for hea in sorted(carpeta.glob("*/*.hea")):
        dx = next(linea for linea in hea.read_text().splitlines() if linea.lstrip("# ").startswith("Dx:"))
        codigos[hea.stem] = dx.split(":", 1)[1].strip()
    return pd.Series(codigos)


ptbxl = pd.read_csv(PTBXL / "ptbxl_database.csv", index_col="ecg_id")
partes = [pd.DataFrame({"dataset": "PTB-XL", "registro": ptbxl.index.astype(str),
                        "codigos": ptbxl["scp_codes"].apply(lambda texto: ",".join(ast.literal_eval(texto))).to_numpy()})]
for nombre, carpeta in CHALLENGE.items():
    dx = codigos_dx(carpeta)
    partes.append(pd.DataFrame({"dataset": nombre, "registro": dx.index, "codigos": dx.to_numpy()}))
codigos = pd.concat(partes, ignore_index=True)

# Mismos diccionarios que el notebook 28 (sección 4): código → nombre
nombre_de = pd.read_csv(PTBXL / "scp_statements.csv", index_col=0)["description"].to_dict()
nombre_de |= pd.concat([pd.read_csv(SNOMED / "dx_mapping_scored.csv"), pd.read_csv(SNOMED / "dx_mapping_unscored.csv")]
                       ).set_index("SNOMEDCTCode")["Dx"].rename(index=str).to_dict()

tabla = pd.read_pickle(TABLA_28)
tabla["registro"] = tabla["registro"].astype(str)
codigos = tabla[["dataset", "registro", "diagnosticos"]].merge(codigos, on=["dataset", "registro"])
iguales = [[nombre_de[c] for c in texto.split(",")] == list(nombres) for texto, nombres in zip(codigos["codigos"], codigos["diagnosticos"])]
assert all(iguales), f"{len(iguales) - sum(iguales)} ECGs con códigos que no dan los nombres del notebook 28"
assert len(codigos) == tabla["dataset"].isin(["PTB-XL", *CHALLENGE]).sum(), "faltan ECGs de la tabla del notebook 28"

codigos[["dataset", "registro", "codigos"]].to_csv(SALIDA, index=False)
print(SALIDA, f"{len(codigos):,} ECGs; todos sus códigos dan los nombres del notebook 28")
