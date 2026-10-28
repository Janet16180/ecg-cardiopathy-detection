"""UI de presentación: registro de un ECG nuevo, EDA de los datasets (notebooks 28 a 30) y modelo, en español, inglés e italiano.

Uso, desde la raíz del repo:
    .venv-mps/bin/python ui/preparar_senales.py                     # una sola vez, unos minutos
    .venv-mps/bin/python ui/preparar_areas.py                       # una sola vez, ~1 min (áreas de interés)
    .venv-mps/bin/uvicorn app:app --app-dir ui --port 8000          # y abrir http://127.0.0.1:8000

uvicorn escucha solo en 127.0.0.1: la UI no queda expuesta a la red.
"""
import base64
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import datos
import experimentos
import historial
import modelo
import registro
from traducciones import traducir, traducir_bloque

app = FastAPI(title="EDA de ECGs", docs_url=None, redoc_url=None)
Idioma = Literal["es", "en", "it"]
LIMITE_MB = 25


@app.middleware("http")
async def revalidar_estaticos(request, call_next):
    """El navegador revalida los archivos de la UI en cada carga, así ve los cambios sin borrar su caché."""
    respuesta = await call_next(request)
    if not request.url.path.startswith("/api/"):
        respuesta.headers["Cache-Control"] = "no-cache"
    return respuesta


class Archivo(BaseModel):
    nombre: str
    contenido: str   # base64; así no hace falta python-multipart


class NuevoECG(BaseModel):
    archivos: list[Archivo] = Field(min_length=1, max_length=4)
    fs: float = Field(500, ge=50, le=10000)   # solo para CSV y NumPy, que no traen la frecuencia de muestreo
    # punto va antes que modelo: dentro de la clase, el campo modelo tapa al módulo modelo
    punto: Literal["99", "95"] = modelo.PUNTO_PREDETERMINADO   # umbral del riesgo: % de sensibilidad en validación
    modelo: str = modelo.PREDETERMINADO


def validar(nombres: list[str], idioma: str) -> tuple[str, ...]:
    desconocidos = set(nombres) - set(datos.DATASETS)
    if desconocidos:
        raise HTTPException(422, traducir(datos.tx("Datasets desconocidos: {nombres}", nombres=", ".join(sorted(desconocidos))), idioma))
    return tuple(d for d in datos.DATASETS if d in nombres)   # siempre en el mismo orden


@app.get("/api/datasets")
def datasets() -> dict:
    return {"originales": datos.ORIGINALES, "nuevo": datos.NUEVO}


@app.get("/api/eda/{seccion}")
def eda(seccion: Literal["resumen", "pacientes", "diagnosticos", "clases", "senales", "duplicados"],
        ds: Annotated[list[str], Query()], lang: Idioma = "es") -> list[dict]:
    bloques = datos.SECCIONES[seccion](validar(ds, lang))
    return [traducir_bloque(b, lang) for b in bloques
            if not isinstance(b["datos"], dict) or b["datos"]["data"]]   # sin tablas vacías


@app.get("/api/ecg")
def ecg(dataset: str, clase: Literal["normal", "con cardiopatía", "sin cardiopatía, con otros hallazgos"] | None = None,
        lang: Idioma = "es") -> dict:
    validar([dataset], lang)
    try:
        ecg = datos.ecg_al_azar(dataset, clase)
    except LookupError as error:
        raise HTTPException(404, traducir(error.args[0], lang))
    return {**ecg,
            "dataset": traducir(ecg["dataset"], lang),
            "clase": traducir(ecg["clase"], lang),
            "cardiopatias": [traducir(c, lang) for c in ecg["cardiopatias"] or []]}


def traducir_areas(a: dict | None, idioma: str) -> dict | None:
    """Nombres de los grupos y motivos de los latidos; los análisis guardados antes de las áreas no las traen."""
    if not a:
        return None
    if not a["disponible"]:
        return {**a, "motivo": traducir(a["motivo"], idioma)}
    return {**a, "grupos": [{**g, "nombre": traducir(g["nombre"], idioma)} for g in a["grupos"]],
            "latidos": [{**l, "motivos": [traducir(x, idioma) for x in l["motivos"]]} for l in a["latidos"]]}


def traducir_resultado(resultado: dict, idioma: str) -> dict:
    m = resultado["modelo"]
    return {**resultado,
            "areas": traducir_areas(resultado.get("areas"), idioma),
            "avisos": [traducir(a, idioma) for a in resultado["avisos"]],
            "bloques": [traducir_bloque(b, idioma) for b in resultado["bloques"]],
            "modelo": {**m, "motivo": traducir(m.get("motivo"), idioma), "nombre": traducir(m.get("nombre"), idioma),
                       **({"etiquetas": [{**e, "etiqueta": traducir(e["etiqueta"], idioma)} for e in m["etiquetas"]]} if "etiquetas" in m else {})}}


def validar_modelo(cual: str, idioma: str) -> str:
    if modelo.MODELOS and cual not in modelo.MODELOS:
        raise HTTPException(422, traducir(datos.tx("Modelo desconocido: {cual}", cual=cual), idioma))
    return cual


@app.post("/api/registrar")
def registrar(nuevo: NuevoECG, lang: Idioma = "es") -> dict:
    """Lee un ECG recién tomado, lo mide, lo pasa por el modelo y lo guarda en el historial local."""
    archivos = {}
    for archivo in nuevo.archivos:
        nombre = Path(archivo.nombre.replace("\\", "/")).name
        if not nombre or nombre.startswith("."):
            raise HTTPException(422, traducir(datos.tx("Nombre de archivo no válido: {nombre}", nombre=archivo.nombre), lang))
        archivos[nombre] = base64.b64decode(archivo.contenido, validate=True)
    if sum(len(c) for c in archivos.values()) > LIMITE_MB * 1024 * 1024:
        raise HTTPException(413, traducir(datos.tx("Los archivos pesan más de {mb} MB.", mb=LIMITE_MB), lang))
    try:
        senal, fs, ficha, avisos = registro.leer(archivos, nuevo.fs)
    except registro.ECGIlegible as error:
        raise HTTPException(422, traducir(error.args[0], lang))
    resultado = registro.analizar(senal, fs, ficha, avisos, validar_modelo(nuevo.modelo, lang), nuevo.punto)
    id_analisis, fecha = historial.guardar(archivos, resultado)
    return {**traducir_resultado(resultado, lang), "id": id_analisis, "fecha": fecha}


@app.get("/api/historial")
def lista_historial(lang: Idioma = "es") -> list[dict]:
    return [{**f, "modelo_nombre": traducir(modelo.NOMBRES_ANTERIORES.get(f["modelo_nombre"], f["modelo_nombre"]), lang)}
            for f in historial.listar()]


@app.get("/api/historial/{id_analisis}")
def analisis_guardado(id_analisis: int, lang: Idioma = "es") -> dict:
    try:
        return traducir_resultado(registro.al_dia(historial.obtener(id_analisis)), lang)
    except LookupError as error:
        raise HTTPException(404, traducir(error.args[0], lang))


@app.delete("/api/historial/{id_analisis}", status_code=204)
def eliminar_analisis(id_analisis: int) -> None:
    if not historial.eliminar(id_analisis):
        raise HTTPException(404)


def traducir_detalle(x: dict, idioma: str) -> dict:
    """Textos del detalle de un modelo (el "+" de Modelos probados); la salida de la interfaz conserva sus llaves."""
    t = lambda v: traducir(v, idioma)
    interfaz = x["salida_interfaz"]
    if interfaz and "etiquetas" in interfaz["salida"]:
        interfaz = {**interfaz, "salida": {**interfaz["salida"], "etiquetas": [{**e, "etiqueta": t(e["etiqueta"])} for e in interfaz["salida"]["etiquetas"]]}}
    return {**x, "pasos": [t(p) for p in x["pasos"]], "ejemplo": traducir_bloque(x["ejemplo"], idioma), "salida_interfaz": interfaz,
            "fuente": t(x["fuente"]),
            "por_que": [t(p) for p in x["por_que"]]}


@app.get("/api/modelos-probados")
def modelos_probados(lang: Idioma = "es") -> dict:
    """La mejor variante de cada tipo de modelo de los experimentos 058 a 062, con sus métricas en validación y su explicación."""
    d = experimentos.modelos_probados()
    return {"presupuestos": [{**p, "nombre": traducir(p["nombre"], lang)} for p in d["presupuestos"]], "nota": traducir(d["nota"], lang),
            "modelos": [{**m, "nombre": traducir(m["nombre"], lang), "familia": traducir(m["familia"], lang),
                         "detalle": traducir_detalle(m["detalle"], lang)} for m in d["modelos"]]}


@app.get("/api/experimentos/{seccion}")
def pagina_experimentos(seccion: Literal["resultados", "caracteristicas"], lang: Idioma = "es") -> list[dict]:
    """Resultados de los experimentos y las características de la señal que usan los modelos."""
    return [traducir_bloque(b, lang) for b in experimentos.SECCIONES[seccion]()]


@app.get("/api/modelos")
def modelos(lang: Idioma = "es") -> dict:
    """Los modelos y los umbrales que se pueden elegir al registrar un ECG."""
    return {"modelos": [{**m, "nombre": traducir(m["nombre"], lang)} for m in modelo.lista()], "predeterminado": modelo.PREDETERMINADO,
            "puntos": [{**p, "nombre": traducir(p["nombre"], lang)} for p in modelo.puntos()],
            "punto_predeterminado": modelo.PUNTO_PREDETERMINADO}


ESTATICOS = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def pagina() -> HTMLResponse:
    """index.html con la versión de cada archivo en su URL: si la UI cambia, el navegador no puede usar una copia vieja."""
    version = {n: hashlib.sha256((ESTATICOS / n).read_bytes()).hexdigest()[:12] for n in ("app.js", "graficas.js", "textos.js", "estilos.css")}
    mapa = {"imports": {f"./{n}": f"./{n}?v={version[n]}" for n in ("graficas.js", "textos.js")}}   # imports de app.js
    html = (ESTATICOS / "index.html").read_text()
    html = html.replace('href="estilos.css"', f'href="estilos.css?v={version["estilos.css"]}"')
    html = html.replace('<script type="module" src="app.js"></script>',
                        f'<script type="importmap">{json.dumps(mapa)}</script>\n  <script type="module" src="app.js?v={version["app.js"]}"></script>')
    return HTMLResponse(html)


app.mount("/", StaticFiles(directory=ESTATICOS, html=True), name="static")
