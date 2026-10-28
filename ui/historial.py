"""Historial de los ECGs analizados, en una base SQLite local (outputs/ui/historial.sqlite, fuera de Git).

Por cada análisis guarda los archivos originales y el resultado tal como se mostró, sin traducir, para volver a
verlo después en cualquier idioma. Se guarda el resultado y no se recalcula, así queda lo que dijo el modelo
en ese momento aunque después cambie. Todo se queda en esta computadora.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime

import datos

RUTA = datos.CACHE / "historial.sqlite"
ESQUEMA = """
CREATE TABLE IF NOT EXISTS analisis (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha               TEXT    NOT NULL,   -- ISO 8601 con zona horaria
    archivo             TEXT    NOT NULL,
    formato             TEXT    NOT NULL,
    fs                  REAL    NOT NULL,
    duracion            REAL    NOT NULL,
    frecuencia_cardiaca REAL,
    latidos             INTEGER,
    avisos              INTEGER NOT NULL,
    modelo_conectado    INTEGER NOT NULL,
    modelo_nombre       TEXT,
    probabilidad        REAL,
    umbral              REAL,
    riesgo              INTEGER,
    resultado           TEXT    NOT NULL    -- JSON del resultado completo, sin traducir
);
CREATE TABLE IF NOT EXISTS archivos (
    analisis_id INTEGER NOT NULL REFERENCES analisis(id) ON DELETE CASCADE,
    nombre      TEXT    NOT NULL,
    contenido   BLOB    NOT NULL
);
"""
COLUMNAS_TABLA = ["id", "fecha", "archivo", "formato", "fs", "duracion", "frecuencia_cardiaca", "latidos", "avisos",
                  "modelo_conectado", "modelo_nombre", "probabilidad", "umbral", "riesgo"]


def conectar() -> sqlite3.Connection:
    """Conexión nueva por consulta; crea las tablas si el archivo es nuevo."""
    conexion = sqlite3.connect(RUTA)
    conexion.row_factory = sqlite3.Row
    conexion.execute("PRAGMA foreign_keys = ON")
    conexion.executescript(ESQUEMA)
    return conexion


def guardar(archivos: dict[str, bytes], resultado: dict) -> tuple[int, str]:
    """Guarda un análisis con sus archivos; devuelve su id y su fecha."""
    fecha = datetime.now().astimezone().isoformat(timespec="seconds")
    ficha, modelo = resultado["ficha"], resultado["modelo"]
    fila = {
        "fecha": fecha,
        "archivo": ficha["archivo"],
        "formato": ficha["formato"],
        "fs": ficha["fs"],
        "duracion": ficha["duracion"],
        "frecuencia_cardiaca": resultado["resumen"]["frecuencia_cardiaca"],
        "latidos": resultado["resumen"]["latidos"],
        "avisos": len(resultado["avisos"]),
        "modelo_conectado": modelo["conectado"],
        "modelo_nombre": modelo.get("nombre"),
        "probabilidad": modelo.get("probabilidad"),
        "umbral": modelo.get("umbral"),
        "riesgo": modelo.get("riesgo"),
        "resultado": json.dumps(resultado, ensure_ascii=False),
    }
    with closing(conectar()) as conexion, conexion:   # el segundo "conexion" confirma la transacción al salir
        cursor = conexion.execute(f"INSERT INTO analisis ({', '.join(fila)}) VALUES ({', '.join('?' * len(fila))})",
                                  list(fila.values()))
        conexion.executemany("INSERT INTO archivos (analisis_id, nombre, contenido) VALUES (?, ?, ?)",
                             [(cursor.lastrowid, nombre, contenido) for nombre, contenido in archivos.items()])
    return cursor.lastrowid, fecha


def listar() -> list[dict]:
    """Todos los análisis, del más reciente al más antiguo, sin los archivos ni el resultado completo."""
    with closing(conectar()) as conexion:
        filas = conexion.execute(f"SELECT {', '.join(COLUMNAS_TABLA)} FROM analisis ORDER BY id DESC").fetchall()
    return [dict(f) | {"modelo_conectado": bool(f["modelo_conectado"]),
                       "riesgo": None if f["riesgo"] is None else bool(f["riesgo"])} for f in filas]


def obtener(id_analisis: int) -> dict:
    """El resultado guardado de un análisis, con su id y su fecha."""
    with closing(conectar()) as conexion:
        fila = conexion.execute("SELECT id, fecha, resultado FROM analisis WHERE id = ?", (id_analisis,)).fetchone()
    if fila is None:
        raise LookupError(datos.tx("No existe el análisis {id} en el historial.", id=id_analisis))
    return {**json.loads(fila["resultado"]), "id": fila["id"], "fecha": fila["fecha"]}


def eliminar(id_analisis: int) -> bool:
    """Borra un análisis y sus archivos; False si no existía."""
    with closing(conectar()) as conexion, conexion:
        return conexion.execute("DELETE FROM analisis WHERE id = ?", (id_analisis,)).rowcount > 0
