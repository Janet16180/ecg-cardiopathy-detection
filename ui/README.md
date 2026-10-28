# App de ECGs

Página web local para explorar los datasets, ver ECGs al azar y analizar un ECG nuevo con los dos modelos entrenados
(gradient boosting binario y multietiqueta, experimento 067).

## Correrla

Desde la raíz del repo, con Python 3.11:

```bash
python3.11 -m venv .venv-ui
.venv-ui/bin/python -m pip install --require-hashes --only-binary :all: --no-deps -r ui/requirements.txt
PYTHONPATH=. .venv-ui/bin/uvicorn app:app --app-dir ui --port 8000
```

En Windows (PowerShell), el último paso es `$env:PYTHONPATH = "."` y luego
`.venv-ui\Scripts\uvicorn app:app --app-dir ui --port 8000`.

Después se abre http://127.0.0.1:8000. Los ECGs que se registran se guardan solo en tu computadora, en
`outputs/ui/historial.sqlite`; ese archivo no se sube a Git.

## Señales del dataset balanceado (opcional)

El repo trae la lista de los 20,000 ECGs del dataset balanceado v3 (`data/processed/balanceado_jovenes_v3/ecgs.csv`) y
su partición, pero no las señales (`senales.npy`, 2.4 GB). Sin ellas la App funciona igual, salvo el visor de ECGs de
la página Señales. Para crearlas idénticas a las originales:

```bash
.venv-ui/bin/python scripts/data/crear_balanceado.py --descargar
```

El script baja de PhysioNet solo los registros que usa el dataset (40,177 archivos, 3.2 GB) y comprueba, ECG por ECG,
que los archivos y las señales son idénticos a los originales (`huellas.csv`). Si ya tienes los datasets en
`data/raw`, córrelo sin `--descargar`. Para ver en el visor ECGs de los datasets originales hacen falta esos datasets
completos en `data/raw`.

## ECGs de prueba para registrar

Con los registros ya en `data/raw` (el paso anterior), este script copia los 2,000 ECGs de prueba del dataset v3 a
`data/processed/balanceado_jovenes_v3/Test/`, en una carpeta `Normal` y otra `Con cardiopatia`, con una carpeta por ECG
y sus archivos originales de PhysioNet. `Test/ecgs.csv` dice qué tiene cada uno. Los modelos no vieron estos ECGs.

```bash
.venv-ui/bin/python scripts/data/exportar_prueba.py
```
