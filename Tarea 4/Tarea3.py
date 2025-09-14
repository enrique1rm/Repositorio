import pandas as pd
import numpy as np

<<<<<<< HEAD
#Carga de datos
df = pd.read_csv('incident_event_log.csv')
=======
# --- Configuración ---
df = pd.read_csv('../incident_event_log.csv', low_memory=False)
>>>>>>> 284e8aec98f5df77bee36149cf45bb6c9ef18415
print (df.head())

# 1) Normalizar strings (recortar) en columnas object
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()

# 2) Parsear fechas relevantes a datetime
# ------------------------
cols_fecha = ["opened_at", "resolved_at", "closed_at", "sys_created_at", "sys_updated_at"]
for c in cols_fecha:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)

# 3) Filtrar SOLO filas que registran cierre
mask_closed = pd.Series(False, index=df.index)
if "incident_state" in df.columns:
    mask_closed = mask_closed | df["incident_state"].astype(str).str.lower().eq("closed")
if "closed_at" in df.columns:
    mask_closed = mask_closed | df["closed_at"].notna()

df_closed = df[mask_closed].copy()

# 4) Un registro por incidente (último por 'number'): ordenar por actualización más reciente
sort_cols = []
if "number" in df_closed.columns:
    sort_cols.append("number")
if "sys_updated_at" in df_closed.columns:
    sort_cols.append("sys_updated_at")
elif "closed_at" in df_closed.columns:
    sort_cols.append("closed_at")
elif "resolved_at" in df_closed.columns:
    sort_cols.append("resolved_at")

if len(sort_cols) > 0:
    df_closed = df_closed.sort_values(by=sort_cols, ascending=True)
if "number" in df_closed.columns:
    df_closed = df_closed.drop_duplicates(subset=["number"], keep="last")

# 5) Target: tiempo_resolucion_horas = resolved_at - opened_at (en horas)
if {"opened_at", "resolved_at"}.issubset(df_closed.columns):
    df_closed = df_closed[df_closed["opened_at"].notna() & df_closed["resolved_at"].notna()].copy()
    df_closed["tiempo_resolucion_horas"] = (df_closed["resolved_at"] - df_closed["opened_at"]).dt.total_seconds() / 3600.0
    # Eliminar negativos/imposibles
    df_closed = df_closed[df_closed["tiempo_resolucion_horas"] >= 0]
    # Winsorizar extremos (1% - 99%) del target
    q1, q99 = df_closed["tiempo_resolucion_horas"].quantile([0.01, 0.99])
    df_closed["tiempo_resolucion_horas"] = df_closed["tiempo_resolucion_horas"].clip(lower=q1, upper=q99)
else:
    raise ValueError("Faltan columnas 'opened_at' y/o 'resolved_at' para calcular el tiempo de resolución.")

# 
# 6) Derivar componentes temporales de opened_at (info conocida al inicio)
if "opened_at" in df_closed.columns:
    df_closed["opened_hour"] = df_closed["opened_at"].dt.hour
    df_closed["opened_weekday"] = df_closed["opened_at"].dt.weekday  # 0 = Lunes, 6 = Domingo
    df_closed["opened_month"] = df_closed["opened_at"].dt.month

# 7) Normalizar tipos y valores faltantes
#Booleanas -> 0/1
#Numéricas NaN -> 0
bool_cols = df_closed.select_dtypes(include=["bool"]).columns.tolist()
for c in bool_cols:
    df_closed[c] = df_closed[c].astype(int)

num_cols = df_closed.select_dtypes(include=[np.number]).columns.tolist()
df_closed[num_cols] = df_closed[num_cols].fillna(0)

<<<<<<< HEAD
# 8) Eliminar columnas constantes (sin información)
const_cols = []
for c in df_closed.columns:
    # nunique con dropna=False para contar NaN como categoría
    if df_closed[c].nunique(dropna=False) <= 1:
        const_cols.append(c)
if len(const_cols) > 0:
    df_closed = df_closed.drop(columns=const_cols, errors="ignore")
=======
import os
os.makedirs("tarea5/data", exist_ok=True)

df_interes.to_csv("tarea5/data/incidentes_limpio.csv", index=False)

print("OK -> tarea5/data/incidentes_limpio.csv")



>>>>>>> 284e8aec98f5df77bee36149cf45bb6c9ef18415

print(df_closed)
print(df)

def load_data():
    return df_closed
