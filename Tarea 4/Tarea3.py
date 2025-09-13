import pandas as pd
from pathlib import Path

# --- Configuración ---
df = pd.read_csv('../incident_event_log.csv', low_memory=False)
print (df.head())

#Conversión de fechas 
for col in ["opened_at", "resolved_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

#tiempo de resolución en horas 
df["tiempo_resolver_horas"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600

#variables de interés
cols_interes = ["tiempo_resolver_horas", "reassignment_count",
    "reopen_count", "impact", "urgency", "priority",]

#Creacion de nuevo dataframe
df_interes = df[cols_interes].copy()

#Filtrado de registros invalidos
df_interes = df_interes.dropna(subset=["tiempo_resolver_horas"])
df_interes = df_interes[df_interes["tiempo_resolver_horas"] >= 0]

#Recortar espacios en categóricas si existiesen
for col in ["impact", "urgency", "priority"]:
    if col in df_interes.columns and pd.api.types.is_object_dtype(df_interes[col]):
        df_interes[col] = df_interes[col].astype(str).str.strip()

#No borrar filas con 0s
for col in ["reassignment_count", "reopen_count"]:
    if col in df_interes.columns:
        df_interes[col] = pd.to_numeric(df_interes[col], errors="coerce").fillna(0)
        df_interes[col] = df_interes[col].clip(lower=0).astype(int)

import re
#extraer valor ordinal de atributos categóricos
def extraer_ordinal(s):
    if pd.isna(s):
        return pd.NA
    m = re.match(r"\s*(\d+)", str(s))
    return int(m.group(1)) if m else pd.NA

for col in ["impact", "urgency", "priority"]:
    if col in df_interes.columns:
        df_interes[col + "_n"] = df_interes[col].apply(extraer_ordinal)

print(df) 

import os
os.makedirs("tarea5/data", exist_ok=True)

df_interes.to_csv("tarea5/data/incidentes_limpio.csv", index=False)

print("OK -> tarea5/data/incidentes_limpio.csv")




