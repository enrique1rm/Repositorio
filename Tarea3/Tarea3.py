import pandas as pd
from pathlib import Path

# --- Configuración ---
df = pd.read_csv('incident_event_log.csv')
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
