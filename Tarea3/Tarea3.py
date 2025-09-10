import pandas as pd
from pathlib import Path

# --- Configuración ---
df = pd.read_csv('incident_event_log.csv')
print (df.head())

#Conversión de fechas 
for col in ["opened_at", "resolved_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

#Variable dependiente: tiempo de resolución en horas 
df["tiempo_resolver_horas"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600.0

