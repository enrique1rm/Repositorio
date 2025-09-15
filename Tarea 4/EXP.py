import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Cargar datos ===
df = pd.read_csv("incident_event_log.csv")

# Transformar fechas
df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", dayfirst=True)

# Variable dependiente: tiempo de resolución en horas
df["tiempo_resolver_horas"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600

# Convertir "made_sla" a numérica (0/1)
df["made_sla_num"] = df["made_sla"].astype("category").cat.codes

# === 2. Selección de variables explicativas ===
variables = ["made_sla_num", "reassignment_count", "reopen_count", "sys_mod_count"]

df_model = df[variables + ["tiempo_resolver_horas"]].dropna()
X = df_model[variables]
y = df_model["tiempo_resolver_horas"]

# === 3. División entrenamiento / prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Entrenamiento del modelo ===
model = LinearRegression()
model.fit(X_train, y_train)

# === 5. Evaluación ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("Resultados del modelo final")
print("Variables explicativas:", variables)
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.4f}")

# === 6. Coeficientes ===
coef_df = pd.DataFrame({
    "Variable": variables,
    "Coeficiente": model.coef_
})
print("\nCoeficientes del modelo:")
print(coef_df.to_string(index=False))

print("\nIntercepto (β0):", model.intercept_)
