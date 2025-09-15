import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar datos
df = pd.read_csv("incident_event_log.csv")

# 2. Procesar fechas y variable dependiente
df["opened_at"]  = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", dayfirst=True)
df["tiempo_resolver_horas"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600

# 3. Crear variables explicativas nuevas
df["made_sla_num"] = df["made_sla"].apply(lambda x: 1 if str(x).lower() == "true" else 0)
df["knowledge_num"] = df["knowledge"].apply(lambda x: 1 if str(x).lower() == "true" else 0)

# Codificar contact_type en números simples
df["contact_type_num"] = df["contact_type"].astype("category").cat.codes

variables = ["sys_mod_count", "made_sla_num", "knowledge_num", "contact_type_num"]

# 4. Dataset final
df_model = df[variables + ["tiempo_resolver_horas"]].dropna()
X = df_model[variables]
y = df_model["tiempo_resolver_horas"]

# 5. Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 6. Modelo
reg = LinearRegression()
reg.fit(X_train, y_train)

# 7. Predicción y evaluación
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)

# 8. Coeficientes
coef_df = pd.DataFrame({"variable": variables, "coef": reg.coef_})
print("\nIntercepto:", reg.intercept_)
print("\nCoeficientes:\n", coef_df)
