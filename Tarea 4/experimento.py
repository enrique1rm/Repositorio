# Tarea4.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 🔹 Importar dataset ya limpio de Tarea 3
from Tarea3 import df_closed  

# --- Variables ---
variables = ["priority", "impact", "reassignment_count", "reopen_count"]

# Quedarse solo con las variables necesarias + target
df_model = df_closed[variables + ["tiempo_resolucion_horas"]].dropna()

# Mapear categóricas a números (más simple que OneHot, y consistente con lo visto antes)
priority_map = {"1 - Critical": 1, "2 - High": 2, "3 - Moderate": 3, "4 - Low": 4}
impact_map   = {"1 - High": 1, "2 - Medium": 2, "3 - Low": 3}

df_model["priority_num"] = df_model["priority"].map(priority_map)
df_model["impact_num"]   = df_model["impact"].map(impact_map)

X = df_model[["priority_num", "impact_num", "reassignment_count", "reopen_count"]]
y = df_model["tiempo_resolucion_horas"]

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- Modelo de Regresión Lineal ---
reg = LinearRegression()
reg.fit(X_train, y_train)

# --- Evaluación ---
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)

# --- Coeficientes ---
coef_df = pd.DataFrame({
    "variable": X.columns,
    "coef": reg.coef_
}).sort_values(by="coef", ascending=False).reset_index(drop=True)

print("\nIntercepto (β0):", reg.intercept_)
print("\nCoeficientes:")
print(coef_df.to_string(index=False))
