import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



import Tarea3  
from Tarea3 import df_closed
df=Tarea3.df_closed

df["opened_at"]  = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", dayfirst=True)
df["tiempo_resolver_horas"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600



df["made_sla_num"] = df["made_sla"].astype(str).str.lower().eq("true").astype(int)
df["knowledge_num"] = df["knowledge"].astype(str).str.lower().eq("true").astype(int)




df["contact_type_num"] = df["contact_type"].astype("category").cat.codes

variables = ["sys_mod_count", "made_sla_num", "knowledge_num", "contact_type_num"]


df_model = df[variables + ["tiempo_resolver_horas"]].dropna()
X = df_model[variables]
y = df_model["tiempo_resolver_horas"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


reg = LinearRegression()
reg.fit(X_train, y_train)


y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)


coef_df = pd.DataFrame({"variable": variables, "coef": reg.coef_})
print("\nIntercepto:", reg.intercept_)
print("\nCoeficientes:\n", coef_df)
