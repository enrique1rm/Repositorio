
import os, re
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px

import joblib
from sklearn.linear_model import LinearRegression


from pathlib import Path
APP_DIR  = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent

DATA_PATH  = REPO_DIR / "Tarea 4" / "tarea5" / "data" / "incidentes_limpio.csv"
MODEL_PATH = APP_DIR / "models" / "modelo_resolucion.pkl"

print(">> DATA_PATH:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(">> CSV cargado. Filas:", len(df), "Columnas:", len(df.columns))
print(">> Columnas:", list(df.columns))
 
df = pd.read_csv(DATA_PATH)
print(">> CSV cargado. Filas:", len(df), "Columnas:", len(df.columns))
for c in ["opened_at","resolved_at","closed_at","sys_updated_at"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

def first_int(x):
    if pd.isna(x): return np.nan
    m = re.match(r"\s*(\d+)", str(x))
    return int(m.group(1)) if m else np.nan

for c in ["impact","urgency","priority"]:
    if c in df.columns and c+"_n" not in df.columns:
        df[c+"_n"] = df[c].apply(first_int)

if "opened_at" in df.columns:
    if "opened_hour" not in df.columns:
        df["opened_hour"] = df["opened_at"].dt.hour
    if "opened_wday" not in df.columns:
        df["opened_wday"] = df["opened_at"].dt.dayofweek
    if "opened_month" not in df.columns:
        df["opened_month"] = df["opened_at"].dt.month

min_date = df["opened_at"].min() if "opened_at" in df.columns else None
max_date = df["opened_at"].max() if "opened_at" in df.columns else None

groups = sorted(df["assignment_group"].dropna().unique()) if "assignment_group" in df.columns else []
cats   = sorted(df["category"].dropna().unique()) if "category" in df.columns else []

baseline_model = None
feat_cols = [c for c in ["impact_n","urgency_n","priority_n","reassignment_count","reopen_count","opened_hour","opened_wday","opened_month"] if c in df.columns]

if os.path.exists(MODEL_PATH):
    try:
        baseline_model = joblib.load(MODEL_PATH)
    except Exception:
        baseline_model = None

if baseline_model is None and "tiempo_resolucion_horas" in df.columns and len(feat_cols) > 0 and len(df) > 50:
    X = df[feat_cols].fillna(0)
    y = df["tiempo_resolucion_horas"].values
    baseline_model = LinearRegression().fit(X, y)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Tablero de Incidentes — Gerente de TI"),

    html.Div([
        html.Div([
            html.Label("Rango de fechas (opened_at)"),
            dcc.DatePickerRange(
                id="f-fechas",
                min_date_allowed=min_date, max_date_allowed=max_date,
                start_date=min_date, end_date=max_date
            )
        ], style={"marginRight":"16px"}),

        html.Div([
            html.Label("Assignment group"),
            dcc.Dropdown(id="f-group",
                         options=[{"label": g, "value": g} for g in groups],
                         multi=True, placeholder="Todos")
        ], style={"width":"320px","marginRight":"16px"}),

        html.Div([
            html.Label("Category"),
            dcc.Dropdown(id="f-cat",
                         options=[{"label": c, "value": c} for c in cats],
                         multi=True, placeholder="Todas")
        ], style={"width":"320px"})
    ], style={"display":"flex","flexWrap":"wrap","alignItems":"end","marginBottom":"12px"}),

    html.Div([
        html.Div(id="kpi-mttr",  className="kpi"),
        html.Div(id="kpi-sla",   className="kpi"),
        html.Div(id="kpi-vol",   className="kpi"),
        html.Div(id="kpi-crit",  className="kpi"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"12px","marginBottom":"12px"}),

    html.Div([
        dcc.Graph(id="g-box-impact"),
        dcc.Graph(id="g-bar-group"),
        dcc.Graph(id="g-scatter-reassign"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":"12px"}),

    html.Hr(),

    html.H3("Predicción de tiempo de resolución"),
    html.Div([
        html.Div([
            html.Label("Impact (1=alto, 2=medio, 3=bajo)"),
            dcc.Dropdown(id="p-impact",
                         options=[{"label": str(v), "value": v} for v in sorted(df["impact_n"].dropna().unique())] if "impact_n" in df.columns else [],
                         placeholder="Selecciona…"),
        ], style={"width":"220px","marginRight":"12px"}),

        html.Div([
            html.Label("Urgency (1,2,3)"),
            dcc.Dropdown(id="p-urgency",
                         options=[{"label": str(v), "value": v} for v in sorted(df["urgency_n"].dropna().unique())] if "urgency_n" in df.columns else [],
                         placeholder="Selecciona…"),
        ], style={"width":"220px","marginRight":"12px"}),

        html.Div([
            html.Label("Priority (1..n)"),
            dcc.Dropdown(id="p-priority",
                         options=[{"label": str(v), "value": v} for v in sorted(df["priority_n"].dropna().unique())] if "priority_n" in df.columns else [],
                         placeholder="Selecciona…"),
        ], style={"width":"220px","marginRight":"12px"}),

        html.Div([
            html.Label("Reassignment count"),
            dcc.Input(id="p-reassign", type="number", value=0, min=0, step=1),
        ], style={"width":"220px","marginRight":"12px"}),

        html.Div([
            html.Label("Reopen count"),
            dcc.Input(id="p-reopen", type="number", value=0, min=0, step=1),
        ], style={"width":"220px","marginRight":"12px"}),

        html.Button("Predecir", id="p-btn", n_clicks=0)
    ], style={"display":"flex","flexWrap":"wrap","alignItems":"end","gap":"8px"}),

    html.Div(id="p-out", style={"marginTop":"12px","fontWeight":"bold"})
], style={"padding":"16px"})

def filtrar(df0, start_date, end_date, groups, cats):
    dff = df0.copy()
    if "opened_at" in dff.columns:
        if start_date: dff = dff[dff["opened_at"] >= pd.to_datetime(start_date)]
        if end_date:   dff = dff[dff["opened_at"] <= pd.to_datetime(end_date)]
    if "assignment_group" in dff.columns and groups:
        dff = dff[dff["assignment_group"].isin(groups)]
    if "category" in dff.columns and cats:
        dff = dff[dff["category"].isin(cats)]
    return dff

@app.callback(
    [Output("kpi-mttr","children"),
     Output("kpi-sla","children"),
     Output("kpi-vol","children"),
     Output("kpi-crit","children"),
     Output("g-box-impact","figure"),
     Output("g-bar-group","figure"),
     Output("g-scatter-reassign","figure")],
    [Input("f-fechas","start_date"), Input("f-fechas","end_date"),
     Input("f-group","value"), Input("f-cat","value")]
)
def actualizar(start_date, end_date, groups, cats):
    dff = filtrar(df, start_date, end_date, groups, cats)

    mttr = dff["tiempo_resolucion_horas"].mean() if "tiempo_resolucion_horas" in dff.columns else np.nan
    kpi_mttr = f"MTTR: {mttr:.2f} h" if pd.notna(mttr) else "MTTR: n/d"

    if "made_sla" in dff.columns and len(dff)>0:
        sla = dff["made_sla"].mean()*100
        kpi_sla = f"% SLA cumplidos: {sla:.1f}%"
    else:
        kpi_sla = "% SLA cumplidos: n/d"

    kpi_vol = f"Volumen: {len(dff)}"

    crit = (dff["impact_n"].eq(1).mean()*100) if "impact_n" in dff.columns and len(dff)>0 else np.nan
    kpi_crit = f"% críticos: {crit:.1f}%" if pd.notna(crit) else "% críticos: n/d"

    fig1 = px.box(dff, x="impact", y="tiempo_resolucion_horas", title="Tiempo por Impact")
    if "assignment_group" in dff.columns:
        tmp = dff.groupby("assignment_group", as_index=False)["tiempo_resolucion_horas"].mean().sort_values("tiempo_resolucion_horas", ascending=False).head(20)
        fig2 = px.bar(tmp, x="assignment_group", y="tiempo_resolucion_horas", title="Tiempo promedio por grupo")
    else:
        fig2 = px.bar(title="Tiempo promedio por grupo (columna no disponible)")

    if "reassignment_count" in dff.columns:
        fig3 = px.scatter(dff, x="reassignment_count", y="tiempo_resolucion_horas",
                          title="Reasignaciones vs Tiempo de resolución", trendline="ols")
    else:
        fig3 = px.scatter(title="Reasignaciones vs Tiempo (columna no disponible)")

    return kpi_mttr, kpi_sla, kpi_vol, kpi_crit, fig1, fig2, fig3

@app.callback(
    Output("p-out","children"),
    Input("p-btn","n_clicks"),
    State("p-impact","value"),
    State("p-urgency","value"),
    State("p-priority","value"),
    State("p-reassign","value"),
    State("p-reopen","value")
)
def predecir(n, imp, urg, pri, reas, reop):
    if not n:
        return ""
    if baseline_model is None or len(feat_cols)==0:
        return "Modelo no disponible. Ejecuta el pipeline (o conecta el .pkl del equipo)."

    row = {
        "impact_n": imp or 2,
        "urgency_n": urg or 2,
        "priority_n": pri or 2,
        "reassignment_count": reas or 0,
        "reopen_count": reop or 0,
        "opened_hour": 12 if "opened_hour" in feat_cols else 12,
        "opened_wday": 2 if "opened_wday" in feat_cols else 2,
        "opened_month": 6 if "opened_month" in feat_cols else 6,
    }
    Xnew = pd.DataFrame([row])[feat_cols].fillna(0)
    pred = baseline_model.predict(Xnew)[0]
    return f"⏱️ Tiempo estimado de resolución: {pred:.2f} horas"

if __name__ == "__main__":
    print(">> Levantando servidor Dash en http://127.0.0.1:8050 ...")
    app.run(debug=True, host="127.0.0.1", port=8050)


