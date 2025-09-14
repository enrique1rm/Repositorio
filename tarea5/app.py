import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from dash import Dash, dcc, html, Input, Output

# ====== RUTA AL CSV EXPORTADO DESDE df_closed ======
# Opción A (recomendada): pathlib con ruta relativa robusta
APP_DIR  = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
DATA_PATH = REPO_DIR / "Tarea 4" / "tarea5" / "data" / "incidentes_limpio.csv"


print(">> DATA_PATH:", DATA_PATH)

parse_cols = ["opened_at", "resolved_at", "closed_at", "sys_updated_at"]
df = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=[c for c in parse_cols if c in pd.read_csv(DATA_PATH, nrows=0).columns])

if "opened_at" in df.columns:
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)

if "made_sla" in df.columns:
 
    df["made_sla"] = pd.to_numeric(df["made_sla"], errors="coerce")

for col in ["assignment_group", "category"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


print(">> CSV cargado. Filas:", len(df), "Columnas:", len(df.columns))

# ====== OPCIONES PARA FILTROS ======
min_date = df["opened_at"].min() if "opened_at" in df.columns else None
max_date = df["opened_at"].max() if "opened_at" in df.columns else None

groups = sorted(df["assignment_group"].dropna().unique().tolist()) if "assignment_group" in df.columns else []
cats   = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []

# ====== APP ======
app = Dash(__name__)
app.title = "Tablero de Incidentes — Gerente de TI"

def kpi_card(id_, title):
    return html.Div([
        html.H4(title),
        html.Div("—", id=id_, style={"fontSize":"1.4rem", "fontWeight":"bold"})
    ], style={"padding":"10px", "border":"1px solid #eee", "borderRadius":"12px", "boxShadow":"0 1px 4px rgba(0,0,0,.05)"})

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
        ], style={"width":"32%"}),

        html.Div([
            html.Label("Assignment group"),
            dcc.Dropdown(options=[{"label":g, "value":g} for g in groups],
                         value=[], multi=True, placeholder="Todos",
                         id="f-group")
        ], style={"width":"32%"}),

        html.Div([
            html.Label("Category"),
            dcc.Dropdown(options=[{"label":c, "value":c} for c in cats],
                         value=[], multi=True, placeholder="Todas",
                         id="f-cat")
        ], style={"width":"32%"})
    ], style={"display":"flex", "gap":"2%", "marginBottom":"14px"}),

    html.Div([
        kpi_card("kpi-mttr", "MTTR (h)"),
        kpi_card("kpi-sla", "% SLA cumplidos"),
        kpi_card("kpi-vol", "Volumen"),
        kpi_card("kpi-crit", "% críticos (impact=1)")
    ], style={"display":"grid", "gridTemplateColumns":"repeat(4, 1fr)", "gap":"12px", "marginBottom":"14px"}),

    html.Div([
        dcc.Graph(id="g-box-impact"),
        dcc.Graph(id="g-bar-group")
    ], style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"12px"}),

    html.Div([
        dcc.Graph(id="g-scatter-reassign")
    ]),
], style={"maxWidth":"1200px", "margin":"20px auto", "fontFamily":"sans-serif"})

# ====== CALLBACK ======
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
def actualizar(start_date, end_date, groups_sel, cats_sel):
    try:
        dff = df.copy()

        # Filtros
        if "opened_at" in dff.columns:
            if start_date: dff = dff[dff["opened_at"] >= pd.to_datetime(start_date)]
            if end_date:   dff = dff[dff["opened_at"] <= pd.to_datetime(end_date)]
        if "assignment_group" in dff.columns and groups_sel:
            dff = dff[dff["assignment_group"].isin(groups_sel)]
        if "category" in dff.columns and cats_sel:
            dff = dff[dff["category"].isin(cats_sel)]

        # KPIs
        if "tiempo_resolucion_horas" in dff.columns and len(dff):
            mttr = dff["tiempo_resolucion_horas"].mean()
            kpi_mttr = f"{mttr:.2f} h"
        else:
            kpi_mttr = "n/d"

        if "made_sla" in dff.columns and len(dff):
            sla = dff["made_sla"].mean() * 100
            kpi_sla = f"{sla:.1f}%"
        else:
            kpi_sla = "n/d"

        kpi_vol = f"{len(dff)}"

        crit = None
        if "impact_n" in dff.columns and len(dff):
            crit = (dff["impact_n"] == 1).mean() * 100
        elif "impact" in dff.columns and len(dff):
            # Heurística si impact viene como texto tipo "1 - High"
            crit = dff["impact"].astype(str).str.startswith("1").mean() * 100
        kpi_crit = f"{crit:.1f}%" if crit is not None else "n/d"

        # Figuras
        if {"impact","tiempo_resolucion_horas"}.issubset(dff.columns):
            fig1 = px.box(dff, x="impact", y="tiempo_resolucion_horas", title="Tiempo por Impact")
        else:
            fig1 = px.box(title="Tiempo por Impact (n/d)")

        if {"assignment_group","tiempo_resolucion_horas"}.issubset(dff.columns) and len(dff):
            tmp = (dff
                   .groupby("assignment_group", as_index=False)["tiempo_resolucion_horas"]
                   .mean()
                   .sort_values("tiempo_resolucion_horas", ascending=False)
                   .head(20))
            fig2 = px.bar(tmp, x="assignment_group", y="tiempo_resolucion_horas",
                          title="Tiempo promedio por grupo (Top 20)")
        else:
            fig2 = px.bar(title="Tiempo promedio por grupo (n/d)")

        if {"reassignment_count","tiempo_resolucion_horas"}.issubset(dff.columns):
            # Si no tienes statsmodels instalado, quita trendline="ols"
            fig3 = px.scatter(dff, x="reassignment_count", y="tiempo_resolucion_horas",
                              title="Reasignaciones vs Tiempo de resolución", trendline="ols")
        else:
            fig3 = px.scatter(title="Reasignaciones vs Tiempo (n/d)")

        return kpi_mttr, kpi_sla, kpi_vol, kpi_crit, fig1, fig2, fig3

    except Exception as e:
        print(">> Error en callback:", e)
        empty = px.scatter(title="(sin datos)")
        return "n/d","n/d","n/d","n/d", empty, empty, empty

if __name__ == "__main__":
    print(">> Levantando servidor Dash en http://127.0.0.1:8050 ...")
    app.run(host="127.0.0.1", port=8050, debug=True)
