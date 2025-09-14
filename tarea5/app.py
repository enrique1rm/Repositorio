# ---------- app.py ----------
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State, callback_context  # <- aquí importo callback_context
import dash.dash_table as dash_table

# ====== RUTA AL CSV EXPORTADO DESDE df_closed ======
APP_DIR  = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
DATA_PATH = REPO_DIR / "Tarea 4" / "tarea5" / "data" / "incidentes_limpio.csv"
print(">> DATA_PATH:", DATA_PATH)

# ====== LECTURA Y NORMALIZACIÓN ======
hdr = pd.read_csv(DATA_PATH, nrows=0)
parse_cols = [c for c in ["opened_at", "resolved_at", "closed_at", "sys_updated_at"] if c in hdr.columns]
df = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=parse_cols)

# Fechas day-first
if "opened_at" in df.columns:
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)

# SLA a num (0/1)
if "made_sla" in df.columns:
    df["made_sla"] = pd.to_numeric(df["made_sla"], errors="coerce")

# Texto limpio (usar .str.strip(), no .strip())
for col in ["assignment_group", "category", "impact", "urgency", "priority"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# Quitar ruiditos en category (ej. "?")
if "category" in df.columns:
    df = df[df["category"] != "?"]

# Si no existen ordinales, intenta crearlos desde "1 - Alto" etc.
def first_int(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return float(s.split()[0]) if s[:1].isdigit() else np.nan

for c in ["impact", "urgency", "priority"]:
    if c in df.columns and (c + "_n") not in df.columns:
        df[c + "_n"] = df[c].apply(first_int)

print(">> CSV cargado. Filas:", len(df), "Columnas:", len(df.columns))

# ====== OPCIONES PARA FILTROS ======
min_date = df["opened_at"].min() if "opened_at" in df.columns else None
max_date = df["opened_at"].max() if "opened_at" in df.columns else None

def opts(series):
    # evita 'nan' como string en opciones
    s = series.copy()
    s = s.where(s.notna(), None)
    return [{"label": v, "value": v} for v in sorted(set([x for x in s.dropna().tolist() if x not in [None, "nan", "None"]]))]

groups_opts = opts(df["assignment_group"]) if "assignment_group" in df.columns else []
cats_opts   = opts(df["category"])         if "category" in df.columns else []
imp_opts    = opts(df["impact"])           if "impact" in df.columns else []
urg_opts    = opts(df["urgency"])          if "urgency" in df.columns else []
pri_opts    = opts(df["priority"])         if "priority" in df.columns else []

# ====== APP ======
app = Dash(__name__)
app.title = "Tablero de Incidentes — Gerente de TI"

def kpi_card(id_, title):
    return html.Div([
        html.Div(title, style={"fontSize":"0.9rem", "color":"#555", "marginBottom":"2px"}),
        html.Div("—", id=id_, style={"fontSize":"1.6rem", "fontWeight":"bold"})
    ], style={
        "padding":"10px","border":"1px solid #eee","borderRadius":"12px",
        "boxShadow":"0 1px 4px rgba(0,0,0,.05)","background":"#fff"
    })

# ====== LAYOUT ======
app.layout = html.Div([
    html.H2("Tablero de Incidentes — Gerente de TI"),

    # Instrucciones (UX)
    html.Details([
        html.Summary("¿Cómo usar este tablero?"),
        html.Ul([
            html.Li("Seleccione el rango de fechas (opened_at) y use los filtros de grupo, categoría, impacto, urgencia y prioridad."),
            html.Li("MTTR es el tiempo promedio (horas) para resolver incidentes del filtro actual."),
            html.Li("% SLA es la proporción con made_sla == 1 en el filtro."),
            html.Li("Los gráficos responden a los filtros; puede seleccionar múltiples valores."),
            html.Li("Use “Reset filtros” para volver a la vista completa.")
        ])
    ], style={"maxWidth":"820px","margin":"0 0 12px 0"}),

    # Fila de filtros (1)
    html.Div([
        html.Div([
            html.Label("Rango de fechas (opened_at)"),
            dcc.DatePickerRange(
                id="f-fechas",
                min_date_allowed=min_date, max_date_allowed=max_date,
                start_date=min_date, end_date=max_date
            )
        ], style={"width":"33%"}),

        html.Div([
            html.Label("Assignment group"),
            dcc.Dropdown(id="f-group", options=groups_opts, value=None, multi=True, placeholder="Todos"),
        ], style={"width":"33%"}),

        html.Div([
            html.Label("Category"),
            dcc.Dropdown(id="f-cat", options=cats_opts, value=None, multi=True, placeholder="Todas"),
        ], style={"width":"33%"}),
    ], style={"display":"flex","gap":"1.5%","marginBottom":"10px"}),

    # Fila de filtros (2)
    html.Div([
        html.Div([
            html.Label("Impacto"),
            dcc.Dropdown(id="f-impact", options=imp_opts, value=None, multi=True, placeholder="Todos"),
        ], style={"width":"24%"}),

        html.Div([
            html.Label("Urgencia"),
            dcc.Dropdown(id="f-urg", options=urg_opts, value=None, multi=True, placeholder="Todas"),
        ], style={"width":"24%"}),

        html.Div([
            html.Label("Prioridad"),
            dcc.Dropdown(id="f-pri", options=pri_opts, value=None, multi=True, placeholder="Todas"),
        ], style={"width":"24%"}),

        html.Div([
            html.Label(" "),
            html.Button("Reset filtros", id="btn-reset", n_clicks=0, style={
                "width":"100%","height":"38px","borderRadius":"8px","border":"1px solid #ddd","background":"#fafafa"
            }),
        ], style={"width":"24%","display":"flex","alignItems":"flex-end"}),
    ], style={"display":"flex","gap":"1.5%","marginBottom":"14px"}),

    # KPIs
    html.Div([
        kpi_card("kpi-mttr", "MTTR (h)"),
        kpi_card("kpi-sla", "% SLA cumplidos"),
        kpi_card("kpi-vol", "Volumen"),
        kpi_card("kpi-crit", "% críticos (impact=1)")
    ], style={"display":"grid", "gridTemplateColumns":"repeat(4, 1fr)", "gap":"12px", "marginBottom":"14px"}),

    # Gráficos 1
    html.Div([
        dcc.Graph(id="g-box-impact", config={"displaylogo": False}),
        dcc.Graph(id="g-bar-group",   config={"displaylogo": False}),
    ], style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"12px", "marginBottom":"14px"}),

    # Gráficos 2 (tendencia + dispersión)
    html.Div([
        dcc.Graph(id="g-trend",            config={"displaylogo": False}),
        dcc.Graph(id="g-scatter-reassign", config={"displaylogo": False}),
    ], style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"12px", "marginBottom":"14px"}),

    # Tabla resumen y descarga
    html.Div([
        html.Div("Resumen por assignment_group", style={"fontWeight":"600","margin":"6px 0"}),
        html.Div(id="tbl-wrap"),
        html.Button("Descargar CSV", id="btn-dw", n_clicks=0,
                    style={"marginTop":"8px","borderRadius":"8px","border":"1px solid #ddd","background":"#fafafa","padding":"6px 10px"}),
        dcc.Download(id="dw")
    ], style={"marginBottom":"24px"}),

], style={"maxWidth":"1200px", "margin":"20px auto", "fontFamily":"Inter, system-ui, sans-serif"})

# ====== CALLBACKS ======

# Reset filtros
@app.callback(
    [Output("f-group","value"),
     Output("f-cat","value"),
     Output("f-impact","value"),
     Output("f-urg","value"),
     Output("f-pri","value"),
     Output("f-fechas","start_date"),
     Output("f-fechas","end_date")],
    Input("btn-reset","n_clicks"),
    prevent_initial_call=True
)
def reset_filters(_):
    return (None, None, None, None, None,
            min_date.date() if pd.notna(min_date) else None,
            max_date.date() if pd.notna(max_date) else None)

@app.callback(
    [Output("kpi-mttr","children"),
     Output("kpi-sla","children"),
     Output("kpi-vol","children"),
     Output("kpi-crit","children"),
     Output("g-box-impact","figure"),
     Output("g-bar-group","figure"),
     Output("g-trend","figure"),
     Output("g-scatter-reassign","figure"),
     Output("tbl-wrap","children"),
     Output("dw","data")],
    [Input("f-fechas","start_date"), Input("f-fechas","end_date"),
     Input("f-group","value"), Input("f-cat","value"),
     Input("f-impact","value"), Input("f-urg","value"), Input("f-pri","value"),
     Input("btn-dw","n_clicks")],
    prevent_initial_call=False
)
def actualizar(start_date, end_date, groups_sel, cats_sel, imp_sel, urg_sel, pri_sel, n_dw):
    try:
        dff = df.copy()

        # --- Filtros ---
        if "opened_at" in dff.columns:
            if start_date: dff = dff[dff["opened_at"] >= pd.to_datetime(start_date)]
            if end_date:   dff = dff[dff["opened_at"] <= pd.to_datetime(end_date)]

        def sel_ok(v):
            return v not in (None, [], ())

        if "assignment_group" in dff.columns and sel_ok(groups_sel):
            dff = dff[dff["assignment_group"].isin(groups_sel)]
        if "category" in dff.columns and sel_ok(cats_sel):
            dff = dff[dff["category"].isin(cats_sel)]
        if "impact" in dff.columns and sel_ok(imp_sel):
            dff = dff[dff["impact"].isin(imp_sel)]
        if "urgency" in dff.columns and sel_ok(urg_sel):
            dff = dff[dff["urgency"].isin(urg_sel)]
        if "priority" in dff.columns and sel_ok(pri_sel):
            dff = dff[dff["priority"].isin(pri_sel)]

        # Estado sin datos
        if dff.empty:
            empty = px.scatter(title="(sin datos con los filtros seleccionados)")
            table = html.Div("No hay datos que mostrar.", style={"padding":"6px"})
            return "n/d","n/d","0","n/d", empty, empty, empty, empty, table, None

        # --- KPIs ---
        if "tiempo_resolucion_horas" in dff.columns:
            kpi_mttr = f"{dff['tiempo_resolucion_horas'].mean():,.2f} h"
        else:
            kpi_mttr = "n/d"

        if "made_sla" in dff.columns:
            kpi_sla = f"{dff['made_sla'].mean()*100:,.1f}%"
        else:
            kpi_sla = "n/d"

        kpi_vol = f"{len(dff):,}"

        if "impact_n" in dff.columns:
            crit = (dff["impact_n"] == 1).mean()*100
            kpi_crit = f"{crit:,.1f}%"
        elif "impact" in dff.columns:
            crit = dff["impact"].astype(str).str.startswith("1").mean()*100
            kpi_crit = f"{crit:,.1f}%"
        else:
            kpi_crit = "n/d"

        # --- Figuras ---
        # 1) Boxplot por Impact
        if {"impact","tiempo_resolucion_horas"}.issubset(dff.columns):
            fig1 = px.box(dff, x="impact", y="tiempo_resolucion_horas",
                          title="Tiempo por Impact",
                          labels={"tiempo_resolucion_horas":"tiempo_resolucion_horas"})
        else:
            fig1 = px.box(title="Tiempo por Impact (n/d)")

        # 2) Barra - tiempo promedio por grupo (Top 20)
        if {"assignment_group","tiempo_resolucion_horas"}.issubset(dff.columns):
            tmp = (dff.groupby("assignment_group", as_index=False)["tiempo_resolucion_horas"]
                      .mean()
                      .sort_values("tiempo_resolucion_horas", ascending=False)
                      .head(20))
            fig2 = px.bar(tmp, x="assignment_group", y="tiempo_resolucion_horas",
                          title="Tiempo promedio por grupo (Top 20)",
                          labels={"tiempo_resolucion_horas":"tiempo_resolucion_horas"})
        else:
            fig2 = px.bar(title="Tiempo promedio por grupo (n/d)")

        # 3) Tendencia semanal del MTTR
        if {"opened_at","tiempo_resolucion_horas"}.issubset(dff.columns):
            trend = (dff.assign(semana=dff["opened_at"].dt.to_period("W").dt.start_time)
                        .groupby("semana", as_index=False)["tiempo_resolucion_horas"].mean())
            fig3 = px.line(trend, x="semana", y="tiempo_resolucion_horas",
                           title="MTTR por semana", markers=True,
                           labels={"tiempo_resolucion_horas":"tiempo_resolucion_horas"})
        else:
            fig3 = px.line(title="MTTR por semana (n/d)")

        # 4) Dispersión Reasignaciones vs Tiempo
        if {"reassignment_count","tiempo_resolucion_horas"}.issubset(dff.columns):
            fig4 = px.scatter(dff, x="reassignment_count", y="tiempo_resolucion_horas",
                              title="Reasignaciones vs Tiempo de resolución",
                              labels={"tiempo_resolucion_horas":"tiempo_resolucion_horas"})
        else:
            fig4 = px.scatter(title="Reasignaciones vs Tiempo (n/d)")

        # --- Tabla resumen por grupo ---
        if {"assignment_group","number","tiempo_resolucion_horas","made_sla"}.issubset(dff.columns):
            summary = (dff.groupby("assignment_group", as_index=False)
                         .agg(vol=("number","count"),
                              mttr=("tiempo_resolucion_horas","mean"),
                              sla=("made_sla","mean")))
            summary["sla"]  = (summary["sla"]*100).round(1)
            summary["mttr"] = summary["mttr"].round(2)
            summary = summary.sort_values(["mttr","vol"], ascending=[False, False])

            table = dash_table.DataTable(
                id="tbl",
                columns=[{"name":n,"id":n} for n in ["assignment_group","vol","mttr","sla"]],
                data=summary.to_dict("records"),
                page_size=10,
                style_table={"overflowX":"auto"},
                style_cell={"fontFamily":"inherit","fontSize":"14px","padding":"6px"},
                style_header={"fontWeight":"600","background":"#fafafa"}
            )
        else:
            summary = pd.DataFrame()
            table = html.Div("No hay columnas suficientes para resumen (assignment_group, number, tiempo_resolucion_horas, made_sla).",
                             style={"padding":"6px"})

        # Descarga (usando callback_context, ya importado)
        download = None
        ctx = callback_context
        if ctx.triggered and "btn-dw" in ctx.triggered[0]["prop_id"] and not summary.empty:
            download = dcc.send_data_frame(summary.to_csv, "resumen_por_grupo.csv", index=False)

        return kpi_mttr, kpi_sla, kpi_vol, kpi_crit, fig1, fig2, fig3, fig4, table, download

    except Exception as e:
        print(">> Error en callback:", e)
        empty = px.scatter(title="(error)")
        table = html.Div("Ocurrió un error al generar la vista.", style={"padding":"6px"})
        return "n/d","n/d","n/d","n/d", empty, empty, empty, empty, table, None

if __name__ == "__main__":
    print(">> Levantando servidor Dash en http://127.0.0.1:8050 ...")
    app.run(host="127.0.0.1", port=8050, debug=True)
