import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Parámetros =====
MAX_SCATTER_PAIRS = 8                 # nº de pares numéricos con mayor |correlación| (global)
TOP_CATS = 15                         # top categorías a graficar en barras por columna categórica
SCATTER_SAMPLE = 50000                # muestreo para dispersión si hay muchas filas
MAX_SCATTER_PER_FEATURE_TOTAL = 20    # límite total de gráficos "por variable"
SAVE_CORR_TO_CSV = False              # <-- Actívalo si quieres guardar la matriz de correlación
OUTDIR = "plots_importantes"          # carpeta para guardar (solo si SAVE_CORR_TO_CSV=True)

# ===== Cargar =====
df = pd.read_csv('incident_event_log.csv')
print("Vista rápida:\n", df.head(), "\n")

# ===== Tipos de datos =====
num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
obj_cols  = df.select_dtypes(include=["object"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

# ===== Estadísticas descriptivas (numéricas) =====
if num_cols:
    desc_num = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc_num["missing_count"] = df[num_cols].isna().sum()
    desc_num["missing_pct"]   = 100 * desc_num["missing_count"] / len(df)
    desc_num["n_unique"]      = [df[c].nunique(dropna=True) for c in num_cols]

    # Imprimir sin truncar
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\n=== Estadísticas descriptivas (numéricas) ===")
    print(desc_num.round(3).to_string())
else:
    desc_num = pd.DataFrame()
    print("No hay columnas numéricas.\n")

# ===== Gráficos (SIN limpiar df) =====

# 1) Histogramas y boxplots por columna numérica
for c in num_cols:
    # Histograma
    plt.figure()
    plt.hist(df[c].dropna().values, bins=30)
    plt.title(f"Histograma - {c}")
    plt.xlabel(c); plt.ylabel("Frecuencia")

    # Boxplot
    plt.figure()
    plt.boxplot(df[c].dropna().values, vert=True, showmeans=True)
    plt.title(f"Boxplot - {c}")
    plt.ylabel(c)

# 2A) Dispersión: pares numéricos con mayor |correlación| (global)
pairs_global = []
corr = pd.DataFrame()
if len(num_cols) >= 2:
    corr = df[num_cols].corr(method="pearson")

    # (Opcional) guardar matriz de correlación a CSV si lo activas
    if SAVE_CORR_TO_CSV:
        os.makedirs(OUTDIR, exist_ok=True)      # <-- crea carpeta ANTES de guardar (arreglo del error)
        corr.to_csv(os.path.join(OUTDIR, "correlacion_numerica.csv"))

    corr_abs = corr.abs().copy()
    # ignorar diagonal y triángulo superior para quedarnos con pares únicos
    corr_abs.values[np.triu_indices_from(corr_abs, 0)] = np.nan
    top_pairs = corr_abs.unstack().dropna().sort_values(ascending=False).head(MAX_SCATTER_PAIRS)

    for (a, b), val in top_pairs.items():
        pairs_global.append((a, b))
        plt.figure()
        # muestreo opcional
        if len(df) > SCATTER_SAMPLE:
            rs = np.random.RandomState(42)
            idx = rs.choice(len(df), size=SCATTER_SAMPLE, replace=False)
            x = df.loc[idx, a].values
            y = df.loc[idx, b].values
        else:
            x = df[a].values
            y = df[b].values
        plt.scatter(x, y, alpha=0.3)
        plt.title(f"Dispersión (top global) - {a} vs {b} (|corr|={val:.2f})")
        plt.xlabel(a); plt.ylabel(b)

# 2B) Dispersión: para CADA numérica, contra su par más correlacionado (sin duplicar)
pairs_by_feature = set()
count_plots = 0
if len(num_cols) >= 2:
    corr_abs = corr.abs() if not corr.empty else df[num_cols].corr(method="pearson").abs()
    for c in num_cols:
        partners = corr_abs[c].drop(labels=[c]).dropna()
        if partners.empty:
            continue
        best = partners.idxmax()  # más correlacionada con c
        pair = tuple(sorted([c, best]))
        if pair in pairs_by_feature:
            continue
        pairs_by_feature.add(pair)

        plt.figure()
        # muestreo opcional
        if len(df) > SCATTER_SAMPLE:
            rs = np.random.RandomState(123)
            idx = rs.choice(len(df), size=SCATTER_SAMPLE, replace=False)
            x = df.loc[idx, pair[0]].values
            y = df.loc[idx, pair[1]].values
        else:
            x = df[pair[0]].values
            y = df[pair[1]].values
        plt.scatter(x, y, alpha=0.3)
        corr_val = (corr if not corr.empty else df[num_cols].corr(method="pearson")).loc[pair[0], pair[1]]
        plt.title(f"Dispersión (por variable) - {pair[0]} vs {pair[1]} (corr={corr_val:.2f})")
        plt.xlabel(pair[0]); plt.ylabel(pair[1])

        count_plots += 1
        if count_plots >= MAX_SCATTER_PER_FEATURE_TOTAL:
            break

# 3) Barras para CATEGÓRICAS (solo las especificadas y existentes)
cat_candidates = [c for c in ["priority", "impact", "urgency", "category", "contact_type", "assignment_group"] if c in df.columns]

for c in cat_candidates:
    vc = df[c].astype(str).value_counts().head(TOP_CATS)
    if vc.empty:
        continue
    plt.figure(figsize=(max(6, len(vc)*0.6), 4))
    plt.bar(vc.index.tolist(), vc.values.tolist())
    plt.title(f"Frecuencia - {c} (top {TOP_CATS})")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frecuencia")

# 4) Barras para booleanas (si existen)
for c in bool_cols:
    vc = df[c].value_counts(dropna=False)
    plt.figure()
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(f"Frecuencia - {c} (boolean)")
    plt.xlabel(c); plt.ylabel("Frecuencia")

# Mostrar todas las figuras en pantalla
plt.show()
