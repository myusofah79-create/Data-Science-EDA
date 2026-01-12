import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

st.set_page_config(page_title="EDA App", layout="wide")

st.title("📊 Simple EDA App (CSV / Excel)")
st.write("Upload fail CSV atau Excel, dan app akan buat ringkasan + graf automatik.")

# =========================
# Helpers
# =========================
def load_data(uploaded_file):
    """Load CSV or Excel into a dataframe."""
    file_name = uploaded_file.name
    ext = Path(file_name).suffix.lower()

    if ext == ".csv":
        # Basic CSV read (if you face encoding issues, see note below)
        df = pd.read_csv(uploaded_file)
        return df

    elif ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(uploaded_file)
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox("Pilih sheet", xls.sheet_names)
        else:
            sheet = xls.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        return df

    else:
        st.error("Format tidak disokong. Sila upload .csv / .xlsx / .xls")
        return None


def plot_histograms(df, numeric_cols):
    st.subheader("📌 Histograms (Numerical)")
    cols = st.columns(2)
    for i, col in enumerate(numeric_cols):
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        cols[i % 2].pyplot(fig)


def plot_boxplots(df, numeric_cols):
    st.subheader("📌 Boxplots (Numerical)")
    cols = st.columns(2)
    for i, col in enumerate(numeric_cols):
        fig, ax = plt.subplots()
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(f"Boxplot: {col}")
        ax.set_ylabel(col)
        cols[i % 2].pyplot(fig)


def plot_correlation_heatmap(df, numeric_cols):
    st.subheader("📌 Correlation Heatmap")
    if len(numeric_cols) < 2:
        st.info("Tak cukup column numerical untuk correlation (perlukan at least 2).")
        return

    corr = df[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots()
    cax = ax.imshow(corr.values)
    ax.set_title("Correlation Heatmap")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90)
    ax.set_yticklabels(numeric_cols)

    fig.colorbar(cax)
    st.pyplot(fig)


def plot_scatter(df, numeric_cols):
    st.subheader("📌 Scatter Plot (Numerical vs Numerical)")
    if len(numeric_cols) < 2:
        st.info("Tak cukup column numerical untuk scatter (perlukan at least 2).")
        return

    x_col = st.selectbox("Pilih X", numeric_cols, key="scatter_x")
    y_col = st.selectbox("Pilih Y", numeric_cols, key="scatter_y")

    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_title(f"Scatter: {x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)


def plot_categorical_counts(df, cat_cols):
    st.subheader("📌 Bar Chart (Categorical Counts)")
    if len(cat_cols) == 0:
        st.info("Tiada column kategori/non-numerical untuk plot.")
        return

    cat_col = st.selectbox("Pilih column kategori", cat_cols, key="cat_count")

    # Top N categories to avoid super long chart
    top_n = st.slider("Show Top N categories", 5, 50, 10)

    vc = df[cat_col].astype(str).value_counts().head(top_n)

    fig, ax = plt.subplots()
    ax.bar(vc.index, vc.values)
    ax.set_title(f"Top {top_n} Categories: {cat_col}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)


def plot_missing_values(df):
    st.subheader("📌 Missing Values per Column")

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        st.success("Tiada missing values ✅")
        return

    fig, ax = plt.subplots()
    ax.bar(missing.index.astype(str), missing.values)
    ax.set_title("Missing Values Count")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)


# =========================
# UI: Upload
# =========================
uploaded_file = st.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.warning("Sila upload fail untuk mula.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()

# =========================
# Basic Preview
# =========================
st.success("Data berjaya dibaca ✅")

st.subheader("👀 Preview Data")
st.dataframe(df.head(50), use_container_width=True)

st.subheader("📌 Data Shape")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values (total)", int(df.isna().sum().sum()))

# =========================
# Detect columns
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# =========================
# Summary Sections
# =========================
st.divider()
st.header("1) Summary Statistics")

# Numerical summary
st.subheader("✅ Numerical Features Summary")
if len(numeric_cols) == 0:
    st.info("Tiada numerical columns dalam dataset.")
else:
    st.write(df[numeric_cols].describe())

# TASK 2: Only show non-numerical summary if exists
st.subheader("✅ Non-Numerical Features Summary")
if len(cat_cols) == 0:
    st.info("Tiada non-numerical features, jadi summary ini tidak dipaparkan.")
else:
    # describe(include="object") is ok, but can include bool/category too
    st.write(df[cat_cols].describe(include="all"))

# =========================
# Graphs (TASK 3)
# =========================
st.divider()
st.header("2) Visualizations")

plot_missing_values(df)

if len(numeric_cols) > 0:
    plot_histograms(df, numeric_cols)
    plot_boxplots(df, numeric_cols)
    plot_correlation_heatmap(df, numeric_cols)
    plot_scatter(df, numeric_cols)

if len(cat_cols) > 0:
    plot_categorical_counts(df, cat_cols)

st.divider()
st.caption("Tip: Kalau CSV ada issue encoding/separator, cuba gunakan pd.read_csv(uploaded_file, sep=';', encoding='latin1').")
