"""
Final Sewage Effluent Explorer
================================
An easy-to-understand Streamlit dashboard for exploring the Environment Agency
"Final Sewage Effluent" water-quality dataset (2000-2025).

The dataset is large: ~8.3 million rows, 454 distinct tests, 11,212 sampling
points. To stay fast and uncluttered the app follows two rules:

  1.  HEADLINE STATS COME FROM A PRECOMPUTED SUMMARY CSV
      (EA_final_sewage_det_summary_full.csv). Counts, percentiles, site counts,
      date ranges and geographic bounds for every test are read from this small
      466-row file, so the 8.3M-row parquet is never scanned just to show a
      number.

  2.  THE PARQUET IS READ ONLY ON DEMAND, ONE TEST AT A TIME
      Using column pushdown + a single Test== filter, even the busiest test
      loads a thin slice rather than the whole file.

Three pages:
  - Overview .............. the shape and scale of the whole dataset
  - Explore a test ........ pick any one test and see its full story
  - Priority tests ........ the 61 regulator-priority determinands, compared

Files expected alongside this app.py:
  - app.py                                       (this file)
  - requirements.txt
  - logo.png                                     (HydroStar logo)
  - EA_final_sewage_effluent_2000_2025_cleaned.parquet
  - EA_final_sewage_det_summary_full.csv
  - EA_final_sewage_test_correlations.csv
  - EA_final_sewage_year_counts.csv
"""

import base64
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Final Sewage Effluent Explorer",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.toolbarMode", "viewer")


# ======================================================
# FILE PATHS
# ======================================================
PARQUET_FILE = "EA_final_sewage_effluent_2000_2025_cleaned.parquet"
SUMMARY_FILE = "EA_final_sewage_det_summary_full.csv"
CORRELATION_FILE = "EA_final_sewage_test_correlations.csv"
YEAR_COUNTS_FILE = "EA_final_sewage_year_counts.csv"
LOGO_FILE = "logo.png"
DETAIL_ROW_LIMIT = 1_200_000


# ======================================================
# BRAND COLOURS (HYDROSTAR + DARK THEME)
# ------------------------------------------------------
# Straight from the HydroStar brand guidelines:
#   #a7d730 primary green, #499823 secondary green,
#   #30343c dark grey, #8c919a light grey. Font: Hind.
# ======================================================
PRIMARY_COLOUR = "#a7d730"
SECONDARY_COLOUR = "#499823"
DARK_GREY = "#30343c"
LIGHT_GREY = "#8c919a"
BACKGROUND = "#0e1117"
PANEL_BG = "#1b222b"
TEXT_COL = "#f2f4f7"
SUBTEXT_COL = LIGHT_GREY
ACCENT_COLOUR = "#86d5f8"
WATER_BLUE = "#4ea8de"
WARN_AMBER = "#f6a609"

# A qualitative palette for categorical charts (years, categories, sites).
# Green-led to stay on brand, with supporting hues for separation.
QUAL_PALETTE = [
    "#a7d730", "#4ea8de", "#f6a609", "#e6679a", "#9b8cff",
    "#2dd4bf", "#f97316", "#84cc16", "#38bdf8", "#c084fc",
    "#facc15", "#fb7185",
]

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_COLOURS = {
    "Spring": "#74c476",
    "Summer": "#f6c453",
    "Autumn": "#d97706",
    "Winter": "#60a5fa",
}
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# The 61 regulator-priority determinands (page 3).
PRIORITY_TESTS = [
    "Alkalinity to pH 4.5 as CaCO3", "Aluminium", "Aluminium, Dissolved",
    "Ammoniacal Nitrogen as N", "Antimony", "Arsenic", "Arsenic, Dissolved",
    "Barium", "Boron", "Boron, Dissolved", "Cadmium", "Cadmium, Dissolved",
    "Calcium", "Calcium, Dissolved", "Chloride", "Chromium",
    "Chromium, Dissolved", "Cobalt", "Conductivity at 20 C",
    "Conductivity at 25 C", "Copper", "Copper, Dissolved",
    "Cyanide : Free as CN", "Cyanide as CN", "Fluoride",
    "Hardness, Total as CaCO3", "Iron", "Iron, Dissolved", "Lead",
    "Lead, Dissolved", "Lithium", "Magnesium", "Magnesium, Dissolved",
    "Manganese", "Manganese, Dissolved", "Mercury", "Mercury, Dissolved",
    "Nickel", "Nickel, Dissolved", "Nitrate as N", "Nitrite as N",
    "Orthophosphate, reactive as P", "Potassium", "Potassium, Dissolved",
    "Salinity : In Situ", "Silver", "Silver, Dissolved", "Sodium",
    "Sodium, Dissolved", "Solids, Suspended at 105 C", "Strontium",
    "Sulphate as SO4", "Sulphate, Dissolved as SO4", "Sulphide as S",
    "Temperature of Water", "Tin", "Turbidity", "Vanadium", "Zinc",
    "Zinc, Dissolved", "pH",
]

# Short, friendly grouping of the priority tests so the user isn't faced with
# a flat list of 61 names. Used as an optional filter on page 3.
PRIORITY_GROUPS = {
    "Nutrients & oxygen demand": [
        "Ammoniacal Nitrogen as N", "Nitrate as N", "Nitrite as N",
        "Orthophosphate, reactive as P", "Sulphide as S",
    ],
    "Heavy metals (total)": [
        "Aluminium", "Antimony", "Arsenic", "Barium", "Cadmium", "Chromium",
        "Cobalt", "Copper", "Iron", "Lead", "Lithium", "Manganese", "Mercury",
        "Nickel", "Silver", "Strontium", "Tin", "Vanadium", "Zinc",
    ],
    "Heavy metals (dissolved)": [
        "Aluminium, Dissolved", "Arsenic, Dissolved", "Cadmium, Dissolved",
        "Chromium, Dissolved", "Copper, Dissolved", "Iron, Dissolved",
        "Lead, Dissolved", "Magnesium, Dissolved", "Manganese, Dissolved",
        "Mercury, Dissolved", "Nickel, Dissolved", "Potassium, Dissolved",
        "Silver, Dissolved", "Sodium, Dissolved", "Zinc, Dissolved",
        "Calcium, Dissolved", "Boron, Dissolved", "Sulphate, Dissolved as SO4",
    ],
    "Major ions & minerals": [
        "Boron", "Calcium", "Chloride", "Fluoride", "Magnesium", "Potassium",
        "Sodium", "Sulphate as SO4",
    ],
    "Physico-chemical": [
        "Alkalinity to pH 4.5 as CaCO3", "Conductivity at 20 C",
        "Conductivity at 25 C", "Hardness, Total as CaCO3", "pH",
        "Salinity : In Situ", "Solids, Suspended at 105 C",
        "Temperature of Water", "Turbidity",
    ],
    "Cyanide": ["Cyanide : Free as CN", "Cyanide as CN"],
}


# ======================================================
# GLOBAL CSS  (dark, on-brand)
# ======================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Hind:wght@300;400;500;600;700&display=swap');

    :root {{
        --hs-primary: {PRIMARY_COLOUR};
        --hs-secondary: {SECONDARY_COLOUR};
        --hs-bg: {BACKGROUND};
        --hs-card: {PANEL_BG};
        --hs-text: {TEXT_COL};
        --hs-subtext: {SUBTEXT_COL};
        --hs-sidebar: {DARK_GREY};
    }}

    html, body, [class*="css"] {{ font-family: 'Hind', sans-serif; }}

    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(167, 215, 48, 0.11) 0%, rgba(14, 17, 23, 0) 35%),
            radial-gradient(circle at bottom left, rgba(78, 168, 222, 0.08) 0%, rgba(14, 17, 23, 0) 40%),
            var(--hs-bg);
        color: var(--hs-text);
    }}
    .block-container {{
        max-width: 1520px;
        padding: 1.8rem clamp(1rem, 2.2vw, 2.4rem) 2rem clamp(1rem, 2.2vw, 2.4rem);
        color: var(--hs-text);
    }}
    h1, h2, h3, h4, h5, h6 {{ color: var(--hs-text) !important; font-weight: 700; letter-spacing: 0.1px; }}
    p, span, label {{ color: var(--hs-text) !important; }}
    .stCaption, .stMarkdown small {{ color: var(--hs-subtext) !important; }}

    section[data-testid="stSidebar"] > div {{
        background:
            linear-gradient(180deg, rgba(48, 52, 60, 0.98) 0%, rgba(28, 34, 43, 0.98) 72%, rgba(20, 25, 32, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {{ color: #ffffff !important; }}
    section[data-testid="stSidebar"] hr {{
        margin: 1rem 0;
        border-color: rgba(255, 255, 255, 0.12);
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] .stToggle > label {{
        font-weight: 700;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        margin-bottom: 0.35rem;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(167, 215, 48, 0.12);
        border-color: rgba(167, 215, 48, 0.30);
    }}

    .sidebar-title-card {{
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(167, 215, 48, 0.16), rgba(78, 168, 222, 0.10));
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 0.75rem;
    }}
    .sidebar-kicker {{
        color: var(--hs-primary) !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 0.25rem 0;
    }}
    .sidebar-title {{
        color: #ffffff !important;
        font-size: 1.3rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0;
    }}
    .sidebar-subtitle {{
        color: rgba(255, 255, 255, 0.74) !important;
        font-size: 0.9rem;
        margin: 0.35rem 0 0 0;
    }}
    .sidebar-section-label {{
        color: rgba(255, 255, 255, 0.66) !important;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin: 0.35rem 0 0.2rem 0;
    }}
    .sidebar-stats-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem;
        margin-top: 0.2rem;
    }}
    .sidebar-stat {{
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.58rem 0.65rem;
    }}
    .sidebar-stat b {{
        display: block;
        color: #ffffff !important;
        font-size: 1rem;
        line-height: 1.1;
    }}
    .sidebar-stat span {{
        display: block;
        color: rgba(255, 255, 255, 0.62) !important;
        font-size: 0.74rem;
        line-height: 1.15;
        margin-top: 0.2rem;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background-color: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.16);
    }}
    .stSelectbox > div > div, .stMultiSelect > div > div {{ background-color: rgba(255, 255, 255, 0.06); }}
    .stSlider > div > div > div {{ background-color: rgba(167, 215, 48, 0.18); }}
    .stSlider [data-testid="stTickBar"] > div {{ background-color: rgba(167, 215, 48, 0.40); }}

    .stButton > button, .stDownloadButton > button {{
        background-color: var(--hs-primary); color: #1d2430;
        font-weight: 700; border: none; border-radius: 8px;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: var(--hs-secondary); color: #ffffff;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(27, 34, 43, 0.96) 0%, rgba(22, 29, 37, 0.96) 100%);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-left: 5px solid var(--hs-primary);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.24);
    }}
    div[data-testid="stMetric"] label {{
        color: var(--hs-subtext) !important;
        font-size: 0.82rem !important; letter-spacing: 0.35px; text-transform: uppercase;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--hs-text) !important; font-weight: 700; line-height: 1.1;
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.25rem 0 0.95rem 0;
    }}
    .metric-tile {{
        min-width: 0;
        background: linear-gradient(180deg, rgba(27, 34, 43, 0.96) 0%, rgba(22, 29, 37, 0.96) 100%);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-left: 5px solid var(--hs-primary);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.24);
    }}
    .metric-label {{
        display: block;
        color: var(--hs-subtext) !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        line-height: 1.15;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        display: block;
        color: var(--hs-text) !important;
        font-size: clamp(1.22rem, 1.7vw, 1.8rem);
        font-weight: 700;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    div[data-testid="stDataFrame"] {{
        background-color: rgba(27, 34, 43, 0.96);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 0.2rem;
    }}
    .stPlotlyChart {{
        background-color: rgba(27, 34, 43, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.55rem 1.45rem 0.25rem 0.55rem;
        margin-bottom: 1.1rem;
        box-sizing: border-box;
        clear: both;
        overflow: visible;
    }}

    /* Hero banner */
    .hero-banner {{
        display: flex; justify-content: space-between; align-items: center; gap: 1.2rem;
        padding: 1.1rem 1.4rem; border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(90deg,
            rgba(12, 16, 24, 0.92) 0%, rgba(18, 30, 22, 0.88) 70%, rgba(29, 52, 33, 0.78) 100%);
        margin-bottom: 1.3rem;
    }}
    .hero-copy {{ max-width: 72%; }}
    .hero-title {{ margin: 0; color: var(--hs-text); font-size: clamp(1.8rem, 2.6vw, 2.6rem); line-height: 1.1; font-weight: 700; }}
    .hero-subtitle {{ margin: 0.4rem 0 0 0; color: var(--hs-subtext); font-size: 1rem; }}
    .hero-logos {{ display: flex; align-items: center; justify-content: flex-end; gap: 1rem; }}
    .hero-logos img {{ height: 96px; width: auto; object-fit: contain; filter: drop-shadow(0 6px 14px rgba(0,0,0,0.35)); }}

    /* Soft info card */
    .soft-card {{
        background: rgba(27, 34, 43, 0.7);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid var(--hs-primary);
        border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.9rem;
    }}
    .soft-card p {{ margin: 0; color: var(--hs-subtext) !important; font-size: 0.95rem; }}

    @media (max-width: 1080px) {{
        .hero-banner {{ flex-direction: column; align-items: flex-start; }}
        .hero-copy {{ max-width: 100%; }}
        .hero-logos {{ justify-content: flex-start; }}
        .hero-logos img {{ height: 74px; }}
        .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 760px) {{
        .block-container {{ padding: 1rem 0.8rem 1.5rem 0.8rem; }}
        .stPlotlyChart {{
            border-radius: 10px;
            padding: 0.35rem 0.5rem 0.1rem 0.35rem;
            margin-bottom: 0.9rem;
        }}
        .hero-banner {{ padding: 0.95rem 1rem; border-radius: 12px; }}
        .hero-title {{ font-size: 1.55rem; }}
        .hero-subtitle {{ font-size: 0.92rem; }}
        .hero-logos img {{ height: 58px; }}
        .metric-grid {{ grid-template-columns: 1fr; gap: 0.65rem; }}
        .metric-tile {{ padding: 0.72rem 0.8rem; border-radius: 10px; }}
        .metric-value {{ font-size: 1.2rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================
# DATA LOADING
# ======================================================
@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    """Load the small precomputed per-(test, unit) summary table.

    This is the backbone of the whole app: every headline number on the
    Overview and Test pages comes from here, so we never scan 8.3M rows just
    to count something.
    """
    df = pd.read_csv(SUMMARY_FILE)
    df["first_sample"] = pd.to_datetime(df["first_sample"], errors="coerce")
    df["last_sample"] = pd.to_datetime(df["last_sample"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def get_primary_unit_summary() -> pd.DataFrame:
    """One row per test, keeping the unit with the most observations.

    A handful of tests (e.g. 'Flow, instantaneous') are reported in several
    units. For headline tables we collapse to each test's dominant unit so the
    user sees one clean row per determinand.
    """
    summary = load_summary()
    idx = summary.groupby("Test")["n_obs"].idxmax()
    primary = summary.loc[idx].copy()
    primary = primary.sort_values("n_obs", ascending=False).reset_index(drop=True)
    return primary


@st.cache_data(show_spinner=False)
def get_dataset_totals() -> dict:
    """Whole-dataset headline figures, derived from the summary only."""
    summary = load_summary()
    primary = get_primary_unit_summary()
    return {
        "total_obs": int(summary["n_obs"].sum()),
        "n_tests": int(summary["Test"].nunique()),
        "n_units": int(summary["Unit"].nunique()),
        "max_sites": int(summary["n_sites"].max()),
        "first": summary["first_sample"].min(),
        "last": summary["last_sample"].max(),
        "n_years": int(summary["n_years"].max()),
        "lat_min": float(summary["lat_min"].min()),
        "lat_max": float(summary["lat_max"].max()),
        "lon_min": float(summary["lon_min"].min()),
        "lon_max": float(summary["lon_max"].max()),
        "n_priority_available": int(primary[primary["Test"].isin(PRIORITY_TESTS)]["Test"].nunique()),
    }


@st.cache_data(show_spinner=False)
def load_correlations() -> pd.DataFrame:
    """Precomputed Spearman correlations between tests using site-year medians."""
    path = Path(CORRELATION_FILE)
    if not path.exists():
        return pd.DataFrame(
            columns=["Test", "Unit", "Other Test", "Other Unit", "Spearman r", "Paired site-years"]
        )
    df = pd.read_csv(path)
    df["Spearman r"] = pd.to_numeric(df["Spearman r"], errors="coerce")
    df["Paired site-years"] = pd.to_numeric(df["Paired site-years"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["Spearman r"])


@st.cache_data(show_spinner=False)
def load_year_counts() -> pd.DataFrame:
    """Precomputed per-test yearly reading counts."""
    path = Path(YEAR_COUNTS_FILE)
    if not path.exists():
        return pd.DataFrame(columns=["Test", "Unit", "SourceYear", "n_obs"])
    df = pd.read_csv(path)
    df["SourceYear"] = pd.to_numeric(df["SourceYear"], errors="coerce").astype("Int64")
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["SourceYear"])


@st.cache_data(show_spinner=False)
def list_all_tests() -> list:
    """All distinct test names, busiest first (for the dropdown)."""
    primary = get_primary_unit_summary()
    return primary["Test"].tolist()


@st.cache_data(show_spinner=False)
def parquet_columns() -> list:
    """Column names of the parquet without reading any rows."""
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(PARQUET_FILE).schema.names)
    except Exception:
        return [
            "Sampling Point", "Type", "Date", "Test", "result", "Unit",
            "Season", "SourceYear", "Category", "Latitude", "Longitude",
        ]


@st.cache_data(show_spinner=True, max_entries=12)
def load_test_data(
    test_name: str,
    unit: str | None = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Load only the rows for a single test from the parquet.

    Uses Arrow predicate pushdown (filters) plus column selection so we read a
    thin slice rather than the full 8.3M-row file. Cached so re-selecting a
    test is instant. At most 12 tests are kept in cache at once.
    """
    cols = ["Sampling Point", "Date", "Test", "result", "Unit",
            "Season", "SourceYear", "Category", "Latitude", "Longitude"]
    available = parquet_columns()
    cols = [c for c in cols if c in available]

    filters = [("Test", "==", test_name)]
    if unit is not None:
        filters.append(("Unit", "==", unit))
    if start_date is not None:
        filters.append(("Date", ">=", pd.Timestamp(start_date)))
    if end_date is not None:
        filters.append(("Date", "<", pd.Timestamp(end_date) + pd.Timedelta(days=1)))

    try:
        df = pd.read_parquet(PARQUET_FILE, columns=cols, filters=filters)
    except Exception:
        # Fallback: some pandas/pyarrow combos dislike filters → load cols, then filter.
        df = pd.read_parquet(PARQUET_FILE, columns=cols)
        df = df[df["Test"] == test_name]
        if unit is not None:
            df = df[df["Unit"] == unit]
        if start_date is not None or end_date is not None:
            df = filter_by_date_range(
                df,
                start_date if start_date is not None else df["Date"].min(),
                end_date if end_date is not None else df["Date"].max(),
            )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "result" in df.columns:
        df["result"] = pd.to_numeric(df["result"], errors="coerce").astype("float32")
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype("string")
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=12)
def get_test_sites(test_name: str, unit: str | None = None) -> pd.DataFrame:
    """Per-sampling-point aggregates for one test: location, count, median.

    Drives the maps and the 'compare locations' views. One row per sampling
    point keeps the map to a few thousand markers at most, not millions.
    """
    df = load_test_data(test_name, unit)
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.dropna(subset=["Latitude", "Longitude"])
        .groupby("Sampling Point")
        .agg(
            n_obs=("result", "count"),
            median=("result", "median"),
            mean=("result", "mean"),
            lat=("Latitude", "median"),
            lon=("Longitude", "median"),
            first=("Date", "min"),
            last=("Date", "max"),
        )
        .reset_index()
    )
    return grouped


@st.cache_data(show_spinner=False)
def encode_logo() -> str:
    p = Path(LOGO_FILE)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")


# ======================================================
# PLOTLY DARK LAYOUT
# ======================================================
def apply_dark_layout(fig, title=None, height=None, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=19, color=TEXT_COL, family="Hind, sans-serif")) if title else None,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COL, family="Hind, sans-serif"),
        margin=dict(l=58, r=24, t=64 if title else 26, b=52),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(
            bgcolor="#111821",
            bordercolor="rgba(167, 215, 48, 0.35)",
            font=dict(color=TEXT_COL, family="Hind, sans-serif", size=13),
        ),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.18)", automargin=True, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.18)", automargin=True, zeroline=False)
    return fig


def hero(title: str, subtitle: str):
    logo_b64 = encode_logo()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="HydroStar logo">'
        if logo_b64 else ""
    )
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-copy">
                <h1 class="hero-title">{title}</h1>
                <p class="hero-subtitle">{subtitle}</p>
            </div>
            <div class="hero-logos">{logo_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(text: str):
    st.markdown(f'<div class="soft-card"><p>{text}</p></div>', unsafe_allow_html=True)


def human_int(n) -> str:
    return f"{int(n):,}"


def human_result_value(value) -> str:
    """Compact result display that keeps trace concentrations visible."""
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}"
    if abs_value >= 100:
        return f"{value:,.1f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if abs_value >= 0.000001:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{value:.2e}"


def estimate_window_rows(summary_row: pd.Series, start_date, end_date) -> int:
    """Estimate rows for a selected date window without scanning the parquet."""
    first = pd.Timestamp(summary_row["first_sample"])
    last = pd.Timestamp(summary_row["last_sample"])
    if pd.isna(first) or pd.isna(last):
        return int(summary_row["n_obs"])
    start = max(pd.Timestamp(start_date), first)
    end = min(pd.Timestamp(end_date), last)
    if start > end:
        return 0
    total_days = max((last - first).days + 1, 1)
    selected_days = max((end - start).days + 1, 1)
    return int(math.ceil(float(summary_row["n_obs"]) * selected_days / total_days))


def smart_round(values: pd.Series) -> int:
    """Choose a sensible number of decimals for display based on magnitude."""
    v = values.dropna().abs()
    v = v[v > 0]
    if v.empty:
        return 2
    med = float(v.median())
    if med >= 100:
        return 1
    if med >= 1:
        return 2
    if med >= 0.01:
        return 4
    return 6


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def render_plotly(fig):
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def metric_grid(items: list[tuple[str, str]]):
    """Render responsive metric tiles that do not depend on fixed columns."""
    cards = []
    for label, value in items:
        cards.append(
            '<div class="metric-tile">'
            f'<span class="metric-label">{html.escape(str(label))}</span>'
            f'<span class="metric-value">{html.escape(str(value))}</span>'
            '</div>'
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def truncate_label(text: str, max_chars: int = 36) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Inclusive date-window filter for row-level views."""
    if df.empty or "Date" not in df.columns:
        return df.copy()
    start_ts = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    dates = pd.to_datetime(df["Date"], errors="coerce")
    return df.loc[(dates >= start_ts) & (dates < end_exclusive)].copy()


def detailed_metric_items(df: pd.DataFrame, unit: str, dec: int = 3) -> list[tuple[str, str]]:
    """Metric values for the currently selected row-level date window."""
    if df.empty:
        return [
            ("Readings", "0"),
            ("Sampling points", "0"),
            ("Years covered", "0"),
            ("Unit", unit),
            ("Median", "n/a"),
            ("P10–P90", "n/a"),
            ("Min / Max", "n/a"),
            ("Period", "No data"),
        ]

    results = pd.to_numeric(df["result"], errors="coerce").dropna() if "result" in df.columns else pd.Series(dtype="float64")
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna() if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    years = dates.dt.year.nunique() if not dates.empty else df["SourceYear"].nunique() if "SourceYear" in df.columns else 0
    period = f"{dates.min().year}–{dates.max().year}" if not dates.empty else "No dates"

    if results.empty:
        median = p_range = min_max = "n/a"
    else:
        median = f"{results.median():,.{dec}f}"
        p_range = f"{results.quantile(0.10):,.{dec}f} – {results.quantile(0.90):,.{dec}f}"
        min_max = f"{results.min():,.{dec}f} / {results.max():,.{dec}f}"

    return [
        ("Readings", human_int(len(df))),
        ("Sampling points", human_int(df["Sampling Point"].nunique()) if "Sampling Point" in df.columns else "0"),
        ("Years covered", human_int(years)),
        ("Unit", unit),
        ("Median", median),
        ("P10–P90", p_range),
        ("Min / Max", min_max),
        ("Period", period),
    ]


def iqr_filtered_frame(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    value_col: str = "result",
    multiplier: float = 3.0,
    min_count: int = 4,
) -> tuple[pd.DataFrame, int]:
    """Hide extreme values per series using Q1 - multiplier*IQR to Q3 + multiplier*IQR."""
    if df.empty or value_col not in df.columns:
        return df.copy(), 0

    d = df.copy()
    values = pd.to_numeric(d[value_col], errors="coerce")
    keep = pd.Series(True, index=d.index)
    groups = [(None, d)] if not group_cols else d.groupby(group_cols, dropna=False, sort=False)

    for _, group in groups:
        vals = values.loc[group.index]
        valid = vals.dropna()
        if len(valid) < min_count:
            continue
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        keep.loc[group.index] = vals.isna() | vals.between(lower, upper)

    hidden = int((~keep & values.notna()).sum())
    return d.loc[keep].copy(), hidden


def site_aggregates_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-site aggregates from an already-selected visualization frame."""
    if df.empty:
        return pd.DataFrame()
    required = {"Latitude", "Longitude", "Sampling Point", "result"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    grouped = (
        df.dropna(subset=["Latitude", "Longitude"])
        .groupby("Sampling Point")
        .agg(
            n_obs=("result", "count"),
            median=("result", "median"),
            mean=("result", "mean"),
            lat=("Latitude", "median"),
            lon=("Longitude", "median"),
            first=("Date", "min"),
            last=("Date", "max"),
        )
        .reset_index()
    )
    return grouped


def box_stats(values: pd.Series) -> dict | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    q1 = vals.quantile(0.25)
    median = vals.quantile(0.5)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    if np.isfinite(iqr) and iqr > 0:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        in_fence = vals[(vals >= lower_bound) & (vals <= upper_bound)]
    else:
        in_fence = vals
    if in_fence.empty:
        in_fence = vals
    return {
        "count": int(vals.count()),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "lowerfence": float(in_fence.min()),
        "upperfence": float(in_fence.max()),
    }


def hidden_counts_by_group(
    before: pd.DataFrame,
    after: pd.DataFrame,
    group_cols: list[str],
) -> pd.Series:
    if before.empty or not group_cols:
        return pd.Series(dtype="int64")
    before_counts = before.groupby(group_cols, dropna=False).size()
    after_counts = after.groupby(group_cols, dropna=False).size() if not after.empty else pd.Series(dtype="int64")
    hidden = before_counts.sub(after_counts, fill_value=0).astype(int)
    return hidden


def site_count_input(label: str, max_sites: int, default: int, key: str) -> int:
    """Numeric control for large site counts without an unwieldy slider."""
    if max_sites <= 0:
        return 0
    return int(st.number_input(
        label,
        min_value=1,
        max_value=int(max_sites),
        value=int(min(default, max_sites)),
        step=1,
        key=key,
    ))


# ======================================================
# MAP BUILDER  (the centrepiece)
# ======================================================
# Plotly renamed the tile-map trace from Scattermapbox (mapbox layout) to
# Scattermap (map layout, MapLibre) in v5.24 and deprecated the old name in v6.
# Detect what's available so the app is warning-free on new Plotly and still
# works if an older version is pinned. Both use free tiles (no token needed).
_HAS_NEW_MAP = hasattr(go, "Scattermap")
_MAP_TRACE = go.Scattermap if _HAS_NEW_MAP else go.Scattermapbox
_MAP_LAYOUT_KEY = "map" if _HAS_NEW_MAP else "mapbox"


def _set_map_layout(fig, *, style, center_lat, center_lon, zoom, **layout_kwargs):
    """Set the tile-map layout under the correct key for this Plotly version."""
    fig.update_layout(**{
        _MAP_LAYOUT_KEY: dict(style=style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        **layout_kwargs,
    })


def build_sites_map(
    sites: pd.DataFrame,
    title: str,
    colour_metric: str | None = None,
    colour_label: str = "",
    unit: str = "",
    height: int = 560,
):
    """A clean, readable map of sampling points.

    Anti-clutter design choices:
      - One marker per sampling point (not per reading), so density never
        explodes into millions of overlapping dots.
      - Marker SIZE encodes how much sampling happened there (sqrt-scaled so a
        site with 100x the data isn't 100x the size).
      - Marker COLOUR optionally encodes a value (e.g. median concentration),
        on a perceptual green scale, with the extreme 2% clipped so a single
        outlier site doesn't wash out the whole colour range.
      - carto-darkmatter base map sits naturally on the dark theme and needs
        no API token.
    """
    if sites.empty:
        st.info("No geographic data available for this selection.")
        return

    pts = sites.copy()
    pts = pts.dropna(subset=["lat", "lon"])
    if pts.empty:
        st.info("No valid coordinates available for this selection.")
        return

    # Size: sqrt scaling, clamped to a comfortable visual range.
    obs = pts["n_obs"].clip(lower=1)
    s_min, s_max = 6.0, 26.0
    sqrt_obs = np.sqrt(obs)
    if sqrt_obs.max() > sqrt_obs.min():
        sizes = s_min + (sqrt_obs - sqrt_obs.min()) / (sqrt_obs.max() - sqrt_obs.min()) * (s_max - s_min)
    else:
        sizes = np.full(len(pts), 12.0)
    pts["_size"] = sizes

    dec = smart_round(pts["median"]) if "median" in pts else 2
    customdata = np.column_stack([
        pts["Sampling Point"].astype(str).values,
        pts["n_obs"].values,
        pts["median"].round(dec).values if "median" in pts else np.zeros(len(pts)),
    ])

    fig = go.Figure()

    if colour_metric and colour_metric in pts.columns and pts[colour_metric].notna().any():
        # Clip extremes so one outlier site doesn't dominate the colour scale.
        cvals = pts[colour_metric].astype(float)
        lo, hi = cvals.quantile(0.02), cvals.quantile(0.98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(cvals.min()), float(cvals.max())
        green_scale = [
            [0.0, "#0d3b2e"], [0.25, "#1f7a4d"], [0.5, "#4fae3f"],
            [0.75, "#a7d730"], [1.0, "#f2f77a"],
        ]
        fig.add_trace(
            _MAP_TRACE(
                lat=pts["lat"], lon=pts["lon"], mode="markers",
                marker=dict(
                    size=pts["_size"], color=cvals, colorscale=green_scale,
                    cmin=lo, cmax=hi, opacity=0.82,
                    colorbar=dict(
                        title=dict(text=colour_label or "Value", side="right"),
                        thickness=14, len=0.7, x=0.99, bgcolor="rgba(0,0,0,0)",
                        tickfont=dict(color=TEXT_COL),
                    ),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Readings: %{customdata[1]:,}<br>"
                    f"Median: %{{customdata[2]:,}} {unit}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            _MAP_TRACE(
                lat=pts["lat"], lon=pts["lon"], mode="markers",
                marker=dict(size=pts["_size"], color=PRIMARY_COLOUR, opacity=0.7),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Readings: %{customdata[1]:,}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Centre on the cloud of points, with a sensible default zoom for GB.
    center_lat = float(pts["lat"].median())
    center_lon = float(pts["lon"].median())
    _set_map_layout(
        fig,
        style="carto-darkmatter",
        center_lat=center_lat, center_lon=center_lon, zoom=5.0,
        margin=dict(l=0, r=0, t=46 if title else 0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=18, color=TEXT_COL, family="Hind, sans-serif")) if title else None,
        font=dict(color=TEXT_COL),
    )
    render_plotly(fig)


# ======================================================
# SHARED CHART BUILDERS
# ======================================================
def yearly_count_chart(df: pd.DataFrame, title: str, colour=PRIMARY_COLOUR):
    """Bar chart of number of readings per year."""
    if df.empty:
        st.info("No data to plot.")
        return
    counts = df.groupby("SourceYear").size().reset_index(name="count")
    counts = counts.sort_values("SourceYear")
    fig = go.Figure(
        go.Bar(
            x=counts["SourceYear"], y=counts["count"],
            marker=dict(color=colour, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
            hovertemplate="<b>%{x}</b><br>%{y:,} readings<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Readings")
    fig.update_xaxes(type="category")
    apply_dark_layout(fig, title, height=430, legend=False)
    render_plotly(fig)


def monthly_trend_chart(
    df: pd.DataFrame,
    title: str,
    unit: str,
    colour=WATER_BLUE,
    remove_outliers: bool = False,
):
    """Median value per calendar month over the full record, with an IQR band.

    Monthly aggregation is the key anti-clutter move: instead of plotting
    hundreds of thousands of individual readings, we show the monthly median
    (the typical value) and a shaded 25-75 percentile band (the usual spread).
    """
    if df.empty or df["result"].notna().sum() == 0:
        st.info("No numeric results to plot a trend.")
        return
    d = df.dropna(subset=["result", "Date"]).copy()
    hidden_total = 0
    if remove_outliers:
        d, hidden_total = iqr_filtered_frame(d)
        if d.empty:
            st.info("No numeric results remain after outlier filtering.")
            return
    d["month"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    agg = d.groupby("month")["result"].agg(
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        n="count",
    ).reset_index()
    if agg.empty:
        st.info("No data to plot a trend.")
        return

    dec = smart_round(d["result"])
    fig = go.Figure()
    # IQR band
    fig.add_trace(go.Scatter(
        x=agg["month"], y=agg["q3"], mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=agg["month"], y=agg["q1"], mode="lines", fill="tonexty",
        line=dict(width=0), fillcolor="rgba(78,168,222,0.18)",
        name="25-75 percentile", hoverinfo="skip",
    ))
    # Median line
    fig.add_trace(go.Scatter(
        x=agg["month"], y=agg["median"], mode="lines",
        line=dict(color=colour, width=2.2), name="Monthly median",
        customdata=agg[["n", "q1", "q3"]].round(dec).values,
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            f"Median: %{{y:,.{dec}f}} {unit}<br>"
            f"Spread: %{{customdata[1]:,.{dec}f}} – %{{customdata[2]:,.{dec}f}} {unit}<br>"
            "Readings: %{customdata[0]:,}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title=f"{unit}" if unit else "Value")
    if hidden_total:
        fig.add_annotation(
            text=f"{hidden_total:,} outlier readings hidden",
            xref="paper", yref="paper", x=1, y=1.08,
            showarrow=False, font=dict(size=12, color=SUBTEXT_COL), align="right",
        )
    apply_dark_layout(fig, title, height=480)
    render_plotly(fig)


def seasonal_box_chart(df: pd.DataFrame, title: str, unit: str, remove_outliers: bool = False):
    """Distribution of values by season (box per season)."""
    if df.empty or df["result"].notna().sum() == 0:
        st.info("No numeric results to plot.")
        return
    d = df.dropna(subset=["result"]).copy()
    if "Season" not in d.columns or d["Season"].isna().all():
        d["Season"] = d["Date"].dt.month.map(SEASON_BY_MONTH)
    d = d[d["Season"].isin(SEASON_ORDER)]
    if d.empty:
        st.info("No seasonal data to plot.")
        return
    source = d.copy()
    hidden_by_season = pd.Series(dtype="int64")
    if remove_outliers:
        d, _ = iqr_filtered_frame(source, ["Season"])
        hidden_by_season = hidden_counts_by_group(source, d, ["Season"])
        if d.empty:
            st.info("No seasonal data remains after outlier filtering.")
            return

    dec = smart_round(d["result"])
    fig = go.Figure()
    for season in SEASON_ORDER:
        vals = d.loc[d["Season"] == season, "result"]
        stats = box_stats(vals)
        if stats is None:
            continue
        hidden = int(hidden_by_season.get(season, 0))
        fig.add_trace(go.Box(
            x=[season],
            q1=[stats["q1"]],
            median=[stats["median"]],
            q3=[stats["q3"]],
            lowerfence=[stats["lowerfence"]],
            upperfence=[stats["upperfence"]],
            mean=[stats["mean"]],
            name=season,
            marker_color=SEASON_COLOURS[season],
            line=dict(color=SEASON_COLOURS[season]),
            fillcolor=SEASON_COLOURS[season],
            opacity=0.78,
            boxpoints=False, boxmean=True,
            customdata=[[stats["count"], stats["min"], stats["max"], stats["mean"], hidden]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Readings shown: %{customdata[0]:,}<br>"
                f"Median: %{{median:,.{dec}f}} {unit}<br>"
                f"Q1: %{{q1:,.{dec}f}} · Q3: %{{q3:,.{dec}f}} {unit}<br>"
                f"Min: %{{customdata[1]:,.{dec}f}} · Max: %{{customdata[2]:,.{dec}f}} {unit}<br>"
                "Outliers hidden: %{customdata[4]:,}<extra></extra>"
            ),
        ))
    fig.update_layout(xaxis_title="Season", yaxis_title=f"{unit}" if unit else "Value")
    fig.update_xaxes(categoryorder="array", categoryarray=SEASON_ORDER)
    apply_dark_layout(fig, title, height=500, legend=False)
    render_plotly(fig)


def yearly_box_chart(
    df: pd.DataFrame,
    title: str,
    unit: str,
    colour=PRIMARY_COLOUR,
    remove_outliers: bool = False,
):
    """Distribution of values per year (precomputed box stats, outliers hidden)."""
    if df.empty or df["result"].notna().sum() == 0:
        st.info("No numeric results to plot.")
        return
    d = df.dropna(subset=["result", "SourceYear"]).copy()
    source = d.copy()
    hidden_by_year = pd.Series(dtype="int64")
    if remove_outliers:
        d, _ = iqr_filtered_frame(source, ["SourceYear"])
        hidden_by_year = hidden_counts_by_group(source, d, ["SourceYear"])
        if d.empty:
            st.info("No yearly data remains after outlier filtering.")
            return

    rows = []
    for yr in sorted(d["SourceYear"].dropna().unique()):
        stats = box_stats(d.loc[d["SourceYear"] == yr, "result"])
        if stats is None:
            continue
        stats["SourceYear"] = str(int(yr)) if float(yr).is_integer() else str(yr)
        stats["hidden"] = int(hidden_by_year.get(yr, 0))
        rows.append(stats)
    if not rows:
        st.info("No data to plot.")
        return
    summary = pd.DataFrame(rows)
    dec = smart_round(d["result"])

    fig = go.Figure(go.Box(
        x=summary["SourceYear"], q1=summary["q1"], median=summary["median"], q3=summary["q3"],
        lowerfence=summary["lowerfence"], upperfence=summary["upperfence"], mean=summary["mean"],
        customdata=summary[["count", "min", "max", "mean", "hidden"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>Readings shown: %{customdata[0]:,}<br>"
            f"Median: %{{median:,.{dec}f}} {unit}<br>"
            f"Q1: %{{q1:,.{dec}f}} · Q3: %{{q3:,.{dec}f}} {unit}<br>"
            f"Min: %{{customdata[1]:,.{dec}f}} · Max: %{{customdata[2]:,.{dec}f}} {unit}<br>"
            "Outliers hidden: %{customdata[4]:,}<extra></extra>"
        ),
        marker_color=colour, line=dict(color=colour),
        fillcolor="rgba(167,215,48,0.20)", boxmean=True, boxpoints=False,
    ))
    fig.update_layout(xaxis_title="Year", yaxis_title=f"{unit}" if unit else "Value")
    fig.update_xaxes(type="category")
    apply_dark_layout(fig, title, height=520, legend=False)
    render_plotly(fig)


def compare_sites_chart(
    df: pd.DataFrame,
    sites: pd.DataFrame,
    title: str,
    unit: str,
    top_n: int = 12,
    remove_outliers: bool = False,
):
    """Compare the busiest N sampling points with median dots and IQR bands.

    With 11k+ sites we never show them all. We rank by number of readings and
    show the top N, which keeps the chart legible and focuses on the sites with
    enough data to be worth comparing.
    """
    if df.empty or sites.empty:
        st.info("No data to compare.")
        return
    top_sites = sites.sort_values("n_obs", ascending=False).head(top_n)["Sampling Point"].tolist()
    source = df[df["Sampling Point"].isin(top_sites)].dropna(subset=["result"]).copy()
    if source.empty:
        st.info("No data to compare.")
        return
    d = source
    hidden_by_site = pd.Series(dtype="int64")
    if remove_outliers:
        d, _ = iqr_filtered_frame(source, ["Sampling Point"])
        hidden_by_site = hidden_counts_by_group(source, d, ["Sampling Point"])
        if d.empty:
            st.info("No site data remains after outlier filtering.")
            return

    g = d.groupby("Sampling Point")["result"]
    stats_df = g.agg(count="count", mean="mean", min="min", max="max")
    q = g.quantile([0.25, 0.5, 0.75]).unstack().rename(
        columns={0.25: "q1", 0.5: "median", 0.75: "q3"}
    )
    stats_df = stats_df.join(q).dropna(subset=["q1", "median", "q3"])
    stats_df = stats_df.reindex([site for site in top_sites if site in stats_df.index]).dropna(subset=["median"])
    if stats_df.empty:
        st.info("No data to compare.")
        return
    if not hidden_by_site.empty:
        stats_df["hidden"] = hidden_by_site.reindex(stats_df.index).fillna(0).astype(int)
    else:
        stats_df["hidden"] = 0
    stats_df = stats_df.reset_index().rename(columns={"Sampling Point": "site"})
    stats_df = stats_df.sort_values("median").reset_index(drop=True)
    stats_df["axis_label"] = [
        f"{i + 1}. {truncate_label(site, 34)}"
        for i, site in enumerate(stats_df["site"])
    ]
    stats_df["iqr_width"] = stats_df["q3"] - stats_df["q1"]
    dec = smart_round(d["result"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats_df["iqr_width"],
        y=stats_df["axis_label"],
        base=stats_df["q1"],
        orientation="h",
        marker=dict(
            color="rgba(167, 215, 48, 0.30)",
            line=dict(color="rgba(167, 215, 48, 0.74)", width=1),
        ),
        width=0.58,
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=stats_df["median"],
        y=stats_df["axis_label"],
        mode="markers",
        marker=dict(
            color=ACCENT_COLOUR,
            size=11,
            line=dict(color="rgba(255,255,255,0.88)", width=1),
        ),
        customdata=stats_df[["site", "count", "q1", "q3", "min", "max", "hidden", "mean"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Readings shown: %{customdata[1]:,}<br>"
            f"Median: %{{x:,.{dec}f}} {unit}<br>"
            f"IQR: %{{customdata[2]:,.{dec}f}} – %{{customdata[3]:,.{dec}f}} {unit}<br>"
            f"Min: %{{customdata[4]:,.{dec}f}} · Max: %{{customdata[5]:,.{dec}f}} {unit}<br>"
            "Outliers hidden: %{customdata[6]:,}<extra></extra>"
        ),
        showlegend=False,
        cliponaxis=False,
    ))
    fig.update_layout(xaxis_title=f"{unit}" if unit else "Value", yaxis_title="")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=stats_df["axis_label"].tolist(),
        tickfont=dict(size=12),
    )
    apply_dark_layout(fig, title, height=max(420, 31 * len(stats_df) + 140), legend=False)
    render_plotly(fig)


def site_season_heatmap(
    df: pd.DataFrame,
    sites: pd.DataFrame,
    title: str,
    unit: str,
    top_n: int = 15,
    remove_outliers: bool = False,
):
    """Heatmap: busiest sites (rows) x season (cols), coloured by median value.

    Answers 'how do the main sites differ, and does season matter?' in one
    glance without a wall of lines.
    """
    if df.empty or sites.empty:
        st.info("No data for a heatmap.")
        return
    top_sites = sites.sort_values("n_obs", ascending=False).head(top_n)["Sampling Point"].tolist()
    d = df[df["Sampling Point"].isin(top_sites)].dropna(subset=["result"]).copy()
    if "Season" not in d.columns or d["Season"].isna().all():
        d["Season"] = d["Date"].dt.month.map(SEASON_BY_MONTH)
    d = d[d["Season"].isin(SEASON_ORDER)]
    if d.empty:
        st.info("No seasonal data for these sites.")
        return
    source = d.copy()
    hidden_by_cell = pd.Series(dtype="int64")
    if remove_outliers:
        d, _ = iqr_filtered_frame(source, ["Sampling Point", "Season"])
        hidden_by_cell = hidden_counts_by_group(source, d, ["Sampling Point", "Season"])
        if d.empty:
            st.info("No seasonal site data remains after outlier filtering.")
            return
    pivot = d.pivot_table(index="Sampling Point", columns="Season", values="result", aggfunc="median")
    pivot = pivot.reindex(columns=[s for s in SEASON_ORDER if s in pivot.columns])
    counts = d.pivot_table(index="Sampling Point", columns="Season", values="result", aggfunc="count").reindex_like(pivot)
    if hidden_by_cell.empty:
        hidden = pd.DataFrame(0, index=pivot.index, columns=pivot.columns)
    else:
        hidden = (
            hidden_by_cell.rename("hidden")
            .reset_index()
            .pivot(index="Sampling Point", columns="Season", values="hidden")
            .reindex_like(pivot)
            .fillna(0)
        )
    # Order rows by overall median (descending) so the strongest sites are on top.
    row_order = d.groupby("Sampling Point")["result"].median().sort_values(ascending=False).index
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    counts = counts.reindex_like(pivot).fillna(0)
    hidden = hidden.reindex_like(pivot).fillna(0)
    dec = smart_round(d["result"])

    green_scale = [[0.0, "#0d3b2e"], [0.5, "#4fae3f"], [1.0, "#f2f77a"]]
    text = pivot.round(dec).astype(str).replace("<NA>", "")
    customdata = np.dstack([counts.values, hidden.values])
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=green_scale,
        colorbar=dict(title=f"Median ({unit})" if unit else "Median"),
        text=text.values,
        texttemplate="%{text}",
        customdata=customdata,
        hovertemplate=(
            "<b>%{y}</b><br>%{x}<br>"
            f"Median: %{{z:,.{dec}f}} {unit}<br>"
            "Readings shown: %{customdata[0]:,}<br>"
            "Outliers hidden: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="", yaxis_title="")
    apply_dark_layout(fig, title, height=max(360, 30 * len(pivot) + 140), legend=False)
    render_plotly(fig)


def zero_share_chart(summary_row: pd.Series, title: str):
    """Simple zero vs non-zero split from the summary table."""
    zero_pct = float(summary_row.get("frac_zero", 0) or 0) * 100
    non_zero_pct = max(0.0, 100.0 - zero_pct)
    fig = go.Figure(go.Bar(
        x=[zero_pct, non_zero_pct],
        y=["Readings"],
        orientation="h",
        marker=dict(color=[WARN_AMBER, PRIMARY_COLOUR]),
        text=[f"{zero_pct:.1f}%", f"{non_zero_pct:.1f}%"],
        textposition="inside",
        hovertemplate="%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Share of readings",
        yaxis_title="",
        barmode="stack",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
    )
    fig.update_traces(name="")
    apply_dark_layout(fig, title, height=260, legend=False)
    render_plotly(fig)


def threshold_exceedance_chart(df: pd.DataFrame, test_name: str, unit: str, key: str):
    """Yearly percentage of readings above a user-selected threshold."""
    if df.empty or "result" not in df.columns or df["result"].notna().sum() == 0:
        st.info("No numeric results for a threshold chart.")
        return
    values = pd.to_numeric(df["result"], errors="coerce").dropna()
    default_threshold = float(values.quantile(0.90)) if not values.empty else 0.0
    threshold = st.number_input(
        "Threshold",
        value=default_threshold,
        format="%.6g",
        help="Shows the percentage of readings above this value each year.",
        key=key,
    )
    d = df.dropna(subset=["result", "SourceYear"]).copy()
    if d.empty:
        st.info("No yearly data for a threshold chart.")
        return
    d["above_threshold"] = pd.to_numeric(d["result"], errors="coerce") > threshold
    yearly = (
        d.groupby("SourceYear")
        .agg(readings=("result", "count"), above=("above_threshold", "sum"))
        .reset_index()
        .sort_values("SourceYear")
    )
    yearly["pct_above"] = np.where(yearly["readings"] > 0, yearly["above"] / yearly["readings"] * 100, 0)

    fig = go.Figure(go.Bar(
        x=yearly["SourceYear"].astype(str),
        y=yearly["pct_above"],
        marker=dict(color=ACCENT_COLOUR),
        customdata=yearly[["above", "readings"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "% above threshold: %{y:.1f}%<br>"
            "Above: %{customdata[0]:,}<br>"
            "Readings: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"% above {human_result_value(threshold)} {unit}",
    )
    fig.update_yaxes(range=[0, max(5, min(100, yearly["pct_above"].max() * 1.15))], ticksuffix="%")
    apply_dark_layout(fig, f"{test_name} — readings above threshold", height=430, legend=False)
    render_plotly(fig)


def correlation_matrix_for_tests(test_names: list[str], min_pairs: int) -> pd.DataFrame:
    """Build a square correlation matrix from the precomputed long table."""
    corr = load_correlations()
    tests = [t for t in dict.fromkeys(test_names) if t]
    if corr.empty or not tests:
        return pd.DataFrame()
    rel = corr[
        corr["Test"].isin(tests)
        & corr["Other Test"].isin(tests)
        & (corr["Paired site-years"] >= min_pairs)
    ]
    matrix = pd.DataFrame(np.nan, index=tests, columns=tests, dtype="float64")
    for test in tests:
        matrix.loc[test, test] = 1.0
    for _, row in rel.iterrows():
        matrix.loc[row["Test"], row["Other Test"]] = row["Spearman r"]
    return matrix


def render_correlation_matrix(test_names: list[str], title: str, min_pairs: int, height: int | None = None):
    matrix = correlation_matrix_for_tests(test_names, min_pairs)
    if matrix.empty:
        st.info("No correlation matrix is available for this selection.")
        return
    labels = [truncate_label(t, 30) for t in matrix.index]
    z = matrix.values
    text = np.where(np.isfinite(z), np.round(z, 2).astype(str), "")
    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        colorscale=[
            [0.0, "#3b1f5f"],
            [0.5, "#202833"],
            [1.0, PRIMARY_COLOUR],
        ],
        colorbar=dict(title="Spearman r"),
        text=text,
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}<br>Spearman r: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="", yaxis_title="")
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    apply_dark_layout(fig, title, height=height or max(480, 14 * len(labels) + 180), legend=False)
    render_plotly(fig)


def render_test_correlation_section(test_name: str, scope_label: str):
    """Top positive/negative correlations and a focused mini matrix."""
    corr = load_correlations()
    if corr.empty:
        st.info("Correlation file is not available.")
        return
    min_pairs = int(st.number_input(
        "Minimum paired site-years",
        min_value=20,
        max_value=5000,
        value=100,
        step=20,
        help="Higher values keep only correlations based on more shared site-year observations.",
        key=f"corr_min_pairs_{scope_label}_{test_name}",
    ))
    rel = corr[(corr["Test"] == test_name) & (corr["Paired site-years"] >= min_pairs)].copy()
    if rel.empty:
        st.info("No correlations meet the current paired-sample threshold.")
        return
    pos = rel[rel["Spearman r"] > 0].sort_values("Spearman r", ascending=False).head(8)
    neg = rel[rel["Spearman r"] < 0].sort_values("Spearman r", ascending=True).head(8)
    view = pd.concat([neg, pos], ignore_index=True)
    if view.empty:
        st.info("No positive or negative correlations meet the current threshold.")
        return
    view["label"] = view["Other Test"].map(lambda s: truncate_label(s, 42))
    colours = np.where(view["Spearman r"] >= 0, PRIMARY_COLOUR, "#e6679a")
    fig = go.Figure(go.Bar(
        x=view["Spearman r"],
        y=view["label"],
        orientation="h",
        marker=dict(color=colours),
        customdata=view[["Other Test", "Other Unit", "Paired site-years"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Spearman r: %{x:.2f}<br>"
            "Unit: %{customdata[1]}<br>"
            "Paired site-years: %{customdata[2]:,}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.35)", width=1))
    fig.update_layout(xaxis_title="Spearman correlation", yaxis_title="")
    fig.update_xaxes(range=[-1, 1])
    apply_dark_layout(fig, f"{test_name} — strongest positive and negative relationships", height=520, legend=False)
    render_plotly(fig)

    matrix_tests = [test_name] + pos["Other Test"].head(5).tolist() + neg["Other Test"].head(5).tolist()
    render_correlation_matrix(
        matrix_tests,
        f"{test_name} — focused correlation matrix",
        min_pairs=min_pairs,
        height=max(520, 28 * len(matrix_tests) + 180),
    )


def overview_pareto_chart(primary: pd.DataFrame):
    """Bar + cumulative line showing concentration of observations across tests."""
    top = primary.sort_values("n_obs", ascending=False).head(30).copy()
    top["cum_pct"] = top["n_obs"].cumsum() / primary["n_obs"].sum() * 100
    labels = top["Test"].map(lambda s: truncate_label(s, 28))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=top["n_obs"],
        marker=dict(color=PRIMARY_COLOUR),
        customdata=top[["Test", "n_obs"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} readings<extra></extra>",
        name="Readings",
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=top["cum_pct"],
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=ACCENT_COLOUR, width=2),
        marker=dict(size=7),
        hovertemplate="Cumulative share: %{y:.1f}%<extra></extra>",
        name="Cumulative share",
    ))
    fig.update_layout(
        xaxis_title="Top tests",
        yaxis_title="Readings",
        yaxis2=dict(title="Cumulative share", overlaying="y", side="right", range=[0, 100], ticksuffix="%"),
    )
    fig.update_xaxes(tickangle=45)
    apply_dark_layout(fig, "How concentrated are readings across tests?", height=560)
    render_plotly(fig)


def overview_richness_scatter(primary: pd.DataFrame):
    """Sites vs readings scatter; highlights tests with enough coverage to support analysis."""
    d = primary.copy()
    d["is_priority"] = d["Test"].isin(PRIORITY_TESTS)
    fig = go.Figure()
    for label, mask, colour in [
        ("Priority", d["is_priority"], PRIMARY_COLOUR),
        ("Other", ~d["is_priority"], WATER_BLUE),
    ]:
        part = d[mask]
        if part.empty:
            continue
        sizes = 8 + (part["n_years"] / max(d["n_years"].max(), 1)) * 18
        fig.add_trace(go.Scatter(
            x=part["n_sites"],
            y=part["n_obs"],
            mode="markers",
            name=label,
            marker=dict(size=sizes, color=colour, opacity=0.72, line=dict(color="rgba(255,255,255,0.25)", width=0.5)),
            customdata=part[["Test", "Unit", "n_years", "median"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Unit: %{customdata[1]}<br>"
                "Sites: %{x:,}<br>"
                "Readings: %{y:,}<br>"
                "Years: %{customdata[2]:,}<br>"
                "Median: %{customdata[3]:.6g}<extra></extra>"
            ),
        ))
    fig.update_layout(xaxis_title="Sampling points", yaxis_title="Readings")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    apply_dark_layout(fig, "Which tests are data-rich enough for detailed analysis?", height=560)
    render_plotly(fig)


def overview_timeline_chart(primary: pd.DataFrame):
    """Date-span strip for the busiest tests."""
    d = primary.sort_values("n_obs", ascending=False).head(35).copy()
    d = d.sort_values("first_sample")
    fig = go.Figure()
    for _, row in d.iterrows():
        label = truncate_label(row["Test"], 36)
        fig.add_trace(go.Scatter(
            x=[row["first_sample"], row["last_sample"]],
            y=[label, label],
            mode="lines+markers",
            line=dict(color=PRIMARY_COLOUR, width=3),
            marker=dict(size=6, color=ACCENT_COLOUR),
            customdata=[[row["Test"], row["n_obs"], row["n_sites"]], [row["Test"], row["n_obs"], row["n_sites"]]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Readings: %{customdata[1]:,}<br>"
                "Sites: %{customdata[2]:,}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(xaxis_title="Sampling period", yaxis_title="")
    apply_dark_layout(fig, "Which tests have long monitoring histories?", height=max(520, 23 * len(d) + 120), legend=False)
    render_plotly(fig)


def overview_data_quality_bars(primary: pd.DataFrame):
    """Zero-rate and high-value-rate bars from the summary file."""
    c1, c2 = st.columns(2)
    with c1:
        top_zero = primary.sort_values("frac_zero", ascending=False).head(15).sort_values("frac_zero")
        fig = go.Figure(go.Bar(
            x=top_zero["frac_zero"] * 100,
            y=top_zero["Test"].map(lambda s: truncate_label(s, 34)),
            orientation="h",
            marker=dict(color=WARN_AMBER),
            customdata=top_zero[["Test", "n_obs"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% zero<br>%{customdata[1]:,} readings<extra></extra>",
        ))
        fig.update_layout(xaxis_title="% zero readings", yaxis_title="")
        fig.update_xaxes(ticksuffix="%")
        apply_dark_layout(fig, "Tests with the most zero readings", height=520, legend=False)
        render_plotly(fig)
    with c2:
        top_high = primary.sort_values("frac_high", ascending=False).head(15).sort_values("frac_high")
        fig = go.Figure(go.Bar(
            x=top_high["frac_high"] * 100,
            y=top_high["Test"].map(lambda s: truncate_label(s, 34)),
            orientation="h",
            marker=dict(color="#e6679a"),
            customdata=top_high[["Test", "n_obs"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% high-value flagged<br>%{customdata[1]:,} readings<extra></extra>",
        ))
        fig.update_layout(xaxis_title="% high-value flagged", yaxis_title="")
        fig.update_xaxes(ticksuffix="%")
        apply_dark_layout(fig, "Tests with the most high-value flags", height=520, legend=False)
        render_plotly(fig)


def priority_group_for_test(test_name: str) -> str:
    for group, tests in PRIORITY_GROUPS.items():
        if test_name in tests:
            return group
    return "Other"


def priority_coverage_heatmap(view: pd.DataFrame):
    year_counts = load_year_counts()
    if year_counts.empty or view.empty:
        st.info("Yearly coverage file is not available.")
        return
    tests = view["Test"].tolist()
    yc = year_counts[year_counts["Test"].isin(tests)].copy()
    if yc.empty:
        st.info("No yearly coverage data for this selection.")
        return
    pivot = yc.pivot_table(index="Test", columns="SourceYear", values="n_obs", aggfunc="sum", fill_value=0)
    order = view.set_index("Test")["n_sites"].sort_values(ascending=False).index.tolist()
    pivot = pivot.reindex([t for t in order if t in pivot.index])
    z = np.log10(pivot.values.astype(float) + 1)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(int(y)) for y in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#202833"], [0.5, "#4fae3f"], [1, "#f2f77a"]],
        colorbar=dict(title="Reading intensity"),
        customdata=pivot.values,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Readings: %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Year", yaxis_title="")
    fig.update_yaxes(tickfont=dict(size=10))
    apply_dark_layout(fig, "Priority monitoring coverage by year", height=max(520, 15 * len(pivot) + 140), legend=False)
    render_plotly(fig)


def priority_detectability_chart(view: pd.DataFrame):
    if view.empty:
        st.info("No priority tests to chart.")
        return
    d = view.sort_values("frac_zero", ascending=False).head(30).sort_values("frac_zero")
    fig = go.Figure(go.Bar(
        x=d["frac_zero"] * 100,
        y=d["Test"].map(lambda s: truncate_label(s, 36)),
        orientation="h",
        marker=dict(color=WARN_AMBER),
        customdata=d[["Test", "n_obs"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% zero readings<br>%{customdata[1]:,} readings<extra></extra>",
    ))
    fig.update_layout(xaxis_title="% zero readings", yaxis_title="")
    fig.update_xaxes(ticksuffix="%")
    apply_dark_layout(fig, "Priority tests with the highest zero-reading share", height=max(430, 23 * len(d) + 120), legend=False)
    render_plotly(fig)


def priority_coverage_value_scatter(view: pd.DataFrame):
    if view.empty:
        st.info("No priority tests to chart.")
        return
    units = view["Unit"].dropna().value_counts().index.tolist()
    if not units:
        return
    default_unit = "mg/l" if "mg/l" in units else units[0]
    unit = st.selectbox(
        "Unit for coverage vs typical value",
        options=units,
        index=units.index(default_unit),
        help="This chart compares tests only within one unit, because medians are not comparable across units.",
        key="priority_coverage_value_unit",
    )
    d = view[view["Unit"] == unit].copy()
    if d.empty:
        st.info("No priority tests for this unit.")
        return
    d["group"] = d["Test"].map(priority_group_for_test)
    fig = go.Figure()
    for i, (group, part) in enumerate(d.groupby("group", sort=False)):
        fig.add_trace(go.Scatter(
            x=part["n_sites"],
            y=part["median"],
            mode="markers",
            name=group,
            marker=dict(
                size=10 + np.sqrt(part["n_obs"].clip(lower=1)) / np.sqrt(d["n_obs"].max()) * 18,
                color=QUAL_PALETTE[i % len(QUAL_PALETTE)],
                opacity=0.8,
                line=dict(color="rgba(255,255,255,0.25)", width=0.5),
            ),
            customdata=part[["Test", "n_obs", "p10", "p90"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Sites: %{x:,}<br>"
                f"Median: %{{y:.6g}} {unit}<br>"
                "Readings: %{customdata[1]:,}<br>"
                "P10-P90: %{customdata[2]:.6g} - %{customdata[3]:.6g}<extra></extra>"
            ),
        ))
    fig.update_layout(xaxis_title="Sampling points", yaxis_title=f"Median ({unit})")
    fig.update_xaxes(type="log")
    if (d["median"] > 0).all():
        fig.update_yaxes(type="log")
    apply_dark_layout(fig, "Priority coverage vs typical value", height=540)
    render_plotly(fig)


def priority_correlation_matrix(view: pd.DataFrame):
    if view.empty:
        st.info("No priority tests selected.")
        return
    min_pairs = int(st.number_input(
        "Minimum paired site-years for matrix",
        min_value=20,
        max_value=5000,
        value=200,
        step=20,
        help="Higher values make the matrix more reliable but hide sparse relationships.",
        key="priority_matrix_min_pairs",
    ))
    tests = view["Test"].tolist()
    render_correlation_matrix(
        tests,
        "Priority test correlation matrix",
        min_pairs=min_pairs,
        height=max(560, 13 * len(tests) + 180),
    )


# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
with st.sidebar:
    totals = get_dataset_totals()
    st.markdown(
        """
        <div class="sidebar-title-card">
            <p class="sidebar-kicker">Environment Agency</p>
            <p class="sidebar-title">Final Sewage Effluent</p>
            <p class="sidebar-subtitle">Water-quality explorer · 2000–2025</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-section-label">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        "Page",
        ["Overview", "Explore a test", "Priority tests"],
        index=0,
        help="Start with Overview to understand the dataset, then drill into a test.",
        label_visibility="collapsed",
    )
    st.markdown('<p class="sidebar-section-label">Filters</p>', unsafe_allow_html=True)
    global_min = totals["first"].date()
    global_max = totals["last"].date()
    st.caption("Drag both handles to set the date window")
    date_range_start, date_range_end = st.slider(
        "Date range",
        min_value=global_min,
        max_value=global_max,
        value=(global_min, global_max),
        format="YYYY-MM-DD",
        help=(
            "Filters detailed maps, selected-test charts, site comparisons and raw rows. "
            "Precomputed overview summary tables and rankings remain all-time."
        ),
    )
    st.markdown('<p class="sidebar-section-label">Display</p>', unsafe_allow_html=True)
    remove_outliers = st.toggle(
        "Remove outliers (3×IQR)",
        value=False,
        help=(
            "For each series, the app finds the middle 50% of values (the IQR) "
            "and hides readings below Q1 - 3×IQR or above Q3 + 3×IQR."
        ),
    )
    st.markdown('<p class="sidebar-section-label">Dataset</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-stats-grid">'
        f'<div class="sidebar-stat"><b>{human_int(totals["total_obs"])}</b><span>Readings</span></div>'
        f'<div class="sidebar-stat"><b>{human_int(totals["n_tests"])}</b><span>Tests</span></div>'
        f'<div class="sidebar-stat"><b>{human_int(totals["max_sites"])}</b><span>Max sites</span></div>'
        f'<div class="sidebar-stat"><b>{totals["first"].year}–{totals["last"].year}</b><span>Period</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ======================================================
# PAGE 1 — OVERVIEW
# ======================================================
def page_overview():
    totals = get_dataset_totals()
    hero(
        "Final Sewage Effluent — Dataset Overview",
        "The scale and shape of England &amp; Wales treated-effluent monitoring, 2000–2025",
    )

    info_card(
        "This dataset records the quality of <b>treated wastewater leaving sewage works</b> "
        "(\u201cFinal Sewage Effluent\u201d) across England &amp; Wales. It is large, so this page "
        "gives you the big picture first: how many readings, how many tests, how many sites, "
        "over what period and where. Use the <b>Explore a test</b> page to dive into any single "
        "determinand."
    )

    # --- Headline metrics ---
    metric_grid([
        ("Total readings", human_int(totals["total_obs"])),
        ("Distinct tests", human_int(totals["n_tests"])),
        ("Sampling points", human_int(totals["max_sites"])),
        ("Years covered", f"{totals['n_years']}"),
        ("First sample", totals["first"].strftime("%d %b %Y")),
        ("Latest sample", totals["last"].strftime("%d %b %Y")),
        ("Units of measure", human_int(totals["n_units"])),
        ("Priority tests tracked", f"{totals['n_priority_available']} / 61"),
    ])

    st.markdown("")

    # --- The df.info() / df.head() / df.tail() trio, made friendly ---
    st.markdown("## What one row looks like")
    st.caption(
        "Each row in the raw dataset is a single lab result: which site, when, which test, "
        "the value and its unit. Below are the column definitions and a few real rows from the "
        "start and end of the record."
    )

    schema_df = pd.DataFrame(
        {
            "Column": ["Sampling Point", "Type", "Date", "Test", "result", "Unit",
                       "Season", "SourceYear", "Category", "Latitude", "Longitude"],
            "Meaning": [
                "Name of the sewage works / monitoring location",
                "Water type (always \u2018Final Sewage Effluent\u2019 here)",
                "Date & time the sample was taken",
                "The determinand measured (e.g. Ammonia, Zinc, pH)",
                "The measured value",
                "Unit of the value (mg/l, pH units, \u00b0C, ...)",
                "Season the sample falls in",
                "Calendar year of the sample",
                "Broad grouping of the test (e.g. standard analyte)",
                "Latitude of the sampling point",
                "Longitude of the sampling point",
            ],
            "Type": ["text", "text", "datetime", "text", "number", "text",
                     "category", "integer", "text", "number", "number"],
        }
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- "describe" that makes sense: per-test summary table ---
    st.markdown("## Every test at a glance")
    st.caption(
        "This is the \u2018describe\u2019 of the dataset — one row per test (showing its dominant "
        "unit), with how many readings exist, how many sites measure it, the date span, and the "
        "typical value (median) with its usual range (10th–90th percentile). Sort or search to explore."
    )
    primary = get_primary_unit_summary()
    table = primary.copy()
    table["Period"] = (
        table["first_sample"].dt.year.astype(str) + "–" + table["last_sample"].dt.year.astype(str)
    )
    table = table[["Test", "Unit", "n_obs", "n_sites", "n_years", "Period",
                   "median", "p10", "p90", "min", "max"]]
    table = table.rename(columns={
        "n_obs": "Readings", "n_sites": "Sites", "n_years": "Years",
        "median": "Median", "p10": "P10", "p90": "P90", "min": "Min", "max": "Max",
    })
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=460,
        column_config={
            "Readings": st.column_config.NumberColumn(format="%d"),
            "Sites": st.column_config.NumberColumn(format="%d"),
            "Median": st.column_config.NumberColumn(format="%.6g"),
            "P10": st.column_config.NumberColumn(format="%.6g"),
            "P90": st.column_config.NumberColumn(format="%.6g"),
            "Min": st.column_config.NumberColumn(format="%.6g"),
            "Max": st.column_config.NumberColumn(format="%.6g"),
        },
    )

    st.markdown("---")

    # --- Two charts: most-measured tests, and how many sites per test ---
    st.markdown("## Which tests dominate the dataset?")
    cc1, cc2 = st.columns(2)
    with cc1:
        top_obs = primary.head(15).sort_values("n_obs")
        fig = go.Figure(go.Bar(
            x=top_obs["n_obs"], y=top_obs["Test"], orientation="h",
            marker=dict(color=PRIMARY_COLOUR),
            hovertemplate="<b>%{y}</b><br>%{x:,} readings<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Readings", yaxis_title="")
        apply_dark_layout(fig, "Top 15 tests by number of readings", height=520, legend=False)
        render_plotly(fig)
    with cc2:
        top_sites = primary.sort_values("n_sites", ascending=False).head(15).sort_values("n_sites")
        fig = go.Figure(go.Bar(
            x=top_sites["n_sites"], y=top_sites["Test"], orientation="h",
            marker=dict(color=WATER_BLUE),
            hovertemplate="<b>%{y}</b><br>measured at %{x:,} sites<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Sampling points", yaxis_title="")
        apply_dark_layout(fig, "Top 15 tests by number of sites", height=520, legend=False)
        render_plotly(fig)

    st.markdown("---")

    st.markdown("## How concentrated is the dataset?")
    st.caption(
        "A few routine determinands dominate the row count. The cumulative line shows how quickly "
        "the dataset total is accounted for as you move through the busiest tests."
    )
    overview_pareto_chart(primary)

    st.markdown("---")

    st.markdown("## Which tests are data-rich?")
    st.caption(
        "Tests in the upper-right have both wide site coverage and many readings. Bubble size shows "
        "how many years the test appears in; priority tests are highlighted."
    )
    overview_richness_scatter(primary)

    st.markdown("---")

    st.markdown("## Monitoring history")
    st.caption("The busiest tests mostly span the full record, but some determinands have shorter histories.")
    overview_timeline_chart(primary)

    st.markdown("---")

    st.markdown("## Data quality signals")
    st.caption(
        "These bars flag tests where zero readings or high-value flags are common. They do not mean "
        "the data is wrong; they highlight tests that need extra care when interpreting medians and trends."
    )
    overview_data_quality_bars(primary)


# ======================================================
# PAGE 2 — EXPLORE A TEST
# ======================================================
def render_test_explorer(test_name: str, scope_label: str):
    """Shared body used by both the single-test page and the priority page."""
    summary = load_summary()
    rows = summary[summary["Test"] == test_name]
    if rows.empty:
        st.warning("No summary information for this test.")
        return

    # If multiple units exist, let the user choose (default = dominant unit).
    units = rows.sort_values("n_obs", ascending=False)["Unit"].tolist()
    if len(units) > 1:
        unit = st.selectbox(
            "Unit",
            options=units,
            help="This test is reported in more than one unit. Pick one to view.",
            key=f"unit_{scope_label}_{test_name}",
        )
    else:
        unit = units[0]
    srow = rows[rows["Unit"] == unit].iloc[0]

    # Gentle data-quality note where relevant.
    notes = []
    if srow.get("frac_zero", 0) and srow["frac_zero"] > 0.02:
        notes.append(f"{srow['frac_zero']*100:.1f}% of readings are exactly zero")
    if srow.get("frac_neg", 0) and srow["frac_neg"] > 0:
        notes.append(f"{srow['frac_neg']*100:.2f}% are negative (often below-detection or sensor noise)")
    if notes:
        st.caption("Data note: " + "; ".join(notes) + ".")

    summary_metric_values = [
        ("Readings", human_int(srow["n_obs"])),
        ("Sampling points", human_int(srow["n_sites"])),
        ("Years covered", human_int(srow["n_years"])),
        ("Unit", unit),
        ("Median", human_result_value(srow["median"])),
        ("P10–P90", f"{human_result_value(srow['p10'])} – {human_result_value(srow['p90'])}"),
        ("Min / Max", f"{human_result_value(srow['min'])} / {human_result_value(srow['max'])}"),
        ("Period", f"{srow['first_sample'].year}–{srow['last_sample'].year}"),
    ]
    load_signature = f"{test_name}::{unit}"
    loaded_key = f"{scope_label}_loaded_test_signature"
    if st.button("Load analysis", type="primary", key=f"load_analysis_{scope_label}"):
        st.session_state[loaded_key] = load_signature

    if st.session_state.get(loaded_key) != load_signature:
        metric_grid(summary_metric_values)
        st.markdown("### Detection pattern")
        zero_share_chart(srow, f"{test_name} — zero vs non-zero readings")
        st.markdown("### Which tests move with this one?")
        st.caption(
            "Correlations are precomputed from site-year median values using Spearman correlation. "
            "They show broad co-movement across the monitoring network, not direct causation."
        )
        render_test_correlation_section(test_name, scope_label)
        st.info("Load the selected test to build the detailed map and charts.")
        return

    estimated_rows = estimate_window_rows(srow, date_range_start, date_range_end)
    if estimated_rows > DETAIL_ROW_LIMIT:
        metric_grid(summary_metric_values)
        st.markdown("### Detection pattern")
        zero_share_chart(srow, f"{test_name} — zero vs non-zero readings")
        st.markdown("### Which tests move with this one?")
        st.caption(
            "Correlations are precomputed from site-year median values using Spearman correlation. "
            "They show broad co-movement across the monitoring network, not direct causation."
        )
        render_test_correlation_section(test_name, scope_label)
        st.warning(
            f"The selected date window is about {human_int(estimated_rows)} readings. "
            f"Narrow the date range below about {human_int(DETAIL_ROW_LIMIT)} readings before loading the detailed charts."
        )
        return

    # --- Load the actual rows for this test (cached, thin slice) ---
    with st.spinner(f"Loading readings for {test_name}…"):
        df = load_test_data(
            test_name,
            unit if len(units) > 1 else None,
            date_range_start,
            date_range_end,
        )
        sites = site_aggregates_from_frame(df)

    dec = smart_round(df["result"]) if "result" in df.columns else 3
    metric_grid(detailed_metric_items(df, unit, dec))
    st.caption(f"Date window: {date_range_start:%Y-%m-%d} to {date_range_end:%Y-%m-%d}.")

    if df.empty:
        st.info("No detailed rows are available for this test in the selected date range.")
        return

    st.markdown("---")

    st.markdown("### Detection pattern")
    zero_share_chart(srow, f"{test_name} — zero vs non-zero readings")

    st.markdown("### Which tests move with this one?")
    st.caption(
        "Correlations are precomputed from site-year median values using Spearman correlation. "
        "They show broad co-movement across the monitoring network, not direct causation."
    )
    render_test_correlation_section(test_name, scope_label)

    st.markdown("---")

    # --- Map of this test's sites, coloured by median value ---
    st.markdown(f"### Where is {test_name} measured?")
    map_sites = sites
    map_hidden = 0
    if remove_outliers:
        map_df, map_hidden = iqr_filtered_frame(df.dropna(subset=["result"]).copy(), ["Sampling Point"])
        map_sites = site_aggregates_from_frame(map_df)
    st.caption(
        "Each dot is a sampling point. Dot size = how often it\u2019s sampled; "
        "colour = the site\u2019s median value (greener/brighter = higher). The extreme 2% of "
        "site values are clipped so one outlier doesn\u2019t wash out the colours."
        + (f" Outlier filter hidden {human_int(map_hidden)} readings for this map." if map_hidden else "")
    )
    build_sites_map(
        map_sites, title="", colour_metric="median",
        colour_label=f"Median ({unit})" if unit else "Median", unit=unit, height=560,
    )

    st.markdown("---")

    # --- Readings per year ---
    st.markdown("### How much data, year by year?")
    yearly_count_chart(df, f"{test_name} — readings per year")

    st.markdown("---")

    # --- Trend over time ---
    st.markdown("### Trend over time")
    st.caption(
        "Monthly median value (line) with the usual spread shaded behind it. Aggregating to "
        "monthly medians keeps the picture clear despite the volume of underlying readings."
    )
    monthly_trend_chart(df, f"{test_name} — monthly median over time", unit, remove_outliers=remove_outliers)

    st.markdown("---")

    st.markdown("### How often is a threshold exceeded?")
    threshold_exceedance_chart(df, test_name, unit, key=f"threshold_{scope_label}_{test_name}")

    st.markdown("---")

    # --- Seasonal and yearly distributions ---
    st.markdown("### Distribution by season and by year")
    seasonal_box_chart(df, f"{test_name} — by season", unit, remove_outliers=remove_outliers)
    yearly_box_chart(df, f"{test_name} — by year", unit, remove_outliers=remove_outliers)

    st.markdown("---")

    # --- Compare locations ---
    st.markdown("### Comparing the busiest locations")
    st.caption(
        f"There are {human_int(len(sites))} sites measuring this test, so we focus on the "
        "most-sampled ones. Adjust how many to compare."
    )
    compare_top_n = site_count_input(
        "Number of busiest sites to compare",
        max_sites=len(sites),
        default=12,
        key=f"topn_compare_{scope_label}_{test_name}",
    )
    compare_sites_chart(
        df, sites, f"{test_name} — median and IQR at the {compare_top_n} busiest sites",
        unit, compare_top_n, remove_outliers=remove_outliers,
    )

    st.markdown("### How do the main sites differ by season?")
    season_top_n = site_count_input(
        "Number of busiest sites to show by season",
        max_sites=len(sites),
        default=18,
        key=f"topn_season_{scope_label}_{test_name}",
    )
    site_season_heatmap(
        df, sites, f"{test_name} — median by site and season",
        unit, top_n=season_top_n, remove_outliers=remove_outliers,
    )

    # --- Raw rows on demand ---
    with st.expander("Show raw readings (first / last 100 rows)", expanded=False):
        show_cols = [c for c in ["Sampling Point", "Date", "result", "Unit",
                                 "Season", "SourceYear", "Latitude", "Longitude"]
                     if c in df.columns]
        d = df.sort_values("Date")
        st.markdown("**First 100**")
        st.dataframe(d[show_cols].head(100), use_container_width=True, hide_index=True)
        st.markdown("**Last 100**")
        st.dataframe(d[show_cols].tail(100), use_container_width=True, hide_index=True)


def page_explore_test():
    hero(
        "Explore a Test",
        "Pick any one determinand and see its full story — coverage, trend, seasonality and sites",
    )
    all_tests = list_all_tests()
    default_test = "Ammoniacal Nitrogen as N"
    default_index = all_tests.index(default_test) if default_test in all_tests else 0
    info_card(
        "Choose a test from the dropdown below. Everything updates to that determinand: a map of "
        "where it\u2019s measured, how the typical value has moved over 25 years, how it varies by "
        "season, and how the busiest monitoring sites compare. Tests are listed busiest-first."
    )
    test_name = st.selectbox(
        "Select a test",
        options=all_tests,
        index=default_index,
        help="All 454 tests are available. Type to search.",
    )
    st.markdown(f"## {test_name}")
    render_test_explorer(test_name, scope_label="all")


# ======================================================
# PAGE 3 — PRIORITY TESTS
# ======================================================
@st.cache_data(show_spinner=False)
def priority_summary_table() -> pd.DataFrame:
    primary = get_primary_unit_summary()
    pr = primary[primary["Test"].isin(PRIORITY_TESTS)].copy()
    pr = pr.sort_values("n_sites", ascending=False).reset_index(drop=True)
    return pr


def page_priority():
    hero(
        "Priority Tests",
        "The 61 regulator-priority determinands — overview, ranking and deep-dive",
    )
    info_card(
        "These are the <b>61 priority determinands</b> of greatest regulatory and environmental "
        "interest (metals, nutrients, major ions and physico-chemical measures). Use the table and "
        "charts to see them as a group, then open any one for the full analysis used on the "
        "<b>Explore a test</b> page."
    )

    pr = priority_summary_table()
    avail = pr["Test"].nunique()

    metric_grid([
        ("Priority tests", f"{avail} / 61"),
        ("Total priority readings", human_int(pr["n_obs"].sum())),
        ("Widest coverage", human_int(pr["n_sites"].max())),
        ("Median sites per test", human_int(pr["n_sites"].median())),
    ])

    st.markdown("---")

    # --- Group filter to tame the list of 61 ---
    st.markdown("## Priority tests at a glance")
    group = st.selectbox(
        "Filter by group",
        options=["All priority tests"] + list(PRIORITY_GROUPS.keys()),
        help="Group the 61 tests into friendly families to make them easier to scan.",
    )
    if group == "All priority tests":
        view = pr.copy()
    else:
        members = PRIORITY_GROUPS[group]
        view = pr[pr["Test"].isin(members)].copy()

    table = view.copy()
    table["Period"] = (
        table["first_sample"].dt.year.astype(str) + "–" + table["last_sample"].dt.year.astype(str)
    )
    table = table[["Test", "Unit", "n_obs", "n_sites", "n_years", "Period", "median", "p10", "p90"]]
    table = table.rename(columns={
        "n_obs": "Readings", "n_sites": "Sites", "n_years": "Years",
        "median": "Median", "p10": "P10", "p90": "P90",
    })
    st.dataframe(
        table, use_container_width=True, hide_index=True, height=420,
        column_config={
            "Readings": st.column_config.NumberColumn(format="%d"),
            "Sites": st.column_config.NumberColumn(format="%d"),
            "Median": st.column_config.NumberColumn(format="%.6g"),
            "P10": st.column_config.NumberColumn(format="%.6g"),
            "P90": st.column_config.NumberColumn(format="%.6g"),
        },
    )

    # --- Coverage ranking chart ---
    st.markdown("## Coverage ranking — how many sites measure each?")
    st.caption("Number of sampling points per priority determinand (current group).")
    rank = view.sort_values("n_sites").tail(30)
    fig = go.Figure(go.Bar(
        x=rank["n_sites"], y=rank["Test"], orientation="h",
        marker=dict(color=PRIMARY_COLOUR),
        customdata=rank[["n_obs", "Unit"]].values,
        hovertemplate="<b>%{y}</b><br>%{x:,} sites<br>%{customdata[0]:,} readings (%{customdata[1]})<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Sampling points", yaxis_title="")
    apply_dark_layout(fig, None, height=max(420, 22 * len(rank) + 120), legend=False)
    render_plotly(fig)

    st.markdown("---")

    st.markdown("## Monitoring coverage by year")
    st.caption(
        "Each cell shows how much monitoring exists for a priority test in a year. Dark cells mean "
        "little or no data; brighter cells mean more readings."
    )
    priority_coverage_heatmap(view)

    st.markdown("---")

    st.markdown("## Priority test correlations")
    st.caption(
        "This matrix uses precomputed Spearman correlations between site-year medians. It is best "
        "read as a screening view for tests that tend to move together across sites and years."
    )
    priority_correlation_matrix(view)

    st.markdown("---")

    st.markdown("## Detectability and typical values")
    st.caption(
        "Zero-heavy tests and tests with sparse coverage need different interpretation from widely "
        "measured routine determinands."
    )
    priority_detectability_chart(view)
    priority_coverage_value_scatter(view)

    st.markdown("---")

    # --- Deep dive into one priority test ---
    st.markdown("## Deep-dive into a priority test")
    options = view["Test"].tolist() if not view.empty else pr["Test"].tolist()
    test_name = st.selectbox(
        "Select a priority test",
        options=options,
        index=0,
        help="Opens the full single-test analysis for this determinand.",
        key="priority_select",
    )
    st.markdown(f"### {test_name}")
    render_test_explorer(test_name, scope_label="priority")


# ======================================================
# ROUTER
# ======================================================
if page == "Overview":
    page_overview()
elif page == "Explore a test":
    page_explore_test()
else:
    page_priority()
