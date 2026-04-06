#!/usr/bin/env python3
"""
NYC LL84 Building Energy & Water Dashboard
==========================================
Local run:
    pip install -r requirements.txt
    python app.py
    Open → http://localhost:8050

Production (Render / Railway / Hugging Face Spaces):
    Point the platform at app.py; set USE_API = True or upload your CSV.
"""

# ============================================================
#  USER SETTINGS  ← edit these
# ============================================================
USE_API   = False                # True  → live NYC Open Data API
                                 # False → local CSV (see CSV_PATH)

CSV_PATH  = "ll84_data.csv"      # Path to the downloaded CSV file
                                 # Download from:
                                 # https://data.cityofnewyork.us/Environment/
                                 # NYC-Building-Energy-and-Water-Data-Disclosure-for-/5zyy-y8am

API_URL   = "https://data.cityofnewyork.us/resource/5zyy-y8am.json"
API_LIMIT = 50_000               # Max rows to pull (dataset ≈ 30k rows/year)
API_YEAR  = None                 # e.g. "2022" to restrict to one year, None for all

HEADER_FILE = "LL84_Header.csv"  # Column inclusion config — TRUE rows are kept,
                                 # FALSE rows are dropped. Must be in the same
                                 # folder as this script.
# ============================================================

import sys
from pathlib import Path
import requests
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, no_update, clientside_callback
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────
BLUE    = "#2563EB"
INDIGO  = "#4F46E5"
DARK    = "#1E293B"
MID     = "#64748B"
LIGHT   = "#F1F5F9"
BORDER  = "#E2E8F0"
WHITE   = "#FFFFFF"
GREEN   = "#16A34A"
PURPLE  = "#7C3AED"
PALETTE = px.colors.qualitative.Bold

LAYOUT_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Inter, system-ui, -apple-system, sans-serif",
                         color=DARK, size=12),
    margin        = dict(t=44, b=40, l=56, r=20),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    hoverlabel    = dict(bgcolor=WHITE, font_size=12, bordercolor=BORDER),
)

# ─────────────────────────────────────────────────────────────
#  COLUMN METADATA  — driven by LL84_Header.csv
# ─────────────────────────────────────────────────────────────

def _normalize_colname(name: str) -> str:
    """Same normalization applied to CSV headers and to LL84_Header.csv names."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_header_config() -> tuple[set[str], dict[str, str]]:
    """
    Read LL84_Header.csv from the same directory as this script.
    Returns:
      include_cols  — set of normalized column names marked TRUE
      display_names — dict mapping normalized name → original human-readable label
    """
    header_path = Path(__file__).parent / HEADER_FILE
    try:
        hdf = pd.read_csv(header_path, encoding="utf-8-sig")
    except FileNotFoundError:
        sys.exit(
            f"\n[ERROR] Header config not found at '{header_path}'.\n"
            f"  → Place '{HEADER_FILE}' in the same folder as app.py.\n"
        )

    # Expect columns: "Column" and "Include"
    hdf.columns = hdf.columns.str.strip()
    if "Column" not in hdf.columns or "Include" not in hdf.columns:
        sys.exit(
            f"[ERROR] '{HEADER_FILE}' must have 'Column' and 'Include' columns.\n"
            f"  Found: {list(hdf.columns)}"
        )

    hdf["norm"] = hdf["Column"].apply(_normalize_colname)
    hdf["inc"]  = hdf["Include"].astype(str).str.strip().str.upper() == "TRUE"

    include_cols  = set(hdf.loc[hdf["inc"], "norm"])
    display_names = dict(zip(hdf["norm"], hdf["Column"].str.strip()))
    return include_cols, display_names


# Override display labels for columns where we want something shorter/nicer.
# Any column NOT listed here falls back to the original name from LL84_Header.csv.
COL_LABEL_OVERRIDES = {
    "primary_property_type_self_selected":             "Property Type",
    "property_gfa_self_reported_ft":                   "GFA (ft²)",
    "site_eui_kbtu_ft":                                "Site EUI (kBtu/ft²)",
    "weather_normalized_site_eui_kbtu_ft":             "WN Site EUI (kBtu/ft²)",
    "total_location_based_ghg_emissions_metric_tons_co2e": "Total GHG (MT CO₂e)",
    "total_location_based_ghg_emissions_intensity_kgco2e_ft": "GHG Intensity (kgCO₂e/ft²)",
    "electricity_use_grid_purchase_kbtu":              "Electricity (kBtu)",
    "natural_gas_use_kbtu":                            "Natural Gas (kBtu)",
    "district_steam_use_kbtu":                         "District Steam (kBtu)",
    "district_hot_water_use_kbtu":                     "District Hot Water (kBtu)",
    "district_chilled_water_use_kbtu":                 "District Chilled Water (kBtu)",
    "electricity_weather_normalized_site_electricity_use_grid_and_onsite_renewables_kwh":
                                                       "WN Electricity (kWh)",
    "electricity_weather_normalized_site_electricity_intensity_grid_and_onsite_renewables_kwh_ft":
                                                       "WN Elec. Intensity (kWh/ft²)",
    "nyc_borough_block_and_lot_bbl":                   "BBL",
    "energy_star_score":                               "ENERGY STAR Score",
    "site_energy_use_kbtu":                            "Site Energy (kBtu)",
    "source_energy_use_kbtu":                          "Source Energy (kBtu)",
    "weather_normalized_site_energy_use_kbtu":         "WN Site Energy (kBtu)",
    "weather_normalized_source_energy_use_kbtu":       "WN Source Energy (kBtu)",
    "largest_property_use_type":                       "Largest Use Type",
    "2nd_largest_property_use_type":                   "2nd Largest Use Type",
    "3rd_largest_property_use_type":                   "3rd Largest Use Type",
    "calendar_year":                                   "Year",
}

# Columns to pin to the left so they're always visible
PIN_LEFT = {"property_name", "address_1", "city", "borough"}

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load LL84 data from either a local CSV or the NYC Open Data Socrata API.
    Controlled by the USE_API flag at the top of this file.
    """
    if USE_API:
        print(f"[data] Fetching up to {API_LIMIT:,} rows from NYC Open Data API …")
        params: dict = {"$limit": API_LIMIT, "$order": ":id"}
        if API_YEAR:
            params["$where"] = f"data_year='{API_YEAR}'"
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
        except requests.RequestException as exc:
            sys.exit(f"\n[ERROR] API request failed: {exc}\n"
                     "Check your internet connection or set USE_API = False.\n")
        df = pd.DataFrame(r.json())
        print(f"[data] Received {len(df):,} rows from API.")
    else:
        print(f"[data] Loading local file: {CSV_PATH} …")
        try:
            df = pd.read_csv(CSV_PATH, low_memory=False)
        except FileNotFoundError:
            sys.exit(
                f"\n[ERROR] CSV not found at '{CSV_PATH}'.\n"
                "  → Download the full dataset from NYC Open Data:\n"
                "    https://data.cityofnewyork.us/Environment/"
                "NYC-Building-Energy-and-Water-Data-Disclosure-for-/5zyy-y8am\n"
                "  → Place the CSV next to app.py and re-run, "
                "or set USE_API = True.\n"
            )

    # ── normalise ──────────────────────────────────────────
    # CSV exports use human-readable headers like "Site EUI (kBtu/ft2)".
    # The API returns snake_case. Normalise both to the same snake_case
    # by replacing every run of non-alphanumeric chars with "_".
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Auto-coerce columns where >60 % of values parse as numbers
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.60:
                df[c] = coerced

    print(f"[data] Ready: {len(df):,} rows × {len(df.columns)} columns.")
    return df


print("[startup] Loading header config …")
INCLUDE_COLS, DISPLAY_NAMES = load_header_config()
print(f"[startup] Header config: {len(INCLUDE_COLS)} columns marked TRUE.")

print("[startup] Loading data …")
DF_RAW: pd.DataFrame = load_data()

# ── Keep only columns flagged TRUE in LL84_Header.csv.
#    Intersect with what actually exists in the data file.
_KEEP = [c for c in DF_RAW.columns if c in INCLUDE_COLS]
if not _KEEP:
    # Fallback warning if nothing matched (e.g. wrong CSV version)
    print("[WARNING] No columns from LL84_Header.csv matched the data file. "
          "Keeping all columns — check that HEADER_FILE points to the right file.")
    _KEEP = list(DF_RAW.columns)

DF: pd.DataFrame = DF_RAW[_KEEP].copy()
print(f"[startup] Kept {len(DF.columns)} of {len(DF_RAW.columns)} columns "
      f"based on LL84_Header.csv.")



# ─────────────────────────────────────────────────────────────
#  AG GRID COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────
def build_col_defs(df: pd.DataFrame) -> list[dict]:
    defs = []
    for c in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[c])
        label  = COL_LABEL_OVERRIDES.get(c) or DISPLAY_NAMES.get(c) or c.replace("_", " ").title()
        pinned = "left" if c in PIN_LEFT else None

        base: dict = {
            "field":      c,
            "headerName": label,
            "sortable":   True,
            "resizable":  True,
            "hide":       False,  # visibility controlled by LL84_Header.csv
            "pinned":     pinned,
            "minWidth":   100,
        }

        if is_num:
            base.update({
                "filter":     "agNumberColumnFilter",
                "type":       "numericColumn",
                "width":      145,
                # Format with commas and up to 2 significant decimal places
                "valueFormatter": {
                    "function": (
                        "params.value != null ? "
                        "d3.format(',.2~f')(params.value) : ''"
                    )
                },
                "filterParams": {
                    "filterOptions": [
                        "equals", "notEqual", "lessThan", "lessThanOrEqual",
                        "greaterThan", "greaterThanOrEqual", "inRange",
                    ],
                    "defaultOption": "inRange",
                    "buttons": ["reset"],
                },
            })
        else:
            base.update({
                "filter":  "agTextColumnFilter",
                "width":   200,
                "filterParams": {
                    "filterOptions": [
                        "contains", "notContains", "startsWith",
                        "endsWith", "equals", "notEqual",
                    ],
                    "defaultOption":       "contains",
                    "caseSensitive":       False,
                    "buttons":             ["reset"],
                    # Cross-platform text matcher: null-safe, handles all
                    # filter options, and treats "contains" as regex-capable.
                    "textMatcher": {
                        "function": (
                            "const { filterOption, value, filterText } = params; "
                            "if (value == null || value === undefined) return false; "
                            "const v = String(value); "
                            "const f = filterText || ''; "
                            "switch (filterOption) { "
                            "  case 'contains': "
                            "    try { return new RegExp(f, 'i').test(v); } "
                            "    catch(e) { return v.toLowerCase().includes(f.toLowerCase()); } "
                            "  case 'notContains': "
                            "    try { return !new RegExp(f, 'i').test(v); } "
                            "    catch(e) { return !v.toLowerCase().includes(f.toLowerCase()); } "
                            "  case 'equals': return v.toLowerCase() === f.toLowerCase(); "
                            "  case 'notEqual': return v.toLowerCase() !== f.toLowerCase(); "
                            "  case 'startsWith': return v.toLowerCase().startsWith(f.toLowerCase()); "
                            "  case 'endsWith': return v.toLowerCase().endsWith(f.toLowerCase()); "
                            "  default: return false; "
                            "}"
                        )
                    },
                },
            })

        defs.append(base)
    return defs


COL_DEFS = build_col_defs(DF)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def first_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first candidate column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    virtualRowData can return numbers as strings.
    Re-coerce any column that is numeric in the original dataframe.
    """
    for c in df.columns:
        if c in DF.columns and pd.api.types.is_numeric_dtype(DF[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _empty_fig(title: str) -> go.Figure:
    return go.Figure().update_layout(
        title=dict(text=title, font_size=14),
        **LAYOUT_BASE,
    )


# ─────────────────────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────────────────────
def _add_percentile_annotations(fig: go.Figure, data: pd.Series,
                                unit: str = "") -> go.Figure:
    """Add vertical lines + labels for 10th, 50th, 90th percentiles."""
    pcts = {10: "#F59E0B", 50: "#10B981", 90: "#EF4444"}
    for p, colour in pcts.items():
        val = data.quantile(p / 100)
        fig.add_vline(x=val, line_dash="dash", line_color=colour,
                      line_width=1.5, opacity=0.85)
        fig.add_annotation(
            x=val, yref="paper", y=1.0,
            text=f"P{p}<br>{val:,.1f}{unit}",
            showarrow=False,
            font=dict(size=10, color=colour),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=colour, borderwidth=1,
            yanchor="bottom", xanchor="center",
        )
    return fig


def fig_eui_histogram(df: pd.DataFrame) -> go.Figure:
    c = first_col(df, "site_eui_kbtu_ft", "source_eui_kbtu_ft")
    if c is None or df.empty:
        return _empty_fig("Site EUI Distribution")

    data = df[c].dropna()
    if len(data) < 2:
        return _empty_fig("Site EUI Distribution (insufficient data)")
    cap  = data.quantile(0.99) if len(data) > 10 else data.max()
    data = data[data <= cap]

    fig = px.histogram(
        data, nbins=60,
        title="Site EUI Distribution",
        labels={c: "Site EUI (kBtu/ft²)", "count": "Buildings"},
        color_discrete_sequence=[BLUE],
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="white")
    fig.update_layout(**{**LAYOUT_BASE, "margin": dict(t=64, b=40, l=56, r=20)},
                      title_font_size=14,
                      xaxis_title="Site EUI (kBtu/ft²)",
                      yaxis_title="Buildings")
    _add_percentile_annotations(fig, data)
    return fig


def fig_property_type_bar(df: pd.DataFrame) -> go.Figure:
    c = first_col(df, "primary_property_type_self_selected", "primary_property_type", "property_type")
    if c is None or df.empty:
        return _empty_fig("Top Property Types")

    counts = df[c].value_counts().head(12).reset_index()
    counts.columns = ["type", "n"]

    fig = px.bar(
        counts, x="n", y="type", orientation="h",
        title="Top Property Types",
        labels={"n": "# Buildings", "type": ""},
        color_discrete_sequence=[BLUE],
    )
    fig.update_layout(
        yaxis_categoryorder="total ascending",
        **LAYOUT_BASE,
        title_font_size=14,
    )
    return fig


def fig_ghg_scatter(df: pd.DataFrame) -> go.Figure:
    eui_c = first_col(df, "site_eui_kbtu_ft", "source_eui_kbtu_ft")
    ghg_c = first_col(df,
                "total_location_based_ghg_emissions_intensity_kgco2e_ft",
                "total_ghg_emissions_metric_tons_co2e",
                "total_ghg_emissions_metric_tons_co_2e")
    bor_c = first_col(df, "borough")

    if eui_c is None or ghg_c is None or df.empty:
        return _empty_fig("GHG Emissions vs. Site EUI")

    plot = df[[eui_c, ghg_c] + ([bor_c] if bor_c else [])].dropna()

    if len(plot) < 2:
        return _empty_fig("GHG Emissions vs. Site EUI (insufficient data)")

    # Remove top-1 % outliers in both axes for readability
    plot = plot[plot[eui_c] <= plot[eui_c].quantile(0.99)]
    plot = plot[plot[ghg_c] <= plot[ghg_c].quantile(0.99)]

    # Sample for render performance when large
    if len(plot) > 4_000:
        plot = plot.sample(4_000, random_state=42)

    color_kw = dict(color=bor_c, color_discrete_sequence=PALETTE) if bor_c else \
               dict(color_discrete_sequence=[BLUE])

    fig = px.scatter(
        plot, x=eui_c, y=ghg_c,
        opacity=0.50,
        title="GHG Emissions vs. Site EUI",
        labels={
            eui_c: "Site EUI (kBtu/ft²)",
            ghg_c: "GHGI (kgCO₂e/ft²)",
        },
        **color_kw,
    )
    fig.update_layout(**LAYOUT_BASE, title_font_size=14,
                      legend_title_text="Borough")
    return fig


def fig_ghg_intensity_histogram(df: pd.DataFrame) -> go.Figure:
    c = first_col(df,
        "total_location_based_ghg_emissions_intensity_kgco2e_ft",
        "location_based_ghg_intensity_kgco2e_ft",
        "ghg_intensity_kgco2e_ft",
    )
    if c is None or df.empty:
        return _empty_fig("GHG Emissions Intensity Distribution")

    data = df[c].dropna()
    if len(data) < 2:
        return _empty_fig("GHG Intensity Distribution (insufficient data)")
    cap  = data.quantile(0.99) if len(data) > 10 else data.max()
    data = data[data <= cap]

    fig = px.histogram(
        data, nbins=60,
        title="GHG Emissions Intensity Distribution",
        labels={c: "GHG Intensity (kgCO₂e/ft²)", "count": "Buildings"},
        color_discrete_sequence=["#7C3AED"],
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="white")
    fig.update_layout(**{**LAYOUT_BASE, "margin": dict(t=64, b=40, l=56, r=20)},
                      title_font_size=14,
                      xaxis_title="GHG Intensity (kgCO₂e/ft²)",
                      yaxis_title="Buildings")
    _add_percentile_annotations(fig, data)
    return fig



def fig_eui_by_decade(df: pd.DataFrame) -> go.Figure:
    """Median Site EUI grouped into 10-year construction-decade bins."""
    year_c = first_col(df, "year_built")
    eui_c  = first_col(df, "site_eui_kbtu_ft", "source_eui_kbtu_ft")

    if year_c is None or eui_c is None or df.empty:
        return _empty_fig("Median Site EUI by Construction Decade")

    tmp = df[[year_c, eui_c]].copy()
    tmp[year_c] = pd.to_numeric(tmp[year_c], errors="coerce")
    tmp[eui_c]  = pd.to_numeric(tmp[eui_c],  errors="coerce")
    tmp = tmp.dropna()

    # Bin into decades; exclude obvious bad years
    tmp = tmp[(tmp[year_c] >= 1800) & (tmp[year_c] <= 2030)]
    tmp["decade"] = (tmp[year_c] // 10 * 10).astype(int)

    grp = (tmp.groupby("decade")[eui_c]
              .agg(median="median", count="count", p25=lambda x: x.quantile(0.25),
                   p75=lambda x: x.quantile(0.75))
              .reset_index())
    grp = grp[grp["count"] >= 5]   # suppress tiny bins
    grp["label"] = grp["decade"].astype(str) + "s"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["label"],
        y=grp["median"],
        marker_color=BLUE,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(grp["p75"] - grp["median"]).tolist(),
            arrayminus=(grp["median"] - grp["p25"]).tolist(),
            color=MID, thickness=1.5, width=4,
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Median EUI: %{y:.1f} kBtu/ft²<br>"
            "Count: %{customdata:,}<extra></extra>"
        ),
        customdata=grp["count"].tolist(),
        name="Median EUI",
    ))
    fig.update_layout(
        **LAYOUT_BASE, title_font_size=14,
        title="Median Site EUI by Construction Decade",
        xaxis_title="Construction Decade",
        yaxis_title="Median Site EUI (kBtu/ft²)",
        bargap=0.25,
    )
    return fig


def fig_gfa_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of gross floor area (log-scaled x-axis for readability)."""
    gfa_c = first_col(df,
        "property_gfa_self_reported_ft",
        "gross_floor_area_ft",
        "gross_floor_area_self_reported_ft",
        "gross_floor_area",
    )
    if gfa_c is None or df.empty:
        return _empty_fig("Building Size Distribution (GFA)")

    data = pd.to_numeric(df[gfa_c], errors="coerce").dropna()
    data = data[data > 0]
    if len(data) < 2:
        return _empty_fig("Building Size Distribution (insufficient data)")

    # Cap at 99th pct; use log bins for wide dynamic range
    cap = data.quantile(0.99)
    data = data[data <= cap]
    log_data = np.log10(data)

    fig = px.histogram(
        log_data, nbins=50,
        title="Building Size Distribution (GFA)",
        color_discrete_sequence=[GREEN],
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="white")

    # Replace log10 tick labels with human-readable ft² values
    tick_vals = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    tick_text = []
    for v in tick_vals:
        val = 10 ** v
        if val >= 1e6:
            tick_text.append(f"{val/1e6:.1f}M")
        elif val >= 1e3:
            tick_text.append(f"{val/1e3:.0f}k")
        else:
            tick_text.append(f"{val:.0f}")

    fig.update_layout(
        **LAYOUT_BASE, title_font_size=14,
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text,
                   title="Gross Floor Area (ft²)"),
        yaxis_title="Buildings",
    )
    return fig


def fig_ghg_by_building_type(df: pd.DataFrame) -> go.Figure:
    """% contribution to total GHG by building type, descending."""
    type_c = first_col(df,
        "primary_property_type_self_selected",
        "primary_property_type",
        "largest_property_use_type",
    )
    ghg_c = first_col(df,
        "total_location_based_ghg_emissions_metric_tons_co2e",
        "total_ghg_emissions_metric_tons_co2e",
        "total_ghg_emissions_metric_tons_co_2e",
    )

    if type_c is None or ghg_c is None or df.empty:
        return _empty_fig("GHG Emissions by Building Type")

    tmp = df[[type_c, ghg_c]].copy()
    tmp[ghg_c] = pd.to_numeric(tmp[ghg_c], errors="coerce")
    tmp = tmp.dropna()

    if tmp.empty or tmp[ghg_c].sum() == 0:
        return _empty_fig("GHG Emissions by Building Type (no data)")

    grp = tmp.groupby(type_c)[ghg_c].sum().reset_index()
    grp.columns = ["type", "ghg"]
    total = grp["ghg"].sum()
    grp["pct"] = grp["ghg"] / total * 100
    grp = grp.sort_values("pct", ascending=True)   # ascending for horizontal bar

    # Collapse tail into "Other" if more than 15 types
    if len(grp) > 15:
        top = grp.tail(15)
        other_pct = grp.head(len(grp) - 15)["pct"].sum()
        other_row = pd.DataFrame([{"type": "Other", "ghg": 0, "pct": other_pct}])
        grp = pd.concat([other_row, top], ignore_index=True)

    fig = go.Figure(go.Bar(
        x=grp["pct"],
        y=grp["type"],
        orientation="h",
        marker_color=PALETTE[:len(grp)] if len(grp) <= len(PALETTE) else PALETTE * 3,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of total GHG<extra></extra>",
    ))
    fig.update_layout(
        **{**LAYOUT_BASE, "margin": dict(t=44, b=40, l=220, r=20)},
        title_font_size=14,
        title="% Contribution to Total GHG by Building Type",
        xaxis_title="% of Total GHG Emissions",
        yaxis_title="",
        xaxis_ticksuffix="%",
    )
    return fig


# Fuel columns to aggregate for the fuel mix chart.
# Each entry: (normalized_col_name, display_label, group)
_FUEL_COLS = [
    ("electricity_use_grid_purchase_kbtu",  "Electricity",      "Electricity"),
    ("natural_gas_use_kbtu",                "Natural Gas",       "Natural Gas"),
    ("fuel_oil_1_use_kbtu",                 "Fuel Oil #1",       "Fuel Oil"),
    ("fuel_oil_2_use_kbtu",                 "Fuel Oil #2",       "Fuel Oil"),
    ("fuel_oil_4_use_kbtu",                 "Fuel Oil #4",       "Fuel Oil"),
    ("fuel_oil_5_6_use_kbtu",               "Fuel Oil #5/6",     "Fuel Oil"),
    ("diesel_2_use_kbtu",                   "Diesel #2",         "Fuel Oil"),
    ("propane_use_kbtu",                    "Propane",           "Fuel Oil"),
    ("kerosene_use_kbtu",                   "Kerosene",          "Fuel Oil"),
    ("district_steam_use_kbtu",             "District Steam",    "District Energy"),
    ("district_hot_water_use_kbtu",         "District Hot Water","District Energy"),
    ("district_chilled_water_use_kbtu",     "District Chilled Water","District Energy"),
]

_FUEL_COLORS = {
    "Electricity":     "#2563EB",
    "Natural Gas":     "#F59E0B",
    "Fuel Oil":        "#DC2626",
    "District Energy": "#10B981",
}


def fig_fuel_mix(df: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bar showing % fuel mix by kBtu."""
    if df.empty:
        return _empty_fig("Fuel Mix (% of Total Energy by kBtu)")

    # Sum each available fuel column; skip columns not in the filtered df
    group_totals: dict[str, float] = {}
    for col, _label, group in _FUEL_COLS:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").sum()
            if val > 0:
                group_totals[group] = group_totals.get(group, 0.0) + val

    if not group_totals or sum(group_totals.values()) == 0:
        return _empty_fig("Fuel Mix (no fuel data available)")

    total = sum(group_totals.values())
    pcts  = {g: v / total * 100 for g, v in group_totals.items()}

    # Sort by descending share for consistent visual order
    groups_sorted = sorted(pcts, key=lambda g: pcts[g], reverse=True)

    fig = go.Figure()
    for g in groups_sorted:
        pct = pcts[g]
        fig.add_trace(go.Bar(
            name=g,
            x=[pct],
            y=["Fuel Mix"],
            orientation="h",
            marker_color=_FUEL_COLORS.get(g, MID),
            text=f"{pct:.1f}%",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{g}</b>: {{x:.1f}}%<extra></extra>",
        ))

    fig.update_layout(**{
        **LAYOUT_BASE,
        "legend": dict(orientation="h", yanchor="bottom", y=1.02,
                       xanchor="left", x=0, font_size=11),
        "margin": dict(t=80, b=40, l=20, r=20),
    }, title_font_size=14,
        title="Fuel Mix — % of Total Site Energy (kBtu)",
        barmode="stack",
        xaxis=dict(title="% of Total kBtu", ticksuffix="%", range=[0, 100]),
        yaxis=dict(title=""),
        height=220,
    )
    return fig

# ─────────────────────────────────────────────────────────────
#  SUMMARY STATS
# ─────────────────────────────────────────────────────────────
def _valid_num(df, col):
    """Return True if col exists in df and has at least one non-NaN numeric value."""
    if col is None or df.empty:
        return False
    series = pd.to_numeric(df[col], errors="coerce")
    return series.notna().any()


def compute_stats(df: pd.DataFrame) -> tuple:
    n     = f"{len(df):,}"
    eui_c = first_col(df, "site_eui_kbtu_ft")
    sc_c  = first_col(df, "energy_star_score")
    # GHG intensity column — normalized name from CSV or API
    ghg_c = first_col(df,
                "total_location_based_ghg_emissions_intensity_kgco2e_ft",
                "location_based_ghg_intensity_kgco2e_ft",
                "ghg_intensity_kgco2e_ft")
    # GFA column name also varies (may include "self_reported" or "ft_" suffix)
    gfa_c = first_col(df,
                "property_gfa_self_reported_ft",
                "gross_floor_area_ft",
                "gross_floor_area_self_reported_ft",
                "gross_floor_area_ft_",
                "gross_floor_area",
                "dof_gross_floor_area")

    # Print diagnostic on first call if columns still not found
    if ghg_c is None or gfa_c is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        print(f"[stats] GHG col resolved to: {ghg_c!r}")
        print(f"[stats] GFA col resolved to: {gfa_c!r}")
        print(f"[stats] Available numeric columns: {numeric_cols[:30]}")

    avg_eui = f"{pd.to_numeric(df[eui_c], errors='coerce').mean():.1f}" if _valid_num(df, eui_c) else "—"
    avg_sc  = f"{pd.to_numeric(df[sc_c],  errors='coerce').mean():.1f}" if _valid_num(df, sc_c)  else "—"

    if _valid_num(df, ghg_c):
        tot = pd.to_numeric(df[ghg_c], errors="coerce").mean()
        tot_ghg = f"{tot:.2f}"
    else:
        tot_ghg = "—"

    if _valid_num(df, gfa_c):
        tot = pd.to_numeric(df[gfa_c], errors="coerce").sum()
        tot_gfa = f"{tot / 1e3:,.0f}k ft²"
    else:
        tot_gfa = "—"

    return n, avg_eui, avg_sc, tot_ghg, tot_gfa


# ─────────────────────────────────────────────────────────────
#  LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────
SOURCE_LABEL = "🌐  Live API"  if USE_API else "📂  Local CSV"
SOURCE_COLOR = GREEN           if USE_API else PURPLE

CARD_SPECS = [
    ("stat-buildings", "Buildings",             "🏢"),
    ("stat-avg-eui",   "Avg Site EUI (kBtu/ft²)", "⚡"),
    ("stat-avg-score", "Avg ENERGY STAR Score",  "⭐"),
    ("stat-tot-ghg",   "Avg GHG Intensity (kgCO₂e/ft²)",    "🌫️"),
    ("stat-tot-gfa",   "Total Floor Area",       "📐"),
]

def stat_card(card_id: str, label: str, icon: str) -> html.Div:
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "6px"}),
        html.Div(id=card_id, children="—",
                 style={"fontSize": "21px", "fontWeight": "700", "color": DARK,
                        "letterSpacing": "-0.5px"}),
        html.Div(label,
                 style={"fontSize": "11px", "color": MID,
                        "marginTop": "4px", "lineHeight": "1.3"}),
    ], style={
        "background":    WHITE,
        "borderRadius":  "12px",
        "padding":       "16px 18px",
        "boxShadow":     f"0 1px 3px rgba(0,0,0,.07), 0 1px 8px rgba(0,0,0,.04)",
        "textAlign":     "center",
        "flex":          "1",
        "minWidth":      "130px",
        "borderTop":     f"3px solid {BLUE}",
    })


def chart_card(graph_id: str) -> html.Div:
    return html.Div([
        dcc.Loading(
            type="circle",
            color=BLUE,
            children=dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "300px"},
            ),
        )
    ], style={
        "flex":         "1",
        "minWidth":     "340px",
        "background":   WHITE,
        "borderRadius": "12px",
        "padding":      "12px 16px",
        "boxShadow":    "0 1px 3px rgba(0,0,0,.07), 0 1px 8px rgba(0,0,0,.04)",
    })


# ─────────────────────────────────────────────────────────────
#  APP + LAYOUT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="NYC LL84 Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server   # expose Flask server for production WSGI deployment

# ── inject global CSS via index_string ───────────────────────
_CSS = f"""
  body {{ margin:0; background:{LIGHT}; }}
  * {{ box-sizing:border-box; }}

  /* AG Grid tweaks */
  .ag-theme-alpine .ag-header-cell-label {{ font-weight:600; font-size:12px; }}
  .ag-theme-alpine .ag-floating-filter-input {{ font-size:12px; }}
  .ag-theme-alpine .ag-row-even  {{ background:{WHITE}; }}
  .ag-theme-alpine .ag-row-odd   {{ background:#F8FAFC; }}
  .ag-theme-alpine .ag-row:hover {{ background:#EFF6FF !important; }}
  .ag-theme-alpine .ag-side-button-label {{ font-size:11px; }}

  /* Export button hover */
  #export-btn:hover {{ opacity:0.88; transform:translateY(-1px); }}
  #reset-btn:hover  {{ opacity:0.88; }}
  #export-btn, #reset-btn {{ transition: all .15s ease; }}
"""

app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{_CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""

app.layout = html.Div([

    # ── header bar ─────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("NYC LL84 Building Energy Dashboard",
                    style={"margin": "0", "fontSize": "19px",
                           "fontWeight": "700", "color": WHITE,
                           "letterSpacing": "-0.3px"}),
            html.Div("Local Law 84 — Energy & Water Benchmarking Disclosure",
                     style={"fontSize": "12px", "color": "rgba(255,255,255,.65)",
                            "marginTop": "3px"}),
        ]),
        html.Div(SOURCE_LABEL, style={
            "background":   SOURCE_COLOR,
            "color":        WHITE,
            "padding":      "5px 14px",
            "borderRadius": "20px",
            "fontSize":     "12px",
            "fontWeight":   "600",
            "alignSelf":    "center",
            "whiteSpace":   "nowrap",
        }),
    ], style={
        "background":      f"linear-gradient(135deg, {DARK} 0%, #334155 100%)",
        "padding":         "16px 28px",
        "display":         "flex",
        "justifyContent":  "space-between",
        "alignItems":      "center",
    }),

    # ── main body ───────────────────────────────────────────
    html.Div([

        # stat cards row
        html.Div(
            [stat_card(cid, lbl, ico) for cid, lbl, ico in CARD_SPECS],
            style={"display": "flex", "gap": "12px",
                   "flexWrap": "wrap", "marginBottom": "20px"},
        ),

        # ── data grid section ───────────────────────────────
        html.Div([

            # toolbar: row count + buttons
            html.Div([
                html.Div(id="row-count",
                         style={"fontSize": "13px", "color": MID,
                                "alignSelf": "center", "fontWeight": "500"}),
                html.Div([
                    html.Button("✕  Reset Filters", id="reset-btn",
                                style={
                                    "background": WHITE,
                                    "color": DARK,
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "8px",
                                    "padding": "7px 14px",
                                    "cursor": "pointer",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "marginRight": "8px",
                                }),
                    html.Button("⬇  Export Filtered CSV", id="export-btn",
                                style={
                                    "background": BLUE,
                                    "color": WHITE,
                                    "border": "none",
                                    "borderRadius": "8px",
                                    "padding": "7px 16px",
                                    "cursor": "pointer",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                }),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={
                "display":        "flex",
                "justifyContent": "space-between",
                "marginBottom":   "10px",
                "alignItems":     "center",
            }),

            # AG Grid
            dag.AgGrid(
                id="main-grid",
                rowData=DF.to_dict("records"),
                columnDefs=COL_DEFS,
                defaultColDef={
                    "floatingFilter": True,
                    "suppressMenu":   False,
                    "wrapHeaderText": True,
                    "autoHeaderHeight": True,
                },
                dashGridOptions={
                    "pagination":              True,
                    "paginationPageSize":       100,
                    "paginationPageSizeSelector": [50, 100, 250, 500],
                    "enableCellTextSelection": True,
                    "suppressRowClickSelection": True,
                    "animateRows":             False,   # off for performance
                    "suppressColumnVirtualisation": False,
                    "rowBuffer":               20,
                    # Sidebar: column chooser + filters panel
                    "sideBar": {
                        "toolPanels": [
                            {
                                "id":           "columns",
                                "labelDefault": "Columns",
                                "iconKey":      "columns",
                                "toolPanel":    "agColumnsToolPanel",
                                "toolPanelParams": {
                                    "suppressRowGroups":  True,
                                    "suppressValues":     True,
                                    "suppressPivotMode":  True,
                                },
                            },
                            {
                                "id":           "filters",
                                "labelDefault": "Filters",
                                "iconKey":      "filter",
                                "toolPanel":    "agFiltersToolPanel",
                            },
                        ],
                        "defaultToolPanel": "",  # collapsed by default
                    },
                },
                columnSize="autoSize",
                style={"height": "500px"},
                className="ag-theme-alpine",
            ),

            # filter tip
            html.Div(
                "💡 Click a column header ▸ filter icon to set numeric ranges or "
                "regex patterns. Use the Columns panel (▦) to show/hide columns.",
                style={"fontSize": "11px", "color": MID,
                       "marginTop": "8px", "fontStyle": "italic"},
            ),

        ], style={
            "background":   WHITE,
            "borderRadius": "12px",
            "padding":      "16px 20px",
            "marginBottom": "20px",
            "boxShadow":    "0 1px 3px rgba(0,0,0,.07), 0 1px 8px rgba(0,0,0,.04)",
        }),

        # ── charts 4×2 grid ─────────────────────────────────
        # Row 1: EUI histogram + GHG intensity histogram (both with percentile overlays)
        html.Div([
            chart_card("chart-eui"),
            chart_card("chart-ghg-intensity"),
        ], style={"display": "flex", "gap": "16px",
                  "flexWrap": "wrap", "marginBottom": "16px"}),

        # Row 2: GHG intensity scatter + property types bar
        html.Div([
            chart_card("chart-ghg"),
            chart_card("chart-types"),
        ], style={"display": "flex", "gap": "16px",
                  "flexWrap": "wrap", "marginBottom": "16px"}),

        # Row 3: EUI by construction decade + GFA histogram
        html.Div([
            chart_card("chart-eui-decade"),
            chart_card("chart-gfa"),
        ], style={"display": "flex", "gap": "16px",
                  "flexWrap": "wrap", "marginBottom": "16px"}),

        # Row 4: GHG by building type (taller for long labels) + fuel mix
        html.Div([
            html.Div([
                dcc.Loading(type="circle", color=BLUE,
                    children=dcc.Graph(id="chart-ghg-type",
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "420px"})),
            ], style={"flex": "1", "minWidth": "340px", "background": WHITE,
                      "borderRadius": "12px", "padding": "12px 16px",
                      "boxShadow": "0 1px 3px rgba(0,0,0,.07), 0 1px 8px rgba(0,0,0,.04)"}),

            html.Div([
                dcc.Loading(type="circle", color=BLUE,
                    children=dcc.Graph(id="chart-fuel-mix",
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "420px"})),
            ], style={"flex": "1", "minWidth": "340px", "background": WHITE,
                      "borderRadius": "12px", "padding": "12px 16px",
                      "boxShadow": "0 1px 3px rgba(0,0,0,.07), 0 1px 8px rgba(0,0,0,.04)"}),
        ], style={"display": "flex", "gap": "16px",
                  "flexWrap": "wrap", "marginBottom": "16px"}),

        # footer
        html.Div(
            f"Data source: NYC Open Data — Local Law 84 (LL84) "
            f"| Dashboard built with Dash + AG Grid",
            style={"textAlign": "center", "fontSize": "11px", "color": MID,
                   "padding": "24px 0 8px"},
        ),

    ], style={"padding": "24px 28px", "minHeight": "calc(100vh - 66px)"}),

    dcc.Download(id="download-csv"),

], style={"fontFamily": "Inter, system-ui, -apple-system, sans-serif", "margin": "0"})


# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────

# ── 1. Charts + stats — driven by virtualRowData ─────────────
@app.callback(
    Output("chart-eui",           "figure"),
    Output("chart-ghg-intensity", "figure"),
    Output("chart-ghg",           "figure"),
    Output("chart-types",         "figure"),
    Output("chart-eui-decade",    "figure"),
    Output("chart-gfa",           "figure"),
    Output("chart-ghg-type",      "figure"),
    Output("chart-fuel-mix",      "figure"),
    Output("stat-buildings",      "children"),
    Output("stat-avg-eui",        "children"),
    Output("stat-avg-score",      "children"),
    Output("stat-tot-ghg",        "children"),
    Output("stat-tot-gfa",        "children"),
    Output("row-count",           "children"),
    Input("main-grid",            "virtualRowData"),
)
def update_visuals(virtual_data):
    """
    Fires whenever AG Grid's filter or sort changes.
    virtualRowData reflects only the rows currently visible in the grid.
    """
    if virtual_data is None:
        # Initial load — use full dataset
        df = DF.copy()
    else:
        df = pd.DataFrame(virtual_data)
        df = coerce_numerics(df)

    n, avg_eui, avg_sc, tot_ghg, tot_gfa = compute_stats(df)

    total_rows = len(DF)
    shown_rows = len(df)
    row_label  = (
        f"Showing {shown_rows:,} of {total_rows:,} rows"
        if shown_rows < total_rows
        else f"Showing all {total_rows:,} rows"
    )

    return (
        fig_eui_histogram(df),
        fig_ghg_intensity_histogram(df),
        fig_ghg_scatter(df),
        fig_property_type_bar(df),
        fig_eui_by_decade(df),
        fig_gfa_histogram(df),
        fig_ghg_by_building_type(df),
        fig_fuel_mix(df),
        n, avg_eui, avg_sc, tot_ghg, tot_gfa,
        row_label,
    )


# ── 2. CSV export ────────────────────────────────────────────
@app.callback(
    Output("download-csv", "data"),
    Input("export-btn",    "n_clicks"),
    State("main-grid",     "virtualRowData"),
    prevent_initial_call=True,
)
def export_csv(_, virtual_data):
    if not virtual_data:
        raise PreventUpdate
    df = coerce_numerics(pd.DataFrame(virtual_data))
    return dcc.send_data_frame(df.to_csv, "ll84_filtered_export.csv", index=False)


# ── 3. Reset all grid filters ────────────────────────────────
@app.callback(
    Output("main-grid", "filterModel"),
    Input("reset-btn",  "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return {}  # empty dict clears all column filters


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
