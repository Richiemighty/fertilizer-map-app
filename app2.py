import io
import re
import html
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import folium
import branca.colormap as cm
from streamlit_folium import st_folium


# =====================================================
# App configuration
# =====================================================

st.set_page_config(
    page_title="HydroGEM fertilizer opportunity map",
    page_icon="🌍",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

DEFAULT_ENERGY_DATASET_PATH = APP_DIR / "Fertilizer Plants_AHDS.xlsx"
DEFAULT_WATER_DATASET_PATH = APP_DIR / "Water availability.xlsx"


# =====================================================
# Styling
# =====================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            font-size: 1rem;
            color: #4B5563;
            margin-bottom: 1.1rem;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.04);
            min-height: 130px;
        }

        .info-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.04);
            margin-bottom: 12px;
        }

        .soft-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
        }

        .error-box {
            background: #FEF2F2;
            border: 1px solid #FCA5A5;
            border-radius: 14px;
            padding: 16px;
            color: #7F1D1D;
        }

        .success-box {
            background: #F0FDF4;
            border: 1px solid #86EFAC;
            border-radius: 14px;
            padding: 16px;
            color: #14532D;
        }

        .small-muted {
            color: #6B7280;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Data models and errors
# =====================================================

@dataclass
class EnergyBundle:
    df_fert: pd.DataFrame
    df_wind: pd.DataFrame
    df_solar: pd.DataFrame
    report: dict
    source_name: str


@dataclass
class WaterBundle:
    df_water: pd.DataFrame
    report: dict
    source_name: str


class DataValidationError(Exception):
    def __init__(self, title, issues):
        self.title = title
        self.issues = issues
        super().__init__(title)


# =====================================================
# Sheet and column aliases
# =====================================================

ENERGY_SHEET_ALIASES = {
    "fertilizer": [
        "Fertilizer Plants",
        "Fertilizer",
        "Fertilizer Plant",
        "Plants"
    ],
    "wind": [
        "Wind Potential",
        "Wind",
        "Wind Data",
        "Wind Resource"
    ],
    "solar": [
        "Solar Potential",
        "Solar",
        "Solar Data",
        "Solar Projects"
    ]
}

WATER_SHEET_ALIASES = {
    "wastewater": [
        "Waste water facilities",
        "Wastewater facilities",
        "Waste Water Facilities",
        "Wastewater",
        "Waste Water",
        "Wastewater Plants"
    ],
    "dams": [
        "Dams",
        "Dam",
        "Reservoirs"
    ],
    "desalination": [
        "Desalination plants",
        "Desalination Plants",
        "Desalination",
        "Desalination facilities"
    ]
}

FERTILIZER_COLUMN_ALIASES = {
    "Name": [
        "Name",
        "Plant Name",
        "Fertilizer Plant",
        "Fertilizer Plant Name"
    ],
    "Country": [
        "Country"
    ],
    "Latitude": [
        "Latitude",
        "Lat"
    ],
    "Longitude": [
        "Longitude",
        "Long",
        "Lng",
        "Lon"
    ],
    "Production_tpa": [
        "Production (tons/ anum)",
        "Production (tons/ annum)",
        "Production tons annum",
        "Production tons per annum",
        "Production",
        "Production_tpa",
        "Capacity",
        "Capacity_tpa"
    ]
}

WIND_COLUMN_ALIASES = {
    "Country": [
        "Country"
    ],
    "Region": [
        "Region",
        "Location",
        "Wind Region",
        "Site"
    ],
    "Latitude": [
        "Latitude",
        "Lat"
    ],
    "Longitude": [
        "Longitude",
        "Long",
        "Lng",
        "Lon"
    ],
    "Wind_Speed_mps_100m": [
        "Wind Speed (m/s) at 100m",
        "Wind Speed",
        "Wind Speed m/s",
        "Wind Speed at 100m",
        "Wind_Speed_mps_100m"
    ],
    "Wind_Density_wm2": [
        "Wind Power Density (W/m²)",
        "Wind Power Density (W/m2)",
        "Wind Density",
        "Wind Power Density",
        "Wind_Density_wm2"
    ]
}

SOLAR_COLUMN_ALIASES = {
    "Country": [
        "Country"
    ],
    "Solar_Site": [
        "Site",
        "Solar Site",
        "Location",
        "Project",
        "Solar_Site"
    ],
    "Latitude": [
        "Latitude",
        "Lat"
    ],
    "Longitude": [
        "Longitude",
        "Long",
        "Lng",
        "Lon"
    ],
    "Solar_Capacity_MW": [
        "Production Capacity(MW)",
        "Production Capacity (MW)",
        "Solar Capacity",
        "Solar Capacity MW",
        "Capacity MW",
        "Solar_Capacity_MW"
    ]
}

WASTEWATER_COLUMN_ALIASES = {
    "Country": [
        "Country"
    ],
    "Water_Name": [
        "Waste water Facility Name",
        "Wastewater Facility Name",
        "Facility Name",
        "Site Name",
        "Name"
    ],
    "Primary_Source": [
        "Primary Source",
        "Water Source",
        "Source Type"
    ],
    "Capacity_Value": [
        "Capacity (m³/d)",
        "Capacity (m3/d)",
        "Capacity (m3/day)",
        "Capacity m3 day",
        "Capacity",
        "Capacity_m3_day"
    ],
    "Latitude": [
        "Latitude",
        "Lat"
    ],
    "Longitude": [
        "Longitude",
        "Long",
        "Lng",
        "Lon"
    ],
    "Source": [
        "Source",
        "Reference"
    ]
}

DAMS_COLUMN_ALIASES = {
    "Country": [
        "Country"
    ],
    "Water_Name": [
        "Site Name",
        "Dam Name",
        "Reservoir Name",
        "Name"
    ],
    "Primary_Source": [
        "Water Souce",
        "Water Source",
        "Source Type"
    ],
    "Capacity_Value": [
        "Capacity MCM",
        "Capacity_MCM",
        "Capacity",
        "Storage Capacity",
        "Reservoir Capacity"
    ],
    "Use": [
        "Use",
        "Purpose"
    ],
    "Latitude": [
        "Latitude",
        "Lat"
    ],
    "Longitude": [
        "Longitude",
        "Long",
        "Lng",
        "Lon"
    ],
    "Source": [
        "Source",
        "Reference"
    ]
}

DESALINATION_COLUMN_ALIASES = {
    "Country": [
        "Country"
    ],
    "Water_Name": [
        "DesalinationPlantName",
        "Desalination Plant Name",
        "Plant Name",
        "Site Name",
        "Name"
    ],
    "Primary_Source": [
        "Primary Source",
        "Water Source",
        "Source Type"
    ],
    "Use": [
        "Use",
        "Purpose"
    ],
    "Capacity_Value": [
        "Capacity (m3/day)",
        "Capacity (m³/day)",
        "Capacity (m3/d)",
        "Capacity",
        "Capacity_m3_day"
    ],
    "Latitude": [
        "Latitude",
        "Lat"
    ],
    "Longitude": [
        "Longitude",
        "Long",
        "Lng",
        "Lon"
    ],
    "Source": [
        "Source",
        "Reference"
    ]
}

# =====================================================
# Utility functions
# =====================================================

def escape_text(value):
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def clean_text(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    return value if value else "Unknown"


def normalize_column_name(value):
    text = str(value).strip().lower()
    text = text.replace("²", "2")
    text = text.replace("³", "3")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_matching_sheet(sheet_names, possible_names):
    normalized_sheets = {
        normalize_column_name(sheet): sheet
        for sheet in sheet_names
    }

    for possible_name in possible_names:
        key = normalize_column_name(possible_name)
        if key in normalized_sheets:
            return normalized_sheets[key]

    return None


def align_columns(df, alias_map, required_columns, sheet_display_name):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    normalized_to_actual = {
        normalize_column_name(col): col
        for col in df.columns
    }

    rename_map = {}
    missing_columns = []

    for canonical_col, aliases in alias_map.items():
        found_col = None

        for alias in aliases:
            alias_key = normalize_column_name(alias)

            if alias_key in normalized_to_actual:
                found_col = normalized_to_actual[alias_key]
                break

        if found_col is None:
            if canonical_col in required_columns:
                missing_columns.append(canonical_col)
            else:
                df[canonical_col] = np.nan
        else:
            rename_map[found_col] = canonical_col

    if missing_columns:
        readable_missing = ", ".join(missing_columns)
        available_columns = ", ".join([str(col) for col in df.columns])

        raise DataValidationError(
            title=f"Missing required columns in the '{sheet_display_name}' sheet",
            issues=[
                f"The following required column(s) could not be found: {readable_missing}.",
                f"Columns found in your sheet: {available_columns}.",
                "Rename your columns using the Data Guide tab, then upload the file again."
            ]
        )

    df = df.rename(columns=rename_map)

    # Ensure all canonical columns exist after renaming.
    for canonical_col in alias_map.keys():
        if canonical_col not in df.columns:
            df[canonical_col] = np.nan

    return df

def parse_number_or_range(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", "")
    text = text.replace("–", "-").replace("—", "-")

    numbers = re.findall(r"\d+\.?\d*", text)

    if len(numbers) == 0:
        return np.nan

    if len(numbers) == 1:
        return float(numbers[0])

    return (float(numbers[0]) + float(numbers[1])) / 2


def make_marker_size(series, min_size=6, max_size=28, use_log=True):
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)

    if values.dropna().empty:
        return pd.Series(min_size, index=series.index)

    if use_log:
        scaled_values = np.log1p(values)
    else:
        scaled_values = values

    min_value = scaled_values.min()
    max_value = scaled_values.max()

    if max_value == min_value:
        return pd.Series((min_size + max_size) / 2, index=series.index)

    marker_sizes = min_size + (
        (scaled_values - min_value) / (max_value - min_value)
    ) * (max_size - min_size)

    return marker_sizes.fillna(min_size).clip(lower=min_size, upper=max_size)


def haversine_distance_km(lat1, lon1, lat2, lon2):
    radius_km = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return radius_km * c


def minmax_score(series):
    values = pd.to_numeric(series, errors="coerce")

    if values.dropna().empty:
        return pd.Series(0, index=series.index)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return pd.Series(100, index=series.index)

    return ((values - min_value) / (max_value - min_value) * 100).fillna(0)


def proximity_score(distance_series, max_distance_km):
    values = pd.to_numeric(distance_series, errors="coerce")

    if max_distance_km <= 0:
        max_distance_km = 1

    score = 100 * (1 - (values / max_distance_km))
    return score.clip(lower=0, upper=100).fillna(0)


def classify_score(score):
    if score >= 75:
        return "High priority"
    if score >= 50:
        return "Moderate priority"
    return "Lower priority"


def format_number(value, decimals=0):
    if pd.isna(value):
        return "N/A"

    if decimals == 0:
        return f"{float(value):,.0f}"

    return f"{float(value):,.{decimals}f}"


def safe_range_slider(label, series, default_label=None):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        st.sidebar.caption(f"{label}: no valid values available")
        return None, None

    min_value = int(np.floor(values.min()))
    max_value = int(np.ceil(values.max()))

    if min_value == max_value:
        st.sidebar.caption(f"{label}: only one value available ({min_value:,})")
        return min_value, max_value

    return st.sidebar.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value)
    )


def apply_numeric_range_filter(df, column, selected_range):
    if df.empty:
        return df

    if selected_range[0] is None or selected_range[1] is None:
        return df

    return df[df[column].between(selected_range[0], selected_range[1])].copy()


# =====================================================
# Energy data cleaning
# =====================================================

def clean_fertilizer_sheet(raw_df, report):
    df = align_columns(
        raw_df,
        FERTILIZER_COLUMN_ALIASES,
        required_columns=["Name", "Country", "Latitude", "Longitude", "Production_tpa"],
        sheet_display_name="Fertilizer Plants"
    )

    raw_rows = len(df)
    df = df.dropna(how="all").copy()

    df["Name"] = df["Name"].apply(clean_text)
    df["Country"] = df["Country"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Production_tpa"] = df["Production_tpa"].apply(parse_number_or_range)

    valid_mask = (
        df["Name"].ne("Unknown")
        & df["Country"].ne("Unknown")
        & df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
        & df["Production_tpa"].notna()
        & (df["Production_tpa"] > 0)
    )

    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()

    if df.empty:
        raise DataValidationError(
            title="No valid fertilizer plant rows found",
            issues=[
                "The Fertilizer Plants sheet exists, but no row could be used.",
                "Each valid row must have Name, Country, Latitude, Longitude, and positive Production.",
                "Latitude and Longitude must be numeric."
            ]
        )

    df["Hover_Text"] = (
        "<b>" + df["Name"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Production: " + df["Production_tpa"].round(0).astype(int).astype(str)
        + " tons/year"
    )

    df["Marker_Size"] = make_marker_size(
        df["Production_tpa"],
        min_size=7,
        max_size=32,
        use_log=True
    )

    report["rows"]["Fertilizer Plants"] = {
        "raw_rows": raw_rows,
        "usable_rows": len(df),
        "dropped_rows": dropped_rows
    }

    return df


def clean_wind_sheet(raw_df, report):
    df = align_columns(
        raw_df,
        WIND_COLUMN_ALIASES,
        required_columns=["Country", "Region", "Latitude", "Longitude", "Wind_Density_wm2"],
        sheet_display_name="Wind Potential"
    )

    raw_rows = len(df)
    df = df.dropna(how="all").copy()

    df["Country"] = df["Country"].apply(clean_text)
    df["Region"] = df["Region"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Wind_Speed_mps_100m"] = df["Wind_Speed_mps_100m"].apply(parse_number_or_range)
    df["Wind_Density_wm2"] = df["Wind_Density_wm2"].apply(parse_number_or_range)

    valid_mask = (
        df["Country"].ne("Unknown")
        & df["Region"].ne("Unknown")
        & df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
        & df["Wind_Density_wm2"].notna()
        & (df["Wind_Density_wm2"] > 0)
    )

    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()

    if df.empty:
        raise DataValidationError(
            title="No valid wind potential rows found",
            issues=[
                "The Wind Potential sheet exists, but no row could be used.",
                "Each valid row must have Country, Region, Latitude, Longitude, and positive Wind Power Density.",
                "Wind Power Density may be a number or a range such as 450-520."
            ]
        )

    df["Hover_Text"] = (
        "<b>" + df["Region"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Wind speed: " + df["Wind_Speed_mps_100m"].round(2).astype(str)
        + " m/s"
        + "<br>Wind density: " + df["Wind_Density_wm2"].round(0).astype(int).astype(str)
        + " W/m²"
    )

    df["Marker_Size"] = make_marker_size(
        df["Wind_Density_wm2"],
        min_size=6,
        max_size=24,
        use_log=True
    )

    report["rows"]["Wind Potential"] = {
        "raw_rows": raw_rows,
        "usable_rows": len(df),
        "dropped_rows": dropped_rows
    }

    return df


def clean_solar_sheet(raw_df, report):
    df = align_columns(
        raw_df,
        SOLAR_COLUMN_ALIASES,
        required_columns=["Country", "Solar_Site", "Latitude", "Longitude", "Solar_Capacity_MW"],
        sheet_display_name="Solar Potential"
    )

    raw_rows = len(df)
    df = df.dropna(how="all").copy()

    df["Country"] = df["Country"].apply(clean_text)
    df["Solar_Site"] = df["Solar_Site"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Solar_Capacity_MW"] = df["Solar_Capacity_MW"].apply(parse_number_or_range)

    valid_mask = (
        df["Country"].ne("Unknown")
        & df["Solar_Site"].ne("Unknown")
        & df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
        & df["Solar_Capacity_MW"].notna()
        & (df["Solar_Capacity_MW"] > 0)
    )

    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()

    if df.empty:
        report["warnings"].append(
            "The Solar Potential sheet was found, but no valid solar rows could be used."
        )
        return pd.DataFrame()

    df["Hover_Text"] = (
        "<b>" + df["Solar_Site"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Solar capacity: " + df["Solar_Capacity_MW"].round(1).astype(str)
        + " MW"
    )

    df["Marker_Size"] = make_marker_size(
        df["Solar_Capacity_MW"],
        min_size=5,
        max_size=22,
        use_log=True
    )

    report["rows"]["Solar Potential"] = {
        "raw_rows": raw_rows,
        "usable_rows": len(df),
        "dropped_rows": dropped_rows
    }

    return df


# =====================================================
# Water data cleaning
# =====================================================

def clean_water_sheet(raw_df, alias_map, required_columns, sheet_display_name, water_type, capacity_unit, report):
    df = align_columns(
        raw_df,
        alias_map,
        required_columns=required_columns,
        sheet_display_name=sheet_display_name
    )

    raw_rows = len(df)
    df = df.dropna(how="all").copy()

    # Ensure optional columns exist, even where a sheet does not provide them.
    # Example: Desalination plants may not have Primary_Source or Use.
    optional_columns = {
        "Primary_Source": "Unknown",
        "Use": "Unknown",
        "Source": "Unknown"
    }

    for column_name, default_value in optional_columns.items():
        if column_name not in df.columns:
            df[column_name] = default_value

    df["Country"] = df["Country"].apply(clean_text)
    df["Water_Name"] = df["Water_Name"].apply(clean_text)
    df["Primary_Source"] = df["Primary_Source"].apply(clean_text)
    df["Use"] = df["Use"].apply(clean_text)
    df["Source"] = df["Source"].apply(clean_text)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Capacity_Value"] = df["Capacity_Value"].apply(parse_number_or_range)

    valid_mask = (
        df["Country"].ne("Unknown")
        & df["Water_Name"].ne("Unknown")
        & df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
        & df["Capacity_Value"].notna()
        & (df["Capacity_Value"] > 0)
    )

    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()

    if df.empty:
        report["warnings"].append(
            f"The {sheet_display_name} sheet was found, but no valid rows could be used."
        )
        return pd.DataFrame()

    df["Water_Type"] = water_type
    df["Capacity_Unit"] = capacity_unit

    df["Hover_Text"] = (
        "<b>" + df["Water_Name"].apply(escape_text) + "</b>"
        + "<br>Type: " + df["Water_Type"].apply(escape_text)
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Capacity: " + df["Capacity_Value"].round(2).astype(str)
        + " " + df["Capacity_Unit"].apply(escape_text)
        + "<br>Primary source: " + df["Primary_Source"].apply(escape_text)
        + "<br>Use: " + df["Use"].apply(escape_text)
    )

    report["rows"][sheet_display_name] = {
        "raw_rows": raw_rows,
        "usable_rows": len(df),
        "dropped_rows": dropped_rows
    }

    keep_columns = [
        "Country",
        "Water_Name",
        "Water_Type",
        "Primary_Source",
        "Use",
        "Capacity_Value",
        "Capacity_Unit",
        "Latitude",
        "Longitude",
        "Source",
        "Hover_Text"
    ]

    return df[keep_columns].copy()

def finalize_water_assets(df_water):
    if df_water.empty:
        return df_water

    frames = []

    for water_type, group in df_water.groupby("Water_Type"):
        group = group.copy()
        group["Type_Capacity_Score"] = minmax_score(group["Capacity_Value"])
        group["Marker_Size"] = make_marker_size(
            group["Capacity_Value"],
            min_size=5,
            max_size=24,
            use_log=True
        )
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


# =====================================================
# Dataset loaders
# =====================================================

@st.cache_data(show_spinner=False)
def load_energy_dataset(file_bytes, source_name):
    report = {
        "source_name": source_name,
        "sheets": {},
        "rows": {},
        "warnings": []
    }

    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError(
            title="Energy workbook could not be opened",
            issues=[
                "Upload a valid .xlsx Excel workbook.",
                "Do not upload CSV for the energy dataset.",
                "If the workbook is open or corrupted, save a fresh copy and upload again."
            ]
        )

    sheet_names = excel_file.sheet_names

    fertilizer_sheet = find_matching_sheet(sheet_names, ENERGY_SHEET_ALIASES["fertilizer"])
    wind_sheet = find_matching_sheet(sheet_names, ENERGY_SHEET_ALIASES["wind"])
    solar_sheet = find_matching_sheet(sheet_names, ENERGY_SHEET_ALIASES["solar"])

    missing_sheets = []

    if fertilizer_sheet is None:
        missing_sheets.append("Fertilizer Plants")

    if wind_sheet is None:
        missing_sheets.append("Wind Potential")

    if missing_sheets:
        raise DataValidationError(
            title="Required energy sheet(s) missing",
            issues=[
                f"Missing sheet(s): {', '.join(missing_sheets)}.",
                f"Sheets found: {', '.join(sheet_names)}.",
                "Use the Data Guide tab to structure the workbook correctly."
            ]
        )

    report["sheets"]["Fertilizer Plants"] = fertilizer_sheet
    report["sheets"]["Wind Potential"] = wind_sheet

    raw_fert = pd.read_excel(io.BytesIO(file_bytes), sheet_name=fertilizer_sheet)
    raw_wind = pd.read_excel(io.BytesIO(file_bytes), sheet_name=wind_sheet)

    df_fert = clean_fertilizer_sheet(raw_fert, report)
    df_wind = clean_wind_sheet(raw_wind, report)

    df_solar = pd.DataFrame()

    if solar_sheet is not None:
        report["sheets"]["Solar Potential"] = solar_sheet
        raw_solar = pd.read_excel(io.BytesIO(file_bytes), sheet_name=solar_sheet)
        df_solar = clean_solar_sheet(raw_solar, report)
    else:
        report["warnings"].append(
            "No Solar Potential sheet was found. Solar will be treated as optional."
        )

    return EnergyBundle(
        df_fert=df_fert,
        df_wind=df_wind,
        df_solar=df_solar,
        report=report,
        source_name=source_name
    )


@st.cache_data(show_spinner=False)
def load_water_dataset(file_bytes, source_name):
    report = {
        "source_name": source_name,
        "sheets": {},
        "rows": {},
        "warnings": []
    }

    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError(
            title="Water workbook could not be opened",
            issues=[
                "Upload a valid .xlsx Excel workbook.",
                "The water workbook should contain sheets such as Waste water facilities, Dams, and Desalination plants."
            ]
        )

    sheet_names = excel_file.sheet_names

    wastewater_sheet = find_matching_sheet(sheet_names, WATER_SHEET_ALIASES["wastewater"])
    dams_sheet = find_matching_sheet(sheet_names, WATER_SHEET_ALIASES["dams"])
    desalination_sheet = find_matching_sheet(sheet_names, WATER_SHEET_ALIASES["desalination"])

    frames = []

    if wastewater_sheet is not None:
        report["sheets"]["Waste water facilities"] = wastewater_sheet
        raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=wastewater_sheet)

        try:
            frames.append(
                clean_water_sheet(
                    raw_df=raw_df,
                    alias_map=WASTEWATER_COLUMN_ALIASES,
                    required_columns=["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"],
                    sheet_display_name="Waste water facilities",
                    water_type="Wastewater facility",
                    capacity_unit="m³/day",
                    report=report
                )
            )
        except DataValidationError as exc:
            report["warnings"].extend(exc.issues)
    else:
        report["warnings"].append("No Waste water facilities sheet was found.")

    if dams_sheet is not None:
        report["sheets"]["Dams"] = dams_sheet
        raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=dams_sheet)

        try:
            frames.append(
                clean_water_sheet(
                    raw_df=raw_df,
                    alias_map=DAMS_COLUMN_ALIASES,
                    required_columns=["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"],
                    sheet_display_name="Dams",
                    water_type="Dam",
                    capacity_unit="MCM",
                    report=report
                )
            )
        except DataValidationError as exc:
            report["warnings"].extend(exc.issues)
    else:
        report["warnings"].append("No Dams sheet was found.")

    if desalination_sheet is not None:
        report["sheets"]["Desalination plants"] = desalination_sheet
        raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=desalination_sheet)

        try:
            frames.append(
                clean_water_sheet(
                    raw_df=raw_df,
                    alias_map=DESALINATION_COLUMN_ALIASES,
                    required_columns=["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"],
                    sheet_display_name="Desalination plants",
                    water_type="Desalination plant",
                    capacity_unit="m³/day",
                    report=report
                )
            )
        except DataValidationError as exc:
            report["warnings"].extend(exc.issues)
    else:
        report["warnings"].append("No Desalination plants sheet was found.")

    frames = [frame for frame in frames if frame is not None and not frame.empty]

    if not frames:
        raise DataValidationError(
            title="No valid water availability rows found",
            issues=[
                "The water workbook was found, but no usable water asset row could be loaded.",
                "At least one of these sheets should be valid: Waste water facilities, Dams, Desalination plants.",
                "Each valid row must have Country, Asset Name, Capacity, Latitude, and Longitude."
            ]
        )

    df_water = pd.concat(frames, ignore_index=True)
    df_water = finalize_water_assets(df_water)

    return WaterBundle(
        df_water=df_water,
        report=report,
        source_name=source_name
    )


# =====================================================
# Asset preparation and scoring
# =====================================================

def prepare_renewable_assets(df_wind, df_solar, selected_renewable_types):
    frames = []

    if "Wind" in selected_renewable_types and not df_wind.empty:
        wind = df_wind.copy()
        wind["Asset_Name"] = wind["Region"]
        wind["Renewable_Type"] = "Wind"
        wind["Quality_Value"] = wind["Wind_Density_wm2"]
        wind["Quality_Unit"] = "W/m²"
        wind["Quality_Score"] = minmax_score(wind["Quality_Value"])

        frames.append(
            wind[[
                "Country",
                "Asset_Name",
                "Renewable_Type",
                "Quality_Value",
                "Quality_Unit",
                "Quality_Score",
                "Latitude",
                "Longitude",
                "Hover_Text",
                "Marker_Size"
            ]]
        )

    if "Solar" in selected_renewable_types and not df_solar.empty:
        solar = df_solar.copy()
        solar["Asset_Name"] = solar["Solar_Site"]
        solar["Renewable_Type"] = "Solar"
        solar["Quality_Value"] = solar["Solar_Capacity_MW"]
        solar["Quality_Unit"] = "MW"
        solar["Quality_Score"] = minmax_score(solar["Quality_Value"])

        frames.append(
            solar[[
                "Country",
                "Asset_Name",
                "Renewable_Type",
                "Quality_Value",
                "Quality_Unit",
                "Quality_Score",
                "Latitude",
                "Longitude",
                "Hover_Text",
                "Marker_Size"
            ]]
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def filter_water_assets(df_water, selected_water_types):
    if df_water.empty:
        return df_water

    return df_water[df_water["Water_Type"].isin(selected_water_types)].copy()


def match_best_asset(
    source_df,
    asset_df,
    name_col,
    type_col,
    country_col,
    quality_value_col,
    quality_unit_col,
    quality_score_col,
    max_distance_km,
    matching_rule,
    prefix
):
    output_columns = [
        f"Nearest_{prefix}_Name",
        f"Nearest_{prefix}_Type",
        f"Nearest_{prefix}_Country",
        f"{prefix}_Quality_Value",
        f"{prefix}_Quality_Unit",
        f"{prefix}_Quality_Score",
        f"Distance_to_{prefix}_km",
        f"{prefix}_Proximity_Score",
        f"{prefix}_Latitude",
        f"{prefix}_Longitude"
    ]

    if source_df.empty:
        return pd.DataFrame(columns=output_columns)

    if asset_df.empty:
        empty = pd.DataFrame(index=source_df.index)

        for col in output_columns:
            empty[col] = np.nan

        empty[f"{prefix}_Quality_Score"] = 0
        empty[f"{prefix}_Proximity_Score"] = 0

        return empty.reset_index(drop=True)

    results = []

    for _, source_row in source_df.iterrows():
        distances = haversine_distance_km(
            source_row["Latitude"],
            source_row["Longitude"],
            asset_df["Latitude"].values,
            asset_df["Longitude"].values
        )

        distance_scores = proximity_score(
            pd.Series(distances),
            max_distance_km=max_distance_km
        ).values

        quality_scores = pd.to_numeric(
            asset_df[quality_score_col],
            errors="coerce"
        ).fillna(0).values

        if matching_rule == "Nearest asset":
            selected_position = int(np.argmin(distances))
        else:
            fit_scores = (quality_scores * 0.50) + (distance_scores * 0.50)
            selected_position = int(np.argmax(fit_scores))

        selected_row = asset_df.iloc[selected_position]

        results.append({
            f"Nearest_{prefix}_Name": selected_row[name_col],
            f"Nearest_{prefix}_Type": selected_row[type_col],
            f"Nearest_{prefix}_Country": selected_row[country_col],
            f"{prefix}_Quality_Value": selected_row[quality_value_col],
            f"{prefix}_Quality_Unit": selected_row[quality_unit_col],
            f"{prefix}_Quality_Score": round(float(selected_row[quality_score_col]), 2),
            f"Distance_to_{prefix}_km": round(float(distances[selected_position]), 2),
            f"{prefix}_Proximity_Score": round(float(distance_scores[selected_position]), 2),
            f"{prefix}_Latitude": selected_row["Latitude"],
            f"{prefix}_Longitude": selected_row["Longitude"]
        })

    return pd.DataFrame(results)


def calculate_effective_weights(raw_weights, has_renewable, has_water):
    active_weights = {}

    active_weights["Production_Score"] = raw_weights["Production scale"]

    if has_renewable:
        active_weights["Renewable_Quality_Score"] = raw_weights["Renewable quality"]
        active_weights["Renewable_Proximity_Score"] = raw_weights["Renewable proximity"]

    if has_water:
        active_weights["Water_Quality_Score"] = raw_weights["Water availability"]
        active_weights["Water_Proximity_Score"] = raw_weights["Water proximity"]

    active_weights = {
        key: value for key, value in active_weights.items()
        if value > 0
    }

    total_weight = sum(active_weights.values())

    if total_weight == 0:
        return {"Production_Score": 1.0}

    return {
        key: value / total_weight
        for key, value in active_weights.items()
    }


def build_opportunity_table(
    df_fert,
    renewable_assets,
    water_assets,
    raw_weights,
    renewable_matching_rule,
    water_matching_rule,
    max_renewable_distance_km,
    max_water_distance_km
):
    if df_fert.empty:
        return pd.DataFrame()

    opportunity = df_fert[[
        "Name",
        "Country",
        "Production_tpa",
        "Latitude",
        "Longitude"
    ]].reset_index(drop=True).copy()

    opportunity["Production_Score"] = minmax_score(opportunity["Production_tpa"])

    renewable_match = match_best_asset(
        source_df=opportunity,
        asset_df=renewable_assets,
        name_col="Asset_Name",
        type_col="Renewable_Type",
        country_col="Country",
        quality_value_col="Quality_Value",
        quality_unit_col="Quality_Unit",
        quality_score_col="Quality_Score",
        max_distance_km=max_renewable_distance_km,
        matching_rule=renewable_matching_rule,
        prefix="Renewable"
    )

    water_match = match_best_asset(
        source_df=opportunity,
        asset_df=water_assets,
        name_col="Water_Name",
        type_col="Water_Type",
        country_col="Country",
        quality_value_col="Capacity_Value",
        quality_unit_col="Capacity_Unit",
        quality_score_col="Type_Capacity_Score",
        max_distance_km=max_water_distance_km,
        matching_rule=water_matching_rule,
        prefix="Water"
    )

    opportunity = pd.concat(
        [
            opportunity.reset_index(drop=True),
            renewable_match.reset_index(drop=True),
            water_match.reset_index(drop=True)
        ],
        axis=1
    )

    opportunity["Renewable_Quality_Score"] = pd.to_numeric(
        opportunity["Renewable_Quality_Score"],
        errors="coerce"
    ).fillna(0)

    opportunity["Renewable_Proximity_Score"] = pd.to_numeric(
        opportunity["Renewable_Proximity_Score"],
        errors="coerce"
    ).fillna(0)

    opportunity["Water_Quality_Score"] = pd.to_numeric(
        opportunity["Water_Quality_Score"],
        errors="coerce"
    ).fillna(0)

    opportunity["Water_Proximity_Score"] = pd.to_numeric(
        opportunity["Water_Proximity_Score"],
        errors="coerce"
    ).fillna(0)

    effective_weights = calculate_effective_weights(
        raw_weights=raw_weights,
        has_renewable=not renewable_assets.empty,
        has_water=not water_assets.empty
    )

    opportunity["Opportunity_Score"] = 0

    for score_col, weight in effective_weights.items():
        opportunity["Opportunity_Score"] += opportunity[score_col] * weight

    opportunity["Opportunity_Score"] = opportunity["Opportunity_Score"].round(2)
    opportunity["Priority_Band"] = opportunity["Opportunity_Score"].apply(classify_score)

    opportunity["Methodology_Weights"] = str({
        key: round(value, 3)
        for key, value in effective_weights.items()
    })

    return opportunity.sort_values(
        "Opportunity_Score",
        ascending=False
    ).reset_index(drop=True)


# =====================================================
# Mapping functions
# =====================================================

def make_colormap(values, colors):
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()

    if numeric_values.empty:
        return cm.LinearColormap(colors, vmin=0, vmax=1)

    vmin = float(numeric_values.min())
    vmax = float(numeric_values.max())

    if vmin == vmax:
        vmax = vmin + 1

    return cm.LinearColormap(colors, vmin=vmin, vmax=vmax)


def add_base_tiles(folium_map, show_labels=True):
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite view",
        overlay=False,
        control=True
    ).add_to(folium_map)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Clean map view",
        overlay=False,
        control=True
    ).add_to(folium_map)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street map view",
        overlay=False,
        control=True
    ).add_to(folium_map)

    if show_labels:
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
            attr="Labels © CARTO and OpenStreetMap contributors",
            name="Country and place labels",
            overlay=True,
            control=True,
            show=True
        ).add_to(folium_map)


def build_popup(title, rows):
    html_rows = ""

    for label, value in rows:
        html_rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;color:#6B7280;'>{escape_text(label)}</td>"
            f"<td style='padding:4px 8px;font-weight:600;'>{escape_text(value)}</td>"
            f"</tr>"
        )

    return f"""
    <div style="font-family: Arial, sans-serif; min-width: 250px;">
        <h4 style="margin:0 0 8px 0;">{escape_text(title)}</h4>
        <table style="border-collapse:collapse;width:100%;">
            {html_rows}
        </table>
    </div>
    """

def add_map_marker_css(folium_map):
    """
    Adds CSS that removes default Leaflet div-icon styling and keeps custom markers clean.
    """
    css = """
    <style>
        .hydrogem-div-icon {
            background: transparent !important;
            border: none !important;
        }

        .hydrogem-marker-inner {
            transform-origin: center center;
            transition: transform 0.12s ease-out;
        }

        .hydrogem-marker-svg {
            filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.45));
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(css))


def build_fixed_marker_icon(
    shape,
    color,
    size=26,
    border_color="#111827",
    label=None,
    label_color="#FFFFFF"
):
    """
    Builds fixed-pixel SVG markers.

    These markers stay visible as users zoom because they are rendered as
    screen-based HTML/SVG icons, not geographic-radius circles.
    """
    size = int(max(18, min(size, 52)))

    if shape == "circle":
        shape_svg = f"""
        <circle cx="50" cy="50" r="38"
            fill="{color}" stroke="{border_color}" stroke-width="6" />
        """

    elif shape == "triangle":
        shape_svg = f"""
        <polygon points="50,8 92,88 8,88"
            fill="{color}" stroke="{border_color}" stroke-width="6" />
        """

    elif shape == "diamond":
        shape_svg = f"""
        <polygon points="50,5 95,50 50,95 5,50"
            fill="{color}" stroke="{border_color}" stroke-width="6" />
        """

    elif shape == "square":
        shape_svg = f"""
        <rect x="15" y="15" width="70" height="70" rx="10"
            fill="{color}" stroke="{border_color}" stroke-width="6" />
        """

    elif shape == "pentagon":
        shape_svg = f"""
        <polygon points="50,6 94,38 78,92 22,92 6,38"
            fill="{color}" stroke="{border_color}" stroke-width="6" />
        """

    else:
        shape_svg = f"""
        <circle cx="50" cy="50" r="38"
            fill="{color}" stroke="{border_color}" stroke-width="6" />
        """

    label_svg = ""

    if label is not None:
        label_svg = f"""
        <text x="50" y="58"
            text-anchor="middle"
            font-size="42"
            font-weight="800"
            font-family="Arial, sans-serif"
            fill="{label_color}">
            {escape_text(label)}
        </text>
        """

    html_icon = f"""
    <div class="hydrogem-marker" style="
        width:{size}px;
        height:{size}px;
        position:relative;
    ">
        <div class="hydrogem-marker-inner" style="
            width:{size}px;
            height:{size}px;
            position:absolute;
            left:50%;
            top:50%;
            transform:translate(-50%, -50%) scale(1);
        ">
            <svg class="hydrogem-marker-svg"
                width="{size}"
                height="{size}"
                viewBox="0 0 100 100"
                xmlns="http://www.w3.org/2000/svg">
                {shape_svg}
                {label_svg}
            </svg>
        </div>
    </div>
    """

    return folium.DivIcon(
        html=html_icon,
        icon_size=(size, size),
        icon_anchor=(size / 2, size / 2),
        class_name="hydrogem-div-icon"
    )


def get_marker_px(row, fallback=24, min_px=20, max_px=44, multiplier=1.35):
    """
    Converts the existing Marker_Size value into a stronger fixed-pixel marker size.
    """
    try:
        value = row.get("Marker_Size", fallback)

        if pd.isna(value):
            value = fallback

        return int(np.clip(float(value) * multiplier, min_px, max_px))

    except Exception:
        return fallback


def add_marker_legend(folium_map):
    """
    Adds a proper on-map legend explaining each marker type.
    """
    legend_html = """
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(255,255,255,0.94);
        border: 1px solid #D1D5DB;
        border-radius: 14px;
        padding: 14px 16px;
        width: 285px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        font-family: Arial, sans-serif;
        color: #111827;
    ">
        <div style="font-weight:800; font-size:15px; margin-bottom:10px;">
            Map legend
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:15px; height:15px; background:#DC2626; border:2px solid #111827; border-radius:50%; display:inline-block;"></span>
            <span>Fertilizer plant</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="
                width:0; height:0;
                border-left:9px solid transparent;
                border-right:9px solid transparent;
                border-bottom:17px solid #2563EB;
                display:inline-block;
            "></span>
            <span>Wind potential</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="
                width:15px; height:15px;
                background:#F59E0B;
                border:2px solid #111827;
                transform:rotate(45deg);
                display:inline-block;
                margin-left:2px;
            "></span>
            <span>Solar potential</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:15px; height:15px; background:#059669; border:2px solid #064E3B; border-radius:50%; display:inline-block;"></span>
            <span>Wastewater facility</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:15px; height:15px; background:#0284C7; border:2px solid #0F172A; display:inline-block;"></span>
            <span>Dam / reservoir</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="
                width:16px; height:16px;
                background:#14B8A6;
                border:2px solid #0F172A;
                clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
                display:inline-block;
            "></span>
            <span>Desalination plant</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="
                width:22px; height:22px;
                background:#166534;
                border:2px solid white;
                border-radius:50%;
                color:white;
                display:inline-flex;
                align-items:center;
                justify-content:center;
                font-size:11px;
                font-weight:800;
                box-shadow:0 1px 4px rgba(0,0,0,0.35);
            ">1</span>
            <span>Ranked hotspot</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:28px; border-top:3px dashed #2563EB; display:inline-block;"></span>
            <span>Plant to renewable asset</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px;">
            <span style="width:28px; border-top:3px dashed #059669; display:inline-block;"></span>
            <span>Plant to water asset</span>
        </div>

        <div style="
            border-top:1px solid #E5E7EB;
            margin-top:10px;
            padding-top:8px;
            font-size:12px;
            color:#6B7280;
            line-height:1.35;
        ">
            Marker size reflects relative scale or capacity. Use the layer control to switch layers on or off.
        </div>
    </div>
    """

    folium_map.get_root().html.add_child(folium.Element(legend_html))


def add_zoom_marker_scaling(folium_map, base_zoom=3, growth_per_zoom=0.045, max_scale=1.45):
    """
    Makes markers remain visible when zooming.

    At minimum, markers keep their base pixel size.
    As the user zooms in, markers grow slightly up to max_scale.
    """
    map_name = folium_map.get_name()

    zoom_script = f"""
    <script>
        (function() {{
            var map = {map_name};

            function resizeHydroGEMMarkers() {{
                var zoom = map.getZoom();
                var scale = 1 + ((zoom - {base_zoom}) * {growth_per_zoom});

                if (scale < 1) {{
                    scale = 1;
                }}

                if (scale > {max_scale}) {{
                    scale = {max_scale};
                }}

                var markers = document.querySelectorAll('.hydrogem-marker-inner');

                markers.forEach(function(marker) {{
                    marker.style.transform = 'translate(-50%, -50%) scale(' + scale + ')';
                }});
            }}

            map.whenReady(resizeHydroGEMMarkers);
            map.on('zoomend', resizeHydroGEMMarkers);
            map.on('layeradd', resizeHydroGEMMarkers);
        }})();
    </script>
    """

    folium_map.get_root().html.add_child(folium.Element(zoom_script))

    
def build_folium_map(
    df_fert,
    renewable_assets,
    water_assets,
    opportunity,
    show_fertilizer,
    show_wind,
    show_solar,
    show_wastewater,
    show_dams,
    show_desalination,
    show_hotspots,
    show_connection_lines,
    show_country_labels,
    top_n_hotspots
):
    folium_map = folium.Map(
        location=[2.0, 20.0],
        zoom_start=3,
        tiles=None,
        control_scale=True
    )

    add_base_tiles(folium_map, show_labels=show_country_labels)
    add_map_marker_css(folium_map)

    fert_cmap = make_colormap(
        df_fert["Production_tpa"] if not df_fert.empty else pd.Series(dtype=float),
        ["#FEE2E2", "#EF4444", "#7F1D1D"]
    )

    renewable_cmap = make_colormap(
        renewable_assets["Quality_Score"] if not renewable_assets.empty else pd.Series(dtype=float),
        ["#DBEAFE", "#2563EB", "#1E3A8A"]
    )

    water_cmap = make_colormap(
        water_assets["Type_Capacity_Score"] if not water_assets.empty else pd.Series(dtype=float),
        ["#D1FAE5", "#059669", "#064E3B"]
    )

    # =================================================
    # Fertilizer plants
    # =================================================
    if show_fertilizer and not df_fert.empty:
        fertilizer_group = folium.FeatureGroup(
            name="Fertilizer plants",
            show=True
        )

        for _, row in df_fert.iterrows():
            popup = build_popup(
                row["Name"],
                [
                    ("Marker", "Red circle"),
                    ("Asset type", "Fertilizer plant"),
                    ("Country", row["Country"]),
                    ("Production", f"{format_number(row['Production_tpa'])} tons/year"),
                    ("Latitude", f"{row['Latitude']:.4f}"),
                    ("Longitude", f"{row['Longitude']:.4f}")
                ]
            )

            size = get_marker_px(
                row,
                fallback=26,
                min_px=22,
                max_px=46,
                multiplier=1.45
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"Fertilizer plant: {row['Name']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=390),
                icon=build_fixed_marker_icon(
                    shape="circle",
                    color=fert_cmap(row["Production_tpa"]),
                    size=size,
                    border_color="#111827"
                )
            ).add_to(fertilizer_group)

        fertilizer_group.add_to(folium_map)

    # =================================================
    # Renewable assets: wind and solar
    # =================================================
    if not renewable_assets.empty:
        wind_assets = renewable_assets[
            renewable_assets["Renewable_Type"] == "Wind"
        ].copy()

        solar_assets = renewable_assets[
            renewable_assets["Renewable_Type"] == "Solar"
        ].copy()

        if show_wind and not wind_assets.empty:
            wind_group = folium.FeatureGroup(
                name="Wind potential",
                show=True
            )

            for _, row in wind_assets.iterrows():
                popup = build_popup(
                    row["Asset_Name"],
                    [
                        ("Marker", "Blue triangle"),
                        ("Asset type", "Wind potential"),
                        ("Country", row["Country"]),
                        ("Quality", f"{format_number(row['Quality_Value'], 1)} {row['Quality_Unit']}"),
                        ("Quality score", f"{format_number(row['Quality_Score'], 1)}"),
                        ("Latitude", f"{row['Latitude']:.4f}"),
                        ("Longitude", f"{row['Longitude']:.4f}")
                    ]
                )

                size = get_marker_px(
                    row,
                    fallback=24,
                    min_px=22,
                    max_px=42,
                    multiplier=1.45
                )

                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    tooltip=f"Wind potential: {row['Asset_Name']} | {row['Country']}",
                    popup=folium.Popup(popup, max_width=390),
                    icon=build_fixed_marker_icon(
                        shape="triangle",
                        color=renewable_cmap(row["Quality_Score"]),
                        size=size,
                        border_color="#111827"
                    )
                ).add_to(wind_group)

            wind_group.add_to(folium_map)

        if show_solar and not solar_assets.empty:
            solar_group = folium.FeatureGroup(
                name="Solar potential",
                show=True
            )

            for _, row in solar_assets.iterrows():
                popup = build_popup(
                    row["Asset_Name"],
                    [
                        ("Marker", "Amber diamond"),
                        ("Asset type", "Solar potential"),
                        ("Country", row["Country"]),
                        ("Quality", f"{format_number(row['Quality_Value'], 1)} {row['Quality_Unit']}"),
                        ("Quality score", f"{format_number(row['Quality_Score'], 1)}"),
                        ("Latitude", f"{row['Latitude']:.4f}"),
                        ("Longitude", f"{row['Longitude']:.4f}")
                    ]
                )

                size = get_marker_px(
                    row,
                    fallback=24,
                    min_px=22,
                    max_px=40,
                    multiplier=1.45
                )

                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    tooltip=f"Solar potential: {row['Asset_Name']} | {row['Country']}",
                    popup=folium.Popup(popup, max_width=390),
                    icon=build_fixed_marker_icon(
                        shape="diamond",
                        color="#F59E0B",
                        size=size,
                        border_color="#111827"
                    )
                ).add_to(solar_group)

            solar_group.add_to(folium_map)

    # =================================================
    # Water assets
    # =================================================
    if not water_assets.empty:
        water_layers = [
            ("Wastewater facility", show_wastewater, "Wastewater facilities", "circle", "#059669"),
            ("Dam", show_dams, "Dams and reservoirs", "square", "#0284C7"),
            ("Desalination plant", show_desalination, "Desalination plants", "pentagon", "#14B8A6")
        ]

        for water_type, should_show, layer_name, marker_shape, default_color in water_layers:
            layer_df = water_assets[
                water_assets["Water_Type"] == water_type
            ].copy()

            if not should_show or layer_df.empty:
                continue

            water_group = folium.FeatureGroup(
                name=layer_name,
                show=True
            )

            for _, row in layer_df.iterrows():
                if water_type == "Wastewater facility":
                    marker_description = "Green circle"
                    marker_color = water_cmap(row["Type_Capacity_Score"])

                elif water_type == "Dam":
                    marker_description = "Blue square"
                    marker_color = default_color

                else:
                    marker_description = "Teal pentagon"
                    marker_color = default_color

                popup = build_popup(
                    row["Water_Name"],
                    [
                        ("Marker", marker_description),
                        ("Asset type", row["Water_Type"]),
                        ("Country", row["Country"]),
                        ("Capacity", f"{format_number(row['Capacity_Value'], 2)} {row['Capacity_Unit']}"),
                        ("Capacity score", f"{format_number(row['Type_Capacity_Score'], 1)}"),
                        ("Primary source", row["Primary_Source"]),
                        ("Use", row["Use"]),
                        ("Latitude", f"{row['Latitude']:.4f}"),
                        ("Longitude", f"{row['Longitude']:.4f}")
                    ]
                )

                size = get_marker_px(
                    row,
                    fallback=23,
                    min_px=20,
                    max_px=40,
                    multiplier=1.45
                )

                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    tooltip=f"{row['Water_Type']}: {row['Water_Name']} | {row['Country']}",
                    popup=folium.Popup(popup, max_width=410),
                    icon=build_fixed_marker_icon(
                        shape=marker_shape,
                        color=marker_color,
                        size=size,
                        border_color="#0F172A"
                    )
                ).add_to(water_group)

            water_group.add_to(folium_map)

    # =================================================
    # Connection lines
    # =================================================
    if show_connection_lines and not opportunity.empty:
        line_group = folium.FeatureGroup(
            name="Hotspot connection lines",
            show=True
        )

        for _, row in opportunity.head(top_n_hotspots).iterrows():
            plant_point = [row["Latitude"], row["Longitude"]]

            if pd.notna(row.get("Renewable_Latitude")) and pd.notna(row.get("Renewable_Longitude")):
                folium.PolyLine(
                    locations=[
                        plant_point,
                        [row["Renewable_Latitude"], row["Renewable_Longitude"]]
                    ],
                    color="#2563EB",
                    weight=3,
                    opacity=0.7,
                    dash_array="7,7",
                    tooltip="Blue dashed line: fertilizer plant to selected renewable asset"
                ).add_to(line_group)

            if pd.notna(row.get("Water_Latitude")) and pd.notna(row.get("Water_Longitude")):
                folium.PolyLine(
                    locations=[
                        plant_point,
                        [row["Water_Latitude"], row["Water_Longitude"]]
                    ],
                    color="#059669",
                    weight=3,
                    opacity=0.7,
                    dash_array="5,7",
                    tooltip="Green dashed line: fertilizer plant to selected water asset"
                ).add_to(line_group)

        line_group.add_to(folium_map)

    # =================================================
    # Ranked hotspots
    # =================================================
    if show_hotspots and not opportunity.empty:
        hotspot_group = folium.FeatureGroup(
            name="Top ranked hotspots",
            show=True
        )

        for rank, (_, row) in enumerate(opportunity.head(top_n_hotspots).iterrows(), start=1):
            popup = build_popup(
                f"#{rank} {row['Name']}",
                [
                    ("Marker", "Green numbered badge"),
                    ("Asset type", "Ranked hotspot"),
                    ("Country", row["Country"]),
                    ("Opportunity score", f"{format_number(row['Opportunity_Score'], 2)}"),
                    ("Priority band", row["Priority_Band"]),
                    ("Production", f"{format_number(row['Production_tpa'])} tons/year"),
                    ("Renewable asset", row.get("Nearest_Renewable_Name", "N/A")),
                    ("Renewable type", row.get("Nearest_Renewable_Type", "N/A")),
                    ("Renewable distance", f"{format_number(row.get('Distance_to_Renewable_km'), 1)} km"),
                    ("Water asset", row.get("Nearest_Water_Name", "N/A")),
                    ("Water type", row.get("Nearest_Water_Type", "N/A")),
                    ("Water distance", f"{format_number(row.get('Distance_to_Water_km'), 1)} km")
                ]
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"#{rank} hotspot: {row['Name']}",
                popup=folium.Popup(popup, max_width=420),
                icon=build_fixed_marker_icon(
                    shape="circle",
                    color="#166534",
                    size=36,
                    border_color="#FFFFFF",
                    label=str(rank),
                    label_color="#FFFFFF"
                )
            ).add_to(hotspot_group)

        hotspot_group.add_to(folium_map)

    # =================================================
    # Controls, legend and zoom behaviour
    # =================================================
    folium.LayerControl(collapsed=False).add_to(folium_map)

    add_marker_legend(folium_map)

    add_zoom_marker_scaling(
        folium_map,
        base_zoom=3,
        growth_per_zoom=0.045,
        max_scale=1.45
    )

    return folium_map

def folium_map_to_html(folium_map):
    return folium_map.get_root().render()


# =====================================================
# Template downloads
# =====================================================

def build_energy_template_workbook():
    output = io.BytesIO()

    fertilizer_template = pd.DataFrame({
        "Name": ["Example Fertilizer Plant"],
        "Country": ["Nigeria"],
        "Latitude": [6.5244],
        "Longitude": [3.3792],
        "Production (tons/ anum)": [1200000]
    })

    wind_template = pd.DataFrame({
        "Country": ["Nigeria"],
        "Region": ["Example Wind Region"],
        "Latitude": [11.9964],
        "Longitude": [8.5167],
        "Wind Speed (m/s) at 100m": [7.2],
        "Wind Power Density (W/m²)": ["450-520"]
    })

    solar_template = pd.DataFrame({
        "Country": ["Nigeria"],
        "Site": ["Example Solar Site"],
        "Latitude": [9.0765],
        "Longitude": [7.3986],
        "Production Capacity(MW)": [100]
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        fertilizer_template.to_excel(writer, sheet_name="Fertilizer Plants", index=False)
        wind_template.to_excel(writer, sheet_name="Wind Potential", index=False)
        solar_template.to_excel(writer, sheet_name="Solar Potential", index=False)

    return output.getvalue()


def build_water_template_workbook():
    output = io.BytesIO()

    wastewater_template = pd.DataFrame({
        "Country": ["Egypt"],
        "Waste water Facility Name": ["Example Wastewater Facility"],
        "Primary Source": ["Municipal wastewater"],
        "Capacity (m³/d)": [250000],
        "Latitude": [30.0444],
        "Longitude": [31.2357],
        "Source": ["Example source"]
    })

    dams_template = pd.DataFrame({
        "Country": ["Ghana"],
        "Site Name": ["Example Dam"],
        "Water Souce": ["River"],
        "Capacity MCM": [1200],
        "Use": ["Hydropower / Irrigation"],
        "Latitude": [6.3000],
        "Longitude": [-0.0500],
        "Source": ["Example source"]
    })

    desalination_template = pd.DataFrame({
        "Country": ["Morocco"],
        "DesalinationPlantName": ["Example Desalination Plant"],
        "Capacity (m3/day)": [100000],
        "Latitude": [33.5731],
        "Longitude": [-7.5898],
        "Source": ["Example source"]
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        wastewater_template.to_excel(writer, sheet_name="Waste water facilities", index=False)
        dams_template.to_excel(writer, sheet_name="Dams", index=False)
        desalination_template.to_excel(writer, sheet_name="Desalination plants", index=False)

    return output.getvalue()


# =====================================================
# Data guide
# =====================================================
def show_dictionary(raw_weights=None, effective_weights=None):
    st.subheader("Dictionary and methodology guide")

    st.markdown(
        """
        This page explains the key terms used in the dashboard. The scoring model is a first-pass screening tool. 
        It helps identify fertilizer production locations that may have stronger overlap with renewable energy and water availability.
        """
    )

    st.markdown("### Core terms")

    core_terms = pd.DataFrame({
        "Term": [
            "Opportunity Score",
            "Hotspot",
            "Fertilizer Plant",
            "Renewable Asset",
            "Water Asset",
            "Matched Asset",
            "Priority Band"
        ],
        "Meaning": [
            "The final 0–100 score used to rank fertilizer plants based on production scale, renewable fit, and water fit.",
            "A high-ranking fertilizer plant location after the selected filters and methodology settings are applied.",
            "An existing fertilizer production facility in the dataset.",
            "A wind or solar location used to assess renewable energy suitability.",
            "A dam, wastewater facility, or desalination plant used to assess water availability.",
            "The renewable or water asset selected by the model as the best fit for a fertilizer plant.",
            "A simple label that groups scores into High priority, Moderate priority, or Lower priority."
        ],
        "How to interpret it": [
            "Higher is better. A higher score means the location appears more promising for further review.",
            "Hotspots are not final investment decisions. They are locations worth investigating first.",
            "Large fertilizer plants may score better because they represent larger conversion or decarbonisation opportunities.",
            "Renewable assets help test whether green power supply may be available near a fertilizer location.",
            "Water assets help test whether water supply may support green hydrogen or green fertilizer production.",
            "The selected asset depends on the matching rule chosen in the sidebar.",
            "Use this for quick prioritisation, not as a substitute for technical feasibility."
        ]
    })

    st.dataframe(
        core_terms,
        width="stretch",
        hide_index=True
    )

    st.markdown("### Score components")

    score_terms = pd.DataFrame({
        "Score": [
            "Production Score",
            "Renewable Quality Score",
            "Renewable Proximity Score",
            "Water Quality Score",
            "Water Proximity Score"
        ],
        "What it measures": [
            "How large the fertilizer plant is compared with other selected fertilizer plants.",
            "How strong the matched renewable resource is.",
            "How close the fertilizer plant is to the matched renewable asset.",
            "How large or strong the matched water asset is.",
            "How close the fertilizer plant is to the matched water asset."
        ],
        "Current logic": [
            "Calculated by normalising fertilizer production capacity to a 0–100 score.",
            "For wind, this uses wind power density. For solar, this uses solar capacity.",
            "Calculated using distance. Closer assets receive higher scores.",
            "Calculated by normalising water asset capacity within its water asset type.",
            "Calculated using distance. Closer water assets receive higher scores."
        ],
        "Interpretation": [
            "A score of 100 means the largest selected fertilizer plant. A score of 0 means the smallest selected plant.",
            "A higher score means the renewable asset is stronger relative to other selected renewable assets.",
            "A higher score means the renewable asset is closer to the fertilizer plant.",
            "A higher score means the water asset has stronger capacity relative to similar water assets.",
            "A higher score means the water asset is closer to the fertilizer plant."
        ]
    })

    st.dataframe(
        score_terms,
        width="stretch",
        hide_index=True
    )

    st.markdown("### Methodology controls")

    methodology_terms = pd.DataFrame({
        "Control": [
            "Renewable resources used in scoring",
            "Water resources used in scoring",
            "Renewable matching rule",
            "Water matching rule",
            "Maximum useful renewable distance",
            "Maximum useful water distance",
            "Score weights",
            "Minimum opportunity score",
            "Number of hotspots to show"
        ],
        "What it does": [
            "Allows users to choose whether wind, solar, or both should influence the opportunity score.",
            "Allows users to choose whether dams, wastewater facilities, desalination plants, or all water assets should influence the score.",
            "Controls how the app selects the renewable asset matched to each fertilizer plant.",
            "Controls how the app selects the water asset matched to each fertilizer plant.",
            "Sets the distance threshold used to score renewable proximity.",
            "Sets the distance threshold used to score water proximity.",
            "Controls how much each score component contributes to the final Opportunity Score.",
            "Filters out locations below the selected Opportunity Score.",
            "Controls how many top-ranked locations are shown as hotspots."
        ],
        "Example": [
            "Selecting only Wind means solar will not affect the score.",
            "Selecting only Dams means wastewater and desalination will not affect the score.",
            "Nearest asset chooses the closest renewable asset. Best quality-proximity fit balances quality and distance.",
            "Nearest asset chooses the closest water asset. Best quality-proximity fit balances capacity and distance.",
            "If set to 600 km, renewable assets close to the plant score better than assets far away.",
            "If set to 300 km, water assets close to the plant score better than assets far away.",
            "If water is more important, increase the water availability and water proximity weights.",
            "If set to 60, only locations scoring 60 and above remain visible in the ranking.",
            "If set to 10, only the top 10 ranked locations are highlighted as hotspots."
        ]
    })

    st.dataframe(
        methodology_terms,
        width="stretch",
        hide_index=True
    )

    st.markdown("### Formula used by the dashboard")

    st.markdown(
        """
        The final score is calculated as:

        ```text
        Opportunity Score =
        Production Score × Production Weight
        + Renewable Quality Score × Renewable Quality Weight
        + Renewable Proximity Score × Renewable Proximity Weight
        + Water Quality Score × Water Availability Weight
        + Water Proximity Score × Water Proximity Weight
        ```

        The app automatically rebalances the weights when a component is unavailable. 
        For example, if no water dataset is loaded, the water weights are removed and the remaining weights are normalised.
        """
    )

    if raw_weights is not None:
        st.markdown("### Current raw weights selected in the sidebar")

        raw_weight_df = pd.DataFrame({
            "Component": list(raw_weights.keys()),
            "Selected Weight": list(raw_weights.values())
        })

        st.dataframe(
            raw_weight_df,
            width="stretch",
            hide_index=True
        )

    if effective_weights is not None:
        st.markdown("### Current effective weights used by the model")

        effective_weight_df = pd.DataFrame({
            "Score Component": list(effective_weights.keys()),
            "Effective Weight (%)": [round(value * 100, 2) for value in effective_weights.values()]
        })

        st.dataframe(
            effective_weight_df,
            width="stretch",
            hide_index=True
        )

    st.markdown("### Important limitations")

    limitations = pd.DataFrame({
        "Limitation": [
            "The score is not a feasibility study",
            "Distance is straight-line distance",
            "Water capacity units differ",
            "Solar and wind quality are not the same metric",
            "Data quality affects the result",
            "Regulatory and commercial issues are not included"
        ],
        "Why it matters": [
            "A high score only means the site should be reviewed further.",
            "The model does not yet calculate road distance, grid distance, pipeline distance, or terrain constraints.",
            "Dams may be measured in million cubic metres, while wastewater and desalination may be measured in cubic metres per day.",
            "Wind uses wind power density, while solar uses capacity. The app normalises them for screening, but they are technically different.",
            "Bad coordinates, missing capacity values, or old data can affect rankings.",
            "The model does not yet assess permits, land, offtake, tariffs, water rights, or capital cost."
        ]
    })

    st.dataframe(
        limitations,
        width="stretch",
        hide_index=True
    )
    
def show_data_guide():
    st.subheader("Data structure guide")

    st.markdown(
        """
        The app works best with two Excel workbooks:

        1. **Energy workbook** containing fertilizer, wind, and optional solar data.
        2. **Water workbook** containing wastewater facilities, dams, and desalination plants.

        The app validates each workbook before processing. If a visitor uploads the wrong file, the app will show what is missing and how to fix it.
        """
    )

    st.markdown("### Energy workbook")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Required sheet: Fertilizer Plants")

        st.dataframe(
            pd.DataFrame({
                "Column": [
                    "Name",
                    "Country",
                    "Latitude",
                    "Longitude",
                    "Production (tons/ anum)"
                ],
                "Required": [
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes"
                ],
                "Example": [
                    "Example Fertilizer Plant",
                    "Nigeria",
                    "6.5244",
                    "3.3792",
                    "1200000"
                ]
            }),
            width="stretch",
            hide_index=True
        )

    with col2:
        st.markdown("#### Required sheet: Wind Potential")

        st.dataframe(
            pd.DataFrame({
                "Column": [
                    "Country",
                    "Region",
                    "Latitude",
                    "Longitude",
                    "Wind Speed (m/s) at 100m",
                    "Wind Power Density (W/m²)"
                ],
                "Required": [
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Optional",
                    "Yes"
                ],
                "Example": [
                    "Nigeria",
                    "Kano Wind Corridor",
                    "11.9964",
                    "8.5167",
                    "7.2",
                    "450-520"
                ]
            }),
            width="stretch",
            hide_index=True
        )

    st.markdown("#### Optional sheet: Solar Potential")

    st.dataframe(
        pd.DataFrame({
            "Column": [
                "Country",
                "Site",
                "Latitude",
                "Longitude",
                "Production Capacity(MW)"
            ],
            "Required": [
                "Yes, if sheet is used",
                "Yes, if sheet is used",
                "Yes, if sheet is used",
                "Yes, if sheet is used",
                "Yes, if sheet is used"
            ],
            "Example": [
                "Nigeria",
                "Example Solar Site",
                "9.0765",
                "7.3986",
                "100"
            ]
        }),
        width="stretch",
        hide_index=True
    )

    st.download_button(
        label="Download energy workbook template",
        data=build_energy_template_workbook(),
        file_name="energy_dataset_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    st.markdown("### Water workbook")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Sheet: Waste water facilities")

        st.dataframe(
            pd.DataFrame({
                "Column": [
                    "Country",
                    "Waste water Facility Name",
                    "Primary Source",
                    "Capacity (m³/d)",
                    "Latitude",
                    "Longitude",
                    "Source"
                ],
                "Required": [
                    "Yes",
                    "Yes",
                    "Optional",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Optional"
                ]
            }),
            width="stretch",
            hide_index=True
        )

        st.markdown("#### Sheet: Dams")

        st.dataframe(
            pd.DataFrame({
                "Column": [
                    "Country",
                    "Site Name",
                    "Water Souce",
                    "Capacity MCM",
                    "Use",
                    "Latitude",
                    "Longitude",
                    "Source"
                ],
                "Required": [
                    "Yes",
                    "Yes",
                    "Optional",
                    "Yes",
                    "Optional",
                    "Yes",
                    "Yes",
                    "Optional"
                ]
            }),
            width="stretch",
            hide_index=True
        )

    with col4:
        st.markdown("#### Sheet: Desalination plants")

        st.dataframe(
            pd.DataFrame({
                "Column": [
                    "Country",
                    "DesalinationPlantName",
                    "Capacity (m3/day)",
                    "Latitude",
                    "Longitude",
                    "Source"
                ],
                "Required": [
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Yes",
                    "Optional"
                ]
            }),
            width="stretch",
            hide_index=True
        )

        st.markdown(
            """
            #### Validation rules

            Rows are dropped when:

            - Latitude or longitude is missing or invalid
            - Capacity is missing, zero, or negative
            - Required asset names are missing
            - Required sheets or columns cannot be matched

            The app accepts numeric ranges such as `450-520` and converts them to the midpoint.
            """
        )

    st.download_button(
        label="Download water workbook template",
        data=build_water_template_workbook(),
        file_name="water_availability_dataset_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div class="main-title">
        HydroGEM fertilizer, renewable energy and water opportunity map
    </div>
    <div class="subtitle">
        An interactive screening dashboard for identifying fertilizer production hotspots supported by renewable energy and water availability.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Sidebar: data source
# =====================================================

st.sidebar.header("Data source")

data_mode = st.sidebar.radio(
    "Choose data mode",
    [
        "Use default root-folder datasets",
        "Upload visitor datasets"
    ]
)

energy_file_bytes = None
water_file_bytes = None
energy_source_name = None
water_source_name = None

if data_mode == "Use default root-folder datasets":
    st.sidebar.caption("The app will look for both files in the same folder as app.py.")

    if DEFAULT_ENERGY_DATASET_PATH.exists():
        energy_file_bytes = DEFAULT_ENERGY_DATASET_PATH.read_bytes()
        energy_source_name = DEFAULT_ENERGY_DATASET_PATH.name
        st.sidebar.success(f"Energy loaded: {DEFAULT_ENERGY_DATASET_PATH.name}")
    else:
        st.sidebar.error(f"Energy file missing: {DEFAULT_ENERGY_DATASET_PATH.name}")

    if DEFAULT_WATER_DATASET_PATH.exists():
        water_file_bytes = DEFAULT_WATER_DATASET_PATH.read_bytes()
        water_source_name = DEFAULT_WATER_DATASET_PATH.name
        st.sidebar.success(f"Water loaded: {DEFAULT_WATER_DATASET_PATH.name}")
    else:
        st.sidebar.warning(f"Water file missing: {DEFAULT_WATER_DATASET_PATH.name}")

else:
    uploaded_energy_file = st.sidebar.file_uploader(
        "Upload energy workbook",
        type=["xlsx"],
        key="energy_uploader"
    )

    uploaded_water_file = st.sidebar.file_uploader(
        "Upload water workbook",
        type=["xlsx"],
        key="water_uploader"
    )

    if uploaded_energy_file is not None:
        energy_file_bytes = uploaded_energy_file.getvalue()
        energy_source_name = uploaded_energy_file.name
        st.sidebar.success(f"Energy uploaded: {uploaded_energy_file.name}")

    if uploaded_water_file is not None:
        water_file_bytes = uploaded_water_file.getvalue()
        water_source_name = uploaded_water_file.name
        st.sidebar.success(f"Water uploaded: {uploaded_water_file.name}")

st.sidebar.divider()


# =====================================================
# Stop if no energy file
# =====================================================

if energy_file_bytes is None:
    tab_start, tab_guide = st.tabs(["Start here", "Data guide"])

    with tab_start:
        st.info("No energy dataset is loaded yet.")

        st.markdown(
            f"""
            Place this file in your root app folder:

            ```text
            {DEFAULT_ENERGY_DATASET_PATH.name}
            ```

            Then refresh the app, or switch to visitor upload mode and upload the workbook manually.
            """
        )

    with tab_guide:
        show_data_guide()

    st.stop()


# =====================================================
# Load datasets
# =====================================================

try:
    with st.spinner("Loading and validating energy dataset..."):
        energy_bundle = load_energy_dataset(
            energy_file_bytes,
            energy_source_name
        )
except DataValidationError as exc:
    tab_error, tab_guide = st.tabs(["Energy upload error", "Data guide"])

    with tab_error:
        st.markdown(
            f"""
            <div class="error-box">
                <h3>{escape_text(exc.title)}</h3>
                <ul>
                    {''.join([f'<li>{escape_text(issue)}</li>' for issue in exc.issues])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_guide:
        show_data_guide()

    st.stop()

water_bundle = None
water_error = None

if water_file_bytes is not None:
    try:
        with st.spinner("Loading and validating water dataset..."):
            water_bundle = load_water_dataset(
                water_file_bytes,
                water_source_name
            )
    except DataValidationError as exc:
        water_error = exc


df_fert = energy_bundle.df_fert
df_wind = energy_bundle.df_wind
df_solar = energy_bundle.df_solar

if water_bundle is not None:
    df_water = water_bundle.df_water
else:
    df_water = pd.DataFrame()


# =====================================================
# Sidebar filters
# =====================================================

st.sidebar.header("Filters")

country_pool = sorted(
    set(df_fert["Country"].dropna().unique())
    | set(df_wind["Country"].dropna().unique())
    | (set(df_solar["Country"].dropna().unique()) if not df_solar.empty else set())
    | (set(df_water["Country"].dropna().unique()) if not df_water.empty else set())
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=country_pool,
    default=country_pool
)

production_range = safe_range_slider(
    "Fertilizer production range, tons/year",
    df_fert["Production_tpa"]
)

wind_range = safe_range_slider(
    "Wind power density range, W/m²",
    df_wind["Wind_Density_wm2"]
)

if not df_solar.empty:
    solar_range = safe_range_slider(
        "Solar capacity range, MW",
        df_solar["Solar_Capacity_MW"]
    )
else:
    solar_range = (None, None)

if not df_water.empty:
    water_capacity_range = safe_range_slider(
        "Water asset capacity range",
        df_water["Capacity_Value"]
    )
else:
    water_capacity_range = (None, None)

st.sidebar.divider()

st.sidebar.header("Methodology controls")

available_renewable_types = ["Wind"]

if not df_solar.empty:
    available_renewable_types.append("Solar")

selected_renewable_types = st.sidebar.multiselect(
    "Renewable resources used in scoring",
    options=available_renewable_types,
    default=available_renewable_types
)

available_water_types = sorted(df_water["Water_Type"].unique()) if not df_water.empty else []

selected_water_types = st.sidebar.multiselect(
    "Water resources used in scoring",
    options=available_water_types,
    default=available_water_types,
    disabled=df_water.empty
)

renewable_matching_rule = st.sidebar.selectbox(
    "Renewable matching rule",
    options=[
        "Best quality-proximity fit",
        "Nearest asset"
    ],
    index=0
)

water_matching_rule = st.sidebar.selectbox(
    "Water matching rule",
    options=[
        "Best quality-proximity fit",
        "Nearest asset"
    ],
    index=0,
    disabled=df_water.empty
)

max_renewable_distance_km = st.sidebar.slider(
    "Maximum useful renewable distance, km",
    min_value=50,
    max_value=2500,
    value=600,
    step=50
)

max_water_distance_km = st.sidebar.slider(
    "Maximum useful water distance, km",
    min_value=10,
    max_value=2500,
    value=300,
    step=10,
    disabled=df_water.empty
)

with st.sidebar.expander("Score weights", expanded=True):
    production_weight = st.slider(
        "Production scale weight",
        min_value=0,
        max_value=100,
        value=30
    )

    renewable_quality_weight = st.slider(
        "Renewable quality weight",
        min_value=0,
        max_value=100,
        value=25
    )

    renewable_proximity_weight = st.slider(
        "Renewable proximity weight",
        min_value=0,
        max_value=100,
        value=20
    )

    water_availability_weight = st.slider(
        "Water availability weight",
        min_value=0,
        max_value=100,
        value=15,
        disabled=df_water.empty
    )

    water_proximity_weight = st.slider(
        "Water proximity weight",
        min_value=0,
        max_value=100,
        value=10,
        disabled=df_water.empty
    )

raw_weights = {
    "Production scale": production_weight,
    "Renewable quality": renewable_quality_weight,
    "Renewable proximity": renewable_proximity_weight,
    "Water availability": water_availability_weight if not df_water.empty else 0,
    "Water proximity": water_proximity_weight if not df_water.empty else 0
}

min_score = st.sidebar.slider(
    "Minimum opportunity score",
    min_value=0,
    max_value=100,
    value=0
)

top_n_hotspots = st.sidebar.slider(
    "Number of hotspots to show",
    min_value=3,
    max_value=30,
    value=10
)

st.sidebar.divider()

st.sidebar.header("Map layers")

show_fertilizer = st.sidebar.checkbox("Show fertilizer plants", value=True)
show_wind = st.sidebar.checkbox("Show wind potential", value=True)
show_solar = st.sidebar.checkbox("Show solar potential", value=False, disabled=df_solar.empty)
show_wastewater = st.sidebar.checkbox("Show wastewater facilities", value=True, disabled=df_water.empty)
show_dams = st.sidebar.checkbox("Show dams", value=True, disabled=df_water.empty)
show_desalination = st.sidebar.checkbox("Show desalination plants", value=True, disabled=df_water.empty)
show_hotspots = st.sidebar.checkbox("Show ranked hotspots", value=True)
show_connection_lines = st.sidebar.checkbox("Show hotspot connection lines", value=True)
show_country_labels = st.sidebar.checkbox("Show country/place labels", value=True)


# =====================================================
# Apply filters
# =====================================================

filtered_fert = df_fert[
    df_fert["Country"].isin(selected_countries)
].copy()

filtered_fert = apply_numeric_range_filter(
    filtered_fert,
    "Production_tpa",
    production_range
)

filtered_wind = df_wind[
    df_wind["Country"].isin(selected_countries)
].copy()

filtered_wind = apply_numeric_range_filter(
    filtered_wind,
    "Wind_Density_wm2",
    wind_range
)

if not df_solar.empty:
    filtered_solar = df_solar[
        df_solar["Country"].isin(selected_countries)
    ].copy()

    filtered_solar = apply_numeric_range_filter(
        filtered_solar,
        "Solar_Capacity_MW",
        solar_range
    )
else:
    filtered_solar = pd.DataFrame()

if not df_water.empty:
    filtered_water = df_water[
        df_water["Country"].isin(selected_countries)
    ].copy()

    filtered_water = filter_water_assets(
        filtered_water,
        selected_water_types
    )

    filtered_water = apply_numeric_range_filter(
        filtered_water,
        "Capacity_Value",
        water_capacity_range
    )
else:
    filtered_water = pd.DataFrame()

renewable_assets = prepare_renewable_assets(
    filtered_wind,
    filtered_solar,
    selected_renewable_types
)

water_assets = filtered_water.copy()

opportunity = build_opportunity_table(
    df_fert=filtered_fert,
    renewable_assets=renewable_assets,
    water_assets=water_assets,
    raw_weights=raw_weights,
    renewable_matching_rule=renewable_matching_rule,
    water_matching_rule=water_matching_rule,
    max_renewable_distance_km=max_renewable_distance_km,
    max_water_distance_km=max_water_distance_km
)

filtered_opportunity = opportunity[
    opportunity["Opportunity_Score"] >= min_score
].copy()

top_hotspots = filtered_opportunity.head(top_n_hotspots).copy()


# =====================================================
# Tabs
# =====================================================

tab_overview, tab_map, tab_hotspots, tab_ranking, tab_quality, tab_dictionary, tab_guide = st.tabs([
    "Overview",
    "Satellite map",
    "Hotspots",
    "Opportunity ranking",
    "Data quality",
    "Dictionary",
    "Data guide"
])


# =====================================================
# Overview
# =====================================================

with tab_overview:
    st.subheader("Executive overview")

    if water_error is not None:
        st.warning(
            f"Water dataset could not be loaded: {water_error.title}. Check the Data Quality tab for details."
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Fertilizer plants", f"{len(filtered_fert):,}")
    col2.metric("Renewable assets", f"{len(renewable_assets):,}")
    col3.metric("Water assets", f"{len(water_assets):,}")
    col4.metric("Scored opportunities", f"{len(filtered_opportunity):,}")

    if not filtered_opportunity.empty:
        col5.metric("Top score", f"{filtered_opportunity['Opportunity_Score'].max():.2f}")
    else:
        col5.metric("Top score", "N/A")

    st.markdown("### What the model is currently prioritising")

    effective_weights = calculate_effective_weights(
        raw_weights=raw_weights,
        has_renewable=not renewable_assets.empty,
        has_water=not water_assets.empty
    )

    weight_df = pd.DataFrame({
        "Score component": list(effective_weights.keys()),
        "Effective weight": [round(value * 100, 1) for value in effective_weights.values()]
    })

    st.dataframe(
        weight_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("### Top hotspots")

    if top_hotspots.empty:
        st.warning("No hotspot matches the selected filters. Reduce filters or adjust methodology settings.")
    else:
        cards = st.columns(min(3, len(top_hotspots)))

        for card_index, (_, row) in enumerate(top_hotspots.head(3).iterrows()):
            with cards[card_index]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h4 style="margin-top:0;">#{card_index + 1} {escape_text(row['Name'])}</h4>
                        <p class="small-muted">{escape_text(row['Country'])}</p>
                        <p><b>Score:</b> {format_number(row['Opportunity_Score'], 2)}</p>
                        <p><b>Production:</b> {format_number(row['Production_tpa'])} tons/year</p>
                        <p><b>Renewable:</b> {escape_text(row.get('Nearest_Renewable_Name', 'N/A'))}</p>
                        <p><b>Water:</b> {escape_text(row.get('Nearest_Water_Name', 'N/A'))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown(
    """
    ### How to read the results

    The dashboard ranks fertilizer plants by how well they combine three conditions:

    1. **Existing production scale**: larger fertilizer plants may represent bigger conversion or decarbonisation opportunities.
    2. **Renewable energy fit**: locations close to stronger wind or solar assets score better.
    3. **Water availability fit**: locations close to stronger water assets score better.

    The output should be read as a **screening result**, not a final investment recommendation. 
    A high-ranking hotspot means the location deserves deeper technical, commercial, environmental, and regulatory feasibility review.
    """
)


# =====================================================
# Map
# =====================================================

with tab_map:
    st.subheader("Satellite map with energy, water and hotspot layers")

    st.caption(
        "Use the map layer control to switch between satellite, clean map, street map, labels, and individual asset layers."
    )

    folium_map = build_folium_map(
        df_fert=filtered_fert,
        renewable_assets=renewable_assets,
        water_assets=water_assets,
        opportunity=filtered_opportunity,
        show_fertilizer=show_fertilizer,
        show_wind=show_wind,
        show_solar=show_solar,
        show_wastewater=show_wastewater,
        show_dams=show_dams,
        show_desalination=show_desalination,
        show_hotspots=show_hotspots,
        show_connection_lines=show_connection_lines,
        show_country_labels=show_country_labels,
        top_n_hotspots=top_n_hotspots
    )

    st_folium(
        folium_map,
        height=760,
        width=1200,
        returned_objects=[]
    )

    st.download_button(
        label="Download current map as HTML",
        data=folium_map_to_html(folium_map),
        file_name="hydrogem_energy_water_hotspot_map.html",
        mime="text/html"
    )


# =====================================================
# Hotspots
# =====================================================

with tab_hotspots:
    st.subheader("Hotspot analysis")

    if top_hotspots.empty:
        st.warning("No hotspots available under the current filter and scoring settings.")
    else:
        st.markdown("### Top hotspot locations")

        hotspot_columns = [
            "Name",
            "Country",
            "Opportunity_Score",
            "Priority_Band",
            "Production_tpa",
            "Nearest_Renewable_Name",
            "Nearest_Renewable_Type",
            "Distance_to_Renewable_km",
            "Nearest_Water_Name",
            "Nearest_Water_Type",
            "Distance_to_Water_km"
        ]

        st.dataframe(
            top_hotspots[hotspot_columns],
            width="stretch",
            hide_index=True
        )

        st.markdown("### Country hotspot summary")

        country_summary = (
            top_hotspots
            .groupby("Country", as_index=False)
            .agg(
                Hotspot_Count=("Name", "count"),
                Average_Score=("Opportunity_Score", "mean"),
                Highest_Score=("Opportunity_Score", "max"),
                Total_Production_tpa=("Production_tpa", "sum"),
                Average_Renewable_Distance_km=("Distance_to_Renewable_km", "mean"),
                Average_Water_Distance_km=("Distance_to_Water_km", "mean")
            )
            .sort_values(["Hotspot_Count", "Highest_Score"], ascending=[False, False])
        )

        country_summary["Average_Score"] = country_summary["Average_Score"].round(2)
        country_summary["Highest_Score"] = country_summary["Highest_Score"].round(2)
        country_summary["Average_Renewable_Distance_km"] = country_summary["Average_Renewable_Distance_km"].round(1)
        country_summary["Average_Water_Distance_km"] = country_summary["Average_Water_Distance_km"].round(1)

        st.dataframe(
            country_summary,
            width="stretch",
            hide_index=True
        )

        st.markdown("### Hotspot logic")

        st.markdown(
            """
            A hotspot is a fertilizer plant that scores strongly after applying the current methodology settings. 
            The methodology is adjustable from the sidebar, so the map and ranking update when you change the weights, distance thresholds, included water assets, included renewable assets, and matching rules.
            """
        )


# =====================================================
# Ranking
# =====================================================

with tab_ranking:
    st.subheader("Full opportunity ranking")

    if filtered_opportunity.empty:
        st.warning("No opportunity records match the selected filters.")
    else:
        display_columns = [
            "Name",
            "Country",
            "Production_tpa",
            "Production_Score",
            "Nearest_Renewable_Name",
            "Nearest_Renewable_Type",
            "Renewable_Quality_Value",
            "Renewable_Quality_Unit",
            "Renewable_Quality_Score",
            "Distance_to_Renewable_km",
            "Renewable_Proximity_Score",
            "Nearest_Water_Name",
            "Nearest_Water_Type",
            "Water_Quality_Value",
            "Water_Quality_Unit",
            "Water_Quality_Score",
            "Distance_to_Water_km",
            "Water_Proximity_Score",
            "Opportunity_Score",
            "Priority_Band"
        ]

        st.dataframe(
            filtered_opportunity[display_columns],
            width="stretch",
            hide_index=True
        )

        st.download_button(
            label="Download filtered ranking as CSV",
            data=filtered_opportunity.to_csv(index=False).encode("utf-8"),
            file_name="hydrogem_opportunity_ranking.csv",
            mime="text/csv"
        )


# =====================================================
# Data quality
# =====================================================

with tab_quality:
    st.subheader("Data quality and validation report")

    st.markdown(
        f"""
        <div class="success-box">
            <b>Energy dataset loaded:</b> {escape_text(energy_bundle.source_name)}
        </div>
        """,
        unsafe_allow_html=True
    )

    if water_bundle is not None:
        st.markdown(
            f"""
            <div class="success-box">
                <b>Water dataset loaded:</b> {escape_text(water_bundle.source_name)}
            </div>
            """,
            unsafe_allow_html=True
        )
    elif water_error is not None:
        st.markdown(
            f"""
            <div class="error-box">
                <h4>{escape_text(water_error.title)}</h4>
                <ul>
                    {''.join([f'<li>{escape_text(issue)}</li>' for issue in water_error.issues])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("No water dataset was loaded.")

    st.markdown("### Energy sheets detected")

    st.dataframe(
        pd.DataFrame([
            {
                "Expected Sheet": expected,
                "Workbook Sheet Used": actual
            }
            for expected, actual in energy_bundle.report["sheets"].items()
        ]),
        width="stretch",
        hide_index=True
    )

    st.markdown("### Energy row validation")

    st.dataframe(
        pd.DataFrame([
            {
                "Sheet": sheet_name,
                "Raw Rows": values["raw_rows"],
                "Usable Rows": values["usable_rows"],
                "Dropped Rows": values["dropped_rows"]
            }
            for sheet_name, values in energy_bundle.report["rows"].items()
        ]),
        width="stretch",
        hide_index=True
    )

    if water_bundle is not None:
        st.markdown("### Water sheets detected")

        st.dataframe(
            pd.DataFrame([
                {
                    "Expected Sheet": expected,
                    "Workbook Sheet Used": actual
                }
                for expected, actual in water_bundle.report["sheets"].items()
            ]),
            width="stretch",
            hide_index=True
        )

        st.markdown("### Water row validation")

        st.dataframe(
            pd.DataFrame([
                {
                    "Sheet": sheet_name,
                    "Raw Rows": values["raw_rows"],
                    "Usable Rows": values["usable_rows"],
                    "Dropped Rows": values["dropped_rows"]
                }
                for sheet_name, values in water_bundle.report["rows"].items()
            ]),
            width="stretch",
            hide_index=True
        )

    st.markdown("### Validation notes")

    warnings = []
    warnings.extend(energy_bundle.report.get("warnings", []))

    if water_bundle is not None:
        warnings.extend(water_bundle.report.get("warnings", []))

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No major validation warnings detected.")

    with st.expander("Preview cleaned fertilizer data"):
        st.dataframe(df_fert.head(50), width="stretch", hide_index=True)

    with st.expander("Preview cleaned wind data"):
        st.dataframe(df_wind.head(50), width="stretch", hide_index=True)

    if not df_solar.empty:
        with st.expander("Preview cleaned solar data"):
            st.dataframe(df_solar.head(50), width="stretch", hide_index=True)

    if not df_water.empty:
        with st.expander("Preview cleaned water assets"):
            st.dataframe(df_water.head(100), width="stretch", hide_index=True)


# =====================================================
# Dictionary
# =====================================================

with tab_dictionary:
    effective_weights_for_dictionary = calculate_effective_weights(
        raw_weights=raw_weights,
        has_renewable=not renewable_assets.empty,
        has_water=not water_assets.empty
    )

    show_dictionary(
        raw_weights=raw_weights,
        effective_weights=effective_weights_for_dictionary
    )


# =====================================================
# Data guide
# =====================================================

with tab_guide:
    show_data_guide()