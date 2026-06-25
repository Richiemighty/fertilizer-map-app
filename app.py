"""
HydroGEM opportunity map application

Purpose
-------
This Streamlit application helps users screen possible hydrogen-related opportunity
locations using multiple datasets:

1. Fertilizer / ammonia production plants
2. Wind potential locations
3. Solar potential locations
4. Water availability assets:
   - Wastewater treatment plants
   - Dams / reservoirs
   - Desalination plants
5. Petroleum processing sites (refineries)

How the score works
-------------------
The default Opportunity Score is calculated as:

    Opportunity Score =
    (Fertilizer / Ammonia Production Plant Score × 35
    + Wind Fit Score × 25
    + Solar Fit Score × 25
    + Water Fit Score × 15) / 100

The score weights can be adjusted in the sidebar.

Important scoring note
----------------------
The app now allows two methods for scoring fertilizer / ammonia plant scale:

1. Log-scaled min-max scoring, recommended
   - Better when one very large plant dominates the dataset.
   - Reduces the extreme gap between very large and smaller plants.

2. Raw min-max scoring
   - Directly compares production values.
   - This is why one plant may score 100 while others score 0.9 or 0.7.

The default is log-scaled scoring because fertilizer / ammonia production capacity
is often highly skewed.
"""


# =====================================================
# 1. Imports
# =====================================================

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
# 2. App configuration
# =====================================================

st.set_page_config(
    page_title="HydroGEM opportunity map",
    page_icon="🌍",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

DEFAULT_MAIN_WORKBOOK_PATH = APP_DIR / "Fertilizer Plants_AHDS.xlsx"
DEFAULT_WATER_WORKBOOK_PATH = APP_DIR / "Water availability.xlsx"
DEFAULT_PETROLEUM_DATA_PATH = APP_DIR / "Petroleum processing sites - Petroleum processing sites.csv"


# =====================================================
# 3. Page styling
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

        .info-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.04);
            margin-bottom: 12px;
            min-height: 150px;
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

        .formula-box {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px 18px;
            font-family: Menlo, Monaco, Consolas, monospace;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 4. Data containers and custom error class
# =====================================================

@dataclass
class MainWorkbookBundle:
    """
    Stores cleaned datasets loaded from the main workbook.

    df_plants:
        Cleaned fertilizer / ammonia production plant data.

    df_wind:
        Cleaned wind potential data.

    df_solar:
        Cleaned solar potential data.

    df_water_from_main:
        Cleaned water data extracted from the main workbook, if available.

    report:
        Data quality and validation report.

    source_name:
        Workbook file name.
    """

    df_plants: pd.DataFrame
    df_wind: pd.DataFrame
    df_solar: pd.DataFrame
    df_water_from_main: pd.DataFrame
    report: dict
    source_name: str


@dataclass
class WaterBundle:
    """
    Stores cleaned water datasets loaded from a separate water workbook.
    """

    df_water: pd.DataFrame
    report: dict
    source_name: str


@dataclass
class PetroleumBundle:
    """
    Stores cleaned petroleum refinery data.
    """

    df_refineries: pd.DataFrame
    report: dict
    source_name: str


class DataValidationError(Exception):
    """
    Friendly validation error for workbook structure problems.
    """

    def __init__(self, title, issues):
        self.title = title
        self.issues = issues
        super().__init__(title)


# =====================================================
# 5. Sheet name and column name aliases
# =====================================================

MAIN_SHEET_ALIASES = {
    "plants": [
        "Fertilizer Plants",
        "Fertilizer",
        "Fertilizer Plant",
        "Plants",
        "Ammonia Plants",
        "Ammonia / Fertilizer Plants",
        "Ammonia / Fertilizer Anchors"
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
    ],
    "wastewater": [
        "Waste water treatment plants",
        "Wastewater treatment plants",
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

PLANT_COLUMN_ALIASES = {
    "Name": [
        "Name",
        "Plant Name",
        "Fertilizer Plant",
        "Fertilizer Plant Name",
        "Ammonia Plant",
        "Ammonia Plant Name"
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
        "Capacity_tpa",
        "Ammonia Capacity",
        "Ammonia Capacity tpa"
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
        "Waste water treatment plant",
        "Wastewater treatment plant",
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

# Petroleum column aliases for flexible column name matching
PETROLEUM_COLUMN_ALIASES = {
    "Refinery_Name": [
        "Refinery Name",
        "Refinery_Name",
        "Name",
        "Refinery"
    ],
    "Country": [
        "Country"
    ],
    "Operator": [
        "Owner/Operator",
        "Owner",
        "Operator",
        "Owner_Operator"
    ],
    "Location": [
        "Location",
        "Site",
        "Region"
    ],
    "Latitude": [
        "Approx. Lat/Long",
        "Lat/Long",
        "Latitude",
        "Lat",
        "Coordinates"
    ],
    "Capacity_bpd": [
        "Capacity (bpd)",
        "Capacity_bpd",
        "Capacity",
        "Capacity bpd"
    ],
    "Annual_Production": [
        "Approx. Annual (barrels)",
        "Annual_Production",
        "Annual Production",
        "Annual"
    ],
    "Status": [
        "Status / Notes (April 2026)",
        "Status",
        "Status_Notes",
        "Notes"
    ]
}


# =====================================================
# 6. General utility functions
# =====================================================

def escape_text(value):
    """
    Escapes text before inserting it into popup HyperText Markup Language.
    """
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def clean_text(value):
    """
    Converts missing or blank text into 'Unknown' and strips extra spaces.
    """
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    return value if value else "Unknown"


def normalize_column_name(value):
    """
    Standardises sheet and column names for flexible matching.

    This allows the app to read columns even when users slightly change names,
    such as 'Lat' instead of 'Latitude' or 'Capacity MCM' instead of 'Capacity'.
    """
    text = str(value).strip().lower()
    text = text.replace("²", "2").replace("³", "3")
    text = text.replace("_", " ").replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_matching_sheet(sheet_names, possible_names):
    """
    Finds the actual workbook sheet name that matches any expected alias.
    """
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
    """
    Renames uploaded columns into the app's internal column names.

    Example:
    - 'Lat' becomes 'Latitude'
    - 'Production (tons/ annum)' becomes 'Production_tpa'

    If a required column is missing, the app returns a friendly error.
    """
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
        raise DataValidationError(
            title=f"Missing required columns in the '{sheet_display_name}' sheet",
            issues=[
                f"Required column(s) not found: {', '.join(missing_columns)}.",
                f"Columns found in your sheet: {', '.join([str(col) for col in df.columns])}.",
                "Rename the columns using the Data Guide tab, then reload the workbook."
            ]
        )

    df = df.rename(columns=rename_map)

    for canonical_col in alias_map.keys():
        if canonical_col not in df.columns:
            df[canonical_col] = np.nan

    return df


def parse_number_or_range(value):
    """
    Converts a single number or a number range into a float.

    Examples:
    - 500 becomes 500.0
    - 450-520 becomes 485.0
    - 450 – 520 becomes 485.0
    """
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


def parse_lat_long(value):
    """
    Parses latitude/longitude from various formats.
    Handles formats like "6.43°N, 3.99°E" or "6.43, 3.99"
    """
    if pd.isna(value):
        return np.nan, np.nan
    
    text = str(value).strip()
    
    # Try to parse "6.43°N, 3.99°E" format
    pattern = r"([0-9.]+)°?\s*([NSEW]?)[,\s]+([0-9.]+)°?\s*([NSEW]?)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        lat_val = float(match.group(1))
        lat_dir = match.group(2).upper()
        lon_val = float(match.group(3))
        lon_dir = match.group(4).upper()
        
        # Convert to decimal degrees
        if lat_dir == 'S':
            lat_val = -lat_val
        if lon_dir == 'W':
            lon_val = -lon_val
            
        return lat_val, lon_val
    
    # Try to parse "6.43, 3.99" format
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    
    return np.nan, np.nan


def make_marker_size(series, min_size=6, max_size=28, use_log=True):
    """
    Converts a numeric series into readable marker sizes.

    Log scaling is used by default so that one extremely large value does not
    make every other marker almost invisible.
    """
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)

    if values.dropna().empty:
        return pd.Series(min_size, index=series.index)

    scaled_values = np.log1p(values) if use_log else values

    min_value = scaled_values.min()
    max_value = scaled_values.max()

    if max_value == min_value:
        return pd.Series((min_size + max_size) / 2, index=series.index)

    marker_sizes = min_size + (
        (scaled_values - min_value) / (max_value - min_value)
    ) * (max_size - min_size)

    return marker_sizes.fillna(min_size).clip(lower=min_size, upper=max_size)


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculates straight-line distance between two coordinates in kilometres.

    This is not road distance, pipeline distance or transmission distance.
    It is a first-pass spatial screening distance.
    """
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
    """
    Converts values into a 0-100 score using raw min-max scaling.

    Formula:
        Score = ((Value - Minimum Value) / (Maximum Value - Minimum Value)) × 100

    Interpretation:
    - Highest value gets 100
    - Lowest value gets 0
    - Other values sit between 0 and 100
    """
    values = pd.to_numeric(series, errors="coerce")

    if values.dropna().empty:
        return pd.Series(0, index=series.index)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return pd.Series(100, index=series.index)

    return ((values - min_value) / (max_value - min_value) * 100).fillna(0)


def log_minmax_score(series):
    """
    Converts values into a 0-100 score using log-scaled min-max scoring.

    Formula:
        Log Value = ln(1 + Value)

        Score =
        ((Log Value - Minimum Log Value)
        / (Maximum Log Value - Minimum Log Value)) × 100

    Why this is useful:
    When one fertilizer / ammonia production plant is much larger than the rest,
    raw min-max scoring compresses smaller plants close to zero. Log scaling
    reduces that distortion while still rewarding larger production capacity.
    """
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)

    if values.dropna().empty:
        return pd.Series(0, index=series.index)

    log_values = np.log1p(values)

    return minmax_score(log_values)


def production_plant_score(series, score_method):
    """
    Scores fertilizer / ammonia production plants based on production capacity.

    score_method options:
    - Log-scaled min-max, recommended
    - Raw min-max
    """
    if score_method == "Raw min-max":
        return minmax_score(series)

    return log_minmax_score(series)


def proximity_score(distance_series, max_distance_km):
    """
    Converts distance into a 0-100 proximity score.

    Formula:
        Proximity Score = max(0, 100 × (1 - Distance / Maximum Useful Distance))

    Interpretation:
    - Asset at the same location gets 100
    - Asset at the maximum useful distance gets 0
    - Asset beyond the maximum useful distance also gets 0
    """
    values = pd.to_numeric(distance_series, errors="coerce")

    if max_distance_km <= 0:
        max_distance_km = 1

    score = 100 * (1 - (values / max_distance_km))

    return score.clip(lower=0, upper=100).fillna(0)


def classify_score(score):
    """
    Converts the numeric Opportunity Score into a simple priority band.
    """
    if score >= 75:
        return "High priority"

    if score >= 50:
        return "Moderate priority"

    return "Lower priority"


def format_number(value, decimals=0):
    """
    Formats numbers for tables and map popups.
    """
    if pd.isna(value):
        return "N/A"

    if decimals == 0:
        return f"{float(value):,.0f}"

    return f"{float(value):,.{decimals}f}"


def safe_range_slider(label, series):
    """
    Creates a sidebar range slider only when the data has valid values.
    """
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
    """
    Filters a dataframe using a numeric range selected in the sidebar.
    """
    if df.empty or selected_range[0] is None or selected_range[1] is None:
        return df

    return df[df[column].between(selected_range[0], selected_range[1])].copy()


# =====================================================
# 7. Data cleaning functions
# =====================================================

def clean_plant_sheet(raw_df, report, source_name, sheet_name):
    """
    Cleans the fertilizer / ammonia production plant sheet.

    The plant sheet is used to define candidate opportunity locations.
    These locations are scored based on production capacity.
    """
    df = align_columns(
        raw_df,
        PLANT_COLUMN_ALIASES,
        ["Name", "Country", "Latitude", "Longitude", "Production_tpa"],
        "Fertilizer Plants"
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
            "No valid fertilizer / ammonia production plant rows found",
            [
                "Each row must have Name, Country, Latitude, Longitude, and positive Production or Capacity."
            ]
        )

    df["Candidate_Type"] = "Fertilizer / ammonia production plant"

    df["Hover_Text"] = (
        "<b>" + df["Name"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Production: " + df["Production_tpa"].round(0).astype(int).astype(str)
        + " tons/year"
    )

    df["Marker_Size"] = make_marker_size(
        df["Production_tpa"],
        min_size=8,
        max_size=34,
        use_log=True
    )

    report["datasets"].append({
        "Dataset": "Fertilizer / ammonia production plants",
        "Workbook": source_name,
        "Sheet": sheet_name,
        "Role in analysis": "Candidate opportunity locations and production plant score",
        "Used in scoring": "Yes",
        "Raw rows": raw_rows,
        "Usable rows": len(df),
        "Dropped rows": dropped_rows
    })

    return df


def clean_wind_sheet(raw_df, report, source_name, sheet_name):
    """
    Cleans the wind potential sheet.

    Wind is used to calculate:
    - Wind Quality Score
    - Wind Proximity Score
    - Wind Fit Score
    """
    df = align_columns(
        raw_df,
        WIND_COLUMN_ALIASES,
        ["Country", "Region", "Latitude", "Longitude", "Wind_Density_wm2"],
        "Wind Potential"
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
            "No valid wind rows found",
            [
                "Each wind row must have Country, Region, Latitude, Longitude, and positive Wind Power Density."
            ]
        )

    df["Hover_Text"] = (
        "<b>" + df["Region"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Wind density: " + df["Wind_Density_wm2"].round(0).astype(int).astype(str)
        + " W/m²"
    )

    df["Quality_Score"] = minmax_score(df["Wind_Density_wm2"])

    df["Marker_Size"] = make_marker_size(
        df["Wind_Density_wm2"],
        min_size=7,
        max_size=26,
        use_log=True
    )

    report["datasets"].append({
        "Dataset": "Wind potential",
        "Workbook": source_name,
        "Sheet": sheet_name,
        "Role in analysis": "Wind quality and wind proximity scoring",
        "Used in scoring": "Yes",
        "Raw rows": raw_rows,
        "Usable rows": len(df),
        "Dropped rows": dropped_rows
    })

    return df


def clean_solar_sheet(raw_df, report, source_name, sheet_name):
    """
    Cleans the solar potential sheet.

    Solar is used to calculate:
    - Solar Quality Score
    - Solar Proximity Score
    - Solar Fit Score
    """
    df = align_columns(
        raw_df,
        SOLAR_COLUMN_ALIASES,
        ["Country", "Solar_Site", "Latitude", "Longitude", "Solar_Capacity_MW"],
        "Solar Potential"
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
            "Solar sheet was found, but no valid solar rows could be used."
        )
        return pd.DataFrame()

    df["Hover_Text"] = (
        "<b>" + df["Solar_Site"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Solar capacity: " + df["Solar_Capacity_MW"].round(1).astype(str)
        + " MW"
    )

    df["Quality_Score"] = minmax_score(df["Solar_Capacity_MW"])

    df["Marker_Size"] = make_marker_size(
        df["Solar_Capacity_MW"],
        min_size=6,
        max_size=24,
        use_log=True
    )

    report["datasets"].append({
        "Dataset": "Solar potential",
        "Workbook": source_name,
        "Sheet": sheet_name,
        "Role in analysis": "Solar quality and solar proximity scoring",
        "Used in scoring": "Yes",
        "Raw rows": raw_rows,
        "Usable rows": len(df),
        "Dropped rows": dropped_rows
    })

    return df


def clean_water_sheet(
    raw_df,
    alias_map,
    required_columns,
    sheet_display_name,
    water_type,
    capacity_unit,
    report,
    source_name,
    sheet_name
):
    """
    Cleans one water sheet and standardises it into a common water asset format.

    Water assets include:
    - Wastewater treatment plants
    - Dams / reservoirs
    - Desalination plants
    """
    df = align_columns(
        raw_df,
        alias_map,
        required_columns,
        sheet_display_name
    )

    raw_rows = len(df)

    df = df.dropna(how="all").copy()

    for column_name, default_value in {
        "Primary_Source": "Unknown",
        "Use": "Unknown",
        "Source": "Unknown"
    }.items():
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
    )

    report["datasets"].append({
        "Dataset": water_type,
        "Workbook": source_name,
        "Sheet": sheet_name,
        "Role in analysis": "Water availability quality and proximity scoring",
        "Used in scoring": "Yes",
        "Raw rows": raw_rows,
        "Usable rows": len(df),
        "Dropped rows": dropped_rows
    })

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
    """
    Adds capacity scores and marker sizes to the combined water asset table.

    Water assets are scored within their own type because capacity units differ:
    - Dams are usually in million cubic metres
    - Wastewater and desalination are usually in cubic metres per day

    This avoids directly comparing dam storage capacity with daily treatment capacity.
    """
    if df_water.empty:
        return df_water

    frames = []

    for water_type, group in df_water.groupby("Water_Type"):
        group = group.copy()

        group["Type_Capacity_Score"] = minmax_score(group["Capacity_Value"])

        group["Marker_Size"] = make_marker_size(
            group["Capacity_Value"],
            min_size=6,
            max_size=26,
            use_log=True
        )

        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def clean_petroleum_data(file_bytes, source_name):
    """
    Cleans and processes petroleum refinery data from CSV file.
    """
    report = {
        "source_name": source_name,
        "datasets": [],
        "warnings": []
    }
    
    try:
        # Try reading with various encodings
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding='cp1252')
    except Exception as e:
        raise DataValidationError(
            "Petroleum data could not be read",
            [f"Error reading CSV file: {str(e)}"]
        )
    
    raw_rows = len(df)
    
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]
    
    # Try to find required columns
    df = align_columns(
        df,
        PETROLEUM_COLUMN_ALIASES,
        ["Refinery_Name", "Country", "Latitude", "Longitude", "Capacity_bpd"],
        "Petroleum Processing Sites"
    )
    
    # Clean text columns
    df["Refinery_Name"] = df["Refinery_Name"].apply(clean_text)
    df["Country"] = df["Country"].apply(clean_text)
    df["Operator"] = df["Operator"].apply(clean_text)
    df["Location"] = df["Location"].apply(clean_text)
    df["Status"] = df["Status"].apply(clean_text)
    
    # Parse latitude and longitude from the Lat/Long column
    lat_vals = []
    lon_vals = []
    
    for _, row in df.iterrows():
        lat, lon = parse_lat_long(row["Latitude"])
        lat_vals.append(lat)
        lon_vals.append(lon)
    
    df["Latitude"] = lat_vals
    df["Longitude"] = lon_vals
    
    # Parse capacity
    df["Capacity_bpd"] = df["Capacity_bpd"].apply(parse_number_or_range)
    df["Annual_Production"] = df["Annual_Production"].apply(parse_number_or_range)
    
    # Keep only operational refineries for the main analysis (but keep all for reference)
    # We'll filter separately for different views
    
    valid_mask = (
        df["Refinery_Name"].ne("Unknown")
        & df["Country"].ne("Unknown")
        & df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
        & df["Capacity_bpd"].notna()
        & (df["Capacity_bpd"] > 0)
    )
    
    dropped_rows = int((~valid_mask).sum())
    
    df = df[valid_mask].copy()
    
    if df.empty:
        raise DataValidationError(
            "No valid petroleum refinery rows found",
            [
                "Each row must have Refinery Name, Country, valid coordinates, and positive Capacity."
            ]
        )
    
    # Create a status category for filtering
    df["Status_Category"] = "Operational"
    df.loc[df["Status"].str.contains("shut|closed|inactive|shutdown", case=False, na=False), "Status_Category"] = "Closed/Shut"
    df.loc[df["Status"].str.contains("planned|proposed|planning|target", case=False, na=False), "Status_Category"] = "Planned/Proposed"
    df.loc[df["Status"].str.contains("rehabilitation|revival|upgrade|expansion", case=False, na=False), "Status_Category"] = "Under Development"
    
    # Create marker information
    df["Refinery_Type"] = "Petroleum Refinery"
    
    df["Hover_Text"] = (
        "<b>" + df["Refinery_Name"].apply(escape_text) + "</b>"
        + "<br>Country: " + df["Country"].apply(escape_text)
        + "<br>Operator: " + df["Operator"].apply(escape_text)
        + "<br>Capacity: " + df["Capacity_bpd"].round(0).astype(int).astype(str)
        + " bpd"
        + "<br>Status: " + df["Status"].apply(escape_text)
    )
    
    df["Marker_Size"] = make_marker_size(
        df["Capacity_bpd"],
        min_size=8,
        max_size=34,
        use_log=True
    )
    
    # Calculate capacity score
    df["Refinery_Score"] = minmax_score(df["Capacity_bpd"])
    
    report["datasets"].append({
        "Dataset": "Petroleum processing sites",
        "Workbook": source_name,
        "Sheet": "CSV",
        "Role in analysis": "Petroleum infrastructure mapping and analysis",
        "Used in scoring": "No (informational)",
        "Raw rows": raw_rows,
        "Usable rows": len(df),
        "Dropped rows": dropped_rows
    })
    
    return PetroleumBundle(
        df_refineries=df,
        report=report,
        source_name=source_name
    )


# =====================================================
# 8. Workbook loading functions
# =====================================================

def load_water_sheets_from_excel(excel_file, file_bytes, source_name, report):
    """
    Looks for water sheets inside a workbook and returns a combined water table.
    """
    sheet_names = excel_file.sheet_names

    frames = []

    sheet_map = [
        (
            "wastewater",
            "Wastewater treatment plant",
            WASTEWATER_COLUMN_ALIASES,
            ["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"],
            "Wastewater treatment plant",
            "m³/day"
        ),
        (
            "dams",
            "Dam / reservoir",
            DAMS_COLUMN_ALIASES,
            ["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"],
            "Dam / reservoir",
            "MCM"
        ),
        (
            "desalination",
            "Desalination plant",
            DESALINATION_COLUMN_ALIASES,
            ["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"],
            "Desalination plant",
            "m³/day"
        )
    ]

    for key, display_name, alias_map, required, water_type, unit in sheet_map:
        matched_sheet = find_matching_sheet(sheet_names, MAIN_SHEET_ALIASES[key])

        if matched_sheet is None:
            report["warnings"].append(
                f"No {display_name} sheet was found in {source_name}."
            )
            continue

        raw_df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=matched_sheet
        )

        try:
            frames.append(
                clean_water_sheet(
                    raw_df,
                    alias_map,
                    required,
                    display_name,
                    water_type,
                    unit,
                    report,
                    source_name,
                    matched_sheet
                )
            )
        except DataValidationError as exc:
            report["warnings"].extend(exc.issues)

    frames = [
        frame for frame in frames
        if frame is not None and not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    return finalize_water_assets(
        pd.concat(frames, ignore_index=True)
    )


@st.cache_data(show_spinner=False)
def load_main_workbook(file_bytes, source_name):
    """
    Loads the main workbook and extracts:

    - Fertilizer / ammonia production plants
    - Wind potential
    - Solar potential
    - Water sheets, if available
    """
    report = {
        "source_name": source_name,
        "datasets": [],
        "warnings": []
    }

    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError(
            "Main workbook could not be opened",
            ["Upload a valid .xlsx Excel workbook."]
        )

    sheet_names = excel_file.sheet_names

    plant_sheet = find_matching_sheet(sheet_names, MAIN_SHEET_ALIASES["plants"])
    wind_sheet = find_matching_sheet(sheet_names, MAIN_SHEET_ALIASES["wind"])
    solar_sheet = find_matching_sheet(sheet_names, MAIN_SHEET_ALIASES["solar"])

    missing = []

    if plant_sheet is None:
        missing.append("Fertilizer Plants")

    if wind_sheet is None:
        missing.append("Wind Potential")

    if missing:
        raise DataValidationError(
            "Required sheet(s) missing",
            [
                f"Missing sheet(s): {', '.join(missing)}.",
                f"Sheets found: {', '.join(sheet_names)}."
            ]
        )

    raw_plants = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=plant_sheet
    )

    raw_wind = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=wind_sheet
    )

    df_plants = clean_plant_sheet(
        raw_plants,
        report,
        source_name,
        plant_sheet
    )

    df_wind = clean_wind_sheet(
        raw_wind,
        report,
        source_name,
        wind_sheet
    )

    df_solar = pd.DataFrame()

    if solar_sheet is not None:
        raw_solar = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=solar_sheet
        )

        df_solar = clean_solar_sheet(
            raw_solar,
            report,
            source_name,
            solar_sheet
        )
    else:
        report["warnings"].append(
            f"No Solar Potential sheet was found in {source_name}."
        )

    df_water_from_main = load_water_sheets_from_excel(
        excel_file,
        file_bytes,
        source_name,
        report
    )

    return MainWorkbookBundle(
        df_plants=df_plants,
        df_wind=df_wind,
        df_solar=df_solar,
        df_water_from_main=df_water_from_main,
        report=report,
        source_name=source_name
    )


@st.cache_data(show_spinner=False)
def load_separate_water_workbook(file_bytes, source_name):
    """
    Loads a separate water workbook when water data is not available in the main workbook.
    """
    report = {
        "source_name": source_name,
        "datasets": [],
        "warnings": []
    }

    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError(
            "Water workbook could not be opened",
            [
                "Upload a valid .xlsx workbook containing wastewater, dams and/or desalination sheets."
            ]
        )

    df_water = load_water_sheets_from_excel(
        excel_file,
        file_bytes,
        source_name,
        report
    )

    if df_water.empty:
        raise DataValidationError(
            "No valid water availability rows found",
            [
                "At least one water sheet should be valid: wastewater, dams, or desalination."
            ]
        )

    return WaterBundle(
        df_water=df_water,
        report=report,
        source_name=source_name
    )


@st.cache_data(show_spinner=False)
def load_petroleum_data(file_bytes, source_name):
    """
    Loads and cleans petroleum refinery data from CSV.
    """
    return clean_petroleum_data(file_bytes, source_name)


# =====================================================
# 9. Asset preparation and scoring functions
# =====================================================

def prepare_wind_assets(df_wind):
    """
    Converts cleaned wind data into a standard resource asset table.
    """
    if df_wind.empty:
        return pd.DataFrame()

    wind = df_wind.copy()

    wind["Asset_Name"] = wind["Region"]
    wind["Asset_Type"] = "Wind"
    wind["Quality_Value"] = wind["Wind_Density_wm2"]
    wind["Quality_Unit"] = "W/m²"
    wind["Quality_Score"] = minmax_score(wind["Quality_Value"])

    return wind[[
        "Country",
        "Asset_Name",
        "Asset_Type",
        "Quality_Value",
        "Quality_Unit",
        "Quality_Score",
        "Latitude",
        "Longitude",
        "Hover_Text",
        "Marker_Size"
    ]]


def prepare_solar_assets(df_solar):
    """
    Converts cleaned solar data into a standard resource asset table.
    """
    if df_solar.empty:
        return pd.DataFrame()

    solar = df_solar.copy()

    solar["Asset_Name"] = solar["Solar_Site"]
    solar["Asset_Type"] = "Solar"
    solar["Quality_Value"] = solar["Solar_Capacity_MW"]
    solar["Quality_Unit"] = "MW"
    solar["Quality_Score"] = minmax_score(solar["Quality_Value"])

    return solar[[
        "Country",
        "Asset_Name",
        "Asset_Type",
        "Quality_Value",
        "Quality_Unit",
        "Quality_Score",
        "Latitude",
        "Longitude",
        "Hover_Text",
        "Marker_Size"
    ]]


def prepare_water_assets(df_water, selected_water_types):
    """
    Filters and standardises water assets for scoring.
    """
    if df_water.empty:
        return pd.DataFrame()

    water = df_water[
        df_water["Water_Type"].isin(selected_water_types)
    ].copy()

    if water.empty:
        return water

    water["Quality_Value"] = water["Capacity_Value"]
    water["Quality_Unit"] = water["Capacity_Unit"]
    water["Quality_Score"] = water["Type_Capacity_Score"]

    return water


def prepare_petroleum_assets(df_refineries, status_filter=None):
    """
    Prepares petroleum refinery data for mapping and analysis.
    
    Args:
        df_refineries: Cleaned refinery dataframe
        status_filter: Optional filter for status category
    """
    if df_refineries.empty:
        return pd.DataFrame()
    
    refineries = df_refineries.copy()
    
    # Apply status filter if provided
    if status_filter and status_filter != "All":
        refineries = refineries[refineries["Status_Category"] == status_filter].copy()
    
    if refineries.empty:
        return refineries
    
    refineries["Asset_Name"] = refineries["Refinery_Name"]
    refineries["Asset_Type"] = "Petroleum Refinery"
    refineries["Quality_Value"] = refineries["Capacity_bpd"]
    refineries["Quality_Unit"] = "bpd"
    refineries["Quality_Score"] = refineries["Refinery_Score"]
    
    return refineries


def match_resource(
    source_df,
    asset_df,
    name_col,
    type_col,
    country_col,
    quality_value_col,
    quality_unit_col,
    quality_score_col,
    max_distance_km,
    match_rule,
    prefix
):
    """
    Matches each fertilizer / ammonia production plant to a resource asset.

    Two matching rules are supported:

    1. Nearest asset:
       Selects the resource with the shortest straight-line distance.

    2. Best quality-distance fit:
       Selects the resource with the best balance of quality and proximity.
       This avoids choosing the closest resource if it is very weak.
    """
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

        if match_rule == "Nearest asset":
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


def combine_quality_and_proximity(
    quality_score,
    proximity_score_value,
    quality_weight_percent
):
    """
    Combines resource quality and distance proximity into one fit score.

    Formula:
        Fit Score =
        (Quality Score × Quality Weight
        + Proximity Score × Proximity Weight) / 100

    If the quality weight is 50, then quality and distance are treated equally.
    """
    q = quality_weight_percent / 100
    p = 1 - q

    return (quality_score * q) + (proximity_score_value * p)


def build_opportunity_table(
    df_plants,
    wind_assets,
    solar_assets,
    water_assets,
    raw_weights,
    max_wind_distance_km,
    max_solar_distance_km,
    max_water_distance_km,
    matching_rule,
    quality_distance_balance,
    production_score_method
):
    """
    Builds the final ranked opportunity table.

    Main steps:
    1. Score each fertilizer / ammonia production plant by production capacity.
    2. Match each plant to wind, solar and water assets.
    3. Calculate quality and proximity scores for each resource.
    4. Combine quality and proximity into fit scores.
    5. Apply the final weighting formula to calculate Opportunity Score.
    """
    if df_plants.empty:
        return pd.DataFrame()

    opportunity = df_plants[[
        "Name",
        "Country",
        "Candidate_Type",
        "Production_tpa",
        "Latitude",
        "Longitude"
    ]].reset_index(drop=True).copy()

    opportunity["Fertilizer_Ammonia_Production_Plant_Score"] = production_plant_score(
        opportunity["Production_tpa"],
        production_score_method
    )

    wind_match = match_resource(
        opportunity,
        wind_assets,
        "Asset_Name",
        "Asset_Type",
        "Country",
        "Quality_Value",
        "Quality_Unit",
        "Quality_Score",
        max_wind_distance_km,
        matching_rule,
        "Wind"
    )

    solar_match = match_resource(
        opportunity,
        solar_assets,
        "Asset_Name",
        "Asset_Type",
        "Country",
        "Quality_Value",
        "Quality_Unit",
        "Quality_Score",
        max_solar_distance_km,
        matching_rule,
        "Solar"
    )

    water_match = match_resource(
        opportunity,
        water_assets,
        "Water_Name",
        "Water_Type",
        "Country",
        "Quality_Value",
        "Quality_Unit",
        "Quality_Score",
        max_water_distance_km,
        matching_rule,
        "Water"
    )

    opportunity = pd.concat(
        [
            opportunity,
            wind_match,
            solar_match,
            water_match
        ],
        axis=1
    )

    for prefix in ["Wind", "Solar", "Water"]:
        opportunity[f"{prefix}_Quality_Score"] = pd.to_numeric(
            opportunity[f"{prefix}_Quality_Score"],
            errors="coerce"
        ).fillna(0)

        opportunity[f"{prefix}_Proximity_Score"] = pd.to_numeric(
            opportunity[f"{prefix}_Proximity_Score"],
            errors="coerce"
        ).fillna(0)

    opportunity["Wind_Fit_Score"] = combine_quality_and_proximity(
        opportunity["Wind_Quality_Score"],
        opportunity["Wind_Proximity_Score"],
        quality_distance_balance
    )

    opportunity["Solar_Fit_Score"] = combine_quality_and_proximity(
        opportunity["Solar_Quality_Score"],
        opportunity["Solar_Proximity_Score"],
        quality_distance_balance
    )

    opportunity["Water_Fit_Score"] = combine_quality_and_proximity(
        opportunity["Water_Quality_Score"],
        opportunity["Water_Proximity_Score"],
        quality_distance_balance
    )

    opportunity["Opportunity_Score"] = (
        opportunity["Fertilizer_Ammonia_Production_Plant_Score"] * (raw_weights["Fertilizer / ammonia production plant"] / 100)
        + opportunity["Wind_Fit_Score"] * (raw_weights["Wind"] / 100)
        + opportunity["Solar_Fit_Score"] * (raw_weights["Solar"] / 100)
        + opportunity["Water_Fit_Score"] * (raw_weights["Water"] / 100)
    )

    opportunity["Opportunity_Score"] = (
        opportunity["Opportunity_Score"]
        .clip(lower=0, upper=100)
        .round(2)
    )

    opportunity["Priority_Band"] = opportunity["Opportunity_Score"].apply(classify_score)
    opportunity["Methodology_Weights"] = str(raw_weights)
    opportunity["Production_Score_Method"] = production_score_method

    return opportunity.sort_values(
        "Opportunity_Score",
        ascending=False
    ).reset_index(drop=True)


# =====================================================
# 10. Mapping functions
# =====================================================

def make_colormap(values, colors):
    """
    Creates a colour scale for markers based on numeric values.
    """
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()

    if numeric_values.empty:
        return cm.LinearColormap(colors, vmin=0, vmax=1)

    vmin = float(numeric_values.min())
    vmax = float(numeric_values.max())

    if vmin == vmax:
        vmax = vmin + 1

    return cm.LinearColormap(colors, vmin=vmin, vmax=vmax)


def add_base_tiles(folium_map, show_labels=True):
    """
    Adds map backgrounds and optional country/place labels.
    """
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
    """
    Builds a neat popup table for a map marker.
    """
    html_rows = ""

    for label, value in rows:
        html_rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;color:#6B7280;'>{escape_text(label)}</td>"
            f"<td style='padding:4px 8px;font-weight:600;'>{escape_text(value)}</td>"
            f"</tr>"
        )

    return f"""
    <div style="font-family: Arial, sans-serif; min-width: 265px;">
        <h4 style="margin:0 0 8px 0;">{escape_text(title)}</h4>
        <table style="border-collapse:collapse;width:100%;">{html_rows}</table>
    </div>
    """


def add_map_marker_css(folium_map):
    """
    Adds custom marker styling to the map.
    """
    css = """
    <style>
        .hydrogem-div-icon { background: transparent !important; border: none !important; }
        .hydrogem-marker-inner { transform-origin: center center; transition: transform 0.12s ease-out; }
        .hydrogem-marker-svg { filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.45)); }
    </style>
    """

    folium_map.get_root().header.add_child(
        folium.Element(css)
    )


def build_fixed_marker_icon(
    shape,
    color,
    size=26,
    border_color="#111827",
    label=None,
    label_color="#FFFFFF"
):
    """
    Builds fixed-pixel Scalable Vector Graphics markers.
    """
    size = int(max(18, min(size, 56)))

    if shape == "circle":
        shape_svg = f"<circle cx='50' cy='50' r='38' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    elif shape == "triangle":
        shape_svg = f"<polygon points='50,8 92,88 8,88' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    elif shape == "diamond":
        shape_svg = f"<polygon points='50,5 95,50 50,95 5,50' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    elif shape == "square":
        shape_svg = f"<rect x='15' y='15' width='70' height='70' rx='10' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    elif shape == "pentagon":
        shape_svg = f"<polygon points='50,6 94,38 78,92 22,92 6,38' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    elif shape == "hexagon":
        shape_svg = f"<polygon points='50,8 92,31 92,69 50,92 8,69 8,31' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    else:
        shape_svg = f"<circle cx='50' cy='50' r='38' fill='{color}' stroke='{border_color}' stroke-width='6' />"

    label_svg = ""

    if label is not None:
        label_svg = (
            f"<text x='50' y='58' text-anchor='middle' font-size='42' "
            f"font-weight='800' font-family='Arial, sans-serif' fill='{label_color}'>"
            f"{escape_text(label)}</text>"
        )

    html_icon = f"""
    <div class="hydrogem-marker" style="width:{size}px;height:{size}px;position:relative;">
        <div class="hydrogem-marker-inner" style="width:{size}px;height:{size}px;position:absolute;left:50%;top:50%;transform:translate(-50%, -50%) scale(1);">
            <svg class="hydrogem-marker-svg" width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
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


def get_marker_px(row, fallback=24, min_px=20, max_px=46, multiplier=1.35):
    """
    Converts the marker size stored in the dataset into a stronger pixel size.
    """
    try:
        value = row.get("Marker_Size", fallback)

        if pd.isna(value):
            value = fallback

        return int(
            np.clip(float(value) * multiplier, min_px, max_px)
        )

    except Exception:
        return fallback


def add_marker_legend(folium_map):
    """
    Adds a detailed on-map legend explaining every marker and line type.
    """
    legend_html = """
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: rgba(255,255,255,0.94); border: 1px solid #D1D5DB; border-radius: 14px; padding: 14px 16px; width: 365px; box-shadow: 0 6px 18px rgba(0,0,0,0.18); font-family: Arial, sans-serif; color: #111827;">
        <div style="font-weight:800; font-size:15px; margin-bottom:10px;">Map legend</div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:16px; height:16px; background:#DC2626; border:2px solid #111827; border-radius:50%; display:inline-block;"></span>
            <span>Fertilizer / ammonia production plant</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:0; height:0; border-left:9px solid transparent; border-right:9px solid transparent; border-bottom:17px solid #2563EB; display:inline-block;"></span>
            <span>Wind potential</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:15px; height:15px; background:#F59E0B; border:2px solid #111827; transform:rotate(45deg); display:inline-block; margin-left:2px;"></span>
            <span>Solar potential</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:15px; height:15px; background:#059669; border:2px solid #064E3B; border-radius:50%; display:inline-block;"></span>
            <span>Wastewater treatment plant</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:15px; height:15px; background:#0284C7; border:2px solid #0F172A; display:inline-block;"></span>
            <span>Dam / reservoir</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:16px; height:16px; background:#14B8A6; border:2px solid #0F172A; clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%); display:inline-block;"></span>
            <span>Desalination plant</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:16px; height:16px; background:#8B5CF6; border:2px solid #111827; clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%); display:inline-block;"></span>
            <span>Petroleum refinery</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:22px; height:22px; background:#166534; border:2px solid white; border-radius:50%; color:white; display:inline-flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; box-shadow:0 1px 4px rgba(0,0,0,0.35);">1</span>
            <span>Ranked opportunity hotspot</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:28px; border-top:3px dashed #2563EB; display:inline-block;"></span>
            <span>Plant to matched wind asset</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:28px; border-top:3px dashed #F59E0B; display:inline-block;"></span>
            <span>Plant to matched solar asset</span>
        </div>

        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;">
            <span style="width:28px; border-top:3px dashed #059669; display:inline-block;"></span>
            <span>Plant to matched water asset</span>
        </div>

        <div style="border-top:1px solid #E5E7EB; margin-top:10px; padding-top:8px; font-size:12px; color:#6B7280; line-height:1.35;">
            Marker size reflects relative production, resource strength or capacity. Use the map layer control to show or hide layers.
        </div>
    </div>
    """

    folium_map.get_root().html.add_child(
        folium.Element(legend_html)
    )


def add_zoom_marker_scaling(
    folium_map,
    base_zoom=3,
    growth_per_zoom=0.045,
    max_scale=1.45
):
    """
    Keeps custom markers readable as the user zooms.
    """
    map_name = folium_map.get_name()

    zoom_script = f"""
    <script>
        (function() {{
            var map = {map_name};

            function resizeHydroGEMMarkers() {{
                var zoom = map.getZoom();
                var scale = 1 + ((zoom - {base_zoom}) * {growth_per_zoom});

                if (scale < 1) {{ scale = 1; }}
                if (scale > {max_scale}) {{ scale = {max_scale}; }}

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

    folium_map.get_root().html.add_child(
        folium.Element(zoom_script)
    )


def build_folium_map(
    df_plants,
    wind_assets,
    solar_assets,
    water_assets,
    petroleum_assets,
    opportunity,
    layer_flags,
    top_n_hotspots
):
    """
    Builds the full interactive Folium map.
    """
    folium_map = folium.Map(
        location=[2.0, 20.0],
        zoom_start=3,
        tiles=None,
        control_scale=True
    )

    add_base_tiles(
        folium_map,
        show_labels=layer_flags["labels"]
    )

    add_map_marker_css(folium_map)

    plant_cmap = make_colormap(
        df_plants["Production_tpa"] if not df_plants.empty else pd.Series(dtype=float),
        ["#FEE2E2", "#EF4444", "#7F1D1D"]
    )

    wind_cmap = make_colormap(
        wind_assets["Quality_Score"] if not wind_assets.empty else pd.Series(dtype=float),
        ["#DBEAFE", "#2563EB", "#1E3A8A"]
    )

    solar_cmap = make_colormap(
        solar_assets["Quality_Score"] if not solar_assets.empty else pd.Series(dtype=float),
        ["#FEF3C7", "#F59E0B", "#92400E"]
    )

    water_cmap = make_colormap(
        water_assets["Quality_Score"] if not water_assets.empty else pd.Series(dtype=float),
        ["#D1FAE5", "#059669", "#064E3B"]
    )

    petroleum_cmap = make_colormap(
        petroleum_assets["Quality_Score"] if not petroleum_assets.empty else pd.Series(dtype=float),
        ["#EDE9FE", "#8B5CF6", "#5B21B6"]
    )

    if layer_flags["plants"] and not df_plants.empty:
        plant_group = folium.FeatureGroup(
            name="Fertilizer / ammonia production plants",
            show=True
        )

        for _, row in df_plants.iterrows():
            popup = build_popup(
                row["Name"],
                [
                    ("Marker", "Red circle"),
                    ("Asset type", row["Candidate_Type"]),
                    ("Country", row["Country"]),
                    ("Production", f"{format_number(row['Production_tpa'])} tons/year"),
                    ("Latitude", f"{row['Latitude']:.4f}"),
                    ("Longitude", f"{row['Longitude']:.4f}")
                ]
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"Fertilizer / ammonia plant: {row['Name']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=420),
                icon=build_fixed_marker_icon(
                    "circle",
                    plant_cmap(row["Production_tpa"]),
                    get_marker_px(
                        row,
                        fallback=28,
                        min_px=23,
                        max_px=48,
                        multiplier=1.45
                    )
                )
            ).add_to(plant_group)

        plant_group.add_to(folium_map)

    if layer_flags["wind"] and not wind_assets.empty:
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
                    ("Quality score", f"{format_number(row['Quality_Score'], 1)}")
                ]
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"Wind: {row['Asset_Name']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=390),
                icon=build_fixed_marker_icon(
                    "triangle",
                    wind_cmap(row["Quality_Score"]),
                    get_marker_px(
                        row,
                        fallback=25,
                        min_px=22,
                        max_px=42,
                        multiplier=1.45
                    )
                )
            ).add_to(wind_group)

        wind_group.add_to(folium_map)

    if layer_flags["solar"] and not solar_assets.empty:
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
                    ("Quality score", f"{format_number(row['Quality_Score'], 1)}")
                ]
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"Solar: {row['Asset_Name']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=390),
                icon=build_fixed_marker_icon(
                    "diamond",
                    solar_cmap(row["Quality_Score"]),
                    get_marker_px(
                        row,
                        fallback=25,
                        min_px=22,
                        max_px=42,
                        multiplier=1.45
                    )
                )
            ).add_to(solar_group)

        solar_group.add_to(folium_map)

    if layer_flags["water"] and not water_assets.empty:
        water_group = folium.FeatureGroup(
            name="Water assets",
            show=True
        )

        for _, row in water_assets.iterrows():
            if row["Water_Type"] == "Wastewater treatment plant":
                shape = "circle"
                marker_label = "Green circle"
                border_color = "#064E3B"
                marker_color = water_cmap(row["Quality_Score"])

            elif row["Water_Type"] == "Dam / reservoir":
                shape = "square"
                marker_label = "Blue square"
                border_color = "#0F172A"
                marker_color = "#0284C7"

            else:
                shape = "pentagon"
                marker_label = "Teal pentagon"
                border_color = "#0F172A"
                marker_color = "#14B8A6"

            popup = build_popup(
                row["Water_Name"],
                [
                    ("Marker", marker_label),
                    ("Asset type", row["Water_Type"]),
                    ("Country", row["Country"]),
                    ("Capacity", f"{format_number(row['Capacity_Value'], 2)} {row['Capacity_Unit']}"),
                    ("Capacity score", f"{format_number(row['Quality_Score'], 1)}")
                ]
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"Water asset: {row['Water_Name']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=410),
                icon=build_fixed_marker_icon(
                    shape,
                    marker_color,
                    get_marker_px(
                        row,
                        fallback=24,
                        min_px=21,
                        max_px=42,
                        multiplier=1.45
                    ),
                    border_color
                )
            ).add_to(water_group)

        water_group.add_to(folium_map)

    if layer_flags["petroleum"] and not petroleum_assets.empty:
        petroleum_group = folium.FeatureGroup(
            name="Petroleum refineries",
            show=True
        )

        for _, row in petroleum_assets.iterrows():
            popup = build_popup(
                row["Refinery_Name"],
                [
                    ("Marker", "Purple hexagon"),
                    ("Asset type", "Petroleum refinery"),
                    ("Country", row["Country"]),
                    ("Operator", row["Operator"]),
                    ("Capacity", f"{format_number(row['Capacity_bpd'])} bpd"),
                    ("Status", row["Status"]),
                    ("Status category", row["Status_Category"])
                ]
            )

            # Different border color based on status
            border_color = "#111827"
            if row["Status_Category"] == "Closed/Shut":
                border_color = "#DC2626"
            elif row["Status_Category"] == "Planned/Proposed":
                border_color = "#F59E0B"
            elif row["Status_Category"] == "Under Development":
                border_color = "#2563EB"

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"Refinery: {row['Refinery_Name']} | {row['Country']} | {row['Status_Category']}",
                popup=folium.Popup(popup, max_width=410),
                icon=build_fixed_marker_icon(
                    "hexagon",
                    petroleum_cmap(row["Quality_Score"]),
                    get_marker_px(
                        row,
                        fallback=26,
                        min_px=22,
                        max_px=44,
                        multiplier=1.45
                    ),
                    border_color
                )
            ).add_to(petroleum_group)

        petroleum_group.add_to(folium_map)

    if layer_flags["lines"] and not opportunity.empty:
        line_group = folium.FeatureGroup(
            name="Hotspot connection lines",
            show=True
        )

        for _, row in opportunity.head(top_n_hotspots).iterrows():
            plant_point = [
                row["Latitude"],
                row["Longitude"]
            ]

            if pd.notna(row.get("Wind_Latitude")) and pd.notna(row.get("Wind_Longitude")):
                folium.PolyLine(
                    [
                        plant_point,
                        [row["Wind_Latitude"], row["Wind_Longitude"]]
                    ],
                    color="#2563EB",
                    weight=3,
                    opacity=0.7,
                    dash_array="7,7",
                    tooltip="Plant to matched wind asset"
                ).add_to(line_group)

            if pd.notna(row.get("Solar_Latitude")) and pd.notna(row.get("Solar_Longitude")):
                folium.PolyLine(
                    [
                        plant_point,
                        [row["Solar_Latitude"], row["Solar_Longitude"]]
                    ],
                    color="#F59E0B",
                    weight=3,
                    opacity=0.7,
                    dash_array="7,7",
                    tooltip="Plant to matched solar asset"
                ).add_to(line_group)

            if pd.notna(row.get("Water_Latitude")) and pd.notna(row.get("Water_Longitude")):
                folium.PolyLine(
                    [
                        plant_point,
                        [row["Water_Latitude"], row["Water_Longitude"]]
                    ],
                    color="#059669",
                    weight=3,
                    opacity=0.7,
                    dash_array="5,7",
                    tooltip="Plant to matched water asset"
                ).add_to(line_group)

        line_group.add_to(folium_map)

    if layer_flags["hotspots"] and not opportunity.empty:
        hotspot_group = folium.FeatureGroup(
            name="Top ranked opportunity hotspots",
            show=True
        )

        for rank, (_, row) in enumerate(
            opportunity.head(top_n_hotspots).iterrows(),
            start=1
        ):
            popup = build_popup(
                f"#{rank} {row['Name']}",
                [
                    ("Marker", "Green numbered badge"),
                    ("Asset type", "Ranked opportunity hotspot"),
                    ("Country", row["Country"]),
                    ("Opportunity score", f"{format_number(row['Opportunity_Score'], 2)}"),
                    ("Priority band", row["Priority_Band"]),
                    ("Nearest wind", row.get("Nearest_Wind_Name", "N/A")),
                    ("Nearest solar", row.get("Nearest_Solar_Name", "N/A")),
                    ("Nearest water", row.get("Nearest_Water_Name", "N/A"))
                ]
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"#{rank} hotspot: {row['Name']}",
                popup=folium.Popup(popup, max_width=430),
                icon=build_fixed_marker_icon(
                    "circle",
                    "#166534",
                    38,
                    "#FFFFFF",
                    label=str(rank),
                    label_color="#FFFFFF"
                )
            ).add_to(hotspot_group)

        hotspot_group.add_to(folium_map)

    folium.LayerControl(
        collapsed=False
    ).add_to(folium_map)

    add_marker_legend(folium_map)

    add_zoom_marker_scaling(
        folium_map,
        base_zoom=3,
        growth_per_zoom=0.045,
        max_scale=1.45
    )

    return folium_map


def folium_map_to_html(folium_map):
    """
    Converts the Folium map into downloadable HyperText Markup Language.
    """
    return folium_map.get_root().render()


# =====================================================
# 11. Methodology explanation, dictionary and data guide
# =====================================================

def show_scoring_methodology(raw_weights, production_score_method, quality_distance_balance):
    """
    Shows the scoring formulas used by the app.
    """
    st.markdown("### Scoring methodology")

    st.markdown(
        """
        The app scores each **fertilizer / ammonia production plant** as a possible opportunity location.
        It then checks how well that plant is supported by wind, solar and water assets.
        """
    )

    st.markdown("#### 1. Fertilizer / ammonia production plant score")

    if production_score_method == "Raw min-max":
        st.markdown(
            """
            <div class="formula-box">
            Plant Score = ((Plant Production - Lowest Plant Production)<br>
            ÷ (Highest Plant Production - Lowest Plant Production)) × 100
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "Raw min-max scoring can make one very large plant score 100 while smaller plants score close to 0. "
            "That is what caused values like 0.9 and 0.7 in your earlier view."
        )

    else:
        st.markdown(
            """
            <div class="formula-box">
            Log Plant Production = ln(1 + Plant Production)<br><br>
            Plant Score = ((Log Plant Production - Lowest Log Plant Production)<br>
            ÷ (Highest Log Plant Production - Lowest Log Plant Production)) × 100
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Log-scaled scoring is recommended here because production capacity is highly skewed. "
            "It still rewards larger plants, but it prevents one mega plant from compressing all other plant scores near zero."
        )

    st.markdown("#### 2. Resource quality score")

    st.markdown(
        """
        <div class="formula-box">
        Quality Score = ((Resource Value - Lowest Resource Value)<br>
        ÷ (Highest Resource Value - Lowest Resource Value)) × 100
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Quality is calculated differently depending on the resource:
        - **Wind quality** uses wind power density.
        - **Solar quality** uses solar capacity.
        - **Water quality** uses water capacity, scored within each water asset type because dams, wastewater plants and desalination plants use different capacity units.
        """
    )

    st.markdown("#### 3. Proximity score")

    st.markdown(
        """
        <div class="formula-box">
        Proximity Score = max(0, 100 × (1 - Distance ÷ Maximum Useful Distance))
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This means closer assets score better. Once an asset is farther than the selected maximum useful distance,
        its proximity score becomes 0.
        """
    )

    st.markdown("#### 4. Fit score")

    st.markdown(
        f"""
        <div class="formula-box">
        Fit Score = (Quality Score × {quality_distance_balance}%)<br>
        + (Proximity Score × {100 - quality_distance_balance}%)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Fit scores are calculated separately for wind, solar and water.  
        The sidebar control lets you decide whether resource strength or closeness should matter more.
        """
    )

    st.markdown("#### 5. Final Opportunity Score")

    st.markdown(
        """
        <div class="formula-box">
        Opportunity Score =<br>
        (Fertilizer / Ammonia Production Plant Score × Plant Weight<br>
        + Wind Fit Score × Wind Weight<br>
        + Solar Fit Score × Solar Weight<br>
        + Water Fit Score × Water Weight) ÷ 100
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Current weights")

    st.dataframe(
        pd.DataFrame({
            "Component": list(raw_weights.keys()),
            "Selected weight": list(raw_weights.values())
        }),
        width="stretch",
        hide_index=True
    )


def show_dictionary(raw_weights=None, production_score_method=None, quality_distance_balance=None):
    """
    Displays plain-language explanations for dashboard terms.
    """
    st.subheader("Dictionary and methodology guide")

    st.markdown(
        """
        This dashboard is a first-pass screening tool. It does not decide where a project should be built.
        It ranks existing fertilizer / ammonia production plant locations based on production scale,
        renewable fit and water fit.
        """
    )

    terms = pd.DataFrame({
        "Term": [
            "Opportunity Score",
            "Hotspot",
            "Fertilizer / ammonia production plant",
            "Fertilizer / ammonia production plant score",
            "Wind Fit Score",
            "Solar Fit Score",
            "Water Fit Score",
            "Quality Score",
            "Proximity Score",
            "Priority Band"
        ],
        "Meaning": [
            "The final 0-100 score used to rank candidate locations.",
            "A top-ranked candidate location after current filters and weights are applied.",
            "An existing fertilizer or ammonia production site used as a possible opportunity location.",
            "A 0-100 score based on production scale. Larger production capacity generally scores higher.",
            "A combined score based on wind resource strength and distance from the production plant.",
            "A combined score based on solar resource strength and distance from the production plant.",
            "A combined score based on water asset capacity and distance from the production plant.",
            "A 0-100 score showing how strong a resource is compared with other resources in the selected dataset.",
            "A 0-100 score showing how close a resource is to the production plant.",
            "A simple grouping of locations into High priority, Moderate priority or Lower priority."
        ],
        "Important note": [
            "It is a screening score, not a feasibility study.",
            "Hotspots should be investigated first, not automatically selected.",
            "The model starts from existing industrial production sites, not empty land parcels.",
            "This depends on the production field in the workbook.",
            "Wind uses wind power density where available.",
            "Solar uses solar capacity where available.",
            "Water uses asset capacity within each water asset type.",
            "Quality scores are relative to the loaded data, not an absolute engineering benchmark.",
            "Straight-line distance is used, not road, grid or pipeline distance.",
            "The bands are directional and should support discussion."
        ]
    })

    st.dataframe(
        terms,
        width="stretch",
        hide_index=True
    )

    if raw_weights is not None and production_score_method is not None:
        show_scoring_methodology(
            raw_weights,
            production_score_method,
            quality_distance_balance
        )


def show_data_guide():
    """
    Shows the expected workbook structure for users uploading their own data.
    """
    st.subheader("Data structure guide")

    st.markdown(
        """
        The app expects one main Excel workbook. It can also use a separate water workbook if water sheets are not inside the main workbook.

        Minimum required sheets:
        - **Fertilizer Plants**
        - **Wind Potential**

        Recommended sheets:
        - **Solar Potential**
        - **Waste water treatment plants**
        - **Dams**
        - **Desalination plants**
        """
    )

    st.markdown("### Fertilizer Plants sheet")

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

    st.markdown("### Wind Potential sheet")

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
                "Example Wind Region",
                "11.9964",
                "8.5167",
                "7.2",
                "450-520"
            ]
        }),
        width="stretch",
        hide_index=True
    )

    st.markdown("### Solar Potential sheet")

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

    st.markdown("### Water sheets")

    st.dataframe(
        pd.DataFrame({
            "Sheet": [
                "Waste water treatment plants",
                "Dams",
                "Desalination plants"
            ],
            "Required columns": [
                "Country, facility/name, capacity, latitude, longitude",
                "Country, dam/site name, capacity, latitude, longitude",
                "Country, plant name, capacity, latitude, longitude"
            ],
            "How it is used": [
                "Scored as water availability asset",
                "Scored as water availability asset",
                "Scored as water availability asset"
            ]
        }),
        width="stretch",
        hide_index=True
    )

    st.markdown("### Petroleum processing sites (CSV)")

    st.dataframe(
        pd.DataFrame({
            "Column": [
                "Refinery Name",
                "Country",
                "Owner/Operator",
                "Location",
                "Approx. Lat/Long",
                "Capacity (bpd)",
                "Approx. Annual (barrels)",
                "Status / Notes (April 2026)"
            ],
            "Required": [
                "Yes",
                "Yes",
                "No",
                "No",
                "Yes",
                "Yes",
                "No",
                "No"
            ],
            "Example": [
                "Dangote Refinery",
                "Nigeria",
                "Dangote Group",
                "Lekki Free Trade Zone, Lagos",
                "6.43°N, 3.99°E",
                "650000",
                "~237 million",
                "Operational"
            ]
        }),
        width="stretch",
        hide_index=True
    )


# =====================================================
# 12. Header
# =====================================================

st.markdown(
    """
    <div class="main-title">
        HydroGEM fertilizer, renewable energy and water opportunity map
    </div>
    <div class="subtitle">
        A first-pass screening dashboard for ranking fertilizer / ammonia production plant locations using wind, solar and water availability.
        Includes petroleum processing sites for infrastructure context.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 13. Sidebar: data source selection
# =====================================================

st.sidebar.header("Data source")

data_mode = st.sidebar.radio(
    "Choose data mode",
    [
        "Use root-folder workbook",
        "Upload workbook"
    ]
)

main_file_bytes = None
main_source_name = None
separate_water_file_bytes = None
separate_water_source_name = None
petroleum_file_bytes = None
petroleum_source_name = None

if data_mode == "Use root-folder workbook":
    st.sidebar.caption(
        "The app will look for the workbook in the same folder as app.py."
    )

    if DEFAULT_MAIN_WORKBOOK_PATH.exists():
        main_file_bytes = DEFAULT_MAIN_WORKBOOK_PATH.read_bytes()
        main_source_name = DEFAULT_MAIN_WORKBOOK_PATH.name

        st.sidebar.success(
            f"Main workbook loaded: {DEFAULT_MAIN_WORKBOOK_PATH.name}"
        )
    else:
        st.sidebar.error(
            f"Main workbook missing: {DEFAULT_MAIN_WORKBOOK_PATH.name}"
        )

    if DEFAULT_WATER_WORKBOOK_PATH.exists():
        separate_water_file_bytes = DEFAULT_WATER_WORKBOOK_PATH.read_bytes()
        separate_water_source_name = DEFAULT_WATER_WORKBOOK_PATH.name

        st.sidebar.info(
            f"Separate water workbook available: {DEFAULT_WATER_WORKBOOK_PATH.name}"
        )

    if DEFAULT_PETROLEUM_DATA_PATH.exists():
        petroleum_file_bytes = DEFAULT_PETROLEUM_DATA_PATH.read_bytes()
        petroleum_source_name = DEFAULT_PETROLEUM_DATA_PATH.name

        st.sidebar.info(
            f"Petroleum data loaded: {DEFAULT_PETROLEUM_DATA_PATH.name}"
        )

else:
    uploaded_main_file = st.sidebar.file_uploader(
        "Upload main workbook",
        type=["xlsx"],
        key="main_workbook_uploader"
    )

    uploaded_water_file = st.sidebar.file_uploader(
        "Optional: upload separate water workbook",
        type=["xlsx"],
        key="water_workbook_uploader"
    )

    uploaded_petroleum_file = st.sidebar.file_uploader(
        "Optional: upload petroleum processing sites CSV",
        type=["csv"],
        key="petroleum_uploader"
    )

    if uploaded_main_file is not None:
        main_file_bytes = uploaded_main_file.getvalue()
        main_source_name = uploaded_main_file.name

        st.sidebar.success(
            f"Main workbook uploaded: {uploaded_main_file.name}"
        )

    if uploaded_water_file is not None:
        separate_water_file_bytes = uploaded_water_file.getvalue()
        separate_water_source_name = uploaded_water_file.name

        st.sidebar.success(
            f"Water workbook uploaded: {uploaded_water_file.name}"
        )

    if uploaded_petroleum_file is not None:
        petroleum_file_bytes = uploaded_petroleum_file.getvalue()
        petroleum_source_name = uploaded_petroleum_file.name

        st.sidebar.success(
            f"Petroleum data uploaded: {uploaded_petroleum_file.name}"
        )

st.sidebar.divider()


# =====================================================
# 14. Stop early if no workbook is loaded
# =====================================================

if main_file_bytes is None:
    tab_start, tab_guide = st.tabs(
        [
            "Start here",
            "Data guide"
        ]
    )

    with tab_start:
        st.info(
            "No main workbook is loaded yet."
        )

        st.markdown(
            f"""
            Place this file in your app root folder:

            ```text
            {DEFAULT_MAIN_WORKBOOK_PATH.name}
            ```

            Then refresh the app, or switch to upload mode.
            """
        )

    with tab_guide:
        show_data_guide()

    st.stop()


# =====================================================
# 15. Load and validate data
# =====================================================

try:
    with st.spinner("Loading and validating main workbook..."):
        main_bundle = load_main_workbook(
            main_file_bytes,
            main_source_name
        )

except DataValidationError as exc:
    tab_error, tab_guide = st.tabs(
        [
            "Workbook error",
            "Data guide"
        ]
    )

    with tab_error:
        st.markdown(
            f"""
            <div class="error-box">
                <h3>{escape_text(exc.title)}</h3>
                <ul>{''.join([f'<li>{escape_text(issue)}</li>' for issue in exc.issues])}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_guide:
        show_data_guide()

    st.stop()

water_bundle = None
water_error = None

if main_bundle.df_water_from_main.empty and separate_water_file_bytes is not None:
    try:
        with st.spinner("Loading separate water workbook..."):
            water_bundle = load_separate_water_workbook(
                separate_water_file_bytes,
                separate_water_source_name
            )
    except DataValidationError as exc:
        water_error = exc

petroleum_bundle = None
petroleum_error = None

if petroleum_file_bytes is not None:
    try:
        with st.spinner("Loading petroleum processing sites..."):
            petroleum_bundle = load_petroleum_data(
                petroleum_file_bytes,
                petroleum_source_name
            )
    except DataValidationError as exc:
        petroleum_error = exc

df_plants = main_bundle.df_plants
df_wind = main_bundle.df_wind
df_solar = main_bundle.df_solar

if not main_bundle.df_water_from_main.empty:
    df_water = main_bundle.df_water_from_main
    water_source_note = "Water data loaded from the main workbook."
elif water_bundle is not None:
    df_water = water_bundle.df_water
    water_source_note = "Water data loaded from the separate water workbook."
else:
    df_water = pd.DataFrame()
    water_source_note = "No usable water data loaded."

df_refineries = petroleum_bundle.df_refineries if petroleum_bundle is not None else pd.DataFrame()
petroleum_source_note = petroleum_bundle.source_name if petroleum_bundle is not None else "No petroleum data loaded"


# =====================================================
# 16. Sidebar: filters and methodology controls
# =====================================================

st.sidebar.header("Filters")

country_pool = sorted(
    set(df_plants["Country"].dropna().unique())
    | set(df_wind["Country"].dropna().unique())
    | (set(df_solar["Country"].dropna().unique()) if not df_solar.empty else set())
    | (set(df_water["Country"].dropna().unique()) if not df_water.empty else set())
    | (set(df_refineries["Country"].dropna().unique()) if not df_refineries.empty else set())
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=country_pool,
    default=country_pool
)

production_range = safe_range_slider(
    "Fertilizer / ammonia production range, tons/year",
    df_plants["Production_tpa"]
)

wind_range = safe_range_slider(
    "Wind power density range, W/m²",
    df_wind["Wind_Density_wm2"]
)

solar_range = (
    safe_range_slider(
        "Solar capacity range, MW",
        df_solar["Solar_Capacity_MW"]
    )
    if not df_solar.empty
    else (None, None)
)

water_capacity_range = (
    safe_range_slider(
        "Water asset capacity range",
        df_water["Capacity_Value"]
    )
    if not df_water.empty
    else (None, None)
)

# Petroleum filters
petroleum_status_options = ["All"] + sorted(df_refineries["Status_Category"].unique()) if not df_refineries.empty else ["All"]
selected_petroleum_status = st.sidebar.selectbox(
    "Petroleum refinery status filter",
    options=petroleum_status_options,
    index=0,
    disabled=df_refineries.empty
)

petroleum_capacity_range = (
    safe_range_slider(
        "Refinery capacity range, bpd",
        df_refineries["Capacity_bpd"]
    )
    if not df_refineries.empty
    else (None, None)
)

st.sidebar.divider()

st.sidebar.header("Methodology controls")

production_score_method = st.sidebar.selectbox(
    "Fertilizer / ammonia production plant scoring method",
    [
        "Log-scaled min-max",
        "Raw min-max"
    ],
    index=0,
    help=(
        "Log-scaled min-max is recommended when one very large plant dominates the dataset. "
        "Raw min-max gives the largest plant 100 and can compress smaller plants close to 0."
    )
)

matching_rule = st.sidebar.selectbox(
    "Resource matching rule",
    [
        "Best quality-distance fit",
        "Nearest asset"
    ],
    index=0
)

quality_distance_balance = st.sidebar.slider(
    "Resource quality weight inside each fit score",
    min_value=0,
    max_value=100,
    value=50,
    help="50 means resource fit is half quality and half distance proximity."
)

max_wind_distance_km = st.sidebar.slider(
    "Maximum useful wind distance, km",
    min_value=50,
    max_value=2500,
    value=600,
    step=50
)

max_solar_distance_km = st.sidebar.slider(
    "Maximum useful solar distance, km",
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

available_water_types = (
    sorted(df_water["Water_Type"].unique())
    if not df_water.empty
    else []
)

selected_water_types = st.sidebar.multiselect(
    "Water asset types used in scoring",
    options=available_water_types,
    default=available_water_types,
    disabled=df_water.empty
)

with st.sidebar.expander("Score weights", expanded=True):
    plant_weight = st.slider(
        "Fertilizer / ammonia production plant weight",
        min_value=0,
        max_value=100,
        value=35
    )

    wind_weight = st.slider(
        "Wind weight",
        min_value=0,
        max_value=100,
        value=25
    )

    solar_weight = st.slider(
        "Solar weight",
        min_value=0,
        max_value=100,
        value=25,
        disabled=df_solar.empty
    )

    water_weight = st.slider(
        "Water weight",
        min_value=0,
        max_value=100,
        value=15,
        disabled=df_water.empty
    )

raw_weights = {
    "Fertilizer / ammonia production plant": plant_weight,
    "Wind": wind_weight,
    "Solar": solar_weight if not df_solar.empty else 0,
    "Water": water_weight if not df_water.empty else 0
}

weight_total = sum(raw_weights.values())

if weight_total != 100:
    st.sidebar.warning(
        f"Selected weights add up to {weight_total}, not 100. "
        "The app will apply the selected values directly."
    )

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

layer_flags = {
    "plants": st.sidebar.checkbox(
        "Show fertilizer / ammonia production plants",
        value=True
    ),
    "wind": st.sidebar.checkbox(
        "Show wind potential",
        value=True
    ),
    "solar": st.sidebar.checkbox(
        "Show solar potential",
        value=True,
        disabled=df_solar.empty
    ),
    "water": st.sidebar.checkbox(
        "Show water assets",
        value=True,
        disabled=df_water.empty
    ),
    "petroleum": st.sidebar.checkbox(
        "Show petroleum refineries",
        value=True,
        disabled=df_refineries.empty
    ),
    "hotspots": st.sidebar.checkbox(
        "Show ranked hotspots",
        value=True
    ),
    "lines": st.sidebar.checkbox(
        "Show hotspot connection lines",
        value=True
    ),
    "labels": st.sidebar.checkbox(
        "Show country/place labels",
        value=True
    )
}


# =====================================================
# 17. Apply filters and build scoring tables
# =====================================================

filtered_plants = df_plants[
    df_plants["Country"].isin(selected_countries)
].copy()

filtered_plants = apply_numeric_range_filter(
    filtered_plants,
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

    filtered_water = filtered_water[
        filtered_water["Water_Type"].isin(selected_water_types)
    ].copy()

    filtered_water = apply_numeric_range_filter(
        filtered_water,
        "Capacity_Value",
        water_capacity_range
    )
else:
    filtered_water = pd.DataFrame()

# Filter petroleum refineries
if not df_refineries.empty:
    filtered_refineries = df_refineries[
        df_refineries["Country"].isin(selected_countries)
    ].copy()
    
    if selected_petroleum_status != "All":
        filtered_refineries = filtered_refineries[
            filtered_refineries["Status_Category"] == selected_petroleum_status
        ].copy()
    
    filtered_refineries = apply_numeric_range_filter(
        filtered_refineries,
        "Capacity_bpd",
        petroleum_capacity_range
    )
else:
    filtered_refineries = pd.DataFrame()

wind_assets = prepare_wind_assets(
    filtered_wind
)

solar_assets = prepare_solar_assets(
    filtered_solar
)

water_assets = prepare_water_assets(
    filtered_water,
    selected_water_types
)

petroleum_assets = prepare_petroleum_assets(
    filtered_refineries,
    status_filter=selected_petroleum_status if selected_petroleum_status != "All" else None
)

opportunity = build_opportunity_table(
    filtered_plants,
    wind_assets,
    solar_assets,
    water_assets,
    raw_weights,
    max_wind_distance_km,
    max_solar_distance_km,
    max_water_distance_km,
    matching_rule,
    quality_distance_balance,
    production_score_method
)

filtered_opportunity = opportunity[
    opportunity["Opportunity_Score"] >= min_score
].copy()

top_hotspots = filtered_opportunity.head(
    top_n_hotspots
).copy()


# =====================================================
# 18. Main page tabs
# =====================================================

tab_overview, tab_map, tab_hotspots, tab_ranking, tab_petroleum, tab_quality, tab_dictionary, tab_guide = st.tabs([
    "Overview",
    "Satellite map",
    "Hotspots",
    "Opportunity ranking",
    "Petroleum refineries",
    "Data quality",
    "Dictionary",
    "Data guide"
])


# =====================================================
# 19. Overview tab
# =====================================================

with tab_overview:
    st.subheader("Executive overview")

    if water_error is not None:
        st.warning(
            f"Separate water workbook could not be loaded: {water_error.title}"
        )

    if petroleum_error is not None:
        st.warning(
            f"Petroleum data could not be loaded: {petroleum_error.title}"
        )

    st.markdown(
        f"""
        <div class="success-box">
            <b>Main workbook used:</b> {escape_text(main_bundle.source_name)}<br>
            <b>Water source:</b> {escape_text(water_source_note)}<br>
            <b>Petroleum source:</b> {escape_text(petroleum_source_note)}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Fertilizer / ammonia production plants",
        f"{len(filtered_plants):,}"
    )

    col2.metric(
        "Wind assets",
        f"{len(wind_assets):,}"
    )

    col3.metric(
        "Solar assets",
        f"{len(solar_assets):,}"
    )

    col4.metric(
        "Water assets",
        f"{len(water_assets):,}"
    )

    col5.metric(
        "Petroleum refineries",
        f"{len(filtered_refineries):,}"
    )

    col6.metric(
        "Top score",
        f"{filtered_opportunity['Opportunity_Score'].max():.2f}"
        if not filtered_opportunity.empty
        else "N/A"
    )

    st.markdown("### Data used in this analysis")

    overview_rows = []
    overview_rows.extend(main_bundle.report["datasets"])

    if water_bundle is not None:
        overview_rows.extend(water_bundle.report["datasets"])
    
    if petroleum_bundle is not None:
        overview_rows.extend(petroleum_bundle.report["datasets"])

    if overview_rows:
        st.dataframe(
            pd.DataFrame(overview_rows),
            width="stretch",
            hide_index=True
        )
    else:
        st.warning("No dataset summary is available.")

    st.markdown("### How to read this page")

    st.markdown(
        """
        This app ranks existing **fertilizer / ammonia production plant locations**.
        A location scores better when it has:

        1. Larger existing production capacity
        2. Stronger or closer wind potential
        3. Stronger or closer solar potential
        4. Stronger or closer water availability

        The result is a screening view. It should guide where deeper feasibility analysis should begin.

        **Petroleum processing sites** are shown for infrastructure context and can help identify:
        - Existing industrial clusters
        - Potential hydrogen off-takers
        - Logistics and port infrastructure
        - Areas with established energy infrastructure
        """
    )

    show_scoring_methodology(
        raw_weights,
        production_score_method,
        quality_distance_balance
    )


# =====================================================
# 20. Map tab
# =====================================================

with tab_map:
    st.subheader("Satellite map")

    st.markdown(
        """
        The map shows the fertilizer / ammonia production plant locations, renewable resource locations, water assets,
        and petroleum refineries used in the analysis.
        
        The numbered green markers show the highest-ranked opportunity hotspots under the current filters and methodology settings.
        """
    )

    folium_map = build_folium_map(
        filtered_plants,
        wind_assets,
        solar_assets,
        water_assets,
        petroleum_assets,
        filtered_opportunity,
        layer_flags,
        top_n_hotspots
    )

    st_folium(
        folium_map,
        height=760,
        width=1200,
        returned_objects=[]
    )

    st.download_button(
        "Download current map as HTML",
        data=folium_map_to_html(folium_map),
        file_name="hydrogem_opportunity_map.html",
        mime="text/html"
    )


# =====================================================
# 21. Hotspots tab
# =====================================================

with tab_hotspots:
    st.subheader("Hotspot analysis")

    st.markdown(
        """
        Hotspots are the top-ranked **fertilizer / ammonia production plant locations** after applying the current filters,
        distance thresholds and score weights.

        The app calculates each location's score by combining:

        - Fertilizer / ammonia production plant score
        - Wind fit score
        - Solar fit score
        - Water fit score

        The table is sorted from highest to lowest Opportunity Score.
        These locations are not final recommendations; they are the places that should be reviewed first.
        """
    )

    show_scoring_methodology(
        raw_weights,
        production_score_method,
        quality_distance_balance
    )

    if top_hotspots.empty:
        st.warning(
            "No hotspot matches the current filters. Reduce the filters or lower the minimum score."
        )
    else:
        hotspot_columns = [
            "Name",
            "Country",
            "Opportunity_Score",
            "Priority_Band",
            "Production_tpa",
            "Fertilizer_Ammonia_Production_Plant_Score",
            "Wind_Fit_Score",
            "Distance_to_Wind_km",
            "Solar_Fit_Score",
            "Distance_to_Solar_km",
            "Water_Fit_Score",
            "Distance_to_Water_km"
        ]

        st.dataframe(
            top_hotspots[hotspot_columns],
            width="stretch",
            hide_index=True
        )

        st.markdown("### Country summary for visible hotspots")

        country_summary = (
            top_hotspots
            .groupby("Country", as_index=False)
            .agg(
                Hotspot_Count=("Name", "count"),
                Average_Score=("Opportunity_Score", "mean"),
                Highest_Score=("Opportunity_Score", "max"),
                Total_Production_tpa=("Production_tpa", "sum"),
                Average_Wind_Distance_km=("Distance_to_Wind_km", "mean"),
                Average_Solar_Distance_km=("Distance_to_Solar_km", "mean"),
                Average_Water_Distance_km=("Distance_to_Water_km", "mean")
            )
            .sort_values(
                [
                    "Hotspot_Count",
                    "Highest_Score"
                ],
                ascending=[
                    False,
                    False
                ]
            )
        )

        st.dataframe(
            country_summary.round(2),
            width="stretch",
            hide_index=True
        )


# =====================================================
# 22. Opportunity ranking tab
# =====================================================

with tab_ranking:
    st.subheader("Full opportunity ranking")

    st.markdown(
        """
        The Opportunity Ranking table shows every fertilizer / ammonia production plant that passed the current filters.

        For each plant, the app selects a matched wind asset, matched solar asset and matched water asset using the selected matching rule.
        It then calculates the fit scores and applies the selected weights to produce the final Opportunity Score.
        """
    )

    show_scoring_methodology(
        raw_weights,
        production_score_method,
        quality_distance_balance
    )

    if filtered_opportunity.empty:
        st.warning("No candidate locations match the current filters.")
    else:
        ranking_columns = [
            "Name",
            "Country",
            "Production_tpa",
            "Fertilizer_Ammonia_Production_Plant_Score",
            "Nearest_Wind_Name",
            "Wind_Quality_Score",
            "Wind_Proximity_Score",
            "Wind_Fit_Score",
            "Distance_to_Wind_km",
            "Nearest_Solar_Name",
            "Solar_Quality_Score",
            "Solar_Proximity_Score",
            "Solar_Fit_Score",
            "Distance_to_Solar_km",
            "Nearest_Water_Name",
            "Nearest_Water_Type",
            "Water_Quality_Score",
            "Water_Proximity_Score",
            "Water_Fit_Score",
            "Distance_to_Water_km",
            "Opportunity_Score",
            "Priority_Band"
        ]

        st.dataframe(
            filtered_opportunity[ranking_columns],
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download filtered ranking as CSV",
            data=filtered_opportunity.to_csv(index=False).encode("utf-8"),
            file_name="hydrogem_opportunity_ranking.csv",
            mime="text/csv"
        )


# =====================================================
# 23. Petroleum refineries tab
# =====================================================

with tab_petroleum:
    st.subheader("Petroleum processing sites")

    st.markdown(
        """
        This tab shows petroleum refinery data loaded from the CSV file. 
        These sites can provide important infrastructure context for hydrogen opportunities:

        - **Existing industrial clusters** that may support hydrogen production and distribution
        - **Potential hydrogen off-takers** in the refining sector
        - **Logistics infrastructure** including ports and pipelines
        - **Established energy workforce** and supply chains

        The data includes operational status, capacity, and location information.
        """
    )

    if filtered_refineries.empty:
        st.warning(
            "No petroleum refineries match the current filters. "
            "Try adjusting the country filter, status filter, or capacity range."
        )
        
        # Show summary of available data
        if not df_refineries.empty:
            st.markdown("### Available petroleum data summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total refineries in dataset", f"{len(df_refineries):,}")
            col2.metric("Countries represented", f"{df_refineries['Country'].nunique():,}")
            
            status_counts = df_refineries["Status_Category"].value_counts()
            col3.metric("Operational refineries", f"{status_counts.get('Operational', 0):,}")
            
            st.dataframe(
                df_refineries[["Refinery_Name", "Country", "Capacity_bpd", "Status_Category", "Status"]],
                width="stretch",
                hide_index=True
            )
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total refineries", f"{len(filtered_refineries):,}")
        col2.metric("Total capacity (bpd)", f"{format_number(filtered_refineries['Capacity_bpd'].sum())}")
        
        status_counts = filtered_refineries["Status_Category"].value_counts()
        col3.metric("Operational", f"{status_counts.get('Operational', 0):,}")
        col4.metric("Countries", f"{filtered_refineries['Country'].nunique():,}")
        
        # Main table
        st.markdown("### Refinery details")
        
        display_columns = [
            "Refinery_Name",
            "Country",
            "Operator",
            "Location",
            "Capacity_bpd",
            "Annual_Production",
            "Status_Category",
            "Status"
        ]
        
        # Format the data for display
        display_df = filtered_refineries[display_columns].copy()
        display_df["Capacity_bpd"] = display_df["Capacity_bpd"].apply(lambda x: format_number(x))
        display_df["Annual_Production"] = display_df["Annual_Production"].apply(
            lambda x: format_number(x) if pd.notna(x) else "N/A"
        )
        
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )
        
        # Country summary
        st.markdown("### Country summary")
        
        country_summary = (
            filtered_refineries
            .groupby("Country", as_index=False)
            .agg(
                Refinery_Count=("Refinery_Name", "count"),
                Total_Capacity_bpd=("Capacity_bpd", "sum"),
                Operational_Count=(
                    "Status_Category",
                    lambda x: (x == "Operational").sum()
                )
            )
            .sort_values("Total_Capacity_bpd", ascending=False)
        )
        
        country_summary["Total_Capacity_bpd"] = country_summary["Total_Capacity_bpd"].apply(
            lambda x: format_number(x)
        )
        
        st.dataframe(
            country_summary,
            width="stretch",
            hide_index=True
        )
        
        # Status distribution
        st.markdown("### Status distribution")
        
        status_dist = filtered_refineries["Status_Category"].value_counts().reset_index()
        status_dist.columns = ["Status Category", "Count"]
        
        st.dataframe(
            status_dist,
            width="stretch",
            hide_index=True
        )
        
        st.download_button(
            "Download petroleum data as CSV",
            data=filtered_refineries.to_csv(index=False).encode("utf-8"),
            file_name="petroleum_refineries.csv",
            mime="text/csv"
        )


# =====================================================
# 24. Data quality tab
# =====================================================

with tab_quality:
    st.subheader("Data quality and validation report")

    st.markdown(
        """
        This tab shows which workbooks and sheets were used, how many rows were loaded,
        and how many rows were dropped during validation.

        Rows are usually dropped because they are missing coordinates, missing names,
        or missing positive capacity/resource values.
        """
    )

    st.markdown(
        f"""
        <div class="success-box">
            <b>Main workbook loaded:</b> {escape_text(main_bundle.source_name)}
        </div>
        """,
        unsafe_allow_html=True
    )

    if main_bundle.report["datasets"]:
        st.dataframe(
            pd.DataFrame(main_bundle.report["datasets"]),
            width="stretch",
            hide_index=True
        )
    else:
        st.warning("No validation summary was generated.")

    if water_bundle is not None:
        st.markdown(
            f"""
            <div class="success-box">
                <b>Separate water workbook loaded:</b> {escape_text(water_bundle.source_name)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            pd.DataFrame(water_bundle.report["datasets"]),
            width="stretch",
            hide_index=True
        )

    if petroleum_bundle is not None:
        st.markdown(
            f"""
            <div class="success-box">
                <b>Petroleum data loaded:</b> {escape_text(petroleum_bundle.source_name)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            pd.DataFrame(petroleum_bundle.report["datasets"]),
            width="stretch",
            hide_index=True
        )

    if water_error is not None:
        st.markdown(
            f"""
            <div class="error-box">
                <h4>{escape_text(water_error.title)}</h4>
                <ul>{''.join([f'<li>{escape_text(issue)}</li>' for issue in water_error.issues])}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    if petroleum_error is not None:
        st.markdown(
            f"""
            <div class="error-box">
                <h4>{escape_text(petroleum_error.title)}</h4>
                <ul>{''.join([f'<li>{escape_text(issue)}</li>' for issue in petroleum_error.issues])}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Validation notes")

    warnings = []
    warnings.extend(main_bundle.report.get("warnings", []))

    if water_bundle is not None:
        warnings.extend(water_bundle.report.get("warnings", []))

    if petroleum_bundle is not None:
        warnings.extend(petroleum_bundle.report.get("warnings", []))

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No major validation warnings detected.")

    with st.expander("Preview cleaned fertilizer / ammonia production plant data"):
        st.dataframe(
            df_plants.head(50),
            width="stretch",
            hide_index=True
        )

    with st.expander("Preview cleaned wind data"):
        st.dataframe(
            df_wind.head(50),
            width="stretch",
            hide_index=True
        )

    if not df_solar.empty:
        with st.expander("Preview cleaned solar data"):
            st.dataframe(
                df_solar.head(50),
                width="stretch",
                hide_index=True
            )

    if not df_water.empty:
        with st.expander("Preview cleaned water data"):
            st.dataframe(
                df_water.head(100),
                width="stretch",
                hide_index=True
            )

    if not df_refineries.empty:
        with st.expander("Preview cleaned petroleum data"):
            st.dataframe(
                df_refineries.head(50),
                width="stretch",
                hide_index=True
            )


# =====================================================
# 25. Dictionary tab
# =====================================================

with tab_dictionary:
    show_dictionary(
        raw_weights=raw_weights,
        production_score_method=production_score_method,
        quality_distance_balance=quality_distance_balance
    )


# =====================================================
# 26. Data guide tab
# =====================================================

with tab_guide:
    show_data_guide()
