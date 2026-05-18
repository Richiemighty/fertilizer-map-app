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
    page_title="HydroGEM hydrogen opportunity map",
    page_icon="🌍",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

DEFAULT_ENERGY_DATASET_PATH = APP_DIR / "Fertilizer Plants_AHDS.xlsx"
DEFAULT_WATER_DATASET_PATH = APP_DIR / "Water availability.xlsx"
DEFAULT_PORT_DATASET_PATH = APP_DIR / "Shipping Ports.xlsx"
DEFAULT_COMMODITY_DATASET_PATH = APP_DIR / "Petroleum_Urea_Ammonia Commodity Report  (1).xlsx"


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

        .info-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.04);
            margin-bottom: 12px;
            min-height: 170px;
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


@dataclass
class PortBundle:
    df_ports: pd.DataFrame
    report: dict
    source_name: str
    used_builtin: bool


@dataclass
class CommodityBundle:
    df_ammonia_country: pd.DataFrame
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
    "fertilizer": ["Fertilizer Plants", "Fertilizer", "Fertilizer Plant", "Plants"],
    "wind": ["Wind Potential", "Wind", "Wind Data", "Wind Resource"],
    "solar": ["Solar Potential", "Solar", "Solar Data", "Solar Projects"]
}

WATER_SHEET_ALIASES = {
    "wastewater": ["Waste water facilities", "Wastewater facilities", "Waste Water Facilities", "Wastewater", "Waste Water", "Wastewater Plants"],
    "dams": ["Dams", "Dam", "Reservoirs"],
    "desalination": ["Desalination plants", "Desalination Plants", "Desalination", "Desalination facilities"]
}

PORT_SHEET_ALIASES = {
    "ports": ["Shipping Ports", "Ports", "Port Locations", "Shipping Port Locations", "Seaports", "Sea Ports"]
}

FERTILIZER_COLUMN_ALIASES = {
    "Name": ["Name", "Plant Name", "Fertilizer Plant", "Fertilizer Plant Name", "Ammonia Plant", "Ammonia Plant Name"],
    "Country": ["Country"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Production_tpa": ["Production (tons/ anum)", "Production (tons/ annum)", "Production tons annum", "Production tons per annum", "Production", "Production_tpa", "Capacity", "Capacity_tpa", "Ammonia Capacity", "Ammonia Capacity tpa"]
}

WIND_COLUMN_ALIASES = {
    "Country": ["Country"],
    "Region": ["Region", "Location", "Wind Region", "Site"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Wind_Speed_mps_100m": ["Wind Speed (m/s) at 100m", "Wind Speed", "Wind Speed m/s", "Wind Speed at 100m", "Wind_Speed_mps_100m"],
    "Wind_Density_wm2": ["Wind Power Density (W/m²)", "Wind Power Density (W/m2)", "Wind Density", "Wind Power Density", "Wind_Density_wm2"]
}

SOLAR_COLUMN_ALIASES = {
    "Country": ["Country"],
    "Solar_Site": ["Site", "Solar Site", "Location", "Project", "Solar_Site"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Solar_Capacity_MW": ["Production Capacity(MW)", "Production Capacity (MW)", "Solar Capacity", "Solar Capacity MW", "Capacity MW", "Solar_Capacity_MW"]
}

WASTEWATER_COLUMN_ALIASES = {
    "Country": ["Country"],
    "Water_Name": ["Waste water Facility Name", "Wastewater Facility Name", "Facility Name", "Site Name", "Name"],
    "Primary_Source": ["Primary Source", "Water Source", "Source Type"],
    "Capacity_Value": ["Capacity (m³/d)", "Capacity (m3/d)", "Capacity (m3/day)", "Capacity m3 day", "Capacity", "Capacity_m3_day"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Source": ["Source", "Reference"]
}

DAMS_COLUMN_ALIASES = {
    "Country": ["Country"],
    "Water_Name": ["Site Name", "Dam Name", "Reservoir Name", "Name"],
    "Primary_Source": ["Water Souce", "Water Source", "Source Type"],
    "Capacity_Value": ["Capacity MCM", "Capacity_MCM", "Capacity", "Storage Capacity", "Reservoir Capacity"],
    "Use": ["Use", "Purpose"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Source": ["Source", "Reference"]
}

DESALINATION_COLUMN_ALIASES = {
    "Country": ["Country"],
    "Water_Name": ["DesalinationPlantName", "Desalination Plant Name", "Plant Name", "Site Name", "Name"],
    "Primary_Source": ["Primary Source", "Water Source", "Source Type"],
    "Use": ["Use", "Purpose"],
    "Capacity_Value": ["Capacity (m3/day)", "Capacity (m³/day)", "Capacity (m3/d)", "Capacity", "Capacity_m3_day"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Source": ["Source", "Reference"]
}

PORT_COLUMN_ALIASES = {
    "Port_Name": ["Port Name", "Port", "Shipping Port", "Seaport", "Name", "Location"],
    "Country": ["Country"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Long", "Lng", "Lon"],
    "Port_Type": ["Port Type", "Type", "Category"],
    "Capacity_Value": ["Capacity", "Throughput", "Annual Throughput", "TEU", "Volume", "Cargo Volume", "Port Capacity"],
    "Source": ["Source", "Reference"]
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
    text = text.replace("²", "2").replace("³", "3")
    text = text.replace("_", " ").replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_matching_sheet(sheet_names, possible_names):
    normalized_sheets = {normalize_column_name(sheet): sheet for sheet in sheet_names}
    for possible_name in possible_names:
        key = normalize_column_name(possible_name)
        if key in normalized_sheets:
            return normalized_sheets[key]
    return None


def align_columns(df, alias_map, required_columns, sheet_display_name):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    normalized_to_actual = {normalize_column_name(col): col for col in df.columns}

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
    scaled_values = np.log1p(values) if use_log else values
    min_value = scaled_values.min()
    max_value = scaled_values.max()
    if max_value == min_value:
        return pd.Series((min_size + max_size) / 2, index=series.index)
    marker_sizes = min_size + ((scaled_values - min_value) / (max_value - min_value)) * (max_size - min_size)
    return marker_sizes.fillna(min_size).clip(lower=min_size, upper=max_size)


def haversine_distance_km(lat1, lon1, lat2, lon2):
    radius_km = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
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


def safe_range_slider(label, series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        st.sidebar.caption(f"{label}: no valid values available")
        return None, None
    min_value = int(np.floor(values.min()))
    max_value = int(np.ceil(values.max()))
    if min_value == max_value:
        st.sidebar.caption(f"{label}: only one value available ({min_value:,})")
        return min_value, max_value
    return st.sidebar.slider(label, min_value=min_value, max_value=max_value, value=(min_value, max_value))


def apply_numeric_range_filter(df, column, selected_range):
    if df.empty or selected_range[0] is None or selected_range[1] is None:
        return df
    return df[df[column].between(selected_range[0], selected_range[1])].copy()


# =====================================================
# Default shipping ports fallback
# =====================================================


def get_builtin_african_ports():
    """
    Fallback reference list of major African shipping ports.
    This keeps the app usable even when a dedicated Shipping Ports.xlsx file is not yet available.
    Replace or override this by adding Shipping Ports.xlsx to the root folder.
    """
    return pd.DataFrame([
        {"Port_Name": "Port Said", "Country": "Egypt", "Latitude": 31.2653, "Longitude": 32.3019, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Alexandria Port", "Country": "Egypt", "Latitude": 31.2001, "Longitude": 29.9187, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Damietta Port", "Country": "Egypt", "Latitude": 31.4550, "Longitude": 31.7690, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Tanger Med", "Country": "Morocco", "Latitude": 35.8870, "Longitude": -5.5000, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Casablanca Port", "Country": "Morocco", "Latitude": 33.6000, "Longitude": -7.6167, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Durban Port", "Country": "South Africa", "Latitude": -29.8680, "Longitude": 31.0456, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Cape Town Port", "Country": "South Africa", "Latitude": -33.9180, "Longitude": 18.4241, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Coega / Ngqura Port", "Country": "South Africa", "Latitude": -33.8000, "Longitude": 25.6833, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Mombasa Port", "Country": "Kenya", "Latitude": -4.0435, "Longitude": 39.6682, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Dar es Salaam Port", "Country": "Tanzania", "Latitude": -6.8235, "Longitude": 39.2695, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Djibouti Port", "Country": "Djibouti", "Latitude": 11.5880, "Longitude": 43.1450, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Lagos Apapa Port", "Country": "Nigeria", "Latitude": 6.4483, "Longitude": 3.3642, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Tin Can Island Port", "Country": "Nigeria", "Latitude": 6.4400, "Longitude": 3.3400, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Lekki Deep Sea Port", "Country": "Nigeria", "Latitude": 6.4300, "Longitude": 4.0250, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Port Harcourt Port", "Country": "Nigeria", "Latitude": 4.7774, "Longitude": 7.0134, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Tema Port", "Country": "Ghana", "Latitude": 5.6500, "Longitude": 0.0167, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Takoradi Port", "Country": "Ghana", "Latitude": 4.8845, "Longitude": -1.7554, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Abidjan Port", "Country": "Côte d'Ivoire", "Latitude": 5.3167, "Longitude": -4.0167, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Dakar Port", "Country": "Senegal", "Latitude": 14.6928, "Longitude": -17.4467, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Luanda Port", "Country": "Angola", "Latitude": -8.7832, "Longitude": 13.2344, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Maputo Port", "Country": "Mozambique", "Latitude": -25.9653, "Longitude": 32.5892, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Beira Port", "Country": "Mozambique", "Latitude": -19.8333, "Longitude": 34.8500, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Walvis Bay Port", "Country": "Namibia", "Latitude": -22.9576, "Longitude": 14.5053, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
        {"Port_Name": "Mombasa Port", "Country": "Kenya", "Latitude": -4.0435, "Longitude": 39.6682, "Port_Type": "Seaport", "Capacity_Value": np.nan, "Source": "Built-in fallback list"},
    ])


# =====================================================
# Cleaning functions
# =====================================================


def clean_fertilizer_sheet(raw_df, report):
    df = align_columns(raw_df, FERTILIZER_COLUMN_ALIASES, ["Name", "Country", "Latitude", "Longitude", "Production_tpa"], "Fertilizer Plants")
    raw_rows = len(df)
    df = df.dropna(how="all").copy()
    df["Name"] = df["Name"].apply(clean_text)
    df["Country"] = df["Country"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Production_tpa"] = df["Production_tpa"].apply(parse_number_or_range)
    valid_mask = df["Name"].ne("Unknown") & df["Country"].ne("Unknown") & df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180) & df["Production_tpa"].notna() & (df["Production_tpa"] > 0)
    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()
    if df.empty:
        raise DataValidationError("No valid ammonia / fertilizer plant rows found", ["Each candidate row must have Name, Country, Latitude, Longitude, and positive Production or Capacity."])
    df["Candidate_Type"] = "Ammonia / fertilizer anchor"
    df["Hover_Text"] = "<b>" + df["Name"].apply(escape_text) + "</b><br>Country: " + df["Country"].apply(escape_text) + "<br>Production: " + df["Production_tpa"].round(0).astype(int).astype(str) + " tons/year"
    df["Marker_Size"] = make_marker_size(df["Production_tpa"], min_size=8, max_size=34, use_log=True)
    report["rows"]["Ammonia / Fertilizer Anchors"] = {"raw_rows": raw_rows, "usable_rows": len(df), "dropped_rows": dropped_rows}
    return df


def clean_wind_sheet(raw_df, report):
    df = align_columns(raw_df, WIND_COLUMN_ALIASES, ["Country", "Region", "Latitude", "Longitude", "Wind_Density_wm2"], "Wind Potential")
    raw_rows = len(df)
    df = df.dropna(how="all").copy()
    df["Country"] = df["Country"].apply(clean_text)
    df["Region"] = df["Region"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Wind_Speed_mps_100m"] = df["Wind_Speed_mps_100m"].apply(parse_number_or_range)
    df["Wind_Density_wm2"] = df["Wind_Density_wm2"].apply(parse_number_or_range)
    valid_mask = df["Country"].ne("Unknown") & df["Region"].ne("Unknown") & df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180) & df["Wind_Density_wm2"].notna() & (df["Wind_Density_wm2"] > 0)
    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()
    if df.empty:
        raise DataValidationError("No valid wind rows found", ["Each wind row must have Country, Region, Latitude, Longitude, and positive Wind Power Density."])
    df["Hover_Text"] = "<b>" + df["Region"].apply(escape_text) + "</b><br>Country: " + df["Country"].apply(escape_text) + "<br>Wind density: " + df["Wind_Density_wm2"].round(0).astype(int).astype(str) + " W/m²"
    df["Quality_Score"] = minmax_score(df["Wind_Density_wm2"])
    df["Marker_Size"] = make_marker_size(df["Wind_Density_wm2"], min_size=7, max_size=26, use_log=True)
    report["rows"]["Wind Potential"] = {"raw_rows": raw_rows, "usable_rows": len(df), "dropped_rows": dropped_rows}
    return df


def clean_solar_sheet(raw_df, report):
    df = align_columns(raw_df, SOLAR_COLUMN_ALIASES, ["Country", "Solar_Site", "Latitude", "Longitude", "Solar_Capacity_MW"], "Solar Potential")
    raw_rows = len(df)
    df = df.dropna(how="all").copy()
    df["Country"] = df["Country"].apply(clean_text)
    df["Solar_Site"] = df["Solar_Site"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Solar_Capacity_MW"] = df["Solar_Capacity_MW"].apply(parse_number_or_range)
    valid_mask = df["Country"].ne("Unknown") & df["Solar_Site"].ne("Unknown") & df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180) & df["Solar_Capacity_MW"].notna() & (df["Solar_Capacity_MW"] > 0)
    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()
    if df.empty:
        report["warnings"].append("Solar sheet was found, but no valid solar rows could be used.")
        return pd.DataFrame()
    df["Hover_Text"] = "<b>" + df["Solar_Site"].apply(escape_text) + "</b><br>Country: " + df["Country"].apply(escape_text) + "<br>Solar capacity: " + df["Solar_Capacity_MW"].round(1).astype(str) + " MW"
    df["Quality_Score"] = minmax_score(df["Solar_Capacity_MW"])
    df["Marker_Size"] = make_marker_size(df["Solar_Capacity_MW"], min_size=6, max_size=24, use_log=True)
    report["rows"]["Solar Potential"] = {"raw_rows": raw_rows, "usable_rows": len(df), "dropped_rows": dropped_rows}
    return df


def clean_water_sheet(raw_df, alias_map, required_columns, sheet_display_name, water_type, capacity_unit, report):
    df = align_columns(raw_df, alias_map, required_columns, sheet_display_name)
    raw_rows = len(df)
    df = df.dropna(how="all").copy()
    for column_name, default_value in {"Primary_Source": "Unknown", "Use": "Unknown", "Source": "Unknown"}.items():
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
    valid_mask = df["Country"].ne("Unknown") & df["Water_Name"].ne("Unknown") & df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180) & df["Capacity_Value"].notna() & (df["Capacity_Value"] > 0)
    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()
    if df.empty:
        report["warnings"].append(f"The {sheet_display_name} sheet was found, but no valid rows could be used.")
        return pd.DataFrame()
    df["Water_Type"] = water_type
    df["Capacity_Unit"] = capacity_unit
    df["Hover_Text"] = "<b>" + df["Water_Name"].apply(escape_text) + "</b><br>Type: " + df["Water_Type"].apply(escape_text) + "<br>Country: " + df["Country"].apply(escape_text) + "<br>Capacity: " + df["Capacity_Value"].round(2).astype(str) + " " + df["Capacity_Unit"].apply(escape_text)
    report["rows"][sheet_display_name] = {"raw_rows": raw_rows, "usable_rows": len(df), "dropped_rows": dropped_rows}
    keep = ["Country", "Water_Name", "Water_Type", "Primary_Source", "Use", "Capacity_Value", "Capacity_Unit", "Latitude", "Longitude", "Source", "Hover_Text"]
    return df[keep].copy()


def finalize_water_assets(df_water):
    if df_water.empty:
        return df_water
    frames = []
    for water_type, group in df_water.groupby("Water_Type"):
        group = group.copy()
        group["Type_Capacity_Score"] = minmax_score(group["Capacity_Value"])
        group["Marker_Size"] = make_marker_size(group["Capacity_Value"], min_size=6, max_size=26, use_log=True)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def clean_ports_sheet(raw_df, report, source_name="Shipping Ports"):
    df = align_columns(raw_df, PORT_COLUMN_ALIASES, ["Port_Name", "Country", "Latitude", "Longitude"], "Shipping Ports")
    raw_rows = len(df)
    df = df.dropna(how="all").copy()
    for column_name, default_value in {"Port_Type": "Seaport", "Source": "Unknown"}.items():
        if column_name not in df.columns:
            df[column_name] = default_value
    df["Port_Name"] = df["Port_Name"].apply(clean_text)
    df["Country"] = df["Country"].apply(clean_text)
    df["Port_Type"] = df["Port_Type"].apply(clean_text)
    df["Source"] = df["Source"].apply(clean_text)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Capacity_Value"] = df["Capacity_Value"].apply(parse_number_or_range)
    valid_mask = df["Port_Name"].ne("Unknown") & df["Country"].ne("Unknown") & df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)
    dropped_rows = int((~valid_mask).sum())
    df = df[valid_mask].copy()
    if df.empty:
        raise DataValidationError("No valid shipping port rows found", ["Each port row must have Port Name, Country, Latitude, and Longitude."])
    df["Port_Capacity_Score"] = minmax_score(df["Capacity_Value"]) if df["Capacity_Value"].notna().any() else 50
    df["Marker_Size"] = make_marker_size(df["Capacity_Value"].fillna(1), min_size=8, max_size=26, use_log=True)
    df["Hover_Text"] = "<b>" + df["Port_Name"].apply(escape_text) + "</b><br>Country: " + df["Country"].apply(escape_text) + "<br>Type: " + df["Port_Type"].apply(escape_text)
    report["rows"][source_name] = {"raw_rows": raw_rows, "usable_rows": len(df), "dropped_rows": dropped_rows}
    return df


# =====================================================
# Dataset loaders
# =====================================================


@st.cache_data(show_spinner=False)
def load_energy_dataset(file_bytes, source_name):
    report = {"source_name": source_name, "sheets": {}, "rows": {}, "warnings": []}
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError("Energy workbook could not be opened", ["Upload a valid .xlsx Excel workbook."])
    sheet_names = excel_file.sheet_names
    fertilizer_sheet = find_matching_sheet(sheet_names, ENERGY_SHEET_ALIASES["fertilizer"])
    wind_sheet = find_matching_sheet(sheet_names, ENERGY_SHEET_ALIASES["wind"])
    solar_sheet = find_matching_sheet(sheet_names, ENERGY_SHEET_ALIASES["solar"])
    missing = []
    if fertilizer_sheet is None:
        missing.append("Fertilizer Plants / Ammonia Plants")
    if wind_sheet is None:
        missing.append("Wind Potential")
    if missing:
        raise DataValidationError("Required energy sheet(s) missing", [f"Missing sheet(s): {', '.join(missing)}.", f"Sheets found: {', '.join(sheet_names)}."])
    report["sheets"]["Ammonia / Fertilizer Anchors"] = fertilizer_sheet
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
        report["warnings"].append("No Solar Potential sheet was found. Solar will be treated as unavailable.")
    return EnergyBundle(df_fert=df_fert, df_wind=df_wind, df_solar=df_solar, report=report, source_name=source_name)


@st.cache_data(show_spinner=False)
def load_water_dataset(file_bytes, source_name):
    report = {"source_name": source_name, "sheets": {}, "rows": {}, "warnings": []}
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError("Water workbook could not be opened", ["Upload a valid .xlsx workbook containing wastewater, dams, and/or desalination sheets."])
    sheet_names = excel_file.sheet_names
    frames = []
    sheet_map = [
        ("wastewater", "Waste water facilities", WASTEWATER_COLUMN_ALIASES, ["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"], "Wastewater facility", "m³/day"),
        ("dams", "Dams", DAMS_COLUMN_ALIASES, ["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"], "Dam", "MCM"),
        ("desalination", "Desalination plants", DESALINATION_COLUMN_ALIASES, ["Country", "Water_Name", "Capacity_Value", "Latitude", "Longitude"], "Desalination plant", "m³/day"),
    ]
    for key, display_name, alias_map, required, water_type, unit in sheet_map:
        matched_sheet = find_matching_sheet(sheet_names, WATER_SHEET_ALIASES[key])
        if matched_sheet is None:
            report["warnings"].append(f"No {display_name} sheet was found.")
            continue
        report["sheets"][display_name] = matched_sheet
        raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=matched_sheet)
        try:
            frames.append(clean_water_sheet(raw_df, alias_map, required, display_name, water_type, unit, report))
        except DataValidationError as exc:
            report["warnings"].extend(exc.issues)
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        raise DataValidationError("No valid water availability rows found", ["At least one water sheet should be valid: Waste water facilities, Dams, or Desalination plants."])
    df_water = pd.concat(frames, ignore_index=True)
    df_water = finalize_water_assets(df_water)
    return WaterBundle(df_water=df_water, report=report, source_name=source_name)


@st.cache_data(show_spinner=False)
def load_port_dataset(file_bytes, source_name):
    report = {"source_name": source_name, "sheets": {}, "rows": {}, "warnings": []}
    if file_bytes is None:
        df_ports = get_builtin_african_ports()
        df_ports = clean_ports_sheet(df_ports, report, source_name="Built-in African ports fallback")
        report["warnings"].append("No Shipping Ports.xlsx file was loaded. The app is using a built-in fallback list of major African ports. Replace this with a proper verified shipping port dataset when available.")
        return PortBundle(df_ports=df_ports, report=report, source_name="Built-in African ports fallback", used_builtin=True)
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        raise DataValidationError("Shipping ports workbook could not be opened", ["Upload a valid .xlsx workbook with a Shipping Ports sheet."])
    sheet_names = excel_file.sheet_names
    port_sheet = find_matching_sheet(sheet_names, PORT_SHEET_ALIASES["ports"])
    if port_sheet is None:
        raise DataValidationError("Shipping Ports sheet missing", [f"Sheets found: {', '.join(sheet_names)}.", "Expected a sheet named Shipping Ports, Ports, Port Locations, or Seaports."])
    report["sheets"]["Shipping Ports"] = port_sheet
    raw_ports = pd.read_excel(io.BytesIO(file_bytes), sheet_name=port_sheet)
    df_ports = clean_ports_sheet(raw_ports, report)
    return PortBundle(df_ports=df_ports, report=report, source_name=source_name, used_builtin=False)


@st.cache_data(show_spinner=False)
def load_ammonia_country_context(file_bytes, source_name):
    report = {"source_name": source_name, "sheets": {}, "rows": {}, "warnings": []}
    if file_bytes is None:
        return CommodityBundle(df_ammonia_country=pd.DataFrame(), report=report, source_name="Not loaded")
    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Ammonia", header=None)
    except Exception:
        report["warnings"].append("Could not load the Ammonia sheet from the commodity report.")
        return CommodityBundle(df_ammonia_country=pd.DataFrame(), report=report, source_name=source_name)
    header_row = None
    for idx, row in raw.iterrows():
        if row.astype(str).str.contains("Country", case=False, na=False).any():
            header_row = idx
            break
    if header_row is None:
        report["warnings"].append("No Country header row was found in the Ammonia sheet.")
        return CommodityBundle(df_ammonia_country=pd.DataFrame(), report=report, source_name=source_name)
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Ammonia", header=header_row)
    df = df.rename(columns={
        "Production Cost/ton\n(USD/unit)": "Production_Cost_USD_per_ton",
        "Avg. Market Price/ton\n(USD/unit)": "Market_Price_USD_per_ton",
        "Total Annual\nExport Value (USD)": "Export_Value_USD",
        "Total Annual\nImport Value (USD)": "Import_Value_USD",
        "Key Notes / 2024 Updates": "Notes"
    })
    if "Country" not in df.columns:
        return CommodityBundle(df_ammonia_country=pd.DataFrame(), report=report, source_name=source_name)
    df["Country"] = df["Country"].apply(clean_text)
    df = df[df["Country"].ne("Unknown")].copy()
    df = df[~df["Country"].astype(str).str.contains("AFRICA TOTALS|Source", case=False, na=False)].copy()
    for col in ["Export_Value_USD", "Import_Value_USD", "Production_Cost_USD_per_ton", "Market_Price_USD_per_ton"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    report["sheets"]["Ammonia country context"] = "Ammonia"
    report["rows"]["Ammonia country context"] = {"raw_rows": len(df), "usable_rows": len(df), "dropped_rows": 0}
    return CommodityBundle(df_ammonia_country=df, report=report, source_name=source_name)


# =====================================================
# Scoring functions
# =====================================================


def match_asset(source_df, asset_df, name_col, type_col, country_col, quality_value_col, quality_unit_col, quality_score_col, max_distance_km, match_rule, prefix):
    output_columns = [
        f"Nearest_{prefix}_Name", f"Nearest_{prefix}_Type", f"Nearest_{prefix}_Country",
        f"{prefix}_Quality_Value", f"{prefix}_Quality_Unit", f"{prefix}_Quality_Score",
        f"Distance_to_{prefix}_km", f"{prefix}_Proximity_Score", f"{prefix}_Latitude", f"{prefix}_Longitude"
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
        distances = haversine_distance_km(source_row["Latitude"], source_row["Longitude"], asset_df["Latitude"].values, asset_df["Longitude"].values)
        distance_scores = proximity_score(pd.Series(distances), max_distance_km=max_distance_km).values
        quality_scores = pd.to_numeric(asset_df[quality_score_col], errors="coerce").fillna(0).values
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


def prepare_wind_assets(df_wind):
    if df_wind.empty:
        return pd.DataFrame()
    wind = df_wind.copy()
    wind["Asset_Name"] = wind["Region"]
    wind["Asset_Type"] = "Wind"
    wind["Quality_Value"] = wind["Wind_Density_wm2"]
    wind["Quality_Unit"] = "W/m²"
    wind["Quality_Score"] = minmax_score(wind["Quality_Value"])
    return wind[["Country", "Asset_Name", "Asset_Type", "Quality_Value", "Quality_Unit", "Quality_Score", "Latitude", "Longitude", "Hover_Text", "Marker_Size"]]


def prepare_solar_assets(df_solar):
    if df_solar.empty:
        return pd.DataFrame()
    solar = df_solar.copy()
    solar["Asset_Name"] = solar["Solar_Site"]
    solar["Asset_Type"] = "Solar"
    solar["Quality_Value"] = solar["Solar_Capacity_MW"]
    solar["Quality_Unit"] = "MW"
    solar["Quality_Score"] = minmax_score(solar["Quality_Value"])
    return solar[["Country", "Asset_Name", "Asset_Type", "Quality_Value", "Quality_Unit", "Quality_Score", "Latitude", "Longitude", "Hover_Text", "Marker_Size"]]


def prepare_water_assets(df_water, selected_water_types):
    if df_water.empty:
        return pd.DataFrame()
    water = df_water[df_water["Water_Type"].isin(selected_water_types)].copy()
    water["Quality_Value"] = water["Capacity_Value"]
    water["Quality_Unit"] = water["Capacity_Unit"]
    water["Quality_Score"] = water["Type_Capacity_Score"]
    return water


def prepare_port_assets(df_ports):
    if df_ports.empty:
        return pd.DataFrame()
    ports = df_ports.copy()
    ports["Asset_Type"] = ports["Port_Type"].fillna("Seaport")
    ports["Quality_Value"] = ports["Capacity_Value"]
    ports["Quality_Unit"] = "capacity / throughput"
    ports["Quality_Score"] = ports["Port_Capacity_Score"] if "Port_Capacity_Score" in ports.columns else 50
    return ports


def calculate_effective_weights(raw_weights, available_components):
    active = {key: raw_weights.get(key, 0) for key in available_components if raw_weights.get(key, 0) > 0}
    total = sum(active.values())
    if total == 0:
        return {"Ammonia_Anchor_Score": 1.0}
    return {key: value / total for key, value in active.items()}


def build_hydrogen_opportunity_table(
    df_candidates,
    df_ports,
    wind_assets,
    solar_assets,
    water_assets,
    raw_weights,
    max_port_distance_km,
    max_wind_distance_km,
    max_solar_distance_km,
    max_water_distance_km,
    matching_rule,
    component_balance
):
    if df_candidates.empty:
        return pd.DataFrame()

    opportunity = df_candidates[["Name", "Country", "Candidate_Type", "Production_tpa", "Latitude", "Longitude"]].reset_index(drop=True).copy()
    opportunity["Ammonia_Anchor_Score"] = minmax_score(opportunity["Production_tpa"])

    port_match = match_asset(
        opportunity, df_ports, "Port_Name", "Asset_Type", "Country",
        "Quality_Value", "Quality_Unit", "Quality_Score", max_port_distance_km, matching_rule, "Port"
    )
    wind_match = match_asset(
        opportunity, wind_assets, "Asset_Name", "Asset_Type", "Country",
        "Quality_Value", "Quality_Unit", "Quality_Score", max_wind_distance_km, matching_rule, "Wind"
    )
    solar_match = match_asset(
        opportunity, solar_assets, "Asset_Name", "Asset_Type", "Country",
        "Quality_Value", "Quality_Unit", "Quality_Score", max_solar_distance_km, matching_rule, "Solar"
    )
    water_match = match_asset(
        opportunity, water_assets, "Water_Name", "Water_Type", "Country",
        "Quality_Value", "Quality_Unit", "Quality_Score", max_water_distance_km, matching_rule, "Water"
    )

    opportunity = pd.concat([opportunity, port_match, wind_match, solar_match, water_match], axis=1)

    for prefix in ["Port", "Wind", "Solar", "Water"]:
        opportunity[f"{prefix}_Quality_Score"] = pd.to_numeric(opportunity[f"{prefix}_Quality_Score"], errors="coerce").fillna(0)
        opportunity[f"{prefix}_Proximity_Score"] = pd.to_numeric(opportunity[f"{prefix}_Proximity_Score"], errors="coerce").fillna(0)

    q = component_balance / 100
    p = 1 - q
    opportunity["Port_Access_Score"] = opportunity["Port_Proximity_Score"]  # Port access is driven by proximity by default.
    opportunity["Wind_Score"] = (opportunity["Wind_Quality_Score"] * q) + (opportunity["Wind_Proximity_Score"] * p)
    opportunity["Solar_Score"] = (opportunity["Solar_Quality_Score"] * q) + (opportunity["Solar_Proximity_Score"] * p)
    opportunity["Water_Score"] = (opportunity["Water_Quality_Score"] * q) + (opportunity["Water_Proximity_Score"] * p)

    available_components = ["Ammonia_Anchor_Score"]
    if not df_ports.empty:
        available_components.append("Port_Access_Score")
    if not wind_assets.empty:
        available_components.append("Wind_Score")
    if not solar_assets.empty:
        available_components.append("Solar_Score")
    if not water_assets.empty:
        available_components.append("Water_Score")

    effective_weights = calculate_effective_weights(raw_weights, available_components)
    opportunity["Opportunity_Score"] = 0.0
    for score_col, weight in effective_weights.items():
        opportunity["Opportunity_Score"] += opportunity[score_col] * weight

    opportunity["Opportunity_Score"] = opportunity["Opportunity_Score"].round(2)
    opportunity["Priority_Band"] = opportunity["Opportunity_Score"].apply(classify_score)
    opportunity["Methodology_Weights"] = str({key: round(value, 3) for key, value in effective_weights.items()})
    return opportunity.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)


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
        name="Satellite view", overlay=False, control=True
    ).add_to(folium_map)
    folium.TileLayer(tiles="CartoDB positron", name="Clean map view", overlay=False, control=True).add_to(folium_map)
    folium.TileLayer(tiles="OpenStreetMap", name="Street map view", overlay=False, control=True).add_to(folium_map)
    if show_labels:
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
            attr="Labels © CARTO and OpenStreetMap contributors",
            name="Country and place labels", overlay=True, control=True, show=True
        ).add_to(folium_map)


def build_popup(title, rows):
    html_rows = ""
    for label, value in rows:
        html_rows += f"<tr><td style='padding:4px 8px;color:#6B7280;'>{escape_text(label)}</td><td style='padding:4px 8px;font-weight:600;'>{escape_text(value)}</td></tr>"
    return f"""
    <div style="font-family: Arial, sans-serif; min-width: 265px;">
        <h4 style="margin:0 0 8px 0;">{escape_text(title)}</h4>
        <table style="border-collapse:collapse;width:100%;">{html_rows}</table>
    </div>
    """


def add_map_marker_css(folium_map):
    css = """
    <style>
        .hydrogem-div-icon { background: transparent !important; border: none !important; }
        .hydrogem-marker-inner { transform-origin: center center; transition: transform 0.12s ease-out; }
        .hydrogem-marker-svg { filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.45)); }
    </style>
    """
    folium_map.get_root().header.add_child(folium.Element(css))


def build_fixed_marker_icon(shape, color, size=26, border_color="#111827", label=None, label_color="#FFFFFF"):
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
    elif shape == "anchor":
        shape_svg = f"<circle cx='50' cy='50' r='38' fill='{color}' stroke='{border_color}' stroke-width='6' /><text x='50' y='62' text-anchor='middle' font-size='48' font-weight='900' font-family='Arial' fill='white'>⚓</text>"
    else:
        shape_svg = f"<circle cx='50' cy='50' r='38' fill='{color}' stroke='{border_color}' stroke-width='6' />"
    label_svg = ""
    if label is not None:
        label_svg = f"<text x='50' y='58' text-anchor='middle' font-size='42' font-weight='800' font-family='Arial, sans-serif' fill='{label_color}'>{escape_text(label)}</text>"
    html_icon = f"""
    <div class="hydrogem-marker" style="width:{size}px;height:{size}px;position:relative;">
        <div class="hydrogem-marker-inner" style="width:{size}px;height:{size}px;position:absolute;left:50%;top:50%;transform:translate(-50%, -50%) scale(1);">
            <svg class="hydrogem-marker-svg" width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">{shape_svg}{label_svg}</svg>
        </div>
    </div>
    """
    return folium.DivIcon(html=html_icon, icon_size=(size, size), icon_anchor=(size / 2, size / 2), class_name="hydrogem-div-icon")


def get_marker_px(row, fallback=24, min_px=20, max_px=46, multiplier=1.35):
    try:
        value = row.get("Marker_Size", fallback)
        if pd.isna(value):
            value = fallback
        return int(np.clip(float(value) * multiplier, min_px, max_px))
    except Exception:
        return fallback


def add_marker_legend(folium_map):
    legend_html = """
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: rgba(255,255,255,0.94); border: 1px solid #D1D5DB; border-radius: 14px; padding: 14px 16px; width: 310px; box-shadow: 0 6px 18px rgba(0,0,0,0.18); font-family: Arial, sans-serif; color: #111827;">
        <div style="font-weight:800; font-size:15px; margin-bottom:10px;">Map legend</div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:16px; height:16px; background:#DC2626; border:2px solid #111827; border-radius:50%; display:inline-block;"></span><span>Ammonia / fertilizer anchor</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:22px; height:22px; background:#7C3AED; border:2px solid #111827; border-radius:50%; color:white; display:inline-flex; align-items:center; justify-content:center; font-size:12px;">⚓</span><span>Shipping port</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:0; height:0; border-left:9px solid transparent; border-right:9px solid transparent; border-bottom:17px solid #2563EB; display:inline-block;"></span><span>Wind potential</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:15px; height:15px; background:#F59E0B; border:2px solid #111827; transform:rotate(45deg); display:inline-block; margin-left:2px;"></span><span>Solar potential</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:15px; height:15px; background:#059669; border:2px solid #064E3B; border-radius:50%; display:inline-block;"></span><span>Water asset</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:22px; height:22px; background:#166534; border:2px solid white; border-radius:50%; color:white; display:inline-flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; box-shadow:0 1px 4px rgba(0,0,0,0.35);">1</span><span>Ranked hydrogen hotspot</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:28px; border-top:3px dashed #7C3AED; display:inline-block;"></span><span>Anchor to selected shipping port</span></div>
        <div style="display:flex; align-items:center; gap:9px; margin-bottom:7px;"><span style="width:28px; border-top:3px dashed #2563EB; display:inline-block;"></span><span>Anchor to renewable asset</span></div>
        <div style="display:flex; align-items:center; gap:9px;"><span style="width:28px; border-top:3px dashed #059669; display:inline-block;"></span><span>Anchor to water asset</span></div>
        <div style="border-top:1px solid #E5E7EB; margin-top:10px; padding-top:8px; font-size:12px; color:#6B7280; line-height:1.35;">Markers are fixed-pixel icons so they remain visible when zooming.</div>
    </div>
    """
    folium_map.get_root().html.add_child(folium.Element(legend_html))


def add_zoom_marker_scaling(folium_map, base_zoom=3, growth_per_zoom=0.045, max_scale=1.45):
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
                markers.forEach(function(marker) {{ marker.style.transform = 'translate(-50%, -50%) scale(' + scale + ')'; }});
            }}
            map.whenReady(resizeHydroGEMMarkers);
            map.on('zoomend', resizeHydroGEMMarkers);
            map.on('layeradd', resizeHydroGEMMarkers);
        }})();
    </script>
    """
    folium_map.get_root().html.add_child(folium.Element(zoom_script))


def build_folium_map(df_candidates, df_ports, wind_assets, solar_assets, water_assets, opportunity, layer_flags, top_n_hotspots):
    folium_map = folium.Map(location=[2.0, 20.0], zoom_start=3, tiles=None, control_scale=True)
    add_base_tiles(folium_map, show_labels=layer_flags["labels"])
    add_map_marker_css(folium_map)

    ammonia_cmap = make_colormap(df_candidates["Production_tpa"] if not df_candidates.empty else pd.Series(dtype=float), ["#FEE2E2", "#EF4444", "#7F1D1D"])
    wind_cmap = make_colormap(wind_assets["Quality_Score"] if not wind_assets.empty else pd.Series(dtype=float), ["#DBEAFE", "#2563EB", "#1E3A8A"])
    solar_cmap = make_colormap(solar_assets["Quality_Score"] if not solar_assets.empty else pd.Series(dtype=float), ["#FEF3C7", "#F59E0B", "#92400E"])
    water_cmap = make_colormap(water_assets["Quality_Score"] if not water_assets.empty else pd.Series(dtype=float), ["#D1FAE5", "#059669", "#064E3B"])

    if layer_flags["anchors"] and not df_candidates.empty:
        anchor_group = folium.FeatureGroup(name="Ammonia / fertilizer anchors", show=True)
        for _, row in df_candidates.iterrows():
            popup = build_popup(row["Name"], [("Marker", "Red circle"), ("Asset type", row["Candidate_Type"]), ("Country", row["Country"]), ("Production", f"{format_number(row['Production_tpa'])} tons/year"), ("Latitude", f"{row['Latitude']:.4f}"), ("Longitude", f"{row['Longitude']:.4f}")])
            folium.Marker(location=[row["Latitude"], row["Longitude"]], tooltip=f"Anchor: {row['Name']} | {row['Country']}", popup=folium.Popup(popup, max_width=410), icon=build_fixed_marker_icon("circle", ammonia_cmap(row["Production_tpa"]), get_marker_px(row, fallback=28, min_px=23, max_px=48, multiplier=1.45))).add_to(anchor_group)
        anchor_group.add_to(folium_map)

    if layer_flags["ports"] and not df_ports.empty:
        port_group = folium.FeatureGroup(name="Shipping ports", show=True)
        for _, row in df_ports.iterrows():
            popup = build_popup(row["Port_Name"], [("Marker", "Purple anchor"), ("Asset type", "Shipping port"), ("Country", row["Country"]), ("Port type", row["Port_Type"]), ("Capacity", format_number(row.get("Capacity_Value", np.nan), 1)), ("Source", row.get("Source", "Unknown"))])
            folium.Marker(location=[row["Latitude"], row["Longitude"]], tooltip=f"Shipping port: {row['Port_Name']} | {row['Country']}", popup=folium.Popup(popup, max_width=410), icon=build_fixed_marker_icon("anchor", "#7C3AED", 30, "#111827")).add_to(port_group)
        port_group.add_to(folium_map)

    if layer_flags["wind"] and not wind_assets.empty:
        wind_group = folium.FeatureGroup(name="Wind potential", show=True)
        for _, row in wind_assets.iterrows():
            popup = build_popup(row["Asset_Name"], [("Marker", "Blue triangle"), ("Asset type", "Wind potential"), ("Country", row["Country"]), ("Quality", f"{format_number(row['Quality_Value'], 1)} {row['Quality_Unit']}"), ("Quality score", f"{format_number(row['Quality_Score'], 1)}")])
            folium.Marker(location=[row["Latitude"], row["Longitude"]], tooltip=f"Wind: {row['Asset_Name']} | {row['Country']}", popup=folium.Popup(popup, max_width=390), icon=build_fixed_marker_icon("triangle", wind_cmap(row["Quality_Score"]), get_marker_px(row, fallback=25, min_px=22, max_px=42, multiplier=1.45))).add_to(wind_group)
        wind_group.add_to(folium_map)

    if layer_flags["solar"] and not solar_assets.empty:
        solar_group = folium.FeatureGroup(name="Solar potential", show=True)
        for _, row in solar_assets.iterrows():
            popup = build_popup(row["Asset_Name"], [("Marker", "Amber diamond"), ("Asset type", "Solar potential"), ("Country", row["Country"]), ("Quality", f"{format_number(row['Quality_Value'], 1)} {row['Quality_Unit']}"), ("Quality score", f"{format_number(row['Quality_Score'], 1)}")])
            folium.Marker(location=[row["Latitude"], row["Longitude"]], tooltip=f"Solar: {row['Asset_Name']} | {row['Country']}", popup=folium.Popup(popup, max_width=390), icon=build_fixed_marker_icon("diamond", solar_cmap(row["Quality_Score"]), get_marker_px(row, fallback=25, min_px=22, max_px=42, multiplier=1.45))).add_to(solar_group)
        solar_group.add_to(folium_map)

    if layer_flags["water"] and not water_assets.empty:
        water_group = folium.FeatureGroup(name="Water assets", show=True)
        for _, row in water_assets.iterrows():
            shape = "circle" if row["Water_Type"] == "Wastewater facility" else "square" if row["Water_Type"] == "Dam" else "pentagon"
            popup = build_popup(row["Water_Name"], [("Marker", "Green water asset"), ("Asset type", row["Water_Type"]), ("Country", row["Country"]), ("Capacity", f"{format_number(row['Capacity_Value'], 2)} {row['Capacity_Unit']}"), ("Capacity score", f"{format_number(row['Quality_Score'], 1)}")])
            folium.Marker(location=[row["Latitude"], row["Longitude"]], tooltip=f"Water: {row['Water_Name']} | {row['Country']}", popup=folium.Popup(popup, max_width=410), icon=build_fixed_marker_icon(shape, water_cmap(row["Quality_Score"]), get_marker_px(row, fallback=24, min_px=21, max_px=42, multiplier=1.45), "#064E3B")).add_to(water_group)
        water_group.add_to(folium_map)

    if layer_flags["lines"] and not opportunity.empty:
        line_group = folium.FeatureGroup(name="Hotspot connection lines", show=True)
        for _, row in opportunity.head(top_n_hotspots).iterrows():
            plant_point = [row["Latitude"], row["Longitude"]]
            if pd.notna(row.get("Port_Latitude")) and pd.notna(row.get("Port_Longitude")):
                folium.PolyLine([plant_point, [row["Port_Latitude"], row["Port_Longitude"]]], color="#7C3AED", weight=3, opacity=0.75, dash_array="7,7", tooltip="Anchor to selected shipping port").add_to(line_group)
            if pd.notna(row.get("Wind_Latitude")) and pd.notna(row.get("Wind_Longitude")):
                folium.PolyLine([plant_point, [row["Wind_Latitude"], row["Wind_Longitude"]]], color="#2563EB", weight=3, opacity=0.7, dash_array="7,7", tooltip="Anchor to selected wind asset").add_to(line_group)
            if pd.notna(row.get("Solar_Latitude")) and pd.notna(row.get("Solar_Longitude")):
                folium.PolyLine([plant_point, [row["Solar_Latitude"], row["Solar_Longitude"]]], color="#F59E0B", weight=3, opacity=0.7, dash_array="7,7", tooltip="Anchor to selected solar asset").add_to(line_group)
            if pd.notna(row.get("Water_Latitude")) and pd.notna(row.get("Water_Longitude")):
                folium.PolyLine([plant_point, [row["Water_Latitude"], row["Water_Longitude"]]], color="#059669", weight=3, opacity=0.7, dash_array="5,7", tooltip="Anchor to selected water asset").add_to(line_group)
        line_group.add_to(folium_map)

    if layer_flags["hotspots"] and not opportunity.empty:
        hotspot_group = folium.FeatureGroup(name="Top ranked hydrogen hotspots", show=True)
        for rank, (_, row) in enumerate(opportunity.head(top_n_hotspots).iterrows(), start=1):
            popup = build_popup(f"#{rank} {row['Name']}", [("Marker", "Green numbered badge"), ("Asset type", "Ranked hydrogen hotspot"), ("Country", row["Country"]), ("Opportunity score", f"{format_number(row['Opportunity_Score'], 2)}"), ("Priority band", row["Priority_Band"]), ("Nearest port", row.get("Nearest_Port_Name", "N/A")), ("Nearest wind", row.get("Nearest_Wind_Name", "N/A")), ("Nearest solar", row.get("Nearest_Solar_Name", "N/A")), ("Nearest water", row.get("Nearest_Water_Name", "N/A"))])
            folium.Marker(location=[row["Latitude"], row["Longitude"]], tooltip=f"#{rank} hotspot: {row['Name']}", popup=folium.Popup(popup, max_width=430), icon=build_fixed_marker_icon("circle", "#166534", 38, "#FFFFFF", label=str(rank), label_color="#FFFFFF")).add_to(hotspot_group)
        hotspot_group.add_to(folium_map)

    folium.LayerControl(collapsed=False).add_to(folium_map)
    add_marker_legend(folium_map)
    add_zoom_marker_scaling(folium_map, base_zoom=3, growth_per_zoom=0.045, max_scale=1.45)
    return folium_map


def folium_map_to_html(folium_map):
    return folium_map.get_root().render()


# =====================================================
# Templates and guides
# =====================================================


def build_port_template_workbook():
    output = io.BytesIO()
    template = pd.DataFrame({
        "Port Name": ["Example Port"],
        "Country": ["Nigeria"],
        "Latitude": [6.4483],
        "Longitude": [3.3642],
        "Port Type": ["Seaport"],
        "Capacity": [np.nan],
        "Source": ["Provide source link or note"]
    })
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template.to_excel(writer, sheet_name="Shipping Ports", index=False)
    return output.getvalue()


def show_dictionary(raw_weights=None, effective_weights=None):
    st.subheader("Dictionary and methodology guide")
    st.markdown("""
    This dashboard is a first-pass screening tool. It does not say where a hydrogen plant should definitely be built. It ranks locations that deserve deeper technical, commercial, regulatory, land, water-rights, grid, pipeline and offtake feasibility work.
    """)

    terms = pd.DataFrame({
        "Term": [
            "Hydrogen opportunity score", "Hydrogen hotspot", "Ammonia / fertilizer anchor", "Shipping port access", "Wind score", "Solar score", "Water score", "Proximity score", "Quality score", "Effective weight"
        ],
        "Meaning": [
            "The final 0-100 score used to rank candidate hydrogen plant locations.",
            "A high-ranked candidate location after current filters and weights are applied.",
            "An existing ammonia or fertilizer production location used as a demand/industrial anchor.",
            "The suitability of a location based on distance to the selected shipping port.",
            "The combined quality and proximity score for the best matched wind asset.",
            "The combined quality and proximity score for the best matched solar asset.",
            "The combined capacity and proximity score for the best matched water asset.",
            "A 0-100 score where closer assets get a higher score within the selected maximum useful distance.",
            "A 0-100 score that normalises resource strength, for example wind density, solar capacity or water capacity.",
            "The actual model weight after the app normalises the raw selected weights to add up to 100%."
        ],
        "How to read it": [
            "Higher is better, but it is still only a screening signal.",
            "Start detailed feasibility review with these locations first.",
            "The model prioritises sites near existing industrial/fertilizer demand because transportation is expensive.",
            "Ports matter because hydrogen/ammonia export and equipment logistics are port-sensitive.",
            "A strong wind resource close to the candidate site improves the score.",
            "A strong solar resource close to the candidate site improves the score.",
            "A large or reliable water source close to the candidate site improves the score.",
            "If the distance exceeds the threshold, proximity score can fall to zero.",
            "Quality is relative to the uploaded/selected dataset, not an absolute engineering guarantee.",
            "For example, raw weights of 25,25,20,20,20 add up to 110, so the app converts them to effective percentages."
        ]
    })
    st.dataframe(terms, width="stretch", hide_index=True)

    st.markdown("### Current formula")
    st.markdown("""
    ```text
    Hydrogen Opportunity Score =
    Ammonia Anchor Score × Ammonia Anchor Weight
    + Port Access Score × Shipping Port Weight
    + Wind Score × Wind Weight
    + Solar Score × Solar Weight
    + Water Score × Water Weight
    ```

    Wind, solar and water scores each combine **resource quality** and **distance/proximity**. The mix is controlled from the sidebar using the quality-vs-proximity slider.
    """)

    if raw_weights is not None:
        st.markdown("### Raw weights selected in the sidebar")
        st.dataframe(pd.DataFrame({"Component": list(raw_weights.keys()), "Raw weight": list(raw_weights.values())}), width="stretch", hide_index=True)

    if effective_weights is not None:
        st.markdown("### Effective weights used by the model")
        st.dataframe(pd.DataFrame({"Score component": list(effective_weights.keys()), "Effective weight (%)": [round(v * 100, 2) for v in effective_weights.values()]}), width="stretch", hide_index=True)

    limitations = pd.DataFrame({
        "Limitation": ["Straight-line distance", "Proxy candidate locations", "Port fallback list", "Mixed units", "Data freshness", "No project economics yet"],
        "Why it matters": [
            "The model does not yet calculate road, pipeline, grid or shipping route distance.",
            "Where a dedicated ammonia plant location file is unavailable, fertilizer plants are used as industrial anchors.",
            "If Shipping Ports.xlsx is missing, a built-in fallback list is used and should be replaced with a verified dataset.",
            "Wind, solar, water and port variables use different units and are normalised for screening only.",
            "Bad coordinates or outdated plant/port/water data will distort results.",
            "Capital expenditure, operating cost, tariffs, permits, land and offtake are not yet scored."
        ]
    })
    st.markdown("### Limitations")
    st.dataframe(limitations, width="stretch", hide_index=True)


def show_data_guide():
    st.subheader("Data structure guide")
    st.markdown("""
    The app works with these root-folder files:

    - `Fertilizer Plants_AHDS.xlsx`: ammonia/fertilizer anchors, wind and solar
    - `Water availability.xlsx`: wastewater, dams and desalination
    - `Shipping Ports.xlsx`: shipping port locations; optional, because the app has a fallback list
    - `Petroleum_Urea_Ammonia Commodity Report  (1).xlsx`: optional country-level ammonia context

    The most important requirement for spatial scoring is **latitude and longitude**. If a dataset has no coordinates, it can inform context but cannot directly place markers on the map.
    """)

    st.markdown("### Shipping Ports workbook template")
    st.dataframe(pd.DataFrame({
        "Column": ["Port Name", "Country", "Latitude", "Longitude", "Port Type", "Capacity", "Source"],
        "Required": ["Yes", "Yes", "Yes", "Yes", "Optional", "Optional", "Optional"],
        "Example": ["Lagos Apapa Port", "Nigeria", "6.4483", "3.3642", "Seaport", "", "Source link or note"]
    }), width="stretch", hide_index=True)

    st.download_button(
        label="Download shipping ports template",
        data=build_port_template_workbook(),
        file_name="Shipping Ports.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div class="main-title">HydroGEM hydrogen opportunity map</div>
    <div class="subtitle">Weighted-priority screening for potential hydrogen plant locations using ammonia/fertilizer anchors, shipping ports, wind, solar and water availability.</div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Sidebar: data source
# =====================================================

st.sidebar.header("Data source")
data_mode = st.sidebar.radio("Choose data mode", ["Use default root-folder datasets", "Upload visitor datasets"])

energy_file_bytes = None
water_file_bytes = None
port_file_bytes = None
commodity_file_bytes = None
energy_source_name = None
water_source_name = None
port_source_name = None
commodity_source_name = None

if data_mode == "Use default root-folder datasets":
    st.sidebar.caption("The app will look for the files in the same folder as app.py.")
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

    if DEFAULT_PORT_DATASET_PATH.exists():
        port_file_bytes = DEFAULT_PORT_DATASET_PATH.read_bytes()
        port_source_name = DEFAULT_PORT_DATASET_PATH.name
        st.sidebar.success(f"Ports loaded: {DEFAULT_PORT_DATASET_PATH.name}")
    else:
        st.sidebar.info("Shipping Ports.xlsx not found; built-in port fallback will be used.")

    if DEFAULT_COMMODITY_DATASET_PATH.exists():
        commodity_file_bytes = DEFAULT_COMMODITY_DATASET_PATH.read_bytes()
        commodity_source_name = DEFAULT_COMMODITY_DATASET_PATH.name
        st.sidebar.success("Commodity context loaded")
else:
    uploaded_energy_file = st.sidebar.file_uploader("Upload energy workbook", type=["xlsx"], key="energy_uploader")
    uploaded_water_file = st.sidebar.file_uploader("Upload water workbook", type=["xlsx"], key="water_uploader")
    uploaded_port_file = st.sidebar.file_uploader("Upload shipping ports workbook", type=["xlsx"], key="port_uploader")
    uploaded_commodity_file = st.sidebar.file_uploader("Upload ammonia commodity report", type=["xlsx"], key="commodity_uploader")
    if uploaded_energy_file is not None:
        energy_file_bytes = uploaded_energy_file.getvalue(); energy_source_name = uploaded_energy_file.name; st.sidebar.success(f"Energy uploaded: {uploaded_energy_file.name}")
    if uploaded_water_file is not None:
        water_file_bytes = uploaded_water_file.getvalue(); water_source_name = uploaded_water_file.name; st.sidebar.success(f"Water uploaded: {uploaded_water_file.name}")
    if uploaded_port_file is not None:
        port_file_bytes = uploaded_port_file.getvalue(); port_source_name = uploaded_port_file.name; st.sidebar.success(f"Ports uploaded: {uploaded_port_file.name}")
    if uploaded_commodity_file is not None:
        commodity_file_bytes = uploaded_commodity_file.getvalue(); commodity_source_name = uploaded_commodity_file.name; st.sidebar.success(f"Commodity uploaded: {uploaded_commodity_file.name}")

st.sidebar.divider()

if energy_file_bytes is None:
    tab_start, tab_guide = st.tabs(["Start here", "Data guide"])
    with tab_start:
        st.info("No energy dataset is loaded yet. Place Fertilizer Plants_AHDS.xlsx in the root folder or upload it from the sidebar.")
    with tab_guide:
        show_data_guide()
    st.stop()

# =====================================================
# Load datasets
# =====================================================

try:
    with st.spinner("Loading and validating energy dataset..."):
        energy_bundle = load_energy_dataset(energy_file_bytes, energy_source_name)
except DataValidationError as exc:
    tab_error, tab_guide = st.tabs(["Energy upload error", "Data guide"])
    with tab_error:
        st.markdown(f"<div class='error-box'><h3>{escape_text(exc.title)}</h3><ul>{''.join([f'<li>{escape_text(issue)}</li>' for issue in exc.issues])}</ul></div>", unsafe_allow_html=True)
    with tab_guide:
        show_data_guide()
    st.stop()

water_bundle = None
water_error = None
if water_file_bytes is not None:
    try:
        with st.spinner("Loading and validating water dataset..."):
            water_bundle = load_water_dataset(water_file_bytes, water_source_name)
    except DataValidationError as exc:
        water_error = exc

try:
    with st.spinner("Loading shipping ports..."):
        port_bundle = load_port_dataset(port_file_bytes, port_source_name if port_source_name else "Built-in African ports fallback")
except DataValidationError as exc:
    st.warning(f"Shipping ports could not be loaded: {exc.title}. Built-in port fallback is being used instead.")
    port_bundle = load_port_dataset(None, "Built-in African ports fallback")

commodity_bundle = load_ammonia_country_context(commodity_file_bytes, commodity_source_name) if commodity_file_bytes is not None else load_ammonia_country_context(None, "Not loaded")

df_candidates = energy_bundle.df_fert
raw_wind = energy_bundle.df_wind
raw_solar = energy_bundle.df_solar
raw_water = water_bundle.df_water if water_bundle is not None else pd.DataFrame()
raw_ports = port_bundle.df_ports

# =====================================================
# Sidebar filters and methodology
# =====================================================

st.sidebar.header("Filters")
country_pool = sorted(set(df_candidates["Country"].dropna().unique()) | set(raw_wind["Country"].dropna().unique()) | (set(raw_solar["Country"].dropna().unique()) if not raw_solar.empty else set()) | (set(raw_water["Country"].dropna().unique()) if not raw_water.empty else set()) | set(raw_ports["Country"].dropna().unique()))
selected_countries = st.sidebar.multiselect("Countries", options=country_pool, default=country_pool)
production_range = safe_range_slider("Ammonia / fertilizer production range, tons/year", df_candidates["Production_tpa"])
wind_range = safe_range_slider("Wind power density range, W/m²", raw_wind["Wind_Density_wm2"])
solar_range = safe_range_slider("Solar capacity range, MW", raw_solar["Solar_Capacity_MW"]) if not raw_solar.empty else (None, None)
water_range = safe_range_slider("Water capacity range", raw_water["Capacity_Value"]) if not raw_water.empty else (None, None)

st.sidebar.divider()
st.sidebar.header("Methodology controls")
matching_rule = st.sidebar.selectbox("Matching rule", ["Best quality-proximity fit", "Nearest asset"], index=0)
max_port_distance_km = st.sidebar.slider("Maximum useful shipping port distance, km", 50, 2500, 500, 50)
max_wind_distance_km = st.sidebar.slider("Maximum useful wind distance, km", 50, 2500, 700, 50)
max_solar_distance_km = st.sidebar.slider("Maximum useful solar distance, km", 50, 2500, 700, 50)
max_water_distance_km = st.sidebar.slider("Maximum useful water distance, km", 10, 2500, 300, 10, disabled=raw_water.empty)
component_balance = st.sidebar.slider("Resource quality share inside wind/solar/water score (%)", 0, 100, 50, 5)

available_water_types = sorted(raw_water["Water_Type"].unique()) if not raw_water.empty else []
selected_water_types = st.sidebar.multiselect("Water resources used in scoring", options=available_water_types, default=available_water_types, disabled=raw_water.empty)

with st.sidebar.expander("Weighted-priority inputs", expanded=True):
    st.caption("Defaults follow the review discussion. They are automatically normalised because 25 + 25 + 20 + 20 + 20 = 110, not 100.")
    ammonia_weight = st.slider("Ammonia / fertilizer anchor weight", 0, 100, 25)
    port_weight = st.slider("Shipping port access weight", 0, 100, 25)
    wind_weight = st.slider("Wind weight", 0, 100, 20)
    solar_weight = st.slider("Solar weight", 0, 100, 20, disabled=raw_solar.empty)
    water_weight = st.slider("Water weight", 0, 100, 20, disabled=raw_water.empty)

raw_weights = {
    "Ammonia_Anchor_Score": ammonia_weight,
    "Port_Access_Score": port_weight,
    "Wind_Score": wind_weight,
    "Solar_Score": solar_weight if not raw_solar.empty else 0,
    "Water_Score": water_weight if not raw_water.empty else 0,
}

min_score = st.sidebar.slider("Minimum hydrogen opportunity score", 0, 100, 0)
top_n_hotspots = st.sidebar.slider("Number of hotspots to show", 3, 30, 10)

st.sidebar.divider()
st.sidebar.header("Map layers")
layer_flags = {
    "anchors": st.sidebar.checkbox("Show ammonia / fertilizer anchors", value=True),
    "ports": st.sidebar.checkbox("Show shipping ports", value=True),
    "wind": st.sidebar.checkbox("Show wind potential", value=True),
    "solar": st.sidebar.checkbox("Show solar potential", value=False, disabled=raw_solar.empty),
    "water": st.sidebar.checkbox("Show water assets", value=True, disabled=raw_water.empty),
    "hotspots": st.sidebar.checkbox("Show ranked hotspots", value=True),
    "lines": st.sidebar.checkbox("Show hotspot connection lines", value=True),
    "labels": st.sidebar.checkbox("Show country/place labels", value=True),
}

# =====================================================
# Apply filters and calculate score
# =====================================================

filtered_candidates = df_candidates[df_candidates["Country"].isin(selected_countries)].copy()
filtered_candidates = apply_numeric_range_filter(filtered_candidates, "Production_tpa", production_range)

filtered_wind = raw_wind[raw_wind["Country"].isin(selected_countries)].copy()
filtered_wind = apply_numeric_range_filter(filtered_wind, "Wind_Density_wm2", wind_range)

if not raw_solar.empty:
    filtered_solar = raw_solar[raw_solar["Country"].isin(selected_countries)].copy()
    filtered_solar = apply_numeric_range_filter(filtered_solar, "Solar_Capacity_MW", solar_range)
else:
    filtered_solar = pd.DataFrame()

if not raw_water.empty:
    filtered_water = raw_water[raw_water["Country"].isin(selected_countries)].copy()
    filtered_water = filtered_water[filtered_water["Water_Type"].isin(selected_water_types)].copy()
    filtered_water = apply_numeric_range_filter(filtered_water, "Capacity_Value", water_range)
else:
    filtered_water = pd.DataFrame()

filtered_ports = raw_ports[raw_ports["Country"].isin(selected_countries)].copy()

wind_assets = prepare_wind_assets(filtered_wind)
solar_assets = prepare_solar_assets(filtered_solar)
water_assets = prepare_water_assets(filtered_water, selected_water_types)
port_assets = prepare_port_assets(filtered_ports)

opportunity = build_hydrogen_opportunity_table(
    df_candidates=filtered_candidates,
    df_ports=port_assets,
    wind_assets=wind_assets,
    solar_assets=solar_assets,
    water_assets=water_assets,
    raw_weights=raw_weights,
    max_port_distance_km=max_port_distance_km,
    max_wind_distance_km=max_wind_distance_km,
    max_solar_distance_km=max_solar_distance_km,
    max_water_distance_km=max_water_distance_km,
    matching_rule=matching_rule,
    component_balance=component_balance
)

filtered_opportunity = opportunity[opportunity["Opportunity_Score"] >= min_score].copy() if not opportunity.empty else opportunity
top_hotspots = filtered_opportunity.head(top_n_hotspots).copy() if not filtered_opportunity.empty else pd.DataFrame()
effective_weights = calculate_effective_weights(raw_weights, ["Ammonia_Anchor_Score", "Port_Access_Score", "Wind_Score", "Solar_Score" if not solar_assets.empty else None, "Water_Score" if not water_assets.empty else None])
effective_weights = {k: v for k, v in effective_weights.items() if k is not None}

# =====================================================
# Tabs
# =====================================================

tab_overview, tab_map, tab_hotspots, tab_ranking, tab_methodology, tab_quality, tab_guide = st.tabs([
    "Overview", "Satellite map", "Hotspots", "Opportunity ranking", "Methodology dictionary", "Data quality", "Data guide"
])

with tab_overview:
    st.subheader("Executive overview")
    if water_error is not None:
        st.warning(f"Water dataset could not be loaded: {water_error.title}")
    if port_bundle.used_builtin:
        st.warning("Shipping Ports.xlsx was not found or not uploaded. The map is using the built-in African ports fallback list. Replace it with a verified port dataset when ready.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Candidate anchors", f"{len(filtered_candidates):,}")
    col2.metric("Shipping ports", f"{len(port_assets):,}")
    col3.metric("Wind assets", f"{len(wind_assets):,}")
    col4.metric("Solar assets", f"{len(solar_assets):,}")
    col5.metric("Water assets", f"{len(water_assets):,}")

    st.markdown("### Current methodology weights")
    st.dataframe(pd.DataFrame({"Score component": list(effective_weights.keys()), "Effective weight (%)": [round(v * 100, 2) for v in effective_weights.values()]}), width="stretch", hide_index=True)

    st.markdown("### Top hydrogen hotspots")
    if top_hotspots.empty:
        st.warning("No hotspots match the current filters.")
    else:
        cols = st.columns(min(3, len(top_hotspots)))
        for idx, (_, row) in enumerate(top_hotspots.head(3).iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class='info-card'>
                    <h4 style='margin-top:0;'>#{idx + 1} {escape_text(row['Name'])}</h4>
                    <p class='small-muted'>{escape_text(row['Country'])}</p>
                    <p><b>Hydrogen opportunity score:</b> {format_number(row['Opportunity_Score'], 2)}</p>
                    <p><b>Nearest port:</b> {escape_text(row.get('Nearest_Port_Name', 'N/A'))}</p>
                    <p><b>Nearest wind:</b> {escape_text(row.get('Nearest_Wind_Name', 'N/A'))}</p>
                    <p><b>Nearest solar:</b> {escape_text(row.get('Nearest_Solar_Name', 'N/A'))}</p>
                    <p><b>Nearest water:</b> {escape_text(row.get('Nearest_Water_Name', 'N/A'))}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    ### How to read the output
    The score ranks candidate hydrogen plant locations by their proximity and suitability across five variables discussed in the review meeting: **ammonia/fertilizer anchors, shipping ports, wind, solar and water**. It is a screening result, not a final investment decision.
    """)

with tab_map:
    st.subheader("Satellite map with hydrogen opportunity layers")
    st.caption("Use the map layer control to switch between satellite, clean map, labels and individual asset layers.")
    folium_map = build_folium_map(filtered_candidates, port_assets, wind_assets, solar_assets, water_assets, filtered_opportunity, layer_flags, top_n_hotspots)
    st_folium(folium_map, height=760, width=1300, returned_objects=[])
    st.download_button("Download current map as HTML", data=folium_map_to_html(folium_map), file_name="hydrogem_hydrogen_opportunity_map.html", mime="text/html")

with tab_hotspots:
    st.subheader("Hotspot analysis")
    if top_hotspots.empty:
        st.warning("No hotspots available under the current filters.")
    else:
        hotspot_columns = ["Name", "Country", "Opportunity_Score", "Priority_Band", "Production_tpa", "Nearest_Port_Name", "Distance_to_Port_km", "Nearest_Wind_Name", "Distance_to_Wind_km", "Nearest_Solar_Name", "Distance_to_Solar_km", "Nearest_Water_Name", "Distance_to_Water_km"]
        st.dataframe(top_hotspots[hotspot_columns], width="stretch", hide_index=True)
        country_summary = top_hotspots.groupby("Country", as_index=False).agg(Hotspot_Count=("Name", "count"), Average_Score=("Opportunity_Score", "mean"), Highest_Score=("Opportunity_Score", "max"), Total_Anchor_Production_tpa=("Production_tpa", "sum"), Average_Port_Distance_km=("Distance_to_Port_km", "mean"))
        for col in ["Average_Score", "Highest_Score", "Average_Port_Distance_km"]:
            country_summary[col] = country_summary[col].round(2)
        st.markdown("### Country hotspot summary")
        st.dataframe(country_summary.sort_values(["Hotspot_Count", "Highest_Score"], ascending=[False, False]), width="stretch", hide_index=True)

with tab_ranking:
    st.subheader("Full opportunity ranking")
    if filtered_opportunity.empty:
        st.warning("No opportunity records match the current filters.")
    else:
        display_columns = [
            "Name", "Country", "Production_tpa", "Ammonia_Anchor_Score",
            "Nearest_Port_Name", "Distance_to_Port_km", "Port_Access_Score",
            "Nearest_Wind_Name", "Wind_Quality_Score", "Wind_Proximity_Score", "Wind_Score",
            "Nearest_Solar_Name", "Solar_Quality_Score", "Solar_Proximity_Score", "Solar_Score",
            "Nearest_Water_Name", "Water_Quality_Score", "Water_Proximity_Score", "Water_Score",
            "Opportunity_Score", "Priority_Band"
        ]
        st.dataframe(filtered_opportunity[display_columns], width="stretch", hide_index=True)
        st.download_button("Download filtered ranking as CSV", data=filtered_opportunity.to_csv(index=False).encode("utf-8"), file_name="hydrogem_hydrogen_opportunity_ranking.csv", mime="text/csv")

with tab_methodology:
    show_dictionary(raw_weights=raw_weights, effective_weights=effective_weights)

with tab_quality:
    st.subheader("Data quality and validation report")
    st.markdown(f"<div class='success-box'><b>Energy dataset loaded:</b> {escape_text(energy_bundle.source_name)}</div>", unsafe_allow_html=True)
    if water_bundle is not None:
        st.markdown(f"<div class='success-box'><b>Water dataset loaded:</b> {escape_text(water_bundle.source_name)}</div>", unsafe_allow_html=True)
    elif water_error is not None:
        st.markdown(f"<div class='error-box'><h4>{escape_text(water_error.title)}</h4><ul>{''.join([f'<li>{escape_text(issue)}</li>' for issue in water_error.issues])}</ul></div>", unsafe_allow_html=True)
    else:
        st.warning("No water dataset was loaded.")
    st.markdown(f"<div class='success-box'><b>Shipping port dataset:</b> {escape_text(port_bundle.source_name)}</div>", unsafe_allow_html=True)

    st.markdown("### Relevant uploaded datasets for this methodology")
    relevance = pd.DataFrame([
        {"Dataset": "Fertilizer Plants_AHDS.xlsx", "Use": "Core spatial dataset", "Reason": "Contains ammonia/fertilizer anchors, wind potential and solar potential with coordinates."},
        {"Dataset": "Water availability.xlsx", "Use": "Core spatial dataset", "Reason": "Contains wastewater, dams and desalination assets with coordinates."},
        {"Dataset": "Shipping Ports.xlsx", "Use": "Core spatial dataset", "Reason": "Required for shipping port access. If absent, app uses fallback port list."},
        {"Dataset": "Petroleum_Urea_Ammonia Commodity Report  (1).xlsx", "Use": "Context dataset", "Reason": "Useful for country-level ammonia market context, but not a plant-location dataset."},
        {"Dataset": "Ammonia_Urea data.xlsx", "Use": "Not used in spatial score", "Reason": "Contains ammonia/urea conversion ratios, not coordinates."},
        {"Dataset": "Cost analysis on different countries.xlsx", "Use": "Future economics module", "Reason": "Useful for cost overlays, but not part of the current weighted-priority location score."},
        {"Dataset": "Africa's Energy project landscape_ Cost & type of electrolyzers.xlsx", "Use": "Future project landscape module", "Reason": "Useful for benchmark context, but not required for this spatial scoring version."},
        {"Dataset": "Cabon intensity.xls", "Use": "Future emissions module", "Reason": "Potentially useful for carbon intensity overlays, but not part of the current scoring request."},
        {"Dataset": "hydrogen for petroleum products.xlsx", "Use": "Future demand module", "Reason": "Useful for refinery hydrogen demand logic, but not a location scoring input here."},
    ])
    st.dataframe(relevance, width="stretch", hide_index=True)

    def show_report(title, report):
        st.markdown(f"### {title}")
        if report.get("sheets"):
            st.dataframe(pd.DataFrame([{"Expected Sheet": k, "Workbook Sheet Used": v} for k, v in report["sheets"].items()]), width="stretch", hide_index=True)
        if report.get("rows"):
            st.dataframe(pd.DataFrame([{"Sheet": k, "Raw Rows": v["raw_rows"], "Usable Rows": v["usable_rows"], "Dropped Rows": v["dropped_rows"]} for k, v in report["rows"].items()]), width="stretch", hide_index=True)
        for warning in report.get("warnings", []):
            st.warning(warning)

    show_report("Energy validation", energy_bundle.report)
    if water_bundle is not None:
        show_report("Water validation", water_bundle.report)
    show_report("Shipping ports validation", port_bundle.report)
    if not commodity_bundle.df_ammonia_country.empty:
        show_report("Ammonia commodity context", commodity_bundle.report)
        with st.expander("Preview ammonia country context"):
            st.dataframe(commodity_bundle.df_ammonia_country.head(50), width="stretch", hide_index=True)
    with st.expander("Preview candidate anchors"):
        st.dataframe(df_candidates.head(100), width="stretch", hide_index=True)
    with st.expander("Preview ports"):
        st.dataframe(raw_ports.head(100), width="stretch", hide_index=True)

with tab_guide:
    show_data_guide()
