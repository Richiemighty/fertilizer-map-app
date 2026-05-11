import io
import re
import html
from dataclasses import dataclass
from pathlib import Path

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
    page_title="African Fertilizer and Renewable Energy Opportunity Map",
    page_icon="🌍",
    layout="wide"
)

DEFAULT_DATASET_PATH = Path("Fertilizer Plants_AHDS.xlsx")


# =====================================================
# Styling
# =====================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 750;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            font-size: 1rem;
            color: #4B5563;
            margin-bottom: 1rem;
        }

        .info-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 12px;
        }

        .guide-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
        }

        .small-muted {
            color: #6B7280;
            font-size: 0.9rem;
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
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Data structures
# =====================================================

@dataclass
class DatasetBundle:
    df_fert: pd.DataFrame
    df_wind: pd.DataFrame
    df_solar: pd.DataFrame
    report: dict
    source_name: str


class DataValidationError(Exception):
    def __init__(self, title, issues):
        self.title = title
        self.issues = issues
        super().__init__(title)


# =====================================================
# Expected sheets and columns
# =====================================================

SHEET_ALIASES = {
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


def align_columns(df, alias_map, sheet_display_name):
    """
    Renames uploaded columns to the internal canonical column names.

    If required columns are missing, the app returns a friendly validation error
    instead of breaking.
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
            missing_columns.append(canonical_col)
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
                "Check the Data Guide tab and rename your columns to match the expected structure."
            ]
        )

    return df.rename(columns=rename_map)


def parse_number_or_range(value):
    """
    Converts numbers or numeric ranges into a float.

    Examples:
    500 -> 500.0
    '450-520' -> 485.0
    '450 – 520' -> 485.0
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


def safe_range_slider(label, series, sidebar=True):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None, None

    min_value = int(np.floor(values.min()))
    max_value = int(np.ceil(values.max()))

    target = st.sidebar if sidebar else st

    if min_value == max_value:
        target.caption(f"{label}: only one value available ({min_value:,})")
        return min_value, max_value

    return target.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value)
    )


# =====================================================
# Cleaning functions
# =====================================================

def clean_fertilizer_sheet(raw_df, report):
    df = align_columns(
        raw_df,
        FERTILIZER_COLUMN_ALIASES,
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
            title="No valid fertilizer plant rows found",
            issues=[
                "The Fertilizer Plants sheet was found, but no valid rows could be used.",
                "Each row must have Name, Country, Latitude, Longitude, and Production.",
                "Latitude and Longitude must be numeric coordinates.",
                "Production must be a positive number."
            ]
        )

    outside_africa = ~(
        df["Latitude"].between(-40, 40)
        & df["Longitude"].between(-25, 60)
    )

    if outside_africa.any():
        report["warnings"].append(
            f"{int(outside_africa.sum())} fertilizer plant row(s) appear to be outside the Africa map range."
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
            title="No valid wind potential rows found",
            issues=[
                "The Wind Potential sheet was found, but no valid rows could be used.",
                "Each row must have Country, Region, Latitude, Longitude, and Wind Power Density.",
                "Wind Power Density must be a positive number or a numeric range such as 450-520."
            ]
        )

    outside_africa = ~(
        df["Latitude"].between(-40, 40)
        & df["Longitude"].between(-25, 60)
    )

    if outside_africa.any():
        report["warnings"].append(
            f"{int(outside_africa.sum())} wind row(s) appear to be outside the Africa map range."
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
            "The Solar Potential sheet was found, but no valid solar rows could be used."
        )
        return pd.DataFrame()

    outside_africa = ~(
        df["Latitude"].between(-40, 40)
        & df["Longitude"].between(-25, 60)
    )

    if outside_africa.any():
        report["warnings"].append(
            f"{int(outside_africa.sum())} solar row(s) appear to be outside the Africa map range."
        )

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
# Dataset loading
# =====================================================

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes, source_name):
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
            title="The file could not be opened",
            issues=[
                "Please upload a valid .xlsx Excel workbook.",
                "Do not upload a CSV file here.",
                "If your file is open or corrupted, save a fresh copy and try again."
            ]
        )

    sheet_names = excel_file.sheet_names

    fertilizer_sheet = find_matching_sheet(
        sheet_names,
        SHEET_ALIASES["fertilizer"]
    )

    wind_sheet = find_matching_sheet(
        sheet_names,
        SHEET_ALIASES["wind"]
    )

    solar_sheet = find_matching_sheet(
        sheet_names,
        SHEET_ALIASES["solar"]
    )

    missing_sheets = []

    if fertilizer_sheet is None:
        missing_sheets.append("Fertilizer Plants")

    if wind_sheet is None:
        missing_sheets.append("Wind Potential")

    if missing_sheets:
        raise DataValidationError(
            title="Required sheet(s) missing",
            issues=[
                f"Missing required sheet(s): {', '.join(missing_sheets)}.",
                f"Sheets found in your workbook: {', '.join(sheet_names)}.",
                "Check the Data Guide tab and structure your workbook correctly."
            ]
        )

    report["sheets"]["Fertilizer Plants"] = fertilizer_sheet
    report["sheets"]["Wind Potential"] = wind_sheet

    raw_fert = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=fertilizer_sheet
    )

    raw_wind = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=wind_sheet
    )

    df_fert = clean_fertilizer_sheet(raw_fert, report)
    df_wind = clean_wind_sheet(raw_wind, report)

    df_solar = pd.DataFrame()

    if solar_sheet is not None:
        report["sheets"]["Solar Potential"] = solar_sheet

        raw_solar = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=solar_sheet
        )

        try:
            df_solar = clean_solar_sheet(raw_solar, report)
        except DataValidationError as exc:
            report["warnings"].extend(exc.issues)
            df_solar = pd.DataFrame()
    else:
        report["warnings"].append(
            "No Solar Potential sheet was found. Solar will be treated as optional and hidden."
        )

    return DatasetBundle(
        df_fert=df_fert,
        df_wind=df_wind,
        df_solar=df_solar,
        report=report,
        source_name=source_name
    )


# =====================================================
# Opportunity scoring
# =====================================================

def find_nearest_resource(
    source_df,
    resource_df,
    resource_name_col,
    resource_country_col,
    resource_value_col
):
    results = []

    if resource_df.empty:
        return pd.DataFrame()

    for _, source_row in source_df.iterrows():
        distances = haversine_distance_km(
            source_row["Latitude"],
            source_row["Longitude"],
            resource_df["Latitude"].values,
            resource_df["Longitude"].values
        )

        nearest_position = np.argmin(distances)
        nearest_row = resource_df.iloc[nearest_position]

        results.append({
            "Nearest Resource": nearest_row[resource_name_col],
            "Nearest Resource Country": nearest_row[resource_country_col],
            "Nearest Resource Value": nearest_row[resource_value_col],
            "Distance to Nearest Resource (km)": round(float(distances[nearest_position]), 2)
        })

    return pd.DataFrame(results)


def build_opportunity_table(df_fert, df_wind, df_solar):
    nearest_wind = find_nearest_resource(
        source_df=df_fert,
        resource_df=df_wind,
        resource_name_col="Region",
        resource_country_col="Country",
        resource_value_col="Wind_Density_wm2"
    )

    nearest_wind = nearest_wind.rename(columns={
        "Nearest Resource": "Nearest_Wind_Region",
        "Nearest Resource Country": "Nearest_Wind_Country",
        "Nearest Resource Value": "Nearest_Wind_Density_wm2",
        "Distance to Nearest Resource (km)": "Distance_to_Wind_km"
    })

    opportunity = pd.concat([
        df_fert[[
            "Name",
            "Country",
            "Production_tpa",
            "Latitude",
            "Longitude"
        ]].reset_index(drop=True),
        nearest_wind.reset_index(drop=True)
    ], axis=1)

    if not df_solar.empty:
        nearest_solar = find_nearest_resource(
            source_df=df_fert,
            resource_df=df_solar,
            resource_name_col="Solar_Site",
            resource_country_col="Country",
            resource_value_col="Solar_Capacity_MW"
        )

        nearest_solar = nearest_solar.rename(columns={
            "Nearest Resource": "Nearest_Solar_Site",
            "Nearest Resource Country": "Nearest_Solar_Country",
            "Nearest Resource Value": "Nearest_Solar_Capacity_MW",
            "Distance to Nearest Resource (km)": "Distance_to_Solar_km"
        })

        opportunity = pd.concat([
            opportunity.reset_index(drop=True),
            nearest_solar.reset_index(drop=True)
        ], axis=1)

    opportunity["Production_Score"] = minmax_score(opportunity["Production_tpa"])
    opportunity["Wind_Quality_Score"] = minmax_score(opportunity["Nearest_Wind_Density_wm2"])
    opportunity["Wind_Distance_Score"] = 100 - minmax_score(opportunity["Distance_to_Wind_km"])

    if not df_solar.empty:
        opportunity["Solar_Quality_Score"] = minmax_score(opportunity["Nearest_Solar_Capacity_MW"])
        opportunity["Solar_Distance_Score"] = 100 - minmax_score(opportunity["Distance_to_Solar_km"])

        opportunity["Best_Renewable_Quality_Score"] = opportunity[
            ["Wind_Quality_Score", "Solar_Quality_Score"]
        ].max(axis=1)

        opportunity["Best_Renewable_Distance_Score"] = opportunity[
            ["Wind_Distance_Score", "Solar_Distance_Score"]
        ].max(axis=1)
    else:
        opportunity["Best_Renewable_Quality_Score"] = opportunity["Wind_Quality_Score"]
        opportunity["Best_Renewable_Distance_Score"] = opportunity["Wind_Distance_Score"]

    opportunity["Opportunity_Score"] = (
        opportunity["Production_Score"] * 0.40
        + opportunity["Best_Renewable_Quality_Score"] * 0.35
        + opportunity["Best_Renewable_Distance_Score"] * 0.25
    ).round(2)

    opportunity["Priority_Band"] = opportunity["Opportunity_Score"].apply(classify_score)

    return opportunity.sort_values(
        "Opportunity_Score",
        ascending=False
    ).reset_index(drop=True)


# =====================================================
# Folium map functions
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


def add_satellite_tiles(folium_map):
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
    <div style="font-family: Arial, sans-serif; min-width: 240px;">
        <h4 style="margin:0 0 8px 0;">{escape_text(title)}</h4>
        <table style="border-collapse:collapse;width:100%;">
            {html_rows}
        </table>
    </div>
    """


def build_folium_map(
    df_fert,
    df_wind,
    df_solar,
    filtered_opportunity,
    show_fertilizer=True,
    show_wind=True,
    show_solar=False,
    show_priority=True
):
    folium_map = folium.Map(
        location=[2.0, 20.0],
        zoom_start=3,
        tiles=None,
        control_scale=True
    )

    add_satellite_tiles(folium_map)

    fertilizer_cmap = make_colormap(
        df_fert["Production_tpa"] if not df_fert.empty else pd.Series(dtype=float),
        ["#FEE2E2", "#DC2626", "#7F1D1D"]
    )

    wind_cmap = make_colormap(
        df_wind["Wind_Density_wm2"] if not df_wind.empty else pd.Series(dtype=float),
        ["#DBEAFE", "#2563EB", "#1E3A8A"]
    )

    solar_cmap = make_colormap(
        df_solar["Solar_Capacity_MW"] if not df_solar.empty else pd.Series(dtype=float),
        ["#FEF3C7", "#F59E0B", "#92400E"]
    )

    if show_fertilizer:
        fertilizer_group = folium.FeatureGroup(
            name="Fertilizer plants",
            show=True
        )

        for _, row in df_fert.iterrows():
            popup = build_popup(
                row["Name"],
                [
                    ("Country", row["Country"]),
                    ("Production", f"{format_number(row['Production_tpa'])} tons/year"),
                    ("Latitude", f"{row['Latitude']:.4f}"),
                    ("Longitude", f"{row['Longitude']:.4f}")
                ]
            )

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=max(5, row["Marker_Size"] / 2),
                tooltip=f"{row['Name']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=360),
                color="#111827",
                weight=1,
                fill=True,
                fill_color=fertilizer_cmap(row["Production_tpa"]),
                fill_opacity=0.78
            ).add_to(fertilizer_group)

        fertilizer_group.add_to(folium_map)

    if show_wind:
        wind_group = folium.FeatureGroup(
            name="Wind potential",
            show=True
        )

        for _, row in df_wind.iterrows():
            popup = build_popup(
                row["Region"],
                [
                    ("Country", row["Country"]),
                    ("Wind density", f"{format_number(row['Wind_Density_wm2'])} W/m²"),
                    ("Wind speed", f"{format_number(row['Wind_Speed_mps_100m'], 2)} m/s"),
                    ("Latitude", f"{row['Latitude']:.4f}"),
                    ("Longitude", f"{row['Longitude']:.4f}")
                ]
            )

            folium.RegularPolygonMarker(
                location=[row["Latitude"], row["Longitude"]],
                number_of_sides=3,
                radius=max(6, row["Marker_Size"] / 2),
                rotation=0,
                tooltip=f"{row['Region']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=360),
                color="#111827",
                weight=1,
                fill=True,
                fill_color=wind_cmap(row["Wind_Density_wm2"]),
                fill_opacity=0.78
            ).add_to(wind_group)

        wind_group.add_to(folium_map)

    if show_solar and not df_solar.empty:
        solar_group = folium.FeatureGroup(
            name="Solar potential",
            show=True
        )

        for _, row in df_solar.iterrows():
            popup = build_popup(
                row["Solar_Site"],
                [
                    ("Country", row["Country"]),
                    ("Solar capacity", f"{format_number(row['Solar_Capacity_MW'], 1)} MW"),
                    ("Latitude", f"{row['Latitude']:.4f}"),
                    ("Longitude", f"{row['Longitude']:.4f}")
                ]
            )

            folium.RegularPolygonMarker(
                location=[row["Latitude"], row["Longitude"]],
                number_of_sides=4,
                radius=max(6, row["Marker_Size"] / 2),
                rotation=45,
                tooltip=f"{row['Solar_Site']} | {row['Country']}",
                popup=folium.Popup(popup, max_width=360),
                color="#111827",
                weight=1,
                fill=True,
                fill_color=solar_cmap(row["Solar_Capacity_MW"]),
                fill_opacity=0.78
            ).add_to(solar_group)

        solar_group.add_to(folium_map)

    if show_priority and not filtered_opportunity.empty:
        priority_group = folium.FeatureGroup(
            name="Top ranked opportunities",
            show=True
        )

        top_opportunities = filtered_opportunity.head(10)

        for rank, (_, row) in enumerate(top_opportunities.iterrows(), start=1):
            popup_rows = [
                ("Country", row["Country"]),
                ("Production", f"{format_number(row['Production_tpa'])} tons/year"),
                ("Nearest wind region", row["Nearest_Wind_Region"]),
                ("Wind density", f"{format_number(row['Nearest_Wind_Density_wm2'])} W/m²"),
                ("Distance to wind", f"{format_number(row['Distance_to_Wind_km'], 1)} km"),
                ("Opportunity score", f"{format_number(row['Opportunity_Score'], 2)}"),
                ("Priority band", row["Priority_Band"])
            ]

            if "Nearest_Solar_Site" in row.index:
                popup_rows.extend([
                    ("Nearest solar site", row.get("Nearest_Solar_Site", "N/A")),
                    ("Distance to solar", f"{format_number(row.get('Distance_to_Solar_km', np.nan), 1)} km")
                ])

            popup = build_popup(
                f"#{rank} {row['Name']}",
                popup_rows
            )

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f"#{rank} opportunity: {row['Name']}",
                popup=folium.Popup(popup, max_width=380),
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:#166534;
                        color:white;
                        border:2px solid white;
                        border-radius:999px;
                        width:30px;
                        height:30px;
                        text-align:center;
                        line-height:27px;
                        font-weight:700;
                        box-shadow:0 1px 5px rgba(0,0,0,0.35);
                    ">
                        {rank}
                    </div>
                    """
                )
            ).add_to(priority_group)

        priority_group.add_to(folium_map)

    folium.LayerControl(collapsed=False).add_to(folium_map)

    return folium_map


def folium_map_to_html(folium_map):
    """
    Converts a Folium map into downloadable HTML.
    """
    return folium_map.get_root().render()

# =====================================================
# Template workbook
# =====================================================

def build_template_workbook():
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
        fertilizer_template.to_excel(
            writer,
            sheet_name="Fertilizer Plants",
            index=False
        )

        wind_template.to_excel(
            writer,
            sheet_name="Wind Potential",
            index=False
        )

        solar_template.to_excel(
            writer,
            sheet_name="Solar Potential",
            index=False
        )

    return output.getvalue()


# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div class="main-title">
        African fertilizer and renewable energy opportunity map
    </div>
    <div class="subtitle">
        A decision-support dashboard for comparing fertilizer production assets with wind and solar resource potential.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Sidebar: data source
# =====================================================

st.sidebar.header("Data source")

data_mode = st.sidebar.radio(
    "Choose data source",
    [
        "Use default dataset",
        "Upload visitor dataset"
    ]
)

file_bytes = None
source_name = None

if data_mode == "Use default dataset":
    st.sidebar.caption(
        "The app will use the Excel file placed in the same folder as app.py."
    )

    if DEFAULT_DATASET_PATH.exists():
        file_bytes = DEFAULT_DATASET_PATH.read_bytes()
        source_name = str(DEFAULT_DATASET_PATH)
        st.sidebar.success(f"Loaded default file: {DEFAULT_DATASET_PATH.name}")
    else:
        st.sidebar.error(
            f"Default file not found: {DEFAULT_DATASET_PATH.name}"
        )

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an Excel workbook",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_name = uploaded_file.name
        st.sidebar.success(f"Uploaded file: {uploaded_file.name}")

st.sidebar.divider()


# =====================================================
# Data guide tab can show even without loaded data
# =====================================================

def show_data_guide():
    st.subheader("Data structure guide")

    st.markdown(
        """
        Your workbook should be an `.xlsx` file with at least **two required sheets** and one optional sheet.

        Required sheets:

        1. **Fertilizer Plants**
        2. **Wind Potential**

        Optional sheet:

        3. **Solar Potential**

        The app is flexible with some column names, but the safest approach is to use the exact structure below.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fertilizer Plants sheet")

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
                    "Dangote Fertilizer Plant",
                    "Nigeria",
                    "6.5244",
                    "3.3792",
                    "1200000"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Wind Potential sheet")

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
                    "Recommended",
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
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("#### Solar Potential sheet")

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
                    "Optional",
                    "Optional",
                    "Optional",
                    "Optional",
                    "Optional"
                ],
                "Example": [
                    "Nigeria",
                    "Abuja Solar Site",
                    "9.0765",
                    "7.3986",
                    "100"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            #### Validation rules

            The app will reject or drop rows where:

            - Latitude or longitude is missing or invalid
            - Production is missing, zero, or negative
            - Wind density is missing, zero, or negative
            - Required sheet names are missing
            - Required columns cannot be matched

            Wind density can be written as a single number such as `500`, or as a range such as `450-520`.
            """
        )

        st.download_button(
            label="Download Excel template",
            data=build_template_workbook(),
            file_name="fertilizer_renewable_data_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =====================================================
# Stop here if no data loaded
# =====================================================

if file_bytes is None:
    tab_dashboard, tab_guide = st.tabs([
        "Start here",
        "Data guide"
    ])

    with tab_dashboard:
        st.info(
            "No dataset is currently loaded. Use the default dataset in your project folder or upload a visitor dataset from the sidebar."
        )

        st.markdown(
            f"""
            Expected default file location:

            ```text
            {DEFAULT_DATASET_PATH}
            ```

            Place the Excel workbook in the same folder as `app.py`, then refresh the app.
            """
        )

    with tab_guide:
        show_data_guide()

    st.stop()


# =====================================================
# Load and validate data
# =====================================================

try:
    with st.spinner("Loading and validating dataset..."):
        bundle = load_dataset(file_bytes, source_name)
except DataValidationError as exc:
    tab_error, tab_guide = st.tabs([
        "Upload error",
        "Data guide"
    ])

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
except Exception as exc:
    tab_error, tab_guide = st.tabs([
        "Upload error",
        "Data guide"
    ])

    with tab_error:
        st.error("An unexpected error occurred while loading the file.")
        st.write(str(exc))

    with tab_guide:
        show_data_guide()

    st.stop()


df_fert = bundle.df_fert
df_wind = bundle.df_wind
df_solar = bundle.df_solar
report = bundle.report

opportunity = build_opportunity_table(df_fert, df_wind, df_solar)


# =====================================================
# Sidebar: filters
# =====================================================

st.sidebar.header("Filters")

all_countries = sorted(
    set(df_fert["Country"].dropna().unique())
    | set(df_wind["Country"].dropna().unique())
    | (set(df_solar["Country"].dropna().unique()) if not df_solar.empty else set())
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=all_countries,
    default=all_countries
)

production_min, production_max = safe_range_slider(
    "Fertilizer production range, tons/year",
    df_fert["Production_tpa"]
)

wind_min, wind_max = safe_range_slider(
    "Wind power density range, W/m²",
    df_wind["Wind_Density_wm2"]
)

if not df_solar.empty:
    solar_min, solar_max = safe_range_slider(
        "Solar capacity range, MW",
        df_solar["Solar_Capacity_MW"]
    )
else:
    solar_min, solar_max = None, None

min_score = st.sidebar.slider(
    "Minimum opportunity score",
    min_value=0,
    max_value=100,
    value=0
)

max_wind_distance = st.sidebar.slider(
    "Maximum distance to nearest wind resource, km",
    min_value=0,
    max_value=int(max(1, np.ceil(opportunity["Distance_to_Wind_km"].max()))),
    value=int(max(1, np.ceil(opportunity["Distance_to_Wind_km"].max())))
)

st.sidebar.divider()

st.sidebar.header("Map layers")

show_fertilizer = st.sidebar.checkbox(
    "Show fertilizer plants",
    value=True
)

show_wind = st.sidebar.checkbox(
    "Show wind potential",
    value=True
)

show_solar = st.sidebar.checkbox(
    "Show solar potential",
    value=False,
    disabled=df_solar.empty
)

show_priority = st.sidebar.checkbox(
    "Show top ranked opportunities",
    value=True
)


# =====================================================
# Apply filters
# =====================================================

filtered_fert = df_fert[
    df_fert["Country"].isin(selected_countries)
    & df_fert["Production_tpa"].between(production_min, production_max)
].copy()

filtered_wind = df_wind[
    df_wind["Country"].isin(selected_countries)
    & df_wind["Wind_Density_wm2"].between(wind_min, wind_max)
].copy()

if not df_solar.empty:
    filtered_solar = df_solar[
        df_solar["Country"].isin(selected_countries)
        & df_solar["Solar_Capacity_MW"].between(solar_min, solar_max)
    ].copy()
else:
    filtered_solar = df_solar.copy()

filtered_opportunity = opportunity[
    opportunity["Country"].isin(selected_countries)
    & opportunity["Production_tpa"].between(production_min, production_max)
    & (opportunity["Opportunity_Score"] >= min_score)
    & (opportunity["Distance_to_Wind_km"] <= max_wind_distance)
].copy()


# =====================================================
# Main tabs
# =====================================================

tab_overview, tab_map, tab_ranking, tab_quality, tab_guide = st.tabs([
    "Overview",
    "Satellite map",
    "Opportunity ranking",
    "Data quality",
    "Data guide"
])


# =====================================================
# Overview tab
# =====================================================

with tab_overview:
    st.subheader("Executive overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fertilizer plants",
        f"{len(filtered_fert):,}"
    )

    col2.metric(
        "Wind locations",
        f"{len(filtered_wind):,}"
    )

    col3.metric(
        "Solar locations",
        f"{len(filtered_solar):,}" if not filtered_solar.empty else "0"
    )

    col4.metric(
        "Ranked opportunities",
        f"{len(filtered_opportunity):,}"
    )

    st.markdown("#### Top opportunity locations")

    if filtered_opportunity.empty:
        st.warning(
            "No opportunities match the selected filters. Reduce the filters in the sidebar."
        )
    else:
        top_three = filtered_opportunity.head(3)

        cols = st.columns(3)

        for index, (_, row) in enumerate(top_three.iterrows()):
            with cols[index]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h4 style="margin-top:0;">#{index + 1} {escape_text(row['Name'])}</h4>
                        <p class="small-muted">{escape_text(row['Country'])}</p>
                        <p><b>Opportunity score:</b> {format_number(row['Opportunity_Score'], 2)}</p>
                        <p><b>Production:</b> {format_number(row['Production_tpa'])} tons/year</p>
                        <p><b>Nearest wind:</b> {escape_text(row['Nearest_Wind_Region'])}</p>
                        <p><b>Distance to wind:</b> {format_number(row['Distance_to_Wind_km'], 1)} km</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("#### Scoring logic")

    st.markdown(
        """
        The opportunity score uses a simple weighted model:

        - **40% fertilizer production scale**
        - **35% renewable resource quality**
        - **25% renewable proximity**

        This is a screening score. It should support early prioritisation, not replace technical feasibility, grid access analysis, land review, regulatory review, or financial modelling.
        """
    )


# =====================================================
# Satellite map tab
# =====================================================

with tab_map:
    st.subheader("Satellite map with country labels")

    st.caption(
        "Use the layer control on the map to switch between satellite, clean map, street map, and labels."
    )

    folium_map = build_folium_map(
        df_fert=filtered_fert,
        df_wind=filtered_wind,
        df_solar=filtered_solar,
        filtered_opportunity=filtered_opportunity,
        show_fertilizer=show_fertilizer,
        show_wind=show_wind,
        show_solar=show_solar,
        show_priority=show_priority
    )

    st_folium(
        folium_map,
        height=760,
        use_container_width=True,
        returned_objects=[]
    )

    map_html = folium_map_to_html(folium_map)

    st.download_button(
        label="Download current map as HTML",
        data=map_html,
        file_name="african_fertilizer_renewable_satellite_map.html",
        mime="text/html"
    )


# =====================================================
# Ranking tab
# =====================================================

with tab_ranking:
    st.subheader("Ranked opportunity table")

    if filtered_opportunity.empty:
        st.warning(
            "No ranked opportunities match the selected filters."
        )
    else:
        display_columns = [
            "Name",
            "Country",
            "Production_tpa",
            "Nearest_Wind_Region",
            "Nearest_Wind_Country",
            "Nearest_Wind_Density_wm2",
            "Distance_to_Wind_km",
            "Opportunity_Score",
            "Priority_Band"
        ]

        if not df_solar.empty:
            display_columns.extend([
                "Nearest_Solar_Site",
                "Nearest_Solar_Country",
                "Nearest_Solar_Capacity_MW",
                "Distance_to_Solar_km"
            ])

        st.dataframe(
            filtered_opportunity[display_columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_opportunity.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download filtered opportunity table as CSV",
            data=csv_data,
            file_name="green_fertilizer_opportunity_ranking.csv",
            mime="text/csv"
        )

    st.markdown("#### Full methodology note")

    st.markdown(
        """
        This ranking is designed as a **first-pass opportunity screen**.

        It prioritises fertilizer plants that combine:

        1. Large existing fertilizer production capacity
        2. Strong nearby wind or solar resource potential
        3. Shorter distance to the nearest renewable resource point

        A high score does not automatically mean the site is investment-ready. It means the site deserves deeper feasibility work.
        """
    )


# =====================================================
# Data quality tab
# =====================================================

with tab_quality:
    st.subheader("Data quality and validation report")

    st.markdown(
        f"""
        <div class="success-box">
            <b>Dataset loaded successfully:</b> {escape_text(bundle.source_name)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Sheets detected")

    st.dataframe(
        pd.DataFrame([
            {
                "Expected Sheet": expected,
                "Workbook Sheet Used": actual
            }
            for expected, actual in report["sheets"].items()
        ]),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Row validation summary")

    st.dataframe(
        pd.DataFrame([
            {
                "Sheet": sheet_name,
                "Raw Rows": values["raw_rows"],
                "Usable Rows": values["usable_rows"],
                "Dropped Rows": values["dropped_rows"]
            }
            for sheet_name, values in report["rows"].items()
        ]),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Validation notes")

    if report["warnings"]:
        for warning in report["warnings"]:
            st.warning(warning)
    else:
        st.success("No major data quality warnings detected.")

    with st.expander("Preview cleaned fertilizer data"):
        st.dataframe(
            df_fert.head(30),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Preview cleaned wind data"):
        st.dataframe(
            df_wind.head(30),
            use_container_width=True,
            hide_index=True
        )

    if not df_solar.empty:
        with st.expander("Preview cleaned solar data"):
            st.dataframe(
                df_solar.head(30),
                use_container_width=True,
                hide_index=True
            )


# =====================================================
# Data guide tab
# =====================================================

with tab_guide:
    show_data_guide()