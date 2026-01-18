from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler

from esda.moran import Moran_Local_BV,Moran_BV
from splot.esda import lisa_cluster
import matplotlib.pyplot as plt                                                                                                                 
import libpysal as lps
from esda.moran import Moran_Local
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from esda.getisord import G_Local
from libpysal.weights import Queen
from libpysal.weights import KNN

import numpy as np
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from esda.moran import Moran
from scipy.stats import pearsonr
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from pysal.explore import esda
import itertools
import statsmodels.api as sm


# Reading & Processing Hospital Data 
def preprocess_hospitals(
    hospital_df: pd.DataFrame,
    state: str
) -> pd.DataFrame:
    """
    Aggregate hospital bed counts by ZIP code for a given state.

    Parameters
    ----------
    hospital_df : pd.DataFrame
        Hospital-level dataset containing ZIP codes and bed counts.
    state : str
        State abbreviation (e.g., 'MD').

    Returns
    -------
    pd.DataFrame
        ZIP-level hospital bed counts.
    """
    df = hospital_df.loc[hospital_df["MSTATE"] == state].copy()
    df["ZIP"] = df["MLOCZIP"].str[:5].astype(int)

    hospital_zip = (
        df.groupby("ZIP", as_index=False)
        .agg(count_hospital_bed=("BDTOT", "sum"))
    )

    return hospital_zip



# ADRD Processing

def preprocess_adrd(
    visits_df: pd.DataFrame,
    diag_cols: list,
    adrd_codes: list,
    hospital_df: pd.DataFrame,
    zip_shapefile: gpd.GeoDataFrame,
    state: str
) -> gpd.GeoDataFrame:
    """
    Process ADRD-related hospital visits and aggregate to ZIP level.

    Returns a GeoDataFrame with demographic, racial composition,
    utilization, and hospital capacity attributes.

    Returns
    -------
    gpd.GeoDataFrame
    """

    # Identify ADRD-related visits
    mask = visits_df[diag_cols].apply(
        lambda col: col.str.slice(0, 4).isin(adrd_codes)
    ).any(axis=1)

    df = visits_df.loc[mask].copy()

    # Dominant ZIP per VisitLink
    df["zip_count"] = df.groupby(["VisitLink", "ZIP"])["ZIP"].transform("count")
    df = (
        df.sort_values(["VisitLink", "zip_count", "ZIP"])
          .drop_duplicates("VisitLink")
          .drop(columns="zip_count")
    )

    df["ZIP"] = pd.to_numeric(df["ZIP"], errors="coerce")

    df = df[[
        "ZIP", "AGE", "KEY", "FEMALE", "RACE", "TOTCHG",
        "ZIPINC_QRTL", "PSTCO_GEO"
    ]].copy()

    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

    # ZIP-level aggregation
    adrd_zip = (
        df.groupby("ZIP", as_index=False)
        .agg(
            AGE=("AGE", "mean"),
            KEY=("KEY", "count"),
            FEMALE=("FEMALE", "sum"),
            ZIPINC_QRTL=("ZIPINC_QRTL", "mean"),
            TOTCHG=("TOTCHG", "sum"),
            PSTCO_GEO=("PSTCO_GEO", "first"),
        )
    )

    # Race percentages
    race_counts = (
        df.groupby(["ZIP", "RACE"])
        .size()
        .reset_index(name="count")
    )
    race_counts["pct"] = race_counts.groupby("ZIP")["count"].transform(
        lambda x: 100 * x / x.sum()
    )

    race_wide = (
        race_counts
        .pivot(index="ZIP", columns="RACE", values="pct")
        .fillna(0)
        .reset_index()
    )

    race_wide.columns = ["ZIP"] + [f"race_pct_{i}" for i in race_wide.columns[1:]]

    adrd_zip = adrd_zip.merge(race_wide, on="ZIP", how="left")

    # Merge hospital beds
    hospital_zip = preprocess_hospitals(hospital_df, state)
    adrd_zip = adrd_zip.merge(hospital_zip, on="ZIP", how="left")
    adrd_zip["count_hospital_bed"] = adrd_zip["count_hospital_bed"].fillna(0)
    adrd_zip["female_pct"] = adrd_zip["FEMALE"] / adrd_zip["KEY"]

    return zip_shapefile.merge(adrd_zip, on="ZIP", how="left")



# County level processing

def aggregate_adrd_by_county(
    visits_df: pd.DataFrame,
    diag_cols: list,
    adrd_codes: list,
    adrd_prefixes: list,
    state_fips: str = "24"
) -> pd.DataFrame:
    """
    Aggregate ADRD-related hospital visits to the county level.

    Parameters
    ----------
    visits_df : pd.DataFrame
        Patient-level hospital visit data.
    adrd_prefixes : list
    state_fips : str, default '24'
        State FIPS code (Maryland = '24').

    Returns
    -------
    pd.DataFrame
        County-level ADRD visit counts.
    """

    df = visits_df.copy()

    # Filter ADRD diagnoses
    mask = visits_df[diag_cols].apply(
        lambda col: col.str.slice(0, 4).isin(adrd_codes)
    ).any(axis=1)

    df = visits_df.loc[mask].copy()

    # Dominant ZIP per VisitLink
    df["zip_count"] = df.groupby(["VisitLink", "ZIP"])["ZIP"].transform("count")
    df = (
        df.sort_values(["VisitLink", "zip_count", "ZIP"])
          .drop_duplicates("VisitLink")
          .drop(columns="zip_count")
    )

    # Clean ZIP
    df["ZIP"] = df["ZIP"].astype(str)
    df = df[df["ZIP"].str.match(r"^\d{5}$")]
    df["ZIP"] = df["ZIP"].astype(int)

    # County filter
    df["PSTCO"] = df["PSTCO"].astype(str)
    df = df.loc[df["PSTCO"].str.startswith(state_fips)]

    county_adrd = (
        df.groupby("PSTCO", as_index=False)
        .agg(adrd_key=("KEY", "count"))
    )

    county_adrd["PSTCO"] = county_adrd["PSTCO"].astype(int)

    return county_adrd


def merge_adrd_with_deaths(
    county_adrd: pd.DataFrame,
    death_df: pd.DataFrame,
    cause_name: str 
) -> pd.DataFrame:
    """
    Merge county-level ADRD visits with mortality data.

    Parameters
    ----------
    county_adrd : pd.DataFrame
        Output from aggregate_adrd_by_county().
    death_df : pd.DataFrame
        County-level mortality data.
    cause_name : str
        Cause of death to filter.

    Returns
    -------
    pd.DataFrame
    """

    df_death = death_df.loc[
        death_df["MCD - ICD-10 113 Cause List"] == cause_name
    ].copy()

    df_death = df_death[["County Code", "Deaths", "Population"]]
    df_death["County Code"] = df_death["County Code"].astype(int)

    merged = df_death.merge(
        county_adrd,
        left_on="County Code",
        right_on="PSTCO",
        how="left"
    )

    return merged


#Plotting

import matplotlib.pyplot as plt


def plot_county_rates(
    gdf: gpd.GeoDataFrame,
    cmap: str = "Reds",
    share_scale: bool = True
):
    """
    Plot county-level ADRD and mortality rates.

    Parameters
    ----------
    gdf : GeoDataFrame
        Output from create_county_rates_gdf().
    cmap : str
        Colormap.
    share_scale : bool
        Whether to use shared color scale.
    """

    if share_scale:
        vmin = min(gdf["death_rate"].min(), gdf["adrd_rate"].min())
        vmax = max(gdf["death_rate"].max(), gdf["adrd_rate"].max())
    else:
        vmin = vmax = None

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    gdf.plot(
        column="death_rate",
        cmap=cmap,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax[0]
    )
    ax[0].set_title("Alzheimer’s Mortality Rate")

    gdf.plot(
        column="adrd_rate",
        cmap=cmap,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax[1]
    )
    ax[1].set_title("ADRD Hospitalization Rate")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


# Sensitivity Analysis
import matplotlib.pyplot as plt
import pandas as pd
import mapclassify


def plot_accessibility_sensitivity(
    gdf_list: list,
    labels: list,
    value_col: str = "acc",
    n_classes: int = 5,
    cmap: str = "crest",
    figsize: tuple = (25, 10)
):
    """


    Parameters
    ----------
    gdf_list : list of GeoDataFrame
        List of GeoDataFrames computed under different threshold values.
    labels : list of str
        Titles corresponding to each GeoDataFrame (e.g., threshold distances).
    value_col : str, default 'acc'
        Column containing accessibility values.
    n_classes : int, default 5

    cmap : str, default 'crest'
        Colormap for plotting.
    figsize : tuple
        Figure size.

    Returns
    -------
    None
    """

    if len(gdf_list) != len(labels):
        raise ValueError("gdf_list and labels must have the same length.")

    # Combine all values to enforce consistent classification
    all_values = pd.concat([gdf[value_col] for gdf in gdf_list]).dropna()


    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, gdf, label in zip(axes, gdf_list, labels):
        gdf.plot(
            column=value_col,
            ax=ax,
            classification_kwds={"k": n_classes},
            cmap=cmap,
            legend=False
        )
        ax.set_title(label, fontsize=14)
        ax.axis("off")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(all_values)

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.03,
        pad=0.08
    )
    cbar.set_label("Accessibility", fontsize=12)

    plt.subplots_adjust(right=0.88, wspace=0.1, hspace=0.15)
    plt.show()
