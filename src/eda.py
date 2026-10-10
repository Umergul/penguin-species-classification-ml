"""Exploratory data analysis for the Palmer Penguins dataset.

Loads data/penguins.csv, prints summary statistics, and saves charts to assets/.
Deterministic: fixed style settings, no randomness.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "penguins.csv"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns and dtypes:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing.to_string())
    print(f"Total rows with any missing value: {int(df.isna().any(axis=1).sum())}")

    print("\nSpecies counts:")
    print(df["species"].value_counts().to_string())

    print("\nNumeric summary (describe):")
    with pd.option_context("display.width", 120):
        print(df.describe().to_string())

    print("\nMedian measurements by species:")
    with pd.option_context("display.width", 120):
        print(
            df.groupby("species")[
                ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
            ]
            .median()
            .round(2)
            .to_string()
        )

    # 1. Species distribution bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["Adelie", "Chinstrap", "Gentoo"]
    counts = df["species"].value_counts().reindex(order)
    palette = {"Adelie": "#4C78A8", "Chinstrap": "#F58518", "Gentoo": "#54A24B"}
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=palette, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    ax.set_title("Palmer Penguins: species distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(ASSETS / "species_distribution.png")
    plt.close(fig)

    # 2. Missing values bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    miss = missing[missing > 0]
    if len(miss) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        sns.barplot(x=miss.index, y=miss.values, ax=ax, color="#E45756")
        for i, v in enumerate(miss.values):
            ax.text(i, v + 0.2, str(int(v)), ha="center", fontweight="bold")
        ax.set_title("Missing values per column")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(ASSETS / "missing_values.png")
    plt.close(fig)

    # 3. Correlation heatmap of numeric features
    numeric = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                square=True, ax=ax)
    ax.set_title("Correlation heatmap of numeric features")
    fig.tight_layout()
    fig.savefig(ASSETS / "correlation_heatmap.png")
    plt.close(fig)

    # 4. Scatter: bill length vs bill depth, colored by species
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm",
                    hue="species", hue_order=order,
                    palette=palette, s=55, alpha=0.8, ax=ax)
    ax.set_title("Bill length vs bill depth by species")
    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Bill depth (mm)")
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(ASSETS / "bill_scatter.png")
    plt.close(fig)

    print("\nCharts saved to assets/:")
    for name in ["species_distribution.png", "missing_values.png",
                 "correlation_heatmap.png", "bill_scatter.png"]:
        print(f"  assets/{name}")


if __name__ == "__main__":
    main()
