#!/usr/bin/env python3
"""
Visualize TimesFM forecasts.

Reads the CSVs produced by forecast.py and the historical Parquet file,
renders fan charts showing historical context, point forecast, and
80% / 60% confidence bands. Saves PNGs to outputs/.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config.settings import DATASET_DIR, OUTPUT_DIR

# Dark theme palette to match the QuestDB blog
BG = "#0f172a"               # slate-900
FG = "#e5e7eb"               # gray-200 (text/axes)
GRID_COLOR = "#334155"       # slate-700
HISTORY_COLOR = "#22d3ee"    # cyan-400
FORECAST_COLOR = "#f472b6"   # pink-400
BAND_80 = "#be185d"          # pink-700 (outer band)
BAND_60 = "#ec4899"          # pink-500 (inner band)
DIVIDER_COLOR = "#94a3b8"    # slate-400

# How many minutes of history to show alongside the forecast
HISTORY_MINUTES = 120


def load_history() -> pd.DataFrame:
    parquet_path = DATASET_DIR / "btc_ohlcv_7d.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run method2_parquet_export.py first."
        )
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_forecast(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{name}_forecast.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run forecast.py first.")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_fan(
    history_ts,
    history_values,
    forecast_df: pd.DataFrame,
    title: str,
    y_label: str,
    out_path,
):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160, facecolor=BG)
    ax.set_facecolor(BG)

    # Historical context
    ax.plot(
        history_ts,
        history_values,
        color=HISTORY_COLOR,
        linewidth=2.0,
        alpha=0.95,
        label="History",
    )

    # 80% band (q10 - q90)
    ax.fill_between(
        forecast_df["timestamp"],
        forecast_df["q10"],
        forecast_df["q90"],
        color=BAND_80,
        alpha=0.85,
        label="80% interval (q10-q90)",
    )

    # 60% band (q20 - q80)
    if "q20" in forecast_df.columns and "q80" in forecast_df.columns:
        ax.fill_between(
            forecast_df["timestamp"],
            forecast_df["q20"],
            forecast_df["q80"],
            color=BAND_60,
            alpha=0.65,
            label="60% interval (q20-q80)",
        )

    # Forecast line - drawn last so it sits on top of the bands
    ax.plot(
        forecast_df["timestamp"],
        forecast_df["forecast"],
        color=FORECAST_COLOR,
        linewidth=3.0,
        label="Point forecast",
    )

    # Forecast boundary
    boundary = forecast_df["timestamp"].iloc[0]
    ax.axvline(
        boundary,
        color=DIVIDER_COLOR,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.annotate(
        "Forecast start",
        xy=(boundary, ax.get_ylim()[1]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=11,
        color=DIVIDER_COLOR,
    )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=16, color=FG)
    ax.set_ylabel(y_label, fontsize=13, color=FG)
    ax.set_xlabel("Time (UTC)", fontsize=13, color=FG)
    ax.tick_params(colors=FG, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.3, linestyle=":", color=GRID_COLOR)
    legend = ax.legend(
        loc="upper left",
        fontsize=12,
        framealpha=0.9,
        facecolor=BG,
        edgecolor=GRID_COLOR,
        labelcolor=FG,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


def main():
    print("=" * 60)
    print("VISUALIZING FORECASTS")
    print("=" * 60)

    history = load_history()
    cutoff = history["timestamp"].iloc[-1] - pd.Timedelta(minutes=HISTORY_MINUTES)
    recent = history[history["timestamp"] >= cutoff].copy()
    print(
        f"\nHistory window: {recent['timestamp'].iloc[0]} "
        f"to {recent['timestamp'].iloc[-1]} ({len(recent)} rows)"
    )

    # Volume
    volume_forecast = load_forecast("volume")
    plot_fan(
        history_ts=recent["timestamp"],
        history_values=recent["volume"].values,
        forecast_df=volume_forecast,
        title="BTC-USDT volume forecast (next 60 minutes)",
        y_label="Volume per minute",
        out_path=OUTPUT_DIR / "volume_forecast.png",
    )

    # Volatility (high - low)
    recent["volatility"] = recent["high"] - recent["low"]
    volatility_forecast = load_forecast("volatility")
    plot_fan(
        history_ts=recent["timestamp"],
        history_values=recent["volatility"].values,
        forecast_df=volatility_forecast,
        title="BTC-USDT volatility forecast (next 60 minutes)",
        y_label="High-low range per minute (USD)",
        out_path=OUTPUT_DIR / "volatility_forecast.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
