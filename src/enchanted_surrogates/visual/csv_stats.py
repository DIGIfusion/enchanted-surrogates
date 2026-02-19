#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import textwrap
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    # Grid defaults
    "axes.grid": True,            # Always show grid
    "grid.linestyle": "--",       # Dashed lines
    "grid.linewidth": 0.5,        # Thin lines
    "grid.alpha": 0.7,            # Transparency
    "grid.color": "gray",         # Neutral color

    # Optional: enable minor grid lines too
    "axes.grid.which": "both"     # Apply to both major and minor ticks
})

# Defaults optimized for scientific papers
plt.rcParams.update({
    "font.size": 10,        # Base font size (body text)
    "axes.titlesize": 12,   # Axis title (slightly larger for emphasis)
    "axes.labelsize": 11,   # Axis labels (x/y)
    "xtick.labelsize": 10,  # Tick labels
    "ytick.labelsize": 10,  # Tick labels
    "legend.fontsize": 9,   # Legend text (slightly smaller, but still readable)
    "legend.title_fontsize": 9,  # Legend title text
    "figure.titlesize": 13  # Overall figure title
})


def shorten(s, maxlen=40):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen] + "..."

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Compute stats + histograms for a CSV file.")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--chunksize", type=int, default=200000,
                        help="Rows per chunk (default: 200k)")
    parser.add_argument("--max-cats", type=int, default=20,
                        help="Max categorical values to show")
    parser.add_argument("--bins", type=int, default=20,
                        help="Number of histogram bins")
    parser.add_argument("--outdir", default="histograms",
                        help="Directory to save histogram plots")
    args = parser.parse_args()

    csv = args.csv
    chunksize = args.chunksize
    maxcats = args.max_cats
    bins = args.bins
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    print_header(f"Reading CSV: {csv}")

    # First pass: detect dtypes
    sample = pd.read_csv(csv, nrows=5000)
    dtypes = sample.dtypes

    numeric_cols = [c for c in dtypes.index if np.issubdtype(dtypes[c], np.number)]
    categorical_cols = [c for c in dtypes.index if c not in numeric_cols]

    print("Detected numeric columns:", numeric_cols)
    print("Detected categorical columns:", categorical_cols)

    # Stats containers
    numeric_stats = {
        col: {
            "count": 0,
            "sum": 0.0,
            "sum2": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "hist": np.zeros(bins, dtype=np.int64),
        }
        for col in numeric_cols
    }

    categorical_counts = {col: {} for col in categorical_cols}

    # First pass: find global min/max for histograms
    print_header("PASS 1: Scanning for numeric ranges")
    for chunk in pd.read_csv(csv, chunksize=chunksize):
        for col in numeric_cols:
            data = chunk[col].dropna()
            if len(data) == 0:
                continue
            numeric_stats[col]["min"] = min(numeric_stats[col]["min"], data.min())
            numeric_stats[col]["max"] = max(numeric_stats[col]["max"], data.max())

    # Prepare histogram bin edges
    bin_edges = {}
    for col in numeric_cols:
        lo = numeric_stats[col]["min"]
        hi = numeric_stats[col]["max"]
        if lo == float("inf"):
            lo, hi = 0, 1
        bin_edges[col] = np.linspace(lo, hi, bins + 1)

    # Reset stats for second pass
    for col in numeric_cols:
        numeric_stats[col]["count"] = 0
        numeric_stats[col]["sum"] = 0.0
        numeric_stats[col]["sum2"] = 0.0
        numeric_stats[col]["hist"][:] = 0

    print_header("PASS 2: Computing stats + histograms")
    for chunk in pd.read_csv(csv, chunksize=chunksize):
        # Numeric stats
        for col in numeric_cols:
            data = chunk[col].dropna()
            if len(data) == 0:
                continue

            numeric_stats[col]["count"] += len(data)
            numeric_stats[col]["sum"] += data.sum()
            numeric_stats[col]["sum2"] += (data**2).sum()

            # Histogram
            hist, _ = np.histogram(data, bins=bin_edges[col])
            numeric_stats[col]["hist"] += hist

        # Categorical stats
        for col in categorical_cols:
            vc = chunk[col].astype(str).value_counts()
            for val, count in vc.items():
                categorical_counts[col][val] = categorical_counts[col].get(val, 0) + count

    # Print numeric stats + save histogram plots
    print_header("NUMERIC COLUMN STATISTICS + HISTOGRAMS")
    for col, st in numeric_stats.items():
        if st["count"] == 0:
            print(f"{col}: no numeric data")
            continue

        mean = st["sum"] / st["count"]
        var = st["sum2"] / st["count"] - mean**2
        std = np.sqrt(max(var, 0))

        print(f"\nColumn: {col}")
        print(f"  Count: {st['count']}")
        print(f"  Mean:  {mean:.6g}")
        print(f"  Std:   {std:.6g}")
        print(f"  Min:   {bin_edges[col][0]}")
        print(f"  Max:   {bin_edges[col][-1]}")

        # Histogram printout
        print("  Histogram:")
        edges = bin_edges[col]
        hist = st["hist"]
        for i in range(len(hist)):
            lo = edges[i]
            hi = edges[i+1]
            print(f"    [{lo:10.4g}, {hi:10.4g}) : {hist[i]}")

        # Save histogram plot
        # --- Quantile-colored histogram ---
        plt.figure(figsize=(4, 3))
        # Compute cumulative distribution (CDF) for coloring
        counts = hist.astype(float)
        total = counts.sum()
        cdf = np.cumsum(counts) / (total + 1e-12)   # values from 0 → 1

        # Use a diverging colormap centered at 0.5 (white)
        cmap = plt.get_cmap("coolwarm")
        colors = cmap(cdf)   # cdf directly maps 0→blue, 0.5→white, 1→red

        # Bar centers and width
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        plt.bar(
            centers,
            counts,
            width=width,
            color=colors,
            edgecolor="black",
            align="center"
        )

        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

        # Colorbar: 0% = left, 50% = median, 100% = right
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])

        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Cumulative percentile (%)")

        tick_vals = [0, 0.25, 0.5, 0.75, 1.0]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{int(t*100)}%" for t in tick_vals])
        plt.tight_layout()

        safe_col = col.replace('/', 'over')
        plt.savefig(os.path.join(outdir, f"{safe_col}_hist.png"))
        plt.close()

    # Print categorical stats
    print_header("CATEGORICAL COLUMN STATISTICS")
    for col, counts in categorical_counts.items():
        print(f"\nColumn: {col}")
        sorted_vals = sorted(counts.items(), key=lambda x: -x[1])
        print(f"Total categories: {len(sorted_vals)}")
        for val, count in sorted_vals[:maxcats]:
            print(f"  {shorten(val):40s}  {count}")

        if len(sorted_vals) > maxcats:
            print(f"  ... ({len(sorted_vals) - maxcats} more values)")
            
    print_header("PLOTTING CATEGORICAL BAR CHARTS")

    def shorten_middle(s, maxlen=40):
        s = str(s)
        if len(s) <= maxlen:
            return s
        return s[:20] + "..." + s[-15:]

    for col, counts in categorical_counts.items():
        print(f"Creating bar chart for categorical column: {col}")

        sorted_vals = sorted(counts.items(), key=lambda x: -x[1])

        # Show ALL categories if maxcats == 0
        if maxcats > 0:
            top_vals = sorted_vals[:maxcats]
        else:
            top_vals = sorted_vals

        # Unique labels
        labels = [f"{shorten_middle(v)}_{i}" for i, (v, _) in enumerate(top_vals)]
        values = [c for _, c in top_vals]

        plt.figure(figsize=(3.5, 3))
        plt.bar(labels, values)
        plt.xticks(rotation=90)
        plt.title(f"Category frequencies for {col}")
        plt.ylabel("Count")
        plt.tight_layout()

        outpath = os.path.join(outdir, f"{col}_barchart.png")
        plt.savefig(outpath)
        plt.close()

        print(f"  ✔ Saved {outpath}")


if __name__ == "__main__":
    main()
