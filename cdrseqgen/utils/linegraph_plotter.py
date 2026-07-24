import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

line_markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def plot_epoch_curves(series, title="Comparison", ylabel="Value", save_dir=".", filename="plot"):
    """Plot any number of epoch-vs-value lines (from epoch_metrics.csv files) on one chart"""
    plt.figure(figsize=(10, 6))
    colors = [plt.cm.turbo(i / max(len(series) - 1, 1)) for i in range(len(series))]

    csv_cache = {}  # a run_dir reused across lines (the breakdown case) is read once
    plotted = False
    for i, (run_dir, column, label) in enumerate(series):
        csv_path = os.path.join(run_dir, "epoch_metrics.csv")
        if csv_path not in csv_cache:
            csv_cache[csv_path] = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        df = csv_cache[csv_path]

        if df is None:
            print(f"❌ Could not find epoch_metrics.csv in {run_dir}")
        elif column not in df.columns:
            print(f"⚠️ Metric '{column}' not found in {csv_path}")
        else:
            plt.plot(
                df["epoch"], df[column],
                color=colors[i], marker=line_markers[i % len(line_markers)],
                linewidth=2, label=label,
            )
            plotted = True

    if not plotted:
        print("🚫 No valid data found to plot. Check your folder paths and metric names!")
        plt.close()
        return

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    output_name = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved as: {output_name}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically compare training runs / metrics on one chart.")
    # Each --run is one line: LABEL FOLDER [METRIC]. METRIC is optional and defaults to --metric,
    # so same-metric comparisons stay terse (metric once) while a per-run METRIC unlocks breakdowns —
    # point several --run at the SAME folder with different metrics to plot whole/core/anchor together.
    parser.add_argument("--run", action="append", nargs="+", metavar="LABEL FOLDER [METRIC]", required=True,
                        help="A line to plot: LABEL FOLDER [METRIC]. Repeat per line. "
                             "Same metric across runs: --run scratch ./A --run esm2 ./B (with --metric). "
                             "Breakdown of one run: --run whole ./R eval_accuracy --run core ./R eval_core_accuracy ...")
    parser.add_argument("--metric", type=str, default="eval_core_accuracy",
                        help="Default column when a --run omits its METRIC: eval_loss, eval_accuracy, "
                             "eval_core_accuracy (default), eval_anchor_accuracy, ...")
    parser.add_argument("--ylabel", type=str, default=None, help="Y-axis label. Defaults to the metric name, or 'Value' when metrics are mixed.")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title for the plot")
    parser.add_argument("--filename", type=str, default=None, help="Optional custom filename for the saved plot (without extension)")
    parser.add_argument("--save_dir", type=str, default="results/pngs", help="Directory to save the plot (created if missing). Defaults to results/pngs.")
    args = parser.parse_args()

    # One (folder, metric, label) line per --run; a run's METRIC falls back to the global --metric
    series = []
    for entry in args.run:
        if len(entry) == 2:
            label, folder, metric = entry[0], entry[1], args.metric
        elif len(entry) == 3:
            label, folder, metric = entry
        else:
            parser.error(f"--run takes 'LABEL FOLDER [METRIC]' (2 or 3 values), got {len(entry)}: {entry}")
        series.append((folder, metric, label))

    # A single shared metric drives the default title / ylabel / filename; mixed metrics (a breakdown)
    # have no one metric to name, so fall back to generic labels.
    metrics_used = {metric for _, metric, _ in series}
    uniform_metric = next(iter(metrics_used)) if len(metrics_used) == 1 else None

    if args.ylabel:
        ylabel = args.ylabel
    elif uniform_metric:
        ylabel = uniform_metric.replace("_", " ").title()
    else:
        ylabel = "Value"

    if args.title:
        plot_title = args.title
    elif uniform_metric:
        plot_title = f"{uniform_metric.replace('_', ' ').title()} Comparison"
    else:
        plot_title = "Metric breakdown"

    # Keep filename in sync with a custom title unless --filename overrides it explicitly
    if args.filename:
        plot_filename = args.filename
    elif args.title:
        plot_filename = slugify(args.title)
    elif uniform_metric:
        plot_filename = f"{uniform_metric}_curve"
    else:
        plot_filename = "breakdown"

    plot_epoch_curves(
        series,
        title=plot_title,
        ylabel=ylabel,
        save_dir=args.save_dir,
        filename=plot_filename,
    )
