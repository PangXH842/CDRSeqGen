import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

line_markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def plot_linegraph(runs_dict, metric="eval_loss", title="Comparison", save_dir=".", filename=None):
    """
    Plots a multi-line comparison of metrics from epoch_metrics.csv.
    """
    plt.figure(figsize=(10, 6))

    found_data = False
    n_runs = max(len(runs_dict), 1)
    # Sample a continuous colormap so every run gets a distinct color, no matter how many runs are compared
    colors = [plt.cm.turbo(i / max(n_runs - 1, 1)) for i in range(n_runs)]

    for i, (label, folder_path) in enumerate(runs_dict.items()):
        csv_path = os.path.join(folder_path, "epoch_metrics.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if metric in df.columns:
                plt.plot(
                    df["epoch"],
                    df[metric],
                    color=colors[i],
                    marker=line_markers[i % len(line_markers)],
                    linewidth=2,
                    label=label
                )

                found_data = True
            else:
                print(f"⚠️ Metric '{metric}' not found in {csv_path}")
        else:
            print(f"❌ Could not find epoch_metrics.csv in {folder_path}")

    if not found_data:
        print("🚫 No valid data found to plot. Check your folder paths!")
        return

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    output_basename = filename if filename else f"{metric}_curve"
    os.makedirs(save_dir, exist_ok=True)
    output_name = os.path.join(save_dir, f"{output_basename}.png")
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved as: {output_name}")
    plt.close()

def plot_metric_breakdown(run_dir, metrics, title="Metric breakdown", save_dir=".", filename=None):
    """Plot SEVERAL metrics from ONE run's epoch_metrics.csv on a single chart.

    The transpose of plot_linegraph (which compares one metric across many runs). Used for the
    whole-span vs core vs anchor breakdown, where the whole point is seeing the three together:
    on H3 they sit at roughly 0.46 / 0.32 / 0.90, and that spread IS the anchor-inflation finding.

    `metrics` is a list of (column_name, legend_label) pairs.
    """
    csv_path = os.path.join(run_dir, "epoch_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Could not find epoch_metrics.csv in {run_dir}")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    colors = [plt.cm.turbo(i / max(len(metrics) - 1, 1)) for i in range(len(metrics))]

    plotted = False
    for i, (metric, label) in enumerate(metrics):
        if metric not in df.columns:
            print(f"⚠️ Metric '{metric}' not found in {csv_path}")
            continue
        plt.plot(
            df["epoch"], df[metric],
            color=colors[i], marker=line_markers[i % len(line_markers)],
            linewidth=2, label=label,
        )
        plotted = True

    if not plotted:
        print("🚫 No valid data found to plot.")
        plt.close()
        return

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Recovery", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    output_name = os.path.join(save_dir, f"{filename or 'metric_breakdown'}.png")
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved as: {output_name}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically compare multiple training runs.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "FOLDER_PATH"), required=True, help="Specify a run. Example: --run 'scratch' './results/run_folder' --run 'esm2' './results/run_folder'")
    # Any column present in epoch_metrics.csv is plottable. This used to be restricted to
    # eval_loss/eval_accuracy, which made the anchor-excluded CORE recovery — the metric the README
    # says to actually report — impossible to plot without editing this file.
    parser.add_argument("--metric", type=str, default="eval_core_accuracy",
                        help="Column from epoch_metrics.csv to plot: eval_loss, eval_accuracy, "
                             "eval_core_accuracy (default), eval_anchor_accuracy, ...")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title for the plot")
    parser.add_argument("--filename", type=str, default=None, help="Optional custom filename for the saved plot (without extension)")
    parser.add_argument("--save_dir", type=str, default="results/pngs", help="Directory to save the plot (created if missing). Defaults to results/pngs.")
    args = parser.parse_args()

    # Convert the list of parsed pairs into a dictionary
    # Example format: {'Label 1': 'path1', 'Label 2': 'path2'}
    runs_to_compare = {label: folder for label, folder in args.run}

    # Generate title if not provided
    default_title = f"{args.metric.replace('_', ' ').title()} Comparison"
    plot_title = args.title if args.title else default_title

    # Keep filename in sync with a custom title unless --filename overrides it explicitly
    if args.filename:
        plot_filename = args.filename
    elif args.title:
        plot_filename = slugify(args.title)
    else:
        plot_filename = None

    plot_linegraph(runs_to_compare, metric=args.metric, title=plot_title, save_dir=args.save_dir, filename=plot_filename)