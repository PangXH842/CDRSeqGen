import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

line_styles = ['-', '--', '-.', ':']
line_colors = ['blue', 'orange', 'green', 'red']
line_markers = ['o', 's', 'D', '^']

def plot_linegraph(runs_dict, metric="eval_loss", title="Comparison", save_dir="."):
    """
    Plots a multi-line comparison of metrics from epoch_metrics.csv.
    """
    plt.figure(figsize=(10, 6))
    
    found_data = False
    
    for i, (label, folder_path) in enumerate(runs_dict.items()):
        csv_path = os.path.join(folder_path, "epoch_metrics.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            if metric in df.columns:
                c_style  = line_styles[i % len(line_styles)]
                c_color  = line_colors[i % len(line_colors)]
                c_marker = line_markers[i % len(line_markers)]
                
                plt.plot(
                    df["epoch"], 
                    df[metric], 
                    linestyle=c_style, 
                    color=c_color, 
                    marker=c_marker, 
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

    output_name = os.path.join(save_dir, f"{metric}_curve.png")
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved as: {output_name}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically compare multiple training runs.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "FOLDER_PATH"), required=True, help="Specify a run. Example: --run 'scratch' './results/run_folder' --run 'esm2' './results/run_folder'")
    parser.add_argument("--metric", type=str, default="eval_accuracy", choices=["eval_loss", "eval_accuracy"], help="Metric to plot (eval_loss or eval_accuracy)")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title for the plot")
    args = parser.parse_args()

    # Convert the list of parsed pairs into a dictionary
    # Example format: {'Label 1': 'path1', 'Label 2': 'path2'}
    runs_to_compare = {label: folder for label, folder in args.run}

    # Generate title if not provided
    default_title = f"{args.metric.replace('_', ' ').title()} Comparison"
    plot_title = args.title if args.title else default_title

    plot_linegraph(runs_to_compare, metric=args.metric, title=plot_title)