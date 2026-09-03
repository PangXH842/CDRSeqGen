"""What a finished run leaves behind for a human: the per-epoch CSV and the curve PNGs.

  * save_training_csv    -- Trainer log history -> epoch_metrics.csv (the per-epoch trajectory)
  * plot_training_curves -- epoch_metrics.csv -> line graph PNGs (training loss and AAR breakdown
                            whole, core, anchor)

Notes:
  * HF log writes extra line for last epoch with best checkpoint (with load_best_model_at_end enabled),
    therefore a deduplication step is needed
"""
import os
import logging

import pandas as pd

from cdrseqgen.visualization.linegraph_plotter import plot_epoch_curves
from cdrseqgen import artifacts

logger = logging.getLogger(__name__)


def save_training_csv(trainer, output_dir):
    """Extracts eval metrics, and saves them to a CSV."""
    history = trainer.state.log_history

    # Filter only the logs that contain evaluation metrics
    eval_logs = [log for log in history if 'eval_loss' in log]

    if not eval_logs:
        logger.warning("No evaluation logs found to plot.")
        return

    # Deduplicate Epochs to prevent post-training evaluation artifacts
    seen_epochs = set()
    unique_eval_logs = []
    for log in eval_logs:
        rounded_epoch = round(log['epoch'], 4)
        if rounded_epoch not in seen_epochs:
            seen_epochs.add(rounded_epoch)
            unique_eval_logs.append(log)

    epochs = [log['epoch'] for log in unique_eval_logs]
    eval_loss = [log['eval_loss'] for log in unique_eval_logs]
    eval_acc = [log.get('eval_accuracy', 0.0) for log in unique_eval_logs]
    eval_core_acc = [log.get('eval_core_accuracy', 0.0) for log in unique_eval_logs]
    eval_anchor_acc = [log.get('eval_anchor_accuracy', 0.0) for log in unique_eval_logs]
    eval_core_cov = [log.get('eval_core_coverage', float('nan')) for log in unique_eval_logs]

    # Save data to CSV
    df = pd.DataFrame({
        "epoch": epochs,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_core_accuracy": eval_core_acc,
        "eval_anchor_accuracy": eval_anchor_acc,
        "eval_core_coverage": eval_core_cov,
    })
    csv_path = os.path.join(output_dir, artifacts.EPOCH_METRICS_FILENAME)
    df.to_csv(csv_path, index=False)

    logger.info(f"📈 Evaluation metrics saved to CSV at: {csv_path}")


def plot_training_curves(output_dir):
    """Generate multi-line graph comparing AAR metrics (whole, core, anchor)"""
    # One chart per metric. The combined breakdown below spans ~0.3-0.9, which compresses
    # within-line movement, so the individual curves stay the readable view of a single trend.
    for metric, title in [
        ("eval_loss", "Evaluation Loss"),
        ("eval_accuracy", "Sequence Recovery — whole-span (anchor-inflated)"),
        ("eval_core_accuracy", "Sequence Recovery — CORE (anchor-excluded)"),
    ]:
        plot_epoch_curves(
            [(output_dir, metric, "Current Run")],
            title=title,
            ylabel=metric.replace("_", " ").title(),
            save_dir=output_dir,
            filename=f"{metric}_curve",
        )

    # Combined AAR view: whole, core, anchor
    plot_epoch_curves(
        [
            (output_dir, "eval_accuracy", "whole-span (anchor-inflated)"),
            (output_dir, "eval_core_accuracy", "core (anchor-excluded)"),
            (output_dir, "eval_anchor_accuracy", "anchor only"),
        ],
        title="Recovery: whole-span vs core vs anchor",
        ylabel="Recovery",
        save_dir=output_dir,
        filename="recovery_breakdown",
    )

    logger.info(f"📊 Training curves successfully saved to {output_dir}")
