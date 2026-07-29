"""What a finished run leaves behind for a human: the per-epoch CSV and the curve PNGs.

  * save_training_csv    -- Trainer log history -> epoch_metrics.csv (the per-epoch trajectory)
  * plot_training_curves -- that CSV -> four PNGs (loss, whole-span, core, and the breakdown)

Notes:
  * epoch_metrics.csv is the TRAJECTORY — the model as it stood at each epoch. It is deliberately
    NOT the same thing as eval_results.json, which HF writes from the post-training evaluate() and
    which (with load_best_model_at_end) describes the BEST checkpoint while carrying the final epoch
    number. They are supposed to disagree; see README "How to read the numbers".
  * De-duplication keeps the FIRST row per epoch for exactly that reason: keeping the last would
    overwrite the real last-epoch point with the best model's numbers and erase any overfitting decay
    from the curve.
"""
import os
import logging

import pandas as pd

from cdrseqgen.utils.linegraph_plotter import plot_epoch_curves

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
    # Fraction of eval rows backing eval_core_accuracy. Recorded so a core score can always be
    # sanity-checked after the fact: well below 1.0 means the loop is too short for the anchor trim
    # and core is noise (see eval/recovery.py). Older runs lack this column.
    eval_core_cov = [log.get('eval_core_coverage', float('nan')) for log in unique_eval_logs]

    # Save data to a CSV for easy multi-line plotting later
    df = pd.DataFrame({
        "epoch": epochs,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_core_accuracy": eval_core_acc,
        "eval_anchor_accuracy": eval_anchor_acc,
        "eval_core_coverage": eval_core_cov,
    })
    csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"📈 Evaluation metrics saved to CSV at: {csv_path}")


def plot_training_curves(output_dir):
    """Auto-generate this run's curves from epoch_metrics.csv.

    All three recovery metrics are recorded per epoch, but only loss + whole-span were ever
    plotted — so every run's auto-plot showed the anchor-inflated metric the README says NOT to
    report, and hid the anchor-excluded CORE metric it says to use. Core is now plotted too, plus a
    combined whole-vs-core-vs-anchor chart that makes the anchor gap visible at a glance (on H3 the
    spread is ~0.46 / ~0.32 / ~0.90).
    """
    # One PNG per metric: <metric>_curve.png — this run's single metric (one line each).
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

    # Combined view: all three recovery metrics on one chart, so the anchor gap is one glance.
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
