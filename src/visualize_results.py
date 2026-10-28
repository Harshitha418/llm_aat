from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

overall_path = RESULTS_DIR / "overall_metrics.csv"
per_label_path = RESULTS_DIR / "per_label_metrics.csv"

if not overall_path.exists():
    raise FileNotFoundError(f"Missing file: {overall_path}")

if not per_label_path.exists():
    raise FileNotFoundError(f"Missing file: {per_label_path}")

overall = pd.read_csv(overall_path)
per_label = pd.read_csv(per_label_path)

print("Overall metrics columns:", overall.columns.tolist())
print("Per-label metrics columns:", per_label.columns.tolist())

# Plot 1: Overall precision, recall and F1.
metric_columns = ["precision", "recall", "f1_score"]

if {"model", *metric_columns}.issubset(overall.columns):
    metrics = overall.set_index("model")[metric_columns]

    ax = metrics.plot(kind="bar", figsize=(9, 5))
    ax.set_title("NER Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "overall_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved overall_performance.png")
else:
    print(
        "Overall plot skipped: expected columns "
        "'model', 'precision', 'recall', 'f1_score'."
    )

# Plot 2: Average inference time.
time_column = "average_inference_ms"

if {"model", time_column}.issubset(overall.columns):
    ax = overall.plot(
        x="model",
        y=time_column,
        kind="bar",
        legend=False,
        figsize=(8, 5),
    )
    ax.set_title("Average Inference Time per Sentence")
    ax.set_xlabel("Model")
    ax.set_ylabel("Time (milliseconds)")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "inference_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved inference_time.png")
else:
    print(
        f"Time plot skipped: expected columns 'model' and "
        f"'{time_column}'. Check the printed column names."
    )

# Plot 3: Per-label F1 comparison.
label_column = "label"
f1_column = "f1_score"

if {"model", label_column, f1_column}.issubset(per_label.columns):
    pivot = per_label.pivot(
        index=label_column,
        columns="model",
        values=f1_column,
    )

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("F1 Score by Entity Class")
    ax.set_xlabel("Entity Class")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "per_label_f1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved per_label_f1.png")
else:
    print(
        "Per-label plot skipped: expected columns "
        "'model', 'label', 'f1_score'. Check the printed column names."
    )

print("\nVisualization script finished.")
print(f"Output directory: {RESULTS_DIR}")