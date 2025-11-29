"""
Plot Evaluation Run Comparisons

Visualizes differences between model evaluation runs from the JSONL log.

Usage:
    uv run plot_runs.py                       # Plot all runs (saves to docs/plots/)
    uv run plot_runs.py --input custom.jsonl  # Custom input file
    uv run plot_runs.py --output plots/       # Save to custom directory
    uv run plot_runs.py --last N              # Only plot last N runs
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_runs(filepath: str) -> list[dict]:
    """Load runs from JSONL file."""
    runs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def plot_model_comparison(runs: list[dict], output_dir: Path = None):
    """Bar chart comparing key metrics across models."""
    if not runs:
        print("No runs to plot")
        return

    # Group runs by model (use latest run per model)
    model_runs = {}
    for run in runs:
        model = run["model"]
        model_runs[model] = run  # Latest overwrites

    models = list(model_runs.keys())
    metrics = ["accuracy", "kappa", "recall_fail", "precision_fail", "f1_fail"]
    metric_labels = ["Accuracy", "Cohen's Kappa", "Fail Recall", "Fail Precision", "Fail F1"]

    # Extract values
    data = {m: [] for m in metrics}
    for model in models:
        run = model_runs[model]
        for metric in metrics:
            data[metric].append(run["metrics"].get(metric, 0))

    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=label)
        # Add value labels on bars
        for bar, val in zip(bars, data[metric]):
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Key Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-2025-08-07', '') for m in models], rotation=15, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Kappa target (0.6)')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Kappa min (0.4)')

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'model_comparison.png'}")
    else:
        plt.show()

    plt.close()


def plot_category_comparison(runs: list[dict], output_dir: Path = None):
    """Heatmap of per-category accuracy across models."""
    if not runs:
        return

    # Group runs by model
    model_runs = {}
    for run in runs:
        model = run["model"]
        model_runs[model] = run

    models = list(model_runs.keys())

    # Collect all categories
    all_categories = set()
    for run in model_runs.values():
        all_categories.update(run.get("per_category", {}).keys())
    categories = sorted(all_categories)

    if not categories:
        print("No per-category data to plot")
        return

    # Build accuracy matrix
    accuracy_matrix = []
    for model in models:
        run = model_runs[model]
        row = []
        for cat in categories:
            cat_data = run.get("per_category", {}).get(cat, {})
            row.append(cat_data.get("accuracy", 0))
        accuracy_matrix.append(row)

    accuracy_matrix = np.array(accuracy_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(models) * 0.8)))

    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticklabels([m.replace('-2025-08-07', '') for m in models])

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(categories)):
            val = accuracy_matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    ax.set_title('Per-Category Accuracy by Model')
    plt.colorbar(im, ax=ax, label='Accuracy')

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "category_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'category_comparison.png'}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrices(runs: list[dict], output_dir: Path = None):
    """Side-by-side confusion matrices for each model."""
    if not runs:
        return

    # Group runs by model
    model_runs = {}
    for run in runs:
        model = run["model"]
        model_runs[model] = run

    models = list(model_runs.keys())
    n_models = len(models)

    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        run = model_runs[model]
        cm = run.get("confusion_matrix", {})
        matrix = np.array([
            [cm.get("tp", 0), cm.get("fn", 0)],
            [cm.get("fp", 0), cm.get("tn", 0)]
        ])

        im = ax.imshow(matrix, cmap='Blues')

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pass', 'Fail'])
        ax.set_yticklabels(['Pass', 'Fail'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(model.replace('-2025-08-07', ''))

        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = 'white' if val > matrix.max() / 2 else 'black'
                ax.text(j, i, str(int(val)), ha='center', va='center', color=color, fontsize=14)

    plt.suptitle('Confusion Matrices by Model', y=1.02)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'confusion_matrices.png'}")
    else:
        plt.show()

    plt.close()


def plot_timeline(runs: list[dict], output_dir: Path = None):
    """Line plot of metrics over time (by run order)."""
    if len(runs) < 2:
        print("Need at least 2 runs for timeline plot")
        return

    metrics = ["accuracy", "kappa", "recall_fail"]
    colors = ['blue', 'green', 'red']

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(runs))
    for metric, color in zip(metrics, colors):
        values = [run["metrics"].get(metric, 0) for run in runs]
        labels = [run["model"].replace('-2025-08-07', '')[:15] for run in runs]
        ax.plot(x, values, marker='o', label=metric.replace('_', ' ').title(), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_xlabel('Run')
    ax.set_title('Metrics Across Runs')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "timeline.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'timeline.png'}")
    else:
        plt.show()

    plt.close()


def print_summary(runs: list[dict]):
    """Print text summary of runs."""
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)

    for i, run in enumerate(runs, 1):
        m = run["metrics"]
        print(f"\n[{i}] {run['model']}")
        print(f"    Timestamp: {run['timestamp']}")
        print(f"    Samples: {run['sample_size']}")
        print(f"    Accuracy: {m['accuracy']*100:.1f}%  |  Kappa: {m['kappa']:.3f}  |  Fail Recall: {m['recall_fail']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation run comparisons")
    parser.add_argument("--input", type=str, default="results/runs.jsonl",
                        help="Input JSONL file (default: results/runs.jsonl)")
    parser.add_argument("--output", type=str, default="docs/plots",
                        help="Output directory for plots (default: docs/plots)")
    parser.add_argument("--last", type=int, default=None,
                        help="Only use last N runs")
    args = parser.parse_args()

    # Load data
    try:
        runs = load_runs(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        print("Run some evaluations first with verify_evaluator.py")
        return

    if not runs:
        print("No runs found in file")
        return

    if args.last:
        runs = runs[-args.last:]

    print(f"Loaded {len(runs)} runs from {args.input}")

    output_dir = Path(args.output) if args.output else None

    # Generate plots
    print_summary(runs)
    plot_model_comparison(runs, output_dir)
    plot_category_comparison(runs, output_dir)
    plot_confusion_matrices(runs, output_dir)

    if len(runs) >= 2:
        plot_timeline(runs, output_dir)

    if output_dir:
        print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
