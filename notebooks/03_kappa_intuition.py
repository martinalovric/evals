#!/usr/bin/env python3
"""
Cohen's Kappa Intuition - Interactive Visualizations
=====================================================
Run with: uv run marimo edit notebooks/03_kappa_intuition.py

This notebook provides visual intuition for Cohen's Kappa
through interactive plots and examples.
"""

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    mo.md(
        """
        # Cohen's Kappa: Visual Intuition

        This notebook helps you develop intuition for Cohen's Kappa through
        interactive visualizations.

        **What you'll explore:**
        1. How Kappa changes with different prediction patterns
        2. Why Kappa beats accuracy for imbalanced data
        3. The relationship between fail recall and Kappa
        """
    )
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 1: Kappa vs Accuracy

    Let's visualize how Kappa and accuracy diverge with imbalanced data.
    """)
    return


@app.cell
def _(mo, np, plt):
    def _():
        def calculate_metrics(tp, tn, fp, fn):
            """Calculate accuracy and kappa from confusion matrix."""
            n = tp + tn + fp + fn
            if n == 0:
                return 0, 0

            # Accuracy
            accuracy = (tp + tn) / n

            # Kappa
            po = accuracy
            p_pred_pass = (tp + fp) / n
            p_true_pass = (tp + fn) / n
            p_pred_fail = (tn + fn) / n
            p_true_fail = (tn + fp) / n
            pe = (p_pred_pass * p_true_pass) + (p_pred_fail * p_true_fail)

            kappa = (po - pe) / (1 - pe) if pe != 1 else 0

            return accuracy, kappa

        # Simulate "always predict pass" with varying true failure rates
        failure_rates = np.linspace(0.01, 0.5, 50)
        n = 100

        accuracies = []
        kappas = []

        for fail_rate in failure_rates:
            n_fail = int(n * fail_rate)
            n_pass = n - n_fail

            # "Always predict pass" means:
            # TP = all passes, FP = all fails (incorrectly called pass)
            # TN = 0, FN = 0
            tp = n_pass
            fp = n_fail
            tn = 0
            fn = 0

            acc, kap = calculate_metrics(tp, tn, fp, fn)
            accuracies.append(acc)
            kappas.append(kap)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(failure_rates * 100, np.array(accuracies) * 100, 'b-', linewidth=2, label='Accuracy')
        ax.plot(failure_rates * 100, np.array(kappas) * 100, 'r-', linewidth=2, label='Kappa (×100)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=40, color='green', linestyle=':', alpha=0.7, label='Kappa target (0.4)')
        ax.set_xlabel('True Failure Rate (%)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Always Predict "Pass": Accuracy vs Kappa', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 50)
        ax.set_ylim(-10, 105)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        _md = mo.md(
            """
            ### "Always Predict Pass" Evaluator

            The plot shows what happens when an evaluator always predicts "pass":

            - **Blue line (Accuracy):** Decreases linearly as failure rate increases
            - **Red line (Kappa):** Always 0 regardless of failure rate!

            **Key insight:** A 90% accurate "always pass" evaluator has Kappa = 0,
            revealing it's no better than guessing.
            """
        )

        import io, base64
        _buf = io.BytesIO()
        fig.savefig(_buf, format="png", dpi=144, bbox_inches="tight")
        _buf.seek(0)
        _img_b64 = base64.b64encode(_buf.read()).decode("ascii")
        _buf.close()
        _img_component = mo.md(f'<img src="data:image/png;base64,{_img_b64}" style="max-width:100%; height:auto;" />')

        return mo.vstack([_md, _img_component])


    _()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 2: Interactive Confusion Matrix

    Adjust the sliders to see how different prediction patterns affect Kappa.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Interactive confusion matrix
    tp_viz = mo.ui.slider(0, 100, value=70, label="True Positives (TP)")
    tn_viz = mo.ui.slider(0, 100, value=20, label="True Negatives (TN)")
    fp_viz = mo.ui.slider(0, 100, value=5, label="False Positives (FP)")
    fn_viz = mo.ui.slider(0, 100, value=5, label="False Negatives (FN)")

    mo.vstack([
        mo.md("### Adjust the confusion matrix:"),
        mo.hstack([tp_viz, tn_viz]),
        mo.hstack([fp_viz, fn_viz]),
    ])
    return fn_viz, fp_viz, tn_viz, tp_viz


@app.cell(hide_code=True)
def _(calculate_metrics, fn_viz, fp_viz, np, plt, tn_viz, tp_viz):
    tp_v = tp_viz.value
    tn_v = tn_viz.value
    fp_v = fp_viz.value
    fn_v = fn_viz.value
    n_v = tp_v + tn_v + fp_v + fn_v

    if n_v == 0:
        fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "Add some samples using the sliders!", ha="center", va="center", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.gca()
    else:
        def _calculate_metrics(_tp, _tn, _fp, _fn):
            _n = _tp + _tn + _fp + _fn
            _acc = (_tp + _tn) / _n if _n > 0 else 0.0
            if _n == 0:
                return _acc, 0.0
            _pred_pos = _tp + _fp
            _pred_neg = _tn + _fn
            _actual_pos = _tp + _fn
            _actual_neg = _tn + _fp
            _pe = ((_pred_pos / _n) * (_actual_pos / _n) + (_pred_neg / _n) * (_actual_neg / _n)) if _n > 0 else 0.0
            _kappa = (_acc - _pe) / (1 - _pe) if (1 - _pe) != 0 else 0.0
            return _acc, _kappa

        try:
            acc_v, kappa_v = calculate_metrics(tp_v, tn_v, fp_v, fn_v)
        except NameError:
            acc_v, kappa_v = _calculate_metrics(tp_v, tn_v, fp_v, fn_v)

        # Calculate additional metrics
        fail_recall_v = tn_v / (tn_v + fn_v) if (tn_v + fn_v) > 0 else 0
        pass_recall_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0

        # Create visualization
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion matrix heatmap
        conf_matrix = np.array([[tp_v, fp_v], [fn_v, tn_v]])
        im = ax1.imshow(conf_matrix, cmap='Blues')

        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Pass', 'Fail'])
        ax1.set_yticklabels(['Pass', 'Fail'])
        ax1.set_xlabel('Actual', fontsize=12)
        ax1.set_ylabel('Predicted', fontsize=12)
        ax1.set_title('Confusion Matrix', fontsize=14)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, conf_matrix[i, j],
                              ha="center", va="center", color="black", fontsize=20)

        # Metrics bar chart
        metrics = ['Accuracy', 'Kappa', 'Fail Recall', 'Pass Recall']
        values = [acc_v, kappa_v, fail_recall_v, pass_recall_v]
        colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

        bars = ax2.bar(metrics, values, color=colors)
        ax2.axhline(y=0.4, color='green', linestyle='--', alpha=0.7, label='Kappa target')
        ax2.axhline(y=0.8, color='orange', linestyle=':', alpha=0.7, label='Fail recall target')
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Evaluation Metrics', fontsize=14)
        plt.gca()

    fig2

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 3: Failure Rate Impact

    How does the true failure rate in your dataset affect evaluation difficulty?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # Simulate a "good" evaluator at different failure rates
    failure_rates_sim = np.linspace(0.05, 0.5, 20)
    n_samples = 200

    # Evaluator characteristics
    pass_accuracy = 0.90  # 90% of passes correctly identified
    fail_accuracy = 0.75  # 75% of failures correctly identified

    results_sim = []

    for fail_rate in failure_rates_sim:
        n_fail = int(n_samples * fail_rate)
        n_pass = n_samples - n_fail

        tp = int(n_pass * pass_accuracy)
        fn = n_pass - tp
        tn = int(n_fail * fail_accuracy)
        fp = n_fail - tn

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total

        po = accuracy
        p_pred_pass = (tp + fp) / total
        p_true_pass = (tp + fn) / total
        pe = (p_pred_pass * p_true_pass) + ((1 - p_pred_pass) * (1 - p_true_pass))
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0

        fail_recall = tn / (tn + fn) if (tn + fn) > 0 else 0

        results_sim.append({
            'fail_rate': fail_rate,
            'accuracy': accuracy,
            'kappa': kappa,
            'fail_recall': fail_recall,
        })

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    fail_rates_plot = [r['fail_rate'] * 100 for r in results_sim]
    ax3.plot(fail_rates_plot, [r['accuracy'] * 100 for r in results_sim], 'b-o', linewidth=2, label='Accuracy', markersize=6)
    ax3.plot(fail_rates_plot, [r['kappa'] * 100 for r in results_sim], 'r-s', linewidth=2, label='Kappa (×100)', markersize=6)
    ax3.plot(fail_rates_plot, [r['fail_recall'] * 100 for r in results_sim], 'g-^', linewidth=2, label='Fail Recall', markersize=6)

    ax3.axhline(y=40, color='red', linestyle=':', alpha=0.7)
    ax3.axhline(y=80, color='green', linestyle=':', alpha=0.7)
    ax3.axvline(x=35.5, color='purple', linestyle='--', alpha=0.5, label='This project (35.5%)')

    ax3.set_xlabel('True Failure Rate (%)', fontsize=12)
    ax3.set_ylabel('Score (%)', fontsize=12)
    ax3.set_title('Metrics vs Dataset Failure Rate\n(90% pass accuracy, 75% fail accuracy)', fontsize=14)
    ax3.legend(loc='lower right')
    ax3.set_xlim(0, 55)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.vstack([
        fig3,
        mo.md(
            """
            ### Observations

            This simulates an evaluator with 90% pass accuracy and 75% fail accuracy:

            1. **Accuracy stays high** (80-95%) regardless of failure rate
            2. **Kappa increases** as failure rate increases (more signal)
            3. **Fail recall is stable** (determined by evaluator quality)

            **Eugene Yan's recommendation:** Target 35.5% failure rate (this project)
            for meaningful signal while keeping evaluation tractable.
            """
        ),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    1. **Kappa reveals the truth** that accuracy hides:
       - "Always pass" has high accuracy but Kappa = 0
       - Kappa penalizes majority-class guessing

    2. **Target ranges** (per Eugene Yan):
       - Kappa: 0.4-0.6 (substantial agreement)
       - Fail Recall: >80% (catching failures)
       - Dataset failure rate: ~35% (meaningful signal)

    3. **Trade-offs:**
       - Higher fail recall often means more false positives (acceptable)
       - Optimizing only for accuracy encourages majority-class guessing

    4. **Human baseline:**
       - Inter-rater reliability is often only Kappa 0.2-0.3
       - Getting Kappa 0.5 with LLM is actually impressive!

    ---

    **Congratulations!** You've completed the notebook series.

    Next steps:
    - Run `uv run verify_evaluator.py --sample 50` for real evaluation
    - Explore `examples/` for code patterns
    - Read `docs/tutorial/` for the full learning progression
    """)
    return


if __name__ == "__main__":
    app.run()
