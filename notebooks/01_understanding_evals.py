#!/usr/bin/env python3
"""
Understanding LLM Evaluations - Interactive Notebook
=====================================================
Run with: uv run marimo edit notebooks/01_understanding_evals.py

This notebook teaches the fundamentals of LLM evaluation through
interactive examples. No API calls needed for the core concepts.
"""

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # Understanding LLM Evaluations

        This interactive notebook teaches you the core concepts of evaluating LLM outputs.

        **What you'll learn:**
        1. Why accuracy is misleading
        2. How Cohen's Kappa works
        3. Confusion matrices explained
        4. Why fail recall matters

        No API calls needed - pure Python for understanding the math.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Problem with Accuracy

    Imagine you have a dataset where **90% of responses are correct** (pass).

    An evaluator that **always predicts "pass"** gets 90% accuracy!

    But is it useful? Let's see...
    """)
    return


@app.cell
def _():
    def _():
        # Simulate an imbalanced dataset
        ground_truth = ["pass"] * 90 + ["fail"] * 10  # 90% pass, 10% fail

        # "Always pass" evaluator
        always_pass_predictions = ["pass"] * 100

        # Calculate accuracy
        correct = sum(p == g for p, g in zip(always_pass_predictions, ground_truth))
        accuracy = correct / len(ground_truth)

        print(f"Ground truth: {ground_truth.count('pass')} pass, {ground_truth.count('fail')} fail")
        print(f"Predictions:  Always 'pass'")
        print(f"Accuracy:     {accuracy:.1%}")
        print()
        print("90% accuracy sounds great, but this evaluator is useless!")
        return print("It never identifies failures - the whole point of evaluation!")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cohen's Kappa: The Solution

    Cohen's Kappa measures agreement **beyond what we'd expect by chance**.

    **Formula:**
    ```
    κ = (observed agreement - expected agreement) / (1 - expected agreement)
    ```

    - **Kappa = 0**: No better than random guessing
    - **Kappa = 1**: Perfect agreement
    - **Kappa < 0**: Worse than random!

    Let's calculate it step by step...
    """)
    return


@app.function
def calculate_kappa_detailed(predictions: list, ground_truth: list) -> dict:
    """Calculate Cohen's Kappa with detailed breakdown."""
    n = len(predictions)

    # Confusion matrix
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == "pass" and g == "pass")
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == "fail" and g == "fail")
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "pass" and g == "fail")
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == "fail" and g == "pass")

    # Observed agreement
    po = (tp + tn) / n

    # Marginal probabilities
    p_pred_pass = (tp + fp) / n
    p_true_pass = (tp + fn) / n
    p_pred_fail = (tn + fn) / n
    p_true_fail = (tn + fp) / n

    # Expected agreement by chance
    pe = (p_pred_pass * p_true_pass) + (p_pred_fail * p_true_fail)

    # Kappa
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0

    return {
        "accuracy": po,
        "observed_agreement": po,
        "expected_agreement": pe,
        "kappa": kappa,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


@app.cell
def _():
    # Same imbalanced scenario
    ground_truth_imb = ["pass"] * 90 + ["fail"] * 10
    always_pass = ["pass"] * 100

    result = calculate_kappa_detailed(always_pass, ground_truth_imb)

    print("Scenario: Always predict 'pass' with 90% pass rate")
    print("=" * 50)
    print(f"Observed agreement (po): {result['observed_agreement']:.3f}")
    print(f"Expected by chance (pe): {result['expected_agreement']:.3f}")
    print(f"Cohen's Kappa:          {result['kappa']:.3f}")
    print()
    print("Interpretation:")
    if result["kappa"] == 0:
        print("  Kappa = 0 means NO BETTER THAN CHANCE!")
    print("  The evaluator is useless despite 90% accuracy.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Interactive: Try Different Scenarios

    Use the sliders below to explore how Kappa changes with different prediction patterns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Create sliders for interactive exploration
    tp_slider = mo.ui.slider(0, 50, value=40, label="True Positives (correctly identified passes)")
    tn_slider = mo.ui.slider(0, 50, value=8, label="True Negatives (correctly identified failures)")
    fp_slider = mo.ui.slider(0, 50, value=2, label="False Positives (false alarms)")
    fn_slider = mo.ui.slider(0, 50, value=0, label="False Negatives (missed failures)")

    mo.vstack([tp_slider, tn_slider, fp_slider, fn_slider])
    return fn_slider, fp_slider, tn_slider, tp_slider


@app.cell
def _(fn_slider, fp_slider, mo, tn_slider, tp_slider):
    try: 
        # show live values (helps confirm reactivity) 
        debug_el = mo.md(f"Debug: tp={tp_slider.value}, tn={tn_slider.value}, fp={fp_slider.value}, fn={fn_slider.value}")
    
        tp_v = tp_slider.value 
        tn_v = tn_slider.value 
        fp_v = fp_slider.value 
        fn_v = fn_slider.value 
        n_v = tp_v + tn_v + fp_v + fn_v
    
        if n_v == 0:
            mo.md("Add some samples using the sliders above!")
        else:
            acc_v = (tp_v + tn_v) / n_v
            p_pred_pass_v = (tp_v + fp_v) / n_v
            p_true_pass_v = (tp_v + fn_v) / n_v
            p_pred_fail_v = (tn_v + fn_v) / n_v
            p_true_fail_v = (tn_v + fp_v) / n_v
            pe_v = (p_pred_pass_v * p_true_pass_v) + (p_pred_fail_v * p_true_fail_v)
            kappa_v = (acc_v - pe_v) / (1 - pe_v) if pe_v != 1 else 0
            fail_recall_v = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0
    
            mo.md(
                f"""
                ## Results
    
                TP={tp_v}, TN={tn_v}, FP={fp_v}, FN={fn_v}, N={n_v}
    
                Accuracy: {acc_v:.1%}  
                Cohen's Kappa: {kappa_v:.3f}  
                Fail recall: {fail_recall_v:.1%}
                """
            )
    except Exception as e: 
        mo.md(f"Error: {type(e).name}: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    1. **Accuracy is misleading** with imbalanced data (most real datasets are imbalanced)

    2. **Cohen's Kappa** accounts for chance agreement:
       - Kappa = 0: No better than guessing
       - Kappa 0.4-0.6: Target range (substantial agreement)
       - Kappa > 0.6: Excellent

    3. **Fail Recall matters more** than overall accuracy:
       - Missing defects (false negatives) is costly
       - False alarms (false positives) are cheap to review

    4. **Human baseline**: Inter-rater reliability is often only Kappa 0.2-0.3

    ---

    **Next:** Open `notebooks/02_position_bias.py` to learn about position bias
    """)
    return


if __name__ == "__main__":
    app.run()
