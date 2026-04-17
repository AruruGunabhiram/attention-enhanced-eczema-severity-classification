import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar

def evaluate_model(model, test_ds, class_names: list) -> dict:
    y_true, y_pred_raw = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        if preds.shape[-1] == 1:
            y_pred_raw.extend((preds > 0.5).astype(int).flatten())
        else:
            y_pred_raw.extend(np.argmax(preds, axis=1))

    f1_macro    = f1_score(y_true, y_pred_raw, average='macro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_raw, average='weighted',  zero_division=0)
    f1_per_class = f1_score(y_true, y_pred_raw, average=None,       zero_division=0)

    # Per-class support counts
    from sklearn.metrics import classification_report as _cr
    report_dict = _cr(y_true, y_pred_raw, target_names=class_names,
                      zero_division=0, output_dict=True)
    support = {name: int(report_dict[name]['support']) for name in class_names}

    print(classification_report(y_true, y_pred_raw, target_names=class_names, zero_division=0))
    print(f"Macro F1:              {f1_macro:.4f}")
    print(f"Weighted F1:           {f1_weighted:.4f}")
    print("Per-class F1:")
    for name, score in zip(class_names, f1_per_class):
        print(f"  {name:<10} {score:.4f}  (support: {support[name]})")

    return {
        'y_true':       y_true,
        'y_pred':       y_pred_raw,
        'f1_macro':     f1_macro,      # primary metric — unweighted, exposes minority class failures
        'f1_weighted':  f1_weighted,   # kept for reference only
        'f1_per_class': {name: float(score) for name, score in zip(class_names, f1_per_class)},
    }

def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    output_path: str = '/content/drive/MyDrive/confusion_matrix_stage3.png',
    class_names: list = None,
):
    if class_names is None:
        class_names = ['Mild', 'Moderate', 'Severe']

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100  # row-normalised %

    # Cell annotations: "42\n(67.7%)"
    annot = np.array([
        [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    accuracy = np.trace(cm) / cm.sum()
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class recall (diagonal of row-normalised matrix)
    per_class_recall = np.diag(cm_pct)
    worst_idx = int(np.argmin(per_class_recall))
    worst_name = class_names[worst_idx]
    worst_recall = per_class_recall[worst_idx]
    best_recall = np.max(np.delete(per_class_recall, worst_idx))

    print(f"Worst recall: {worst_name} at {worst_recall:.1f}%  "
          f"({best_recall - worst_recall:.1f} pp below the next best class)")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt='',                  # annotations are pre-formatted strings
        cmap='RdYlGn',           # red (low) → yellow → green (high) highlights off-diagonal errors
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor='white',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={'size': 11},
    )

    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=8)
    ax.set_ylabel('True Label', fontsize=12, labelpad=8)
    ax.set_title(
        f'Confusion Matrix — Stage 3 Severity\n'
        f'Accuracy: {accuracy:.1%}   Macro F1: {f1_macro:.4f}',
        fontsize=13, pad=14,
    )
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {output_path}")


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    """Thin wrapper kept for backwards compatibility."""
    plot_and_save_confusion_matrix(y_true, y_pred, output_path=save_path, class_names=class_names)

def run_mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, y_true: np.ndarray,
                     model_a_name: str = "Model A (MobileNetV2)",
                     model_b_name: str = "Model B (EfficientNetB0+CBAM)") -> dict:
    # McNemar's test is the correct statistical test here — NOT a t-test.
    # A t-test assumes independent samples. These predictions are *paired*:
    # both models saw exactly the same test images in the same order, so their
    # errors are correlated. McNemar's test is designed for exactly this case:
    # it asks "do the two models disagree on different subsets of examples?"
    # and only uses the off-diagonal cells (b, c) where the models *disagree*.
    # The cells where both are right or both are wrong carry no information
    # about which model is better.
    y_true  = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # 2×2 contingency table:
    #                  B correct    B wrong
    #   A correct   [ n_both_right, n_a_only ]
    #   A wrong     [ n_b_only,     n_both_wrong ]
    n_both_right = int(np.sum( correct_a &  correct_b))
    n_a_only     = int(np.sum( correct_a & ~correct_b))
    n_b_only     = int(np.sum(~correct_a &  correct_b))
    n_both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = np.array([[n_both_right, n_a_only],
                      [n_b_only,     n_both_wrong]])

    print("Contingency table:")
    print(f"  {'':20s}  B correct  B wrong")
    print(f"  {'A correct':20s}  {n_both_right:>9}  {n_a_only:>7}")
    print(f"  {'A wrong':20s}  {n_b_only:>9}  {n_both_wrong:>7}")
    print(f"\n  Discordant pairs: b={n_a_only} (A only), c={n_b_only} (B only)")

    # exact=False uses chi-square approximation (valid when b+c >= 25).
    # exact=True uses binomial and is safer for small test sets like ours (~310 images).
    use_exact = (n_a_only + n_b_only) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    p      = result.pvalue
    stat   = result.statistic
    sig    = p < 0.05
    method = "binomial (exact)" if use_exact else "chi-square"

    print(f"\nMcNemar's test ({method}):")
    if sig:
        print(f"  {model_b_name} IS statistically significantly better than {model_a_name}")
    else:
        print(f"  {model_b_name} is NOT statistically significantly better than {model_a_name}")
    print(f"  Statistic = {stat:.4f},  p = {p:.4f}  (threshold: p < 0.05)")

    # Paper framing for a non-significant result:
    # "Although EfficientNetB0+CBAM achieved higher accuracy (64.5% vs 61.0%),
    #  McNemar's test did not reach significance (χ²=X.XX, p=0.XXX), indicating
    #  the improvement is directional but not conclusive at the current test set
    #  size (~310 images). A larger evaluation cohort would be required to confirm
    #  statistical significance."
    # Do NOT write "the models perform equally well" — that conflates failure to
    # reject H0 with evidence of equivalence. They are not the same claim.
    if not sig:
        print("\n  [Paper note] Non-significant result — report as a directional trend,")
        print("  not as equivalence. Mention test set size as the limiting factor.")

    return {
        'n_both_right': n_both_right,
        'n_a_only':     n_a_only,
        'n_b_only':     n_b_only,
        'n_both_wrong': n_both_wrong,
        'statistic':    float(stat),
        'p_value':      float(p),
        'significant':  sig,
        'method':       method,
    }


def save_metrics_table(metrics_dict: dict, save_path: str):
    """Append a row of results to results/tables/metrics.csv.

    Flattens per-class F1 dict into columns (f1_mild, f1_moderate, f1_severe)
    and drops raw y_true/y_pred arrays — those belong in a confusion matrix,
    not a summary table.
    """
    row = {k: v for k, v in metrics_dict.items() if k not in ('y_true', 'y_pred', 'f1_per_class')}
    if 'f1_per_class' in metrics_dict:
        for name, score in metrics_dict['f1_per_class'].items():
            row[f'f1_{name}'] = score
    # f1_macro is the primary column; f1_weighted is secondary
    col_order = ['f1_macro', 'f1_weighted'] + [f'f1_{n}' for n in ('mild', 'moderate', 'severe')]
    col_order += [c for c in row if c not in col_order]
    df = pd.DataFrame([{c: row.get(c) for c in col_order}])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, index=False)
