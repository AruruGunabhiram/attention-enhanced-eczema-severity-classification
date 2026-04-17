"""
Results showcase for the attention-enhanced eczema severity classification project.
Generates a multi-panel figure: training curves, model comparison, and Grad-CAM outputs.
Run from the project root: python showcase_results.py
"""

import os
import re
import textwrap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

LOGS = "logs"
OUTPUTS = "outputs"
SAVE_PATH = "outputs/results_showcase.png"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_full_history(warmup_csv, finetune_csv):
    """Concatenate warmup + finetune epochs into one continuous DataFrame."""
    parts = []
    offset = 0
    for path in [warmup_csv, finetune_csv]:
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path)
            df["epoch"] = df["epoch"] + offset
            offset = df["epoch"].iloc[-1] + 1
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def parse_report(path):
    """Return (accuracy, per_class_f1_dict) from a saved report txt."""
    if not os.path.exists(path):
        return None, {}
    text = open(path).read()
    acc_match = re.search(r"(?:Final val(?:idation)? accuracy|accuracy)\s*[:\=]\s*([\d.]+)", text)
    acc = float(acc_match.group(1)) if acc_match else None
    f1 = {}
    for m in re.finditer(r"(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text):
        label = m.group(1)
        if label in ("accuracy", "macro", "weighted"):
            continue
        f1[label] = float(m.group(4))  # f1-score column
    return acc, f1


# ── load data ─────────────────────────────────────────────────────────────────

s2_mv2 = load_full_history(
    f"{LOGS}/stage2_mobilenetv2_warmup_history.csv",
    f"{LOGS}/stage2_mobilenetv2_finetune_history.csv",
)
s2_cbam = load_full_history(
    f"{LOGS}/stage2_cbam_warmup_history.csv",
    f"{LOGS}/stage2_cbam_finetune_history.csv",
)
s2_v2s = load_full_history(
    f"{LOGS}/stage2_cbam_v2s_warmup_history.csv",
    f"{LOGS}/stage2_cbam_v2s_finetune_history.csv",
)
s3_cbam = load_full_history(
    f"{LOGS}/stage3_cbam_warmup_history.csv",
    f"{LOGS}/stage3_cbam_finetune_history.csv",
)
s3_v2s = load_full_history(
    f"{LOGS}/stage3_cbam_v2s_warmup_history.csv",
    f"{LOGS}/stage3_cbam_v2s_finetune_history.csv",
)

_, s2_mv2_f1  = parse_report(f"{LOGS}/stage2_mobilenetv2_report.txt")
_, s2_cbam_f1 = parse_report(f"{LOGS}/stage2_cbam_report.txt")
_, s2_v2s_f1  = parse_report(f"{LOGS}/stage2_cbam_v2s_report.txt")
_, s3_cbam_f1 = parse_report(f"{LOGS}/stage3_cbam_report.txt")
_, s3_v2s_f1  = parse_report(f"{LOGS}/stage3_cbam_v2s_report.txt")

gradcam_imgs = sorted([
    os.path.join(OUTPUTS, f)
    for f in os.listdir(OUTPUTS)
    if f.lower().endswith(".png") and f.startswith("gradcam")
])

# ── layout ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 22), facecolor="#0f1117")
fig.suptitle(
    "Attention-Enhanced Eczema Severity Classification — Results Showcase",
    fontsize=18, fontweight="bold", color="white", y=0.98,
)

gs = gridspec.GridSpec(
    4, 3,
    figure=fig,
    hspace=0.45, wspace=0.35,
    top=0.94, bottom=0.04, left=0.07, right=0.97,
)

DARK   = "#0f1117"
PANEL  = "#1a1d27"
TEXT   = "#e0e0e0"
MUTED  = "#888888"
COLORS = {"MobileNetV2": "#4fc3f7", "CBAM-EffB0": "#81c784", "CBAM-V2S": "#ffb74d"}
S3COL  = {"CBAM-EffB0": "#81c784", "CBAM-V2S": "#ffb74d"}

def styled_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, facecolor="#222233", labelcolor=TEXT, fontsize=8, framealpha=0.8)
    return ax


# ── ROW 0: Stage 2 training curves ────────────────────────────────────────────

ax0 = fig.add_subplot(gs[0, 0])
styled_ax(ax0, "Stage 2 — Val Accuracy (all models)")
for df, label, c in [
    (s2_mv2, "MobileNetV2", COLORS["MobileNetV2"]),
    (s2_cbam, "CBAM-EffB0", COLORS["CBAM-EffB0"]),
    (s2_v2s, "CBAM-V2S", COLORS["CBAM-V2S"]),
]:
    if not df.empty and "val_accuracy" in df.columns:
        ax0.plot(df["epoch"], df["val_accuracy"], color=c, lw=2, label=label)
ax0.set_xlabel("Epoch"); ax0.set_ylabel("Accuracy")
ax0.set_ylim(0.5, 1.0)

ax1 = fig.add_subplot(gs[0, 1])
styled_ax(ax1, "Stage 2 — Train vs Val Loss (CBAM-V2S)")
if not s2_v2s.empty:
    ax1.plot(s2_v2s["epoch"], s2_v2s["loss"],     color="#ef5350", lw=2, label="Train loss")
    ax1.plot(s2_v2s["epoch"], s2_v2s["val_loss"], color="#ffb74d", lw=2, linestyle="--", label="Val loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

ax2 = fig.add_subplot(gs[0, 2])
styled_ax(ax2, "Stage 2 — Val AUC (models with AUC)")
for df, label, c in [
    (s2_mv2, "MobileNetV2", COLORS["MobileNetV2"]),
    (s2_cbam, "CBAM-EffB0", COLORS["CBAM-EffB0"]),
    (s2_v2s, "CBAM-V2S", COLORS["CBAM-V2S"]),
]:
    if not df.empty and "val_auc" in df.columns:
        ax2.plot(df["epoch"], df["val_auc"], color=c, lw=2, label=label)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
ax2.set_ylim(0.5, 1.0)


# ── ROW 1: Stage 3 training curves ────────────────────────────────────────────

ax3 = fig.add_subplot(gs[1, 0])
styled_ax(ax3, "Stage 3 — Val Accuracy")
for df, label, c in [
    (s3_cbam, "CBAM-EffB0", S3COL["CBAM-EffB0"]),
    (s3_v2s,  "CBAM-V2S",  S3COL["CBAM-V2S"]),
]:
    if not df.empty and "val_accuracy" in df.columns:
        ax3.plot(df["epoch"], df["val_accuracy"], color=c, lw=2, label=label)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy")

ax4 = fig.add_subplot(gs[1, 1])
styled_ax(ax4, "Stage 3 — Train vs Val Loss (CBAM-V2S)")
if not s3_v2s.empty:
    ax4.plot(s3_v2s["epoch"], s3_v2s["loss"],     color="#ef5350", lw=2, label="Train loss")
    ax4.plot(s3_v2s["epoch"], s3_v2s["val_loss"], color="#ffb74d", lw=2, linestyle="--", label="Val loss")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")


# ── ROW 1, col 2: Model comparison bar chart ──────────────────────────────────

ax5 = fig.add_subplot(gs[1, 2])
styled_ax(ax5, "Model Comparison — Val Accuracy")

models  = ["MobileNetV2\n(S2)", "CBAM-EffB0\n(S2)", "CBAM-V2S\n(S2)", "CBAM-EffB0\n(S3)", "CBAM-V2S\n(S3)"]
accs    = [0.8515, 0.8445, 0.8701, 0.7742, 0.7730]
palette = [COLORS["MobileNetV2"], COLORS["CBAM-EffB0"], COLORS["CBAM-V2S"],
           S3COL["CBAM-EffB0"], S3COL["CBAM-V2S"]]
bars = ax5.bar(models, accs, color=palette, edgecolor="#333344", linewidth=0.8)
for bar, val in zip(bars, accs):
    ax5.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
        f"{val:.1%}", ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold",
    )
ax5.set_ylim(0.6, 0.95)
ax5.set_ylabel("Val Accuracy")
ax5.tick_params(axis="x", labelsize=8)
if ax5.get_legend():
    ax5.get_legend().remove()


# ── ROW 2: Per-class F1 grouped bar charts ────────────────────────────────────

def f1_bar(ax, title, reports_labels):
    """reports_labels: list of (f1_dict, label, color)"""
    all_classes = []
    for f1, _, _ in reports_labels:
        all_classes += list(f1.keys())
    classes = list(dict.fromkeys(all_classes))
    n = len(reports_labels)
    x = np.arange(len(classes))
    width = 0.8 / n
    for i, (f1, label, c) in enumerate(reports_labels):
        vals = [f1.get(cls, 0.0) for cls in classes]
        ax.bar(x + i * width - (n - 1) * width / 2, vals, width, label=label, color=c, edgecolor="#333344")
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
    styled_ax(ax, title)

ax6 = fig.add_subplot(gs[2, 0])
f1_bar(ax6, "Stage 2 — Per-class F1", [
    (s2_mv2_f1,  "MobileNetV2", COLORS["MobileNetV2"]),
    (s2_cbam_f1, "CBAM-EffB0",  COLORS["CBAM-EffB0"]),
    (s2_v2s_f1,  "CBAM-V2S",   COLORS["CBAM-V2S"]),
])

ax7 = fig.add_subplot(gs[2, 1])
f1_bar(ax7, "Stage 3 — Per-class F1 (severity)", [
    (s3_cbam_f1, "CBAM-EffB0", S3COL["CBAM-EffB0"]),
    (s3_v2s_f1,  "CBAM-V2S",  S3COL["CBAM-V2S"]),
])


# ── ROW 2, col 2 + ROW 3: Grad-CAM images ────────────────────────────────────

gradcam_axes = [
    fig.add_subplot(gs[2, 2]),
    fig.add_subplot(gs[3, 0]),
    fig.add_subplot(gs[3, 1]),
]

for i, ax in enumerate(gradcam_axes):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    if i < len(gradcam_imgs):
        img = mpimg.imread(gradcam_imgs[i])
        ax.imshow(img)
        name = os.path.basename(gradcam_imgs[i]).replace("gradcam_", "").replace(".png", "")
        ax.set_title(f"Grad-CAM: {name}", color=TEXT, fontsize=10, fontweight="bold", pad=6)
    else:
        ax.text(0.5, 0.5, "No Grad-CAM\noutput", ha="center", va="center",
                color=MUTED, fontsize=11, transform=ax.transAxes)
        ax.set_title("Grad-CAM", color=TEXT, fontsize=10, fontweight="bold", pad=6)
    ax.axis("off")


# ── ROW 3, col 2: Summary text panel ─────────────────────────────────────────

ax_txt = fig.add_subplot(gs[3, 2])
ax_txt.set_facecolor(PANEL)
for spine in ax_txt.spines.values():
    spine.set_edgecolor("#333344")
ax_txt.axis("off")
ax_txt.set_title("Key Results", color=TEXT, fontsize=11, fontweight="bold", pad=8)

summary = textwrap.dedent("""\
    Stage 2 — Eczema Detection
    ──────────────────────────
    Best model : CBAM + EfficientNetV2S
    Val Accuracy : 87.01%
    Macro F1     : 0.870
    Threshold    : 0.480 (optimized)

    Stage 3 — Severity Grading
    ──────────────────────────
    Best model : CBAM + EfficientNetV2S
    Val Accuracy : 77.30%
    Macro F1     : 0.663
    Note: mild class underrepresented

    Architecture
    ────────────
    Backbone  : EfficientNetV2S (ImageNet)
    Attention : CBAM (channel + spatial)
    Explain.  : Grad-CAM (last Conv2D)
""")

ax_txt.text(
    0.05, 0.97, summary,
    transform=ax_txt.transAxes,
    va="top", ha="left",
    color=TEXT, fontsize=8.5,
    fontfamily="monospace",
    linespacing=1.6,
)

# ── save ──────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUTS, exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {SAVE_PATH}")
plt.show()
