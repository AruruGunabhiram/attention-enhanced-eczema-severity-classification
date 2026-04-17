import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def evaluate_model(model, test_ds, class_names: list) -> dict:
    y_true, y_pred_raw = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        if preds.shape[-1] == 1:
            y_pred_raw.extend((preds > 0.5).astype(int).flatten())
        else:
            y_pred_raw.extend(np.argmax(preds, axis=1))
    print(
        classification_report(
            y_true, y_pred_raw, target_names=class_names, zero_division=0
        )
    )
    f1 = f1_score(y_true, y_pred_raw, average='weighted')
    return {'y_true': y_true, 'y_pred': y_pred_raw, 'f1_weighted': f1}

def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def save_metrics_table(metrics_dict: dict, save_path: str):
    """Append a row of results to results/tables/metrics.csv"""
    df = pd.DataFrame([metrics_dict])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, index=False)
