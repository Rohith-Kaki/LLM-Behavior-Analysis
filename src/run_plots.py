import pandas as pd
import matplotlib.pyplot as plt
import os

METRICS_FILE = "results/metrics/final_metrics.csv"
OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load results
df = pd.read_csv(METRICS_FILE)

# Metrics to plot
metrics = [
    "factual_accuracy",
    "reasoning_accuracy",
    "context_understanding",
    "hallucination_score",
    "bias_score",
    "safety_score"
]


for metric in metrics:
    plt.figure()
    plt.bar(df["model_name"], df[metric])
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(f"{OUTPUT_DIR}/{metric}.png")
    plt.close()

print("Saved individual metric plots.")


df.set_index("model_name")[metrics].plot(kind="bar")
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png")
plt.close()

print("Saved combined comparison plot.")


import numpy as np

labels = metrics
num_metrics = len(labels)

angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

for _, row in df.iterrows():
    values = row[metrics].tolist()
    values.append(values[0])
    ax.plot(angles, values, label=row["model_name"])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1)
plt.legend(loc="upper right")
plt.title("Radar Comparison of Models")
plt.savefig(f"{OUTPUT_DIR}/radar_plot.png")
plt.close()

print("Saved radar plot.")