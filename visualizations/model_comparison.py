import matplotlib.pyplot as plt

models = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "RF + Engineered"
]

r2_scores = [
    0.651,
    0.707,
    0.708,
    0.709
]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, r2_scores)

plt.title("Model Comparison - R² Score")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.ylim(0, 0.8)

for bar, score in zip(bars, r2_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "visualizations/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()