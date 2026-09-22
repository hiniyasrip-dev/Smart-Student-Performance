import matplotlib.pyplot as plt

features = [
    "weekly_self_study_hours",
    "study_attendance_interaction",
    "engagement_score",
    "class_participation",
    "attendance_percentage"
]

importance = [
    0.896014,
    0.033837,
    0.025546,
    0.024179,
    0.020424
]

plt.figure(figsize=(10, 6))

bars = plt.barh(features[::-1], importance[::-1])

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

for bar, value in zip(bars, importance[::-1]):
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center"
    )

plt.xlim(0, 1)
plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()