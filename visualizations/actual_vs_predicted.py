import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load processed dataset
df = pd.read_csv(
    "data/processed/student_performance_engineered.csv"
)

# Features and target
features = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "study_attendance_interaction",
    "engagement_score"
]

X = df[features]
y = df["total_score"]

# Same train-test split used during model evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train the final model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction reference line
min_score = min(y_test.min(), y_pred.min())
max_score = max(y_test.max(), y_pred.max())

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    linestyle="--"
)

plt.title("Actual vs Predicted Student Performance")
plt.xlabel("Actual Total Score")
plt.ylabel("Predicted Total Score")

plt.tight_layout()

plt.savefig(
    "visualizations/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()