from pathlib import Path
import joblib
import pandas as pd


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Model paths
MODEL_PATH = BASE_DIR / "models" / "random_forest_student_performance.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"


# Load trained model and feature names
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)


def predict_student_performance(
    weekly_self_study_hours,
    attendance_percentage,
    class_participation
):

    # Input validation
    if weekly_self_study_hours < 0:
        raise ValueError("Study hours cannot be negative.")

    if attendance_percentage < 0 or attendance_percentage > 100:
        raise ValueError("Attendance percentage must be between 0 and 100.")

    if class_participation < 0:
        raise ValueError("Class participation cannot be negative.")

    # Create engineered features
    study_attendance_interaction = (
        weekly_self_study_hours * attendance_percentage
    )

    engagement_score = (
        attendance_percentage + class_participation
    ) / 2

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "weekly_self_study_hours": weekly_self_study_hours,
        "attendance_percentage": attendance_percentage,
        "class_participation": class_participation,
        "study_attendance_interaction": study_attendance_interaction,
        "engagement_score": engagement_score
    }])

    # Ensure correct feature order
    input_data = input_data[feature_names]

    # Make prediction
    prediction = model.predict(input_data)[0]

    return prediction