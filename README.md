# 🎓 Smart Student Performance Predictive ML System

An end-to-end Machine Learning project that predicts a student's **total performance score** using academic engagement indicators such as self-study hours, attendance, and class participation.

The project covers the complete ML workflow — from data understanding and exploratory analysis to feature engineering, model comparison, evaluation, model deployment, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Student performance can be influenced by multiple academic engagement factors. This project uses historical student data to build a regression model that estimates a student's total performance score.

The final system uses a **Random Forest Regression model** and provides predictions through an interactive **Streamlit dashboard**.

### Key Highlights

* 5,010 student records
* Data cleaning and exploratory data analysis
* Feature engineering
* Multiple regression models compared
* 5-fold cross-validation
* Residual and error analysis
* Feature importance analysis
* Trained Random Forest model saved with Joblib
* Reusable prediction pipeline
* Interactive Streamlit dashboard

---

## 🎯 Objectives

* Understand the relationship between student engagement and academic performance.
* Clean and prepare the dataset for machine learning.
* Perform exploratory data analysis to identify patterns.
* Engineer additional predictive features.
* Compare multiple regression algorithms.
* Evaluate model performance using MAE, RMSE, and R².
* Analyze prediction errors and feature importance.
* Deploy the trained model through a Streamlit application.

---

## 📊 Dataset

The dataset contains **5,010 student records**.

### Input Features

| Feature                   | Description                       |
| ------------------------- | --------------------------------- |
| `weekly_self_study_hours` | Student's weekly self-study hours |
| `attendance_percentage`   | Student attendance percentage     |
| `class_participation`     | Class participation measure       |

### Target Variable

| Target        | Description                       |
| ------------- | --------------------------------- |
| `total_score` | Student's total performance score |

### Engineered Features

Two additional features were created:

* `study_attendance_interaction`
* `engagement_score`

The engineered features were evaluated along with the original variables during model development.

---

## 🔍 Exploratory Data Analysis

The project includes analysis of:

* Feature distributions
* Target distribution
* Correlation between variables
* Relationships between predictors and total score
* Potential patterns and outliers

The analysis showed that `weekly_self_study_hours` had the strongest relationship with `total_score` in this dataset.

---

## ⚙️ Feature Engineering

Two features were created to capture additional relationships.

### Study × Attendance Interaction

```python
study_attendance_interaction = (
    weekly_self_study_hours * attendance_percentage
)
```

### Engagement Score

```python
engagement_score = (
    attendance_percentage + class_participation
) / 2
```

These engineered features were included in the final Random Forest model.

---

## 🤖 Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Random Forest Regressor with engineered features

### Model Comparison

| Model             | Features              |       MAE |      RMSE |        R² |
| ----------------- | --------------------- | --------: | --------: | --------: |
| Linear Regression | Original              |     7.338 |     9.257 |     0.651 |
| Decision Tree     | Original              |     6.221 |     8.471 |     0.707 |
| Random Forest     | Original              |     6.198 |     8.458 |     0.708 |
| Random Forest     | Original + Engineered | **6.193** | **8.445** | **0.709** |

The final model is the **Random Forest Regressor with engineered features**.

---

## 📈 Model Evaluation

### Final Model Performance

* **MAE:** 6.193
* **RMSE:** 8.445
* **R²:** 0.709

The model was evaluated on a held-out test set using an **80:20 train-test split**.

### Cross-Validation

A 5-fold cross-validation was also performed.

* Mean R²: **0.703**
* Standard deviation: **0.011**

The cross-validation results were relatively consistent across the five folds.

---

## 🌲 Feature Importance

The Random Forest model identified the following feature importance values:

| Feature                        | Importance |
| ------------------------------ | ---------: |
| `weekly_self_study_hours`      |      0.896 |
| `study_attendance_interaction` |      0.034 |
| `engagement_score`             |      0.026 |
| `class_participation`          |      0.024 |
| `attendance_percentage`        |      0.020 |

`weekly_self_study_hours` was the dominant feature in the trained model.

**Important:** Feature importance indicates the model's reliance on a feature for prediction. It does not establish that the feature causally determines academic performance.

---

## 🧪 Error Analysis

Residual analysis was performed to understand where the model makes larger prediction errors.

The overall Mean Absolute Error was approximately **6.19 points**.

The model showed larger errors for some students with unusual or extreme outcomes, indicating that predictions are less reliable for certain atypical cases.

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train / Test Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Cross-Validation
     ↓
Residual & Error Analysis
     ↓
Final Random Forest Model
     ↓
Model Serialization
     ↓
Prediction Pipeline
     ↓
Streamlit Dashboard
```

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard where users can enter:

* Weekly self-study hours
* Attendance percentage
* Class participation

The application then generates a predicted total performance score.

### Run the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The application will open in your browser.

---

## 🧩 Prediction Pipeline

The trained model and feature names are saved using Joblib:

```text
models/
├── feature_names.pkl
└── random_forest_student_performance.pkl
```

The reusable prediction logic is implemented in:

```text
src/predict.py
```

The prediction pipeline automatically creates the engineered features required by the trained model before generating a prediction.

---

## 📁 Project Structure

```text
Smart-Student-Performance/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── student_performance_analysis.xlsx
│   │
│   └── processed/
│       ├── student_performance_engineered.csv
│       └── student_performance_model_data.csv
│
├── models/
│   ├── feature_names.pkl
│   └── random_forest_student_performance.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
│
├── src/
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Jupyter Notebook
* Git & GitHub

---

## ⚠️ Limitations

* The dataset contains a limited set of academic engagement variables.
* The model's predictions are estimates rather than guaranteed academic outcomes.
* Feature importance should not be interpreted as causal evidence.
* Some students with unusual outcomes may have larger prediction errors.
* Additional academic, demographic, behavioral, and historical variables could potentially improve the model.

---

## 🚀 Future Enhancements

Possible future improvements include:

* Hyperparameter optimization
* Additional student behavioral and academic features
* Explainable AI using SHAP
* Model monitoring
* Prediction confidence or uncertainty estimates
* Improved dashboard visualizations
* Deployment to a cloud platform
* Database integration
* Student-level performance recommendations

---

## 👩‍💻 Author

**Hiniyasri P.**

BE Computer Science Engineering Student
Interested in **Data Analytics, Machine Learning, and Data-Driven Solutions**.

---

## ⭐ Project Purpose

This project was developed as a practical demonstration of an end-to-end Machine Learning workflow, combining **data analysis, feature engineering, model development, evaluation, and deployment** into a single project.
