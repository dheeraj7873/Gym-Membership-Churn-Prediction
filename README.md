# Gym Membership Churn Prediction

This project aims to predict whether a gym member will churn (cancel their membership) using machine learning techniques based on behavioral and profile data.

## Dataset

The dataset contains the following features:
- age
- gender
- membership_type
- avg_checkins_per_week
- last_checkin_days_ago
- class_attendance_rate
- complaints
- satisfaction_score
- membership_duration_months
- churn (target variable: 0 = no churn, 1 = churn)

## Problem Statement

Churn prediction helps fitness centers retain customers by identifying those likely to leave so that retention strategies can be applied early.

## Project Workflow

1. Data Cleaning and Preprocessing
2. Label Encoding of categorical features
3. Feature Scaling using StandardScaler
4. Model Training using RandomForestClassifier
5. Evaluation using Accuracy, ROC AUC, Confusion Matrix, and Classification Report
6. Visualization of Feature Importance and Confusion Matrix

## Results

- Accuracy: ~95%
- AUC Score: ~0.98
- Most important features: satisfaction_score, last_checkin_days_ago, avg_checkins_per_week

## Files

- gym_churn_dataset.csv
- churn_prediction.ipynb
- README.md

## How to Run

1. Clone this repository
2. Install required libraries:

   pip install pandas scikit-learn matplotlib seaborn

3. Open and run churn_prediction.ipynb in Jupyter Notebook or any Python IDE

## Author

Dheeraj Choudhary  
LinkedIn: https://www.linkedin.com/in/dheerajchoudhary/
