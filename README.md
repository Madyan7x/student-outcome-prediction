# student-outcome-prediction
ML classification model to predict student academic outcomes using Python, Scikit-learn &amp; Streamlit — 85% accuracy with Random Forest

# Student Outcome Prediction — ML Classification Model

A machine learning project that predicts whether a student will Pass or Fail
based on behavioral and demographic data from the Open University Learning Analytics Dataset (OULAD).

## Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 82% |
| Random Forest (GridSearchCV) | 85% |

## Features
- Processed 7 datasets (32K+ student records, 10M+ VLE interaction logs)
- Feature engineering: total clicks, average scores, days active, withdrawal flags
- Class balancing using SMOTE
- Hyperparameter tuning via GridSearchCV
- Real-time prediction via Streamlit web app

## Tech Stack
Python • Scikit-learn • Pandas • NumPy • SMOTE • Streamlit • Joblib • Google Colab

## How to Run
pip install -r requirements.txt
streamlit run streamlit_app.py
