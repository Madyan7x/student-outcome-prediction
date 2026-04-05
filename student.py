import pandas as pd

# Load CSV files
student_info = pd.read_csv("studentInfo.csv")
student_vle = pd.read_csv("studentVle.csv")
student_assessment = pd.read_csv("studentAssessment.csv")
student_registration = pd.read_csv("studentRegistration.csv")
courses = pd.read_csv("courses.csv")
assessments = pd.read_csv("assessments.csv")
vle = pd.read_csv("vle.csv")

# Quick preview of data
print("📌 studentInfo:")
print(student_info.head())
print("___######___")
print("📌 studentVle:")
print(student_vle.head())
print("___######___")
print("📌 studentAssessment:")
print(student_assessment.head())
print("___######___")
print("📌 studentRegistration:")
print(student_registration.head())
print("___######___")
print("📌 courses:")
print(courses.head())
print("___######___")
print("📌 assessments:")
print(assessments.head())
print("___######___")
print("📌 vle:")
print(vle.head())
print("___######___")

# Print shape of each dataset
print("___Shape___")
print("studentInfo:", student_info.shape)
print("studentVle:", student_vle.shape)
print("studentAssessment:", student_assessment.shape)
print("studentRegistration:", student_registration.shape)
print("courses:", courses.shape)
print("assessments:", assessments.shape)
print("vle:", vle.shape)

# Check for missing values
print("___######___")
print("___Null Values___")
print(student_info.isnull().sum())
print(student_vle.isnull().sum())
print(student_assessment.isnull().sum())
print(student_registration.isnull().sum())

# =============== Data Cleaning ===============
student_info['imd_band'] = student_info['imd_band'].fillna("Unknown")
student_assessment = student_assessment.dropna(subset=['score'])
student_registration['date_registration'] = student_registration['date_registration'].fillna(0)

student_info['gender'] = student_info['gender'].map({'M': 0, 'F': 1})
student_info['disability'] = student_info['disability'].map({'N': 0, 'Y': 1})
student_info = student_info[student_info['final_result'].isin(['Pass', 'Fail'])]
student_info['final_result'] = student_info['final_result'].map({'Fail': 0, 'Pass': 1})

# One-hot encode categorical columns
student_info = pd.get_dummies(student_info, columns=['region', 'highest_education', 'imd_band', 'age_band'])

# =============== Feature Engineering ===============
# Total clicks per student
total_clicks = student_vle.groupby('id_student')['sum_click'].sum().reset_index()
total_clicks.columns = ['id_student', 'total_clicks']

# Average score per student
avg_score = student_assessment.groupby('id_student')['score'].mean().reset_index()
avg_score.columns = ['id_student', 'avg_score']

# Days active
days_active = student_vle.groupby('id_student')['date'].nunique().reset_index()
days_active.columns = ['id_student', 'days_active']

# Withdrawal flag
student_registration['withdrew'] = student_registration['date_unregistration'].notnull().astype(int)
withdraw_df = student_registration[['id_student', 'withdrew']].drop_duplicates()

# =============== Merging ===============
main_df = student_info.copy()
main_df = main_df.merge(total_clicks, on='id_student', how='left').fillna({'total_clicks': 0})
main_df = main_df.merge(avg_score, on='id_student', how='left').fillna({'avg_score': 0})
main_df = main_df.merge(days_active, on='id_student', how='left').fillna({'days_active': 0})
main_df = main_df.merge(withdraw_df, on='id_student', how='left').fillna({'withdrew': 0})

print("📦 Final Merged DataFrame:")
print(main_df.head())
print(main_df.shape)
print(main_df.isnull().sum())

# =============== Model Preparation ===============
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = main_df.drop(columns=['id_student', 'code_module', 'code_presentation', 'final_result'])
y = main_df['final_result']

# Balance the dataset using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Show the number of training and testing samples
print("Training data size:", X_train.shape[0])
print("Testing data size:", X_test.shape[0])
print("Total data after resampling:", X_res.shape[0])

# =============== Model Training (Original Models) ===============
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("🔍 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=["Fail", "Pass"]))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=125, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["Fail", "Pass"]))
# =============== GridSearchCV for Best Model ===============
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 125, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("✅ Best Model from GridSearchCV:", grid.best_estimator_)
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print("🔥 Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=["Fail", "Pass"]))

# =============== LazyPredict Evaluation ===============
# from lazypredict.Supervised import LazyClassifier
# lazy = LazyClassifier(verbose=0, ignore_warnings=True)
# models, predictions = lazy.fit(X_train, X_test, y_train, y_test)
# print("📊 LazyPredict Summary:")
# print(models)

# =============== Save for Streamlit ===============
import joblib
joblib.dump(best_model, "student_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")

# =============== Prepare Power BI ===============
main_df['gender'] = main_df['gender'].map({0: 'Male', 1: 'Female'})
main_df['disability'] = main_df['disability'].map({0: 'No', 1: 'Yes'})
main_df['withdrew'] = main_df['withdrew'].map({0: 'No', 1: 'Yes'})

imd_columns = [col for col in main_df.columns if col.startswith('imd_band_')]
main_df['imd_band'] = main_df[imd_columns].idxmax(axis=1).str.replace('imd_band_', '')
main_df.drop(columns=imd_columns, inplace=True)

main_df.to_csv("main_df_final_powerbi.csv", index=False)
