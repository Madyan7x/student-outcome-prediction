import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("student_model.pkl")
features = joblib.load("model_features.pkl")

# Load prepared student data
students_df = pd.read_csv("main_df_model_input.csv")

st.set_page_config(page_title="🎯 Student Success Prediction", layout="centered")
st.title("🎓 Student Outcome Predictor")

# ========== First Section: View by Student ID ==========
st.header("🔍 Look up Student by ID")

# Student ID input
student_id = st.number_input("Enter Student ID", min_value=int(students_df['id_student'].min()),
                             max_value=int(students_df['id_student'].max()), step=1)

# Fetch student row
student_row = students_df[students_df['id_student'] == student_id]

if not student_row.empty:
    st.subheader("📋 Student Features")
    st.dataframe(student_row.drop(columns=['id_student']), use_container_width=True)

    # Get only features required for model
    model_input = student_row[features]

    # Predict
    prediction = model.predict(model_input)[0]
    prediction_proba = model.predict_proba(model_input)[0][prediction]

    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader("🔮 Prediction")

    if prediction == 1:
        st.success(f"{result}  (Confidence: {prediction_proba:.2%})")
    else:
        st.error(f"{result}  (Confidence: {prediction_proba:.2%})")

else:
    st.warning("No student found with this ID.")

# ========== Second Section: Manual Input ==========
st.header("✍️ Manual Prediction Input")

st.info("Fill only the following important features:")

avg_score = st.slider("📈 Average Score", 0.0, 100.0, 70.0)  # Default: 70
total_clicks = st.number_input("🖱️ Total Clicks", min_value=0, value=2000)  # Default: 2000
days_active = st.number_input("📅 Days Active", min_value=0, value=25)  # Default: 25
withdrew = st.selectbox("📤 Withdrew?", options=[0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
disability = st.selectbox("♿ Disability?", options=[0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
gender = st.selectbox("⚧️ Gender", options=[0, 1], index=0, format_func=lambda x: "Male" if x == 0 else "Female")

# Prepare a blank row for model input
input_dict = {
    'avg_score': avg_score,
    'total_clicks': total_clicks,
    'days_active': days_active,
    'withdrew': withdrew,
    'disability': disability,
    'gender': gender
}

# Fill the rest of the required features with 0
manual_input = {col: 0 for col in features}
manual_input.update(input_dict)

# Predict manually
if st.button("🚀 Predict Outcome"):
    input_df = pd.DataFrame([manual_input])
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][pred]
    label = "✅ Pass" if pred == 1 else "❌ Fail"
    
    st.subheader("🔮 Manual Prediction Result")
    if pred == 1:
        st.success(f"{label}  (Confidence: {pred_proba:.2%})")
    else:
        st.error(f"{label}  (Confidence: {pred_proba:.2%})")
