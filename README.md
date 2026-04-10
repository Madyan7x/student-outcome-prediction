# 🎓 Student Outcome Prediction — ML Classification Model & Power BI Dashboard

A machine learning project that predicts whether a university student will **Pass** or **Fail** based on behavioral, demographic, and assessment data from the **Open University Learning Analytics Dataset (OULAD)**.

Built with Python, Scikit-learn, and deployed as an interactive **Streamlit web application**.

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 82% |
| Random Forest (GridSearchCV) | **85%** ✅ |

---

## 📸 Power BI Dashboard

An interactive Power BI dashboard built on the cleaned student data, visualizing:
- Student outcome distribution (Pass / Fail)
- Performance trends by demographics and region
- Assessment scores and activity patterns

📊 [View Dashboard File](Student.pbix)

---

## 🗂️ Dataset

This project uses the **OULAD** dataset — a real-world open education dataset from the UK Open University.

📥 **Download the dataset here:**  
[Open University Learning Analytics Dataset (OULAD) — Google Drive](https://share.google/xWOBKfXZOkKIAWv8q)

Place all CSV files in the **root project folder** before running.

| File | Description |
|------|-------------|
| `studentInfo.csv` | Demographics & final outcomes |
| `studentVle.csv` | Student activity logs (10M+ rows) |
| `studentAssessment.csv` | Assessment scores |
| `studentRegistration.csv` | Registration & withdrawal info |
| `courses.csv` | Course-level info |
| `assessments.csv` | Assessment metadata |
| `vle.csv` | Virtual learning environment tools |

---

## ⚙️ Project Pipeline

### 1. Data Loading & Exploration
- Loaded 7 CSV files using Pandas
- Explored shapes, data types, and null values

### 2. Data Cleaning
- Filled missing `imd_band` values with `"Unknown"`
- Dropped rows with missing assessment scores
- Encoded `gender` and `disability` as binary (0/1)
- Filtered to binary classification: **Pass / Fail** (excluded Withdrawn)
- One-hot encoded: `region`, `highest_education`, `imd_band`, `age_band`

### 3. Feature Engineering
- **Total Clicks** — sum of VLE interactions per student
- **Average Score** — mean assessment score per student
- **Days Active** — number of unique active days in VLE
- **Withdrawal Flag** — whether student unregistered

### 4. Model Training
- Applied **SMOTE** to handle class imbalance
- 80/20 stratified train-test split
- Trained: Logistic Regression & Random Forest
- Optimized Random Forest via **GridSearchCV** (n_estimators, max_depth, min_samples_split)

### 5. Deployment
- Saved best model using `joblib`
- Built interactive **Streamlit app** for real-time prediction

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the ML pipeline
```bash
python student.py
```
This will train the model and generate:
- `student_model.pkl`
- `model_features.pkl`
- `main_df_final_powerbi.csv`

### Step 3 — Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🖥️ Streamlit App Features

**🔍 Lookup by Student ID**
- Enter any student ID from the dataset
- View all student features
- Get instant Pass/Fail prediction with confidence score

**✍️ Manual Prediction**
- Input: Average Score, Total Clicks, Days Active, Withdrawal status, Gender, Disability
- Instant prediction with confidence score

---

## 📦 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, imbalanced-learn (SMOTE) |
| Visualization | Power BI |
| Deployment | Streamlit, Joblib |
| Environment | Google Colab / VS Code |

---

## 📁 Project Structure

```
student-outcome-prediction/
│
├── student.py                  # ML pipeline (cleaning, training, saving model)
├── app.py                      # Streamlit web application
├── Student.pbix                # Power BI dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── student_model.pkl           # Saved best model (generated after running student.py)
├── model_features.pkl          # Feature list (generated after running student.py)
├── main_df_final_powerbi.csv   # Cleaned data for Power BI (generated after running student.py)
│
└── data/                       # Place OULAD CSV files here
    ├── studentInfo.csv
    ├── studentVle.csv
    ├── studentAssessment.csv
    ├── studentRegistration.csv
    ├── courses.csv
    ├── assessments.csv
    └── vle.csv
```

---

## 👤 Author

**Madyan Alammari**  
Computer Science — King Abdulaziz University  
📧 Madyan3172001@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/madyan-alammari-73852a170)  
🐙 [GitHub](https://github.com/MadyanAlammari)
