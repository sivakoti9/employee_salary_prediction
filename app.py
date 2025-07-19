import streamlit as st
import pandas as pd
import joblib

# Load trained model (not using pipeline)
model = joblib.load("best_model.pkl")

# Updated education mapping
education_mapping = {
    "5th-6th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

# App layout
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** using personal and professional attributes.")

st.sidebar.header("Enter Employee Details")

# Input form
age = st.sidebar.slider("Age", 18, 90, 30)

workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
    "State-gov", "Without-pay", "Never-worked"
])

fnlwgt = st.sidebar.slider("FNLWGT (Final Weight)", 10000, 1000000, 100000)

education = st.sidebar.selectbox("Education", list(education_mapping.keys()))
educational_num = education_mapping[education]

marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent"
])

occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
    "Priv-house-serv", "Protective-serv", "Armed-Forces"
])

relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])

race = st.sidebar.selectbox("Race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
])

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

capital_gain = st.sidebar.slider("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.slider("Capital Loss", 0, 99999, 0)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "England", "Other"
])

# Input as DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Preview input
st.subheader("🔎 Input Preview")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted Salary: {prediction[0]}")

# Batch prediction section
st.markdown("---")
st.markdown("### 📂 Batch Prediction using CSV")

uploaded_file = st.file_uploader("Upload a CSV file with employee records", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    if 'education' in batch_df.columns and 'educational-num' not in batch_df.columns:
        batch_df['educational-num'] = batch_df['education'].map(education_mapping)
        batch_df.drop('education', axis=1, inplace=True)

    st.write("📄 Uploaded Data Preview", batch_df.head())

    predictions = model.predict(batch_df)
    batch_df['PredictedClass'] = predictions

    st.write("✅ Batch Predictions:")
    st.write(batch_df.head())

    csv = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
