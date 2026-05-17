import streamlit as st
import pandas as pd
import joblib

# Load the new pipeline model
model = joblib.load("best_pipeline.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Collect ONLY the 13 features the model was trained on
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", ["Private", "Local-gov", "Others", "Self-emp-not-inc", "State-gov", "Self-emp-inc", "Federal-gov"])
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=200000)
educational_num = st.sidebar.slider("Years of Education", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Widowed", "Divorced", "Separated", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "Others"])
relationship = st.sidebar.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.sidebar.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy", "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia", "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "Greece", "France", "Ecuador", "Ireland", "Hong", "Trinadad&Tobago", "Cambodia", "Laos", "Thailand", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Honduras", "Hungary", "Scotland", "Holand-Netherlands"])

# Create a dataframe with the exact 13 column names expected by the model
input_df = pd.DataFrame({
    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt], 
    'educational-num': [educational_num], 'marital-status': [marital_status], 
    'occupation': [occupation], 'relationship': [relationship], 'race': [race], 
    'gender': [gender], 'capital-gain': [capital_gain], 'capital-loss': [capital_loss], 
    'hours-per-week': [hours_per_week], 'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.dataframe(input_df)

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction (must match census format)", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Clean the batch data just like we cleaned the training data
    clean_batch = batch_data.copy()
    if 'income' in clean_batch.columns:
        clean_batch = clean_batch.drop(columns=['income'])
    if 'education' in clean_batch.columns:
        clean_batch = clean_batch.drop(columns=['education'])

    try:
        batch_preds = model.predict(clean_batch)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.dataframe(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")


