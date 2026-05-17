import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pipeline model
model = joblib.load("best_pipeline.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="wide")

st.title("💼 Employee Salary Predictor & Analytics")
st.markdown("Predict income brackets and analyze feature influence.")

# Layout: Sidebar for inputs, Main for results
st.sidebar.header("Employee Demographics")

# Collect Features
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
native_country = st.sidebar.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "India", "England", "China", "Others"])

# Create DataFrame
input_df = pd.DataFrame({
    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt], 
    'educational-num': [educational_num], 'marital-status': [marital_status], 
    'occupation': [occupation], 'relationship': [relationship], 'race': [race], 
    'gender': [gender], 'capital-gain': [capital_gain], 'capital-loss': [capital_loss], 
    'hours-per-week': [hours_per_week], 'native-country': [native_country]
})

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 🔎 Candidate Profile")
    st.dataframe(input_df.T, use_container_width=True)

with col2:
    st.write("### 📊 Prediction Analysis")
    if st.button("Run AI Analysis", type="primary"):
        # Get prediction and probabilities
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Display professional metrics
        if prediction == ">50K":
            st.success(f"**Outcome: High Income Predicted (>50K)**")
            confidence = probabilities[1]
        else:
            st.warning(f"**Outcome: Standard Income Predicted (<=50K)**")
            confidence = probabilities[0]

        st.metric(label="Model Confidence Score", value=f"{confidence:.1%}")
        st.progress(float(confidence))

        st.info("💡 **Insight:** The model relies heavily on 'Capital Gain', 'Marital Status', and 'Years of Education' for this demographic.")

st.markdown("---")
st.markdown("#### 📂 Enterprise Batch Processing")
uploaded_file = st.file_uploader("Upload census-formatted CSV for bulk predictions", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    clean_batch = batch_data.copy().drop(columns=['income', 'education'], errors='ignore')

    try:
        batch_preds = model.predict(clean_batch)
        batch_data['PredictedClass'] = batch_preds
        st.write(f"✅ Successfully processed {len(batch_data)} records:")
        st.dataframe(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results (CSV)", csv, file_name='batch_predictions.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Format Error: {e}")
