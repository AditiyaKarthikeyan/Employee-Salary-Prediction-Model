# Employee Salary Prediction Model 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## Project Context
This model was developed as part of the **Virtual Internship on Artificial Intelligence & Machine Learning** (June 2025 - Aug 2025), a program facilitated in collaboration with **AICTE, IBM, and Edunet Foundation**.

## Overview
Understanding the socioeconomic factors that influence income levels is crucial for economic modeling and business planning. This project implements a machine learning classification system designed to predict whether an individual's income exceeds $50,000 per year based on census data demographics and employment features.

---

## Technical Architecture

### Dataset & Exploratory Data Analysis (EDA)
* **Dataset:** Utilized the Adult Census Income dataset (`adult.csv`), evaluating features such as age, working class, education level, marital status, and hours worked per week.
* **Data Cleaning:** Missing values (represented as '?') in the `workclass` and `occupation` columns were isolated and categorized as 'Others'. Records indicating 'Without-pay' and 'Never-worked' were removed to focus on the active workforce.
* **Outlier Handling:** Applied strict bounding based on boxplot analysis, restricting the age range to 17-75 years and educational numerical values between 5 and 16.
* **Feature Engineering:** Categorical features were numerically transformed using `LabelEncoder` to prepare the data for pipeline ingestion.

### Model Evaluation & Selection
Multiple classification algorithms were trained and evaluated against a 20% test split:
* **Logistic Regression:** 81.49% 
* **K-Nearest Neighbors (KNN):** 82.45% 
* **Support Vector Machine (SVM):** 83.96% 
* **Random Forest:** 85.02% 
* **Gradient Boosting:** Achieved the highest accuracy at 85.71% and was selected as the final production model.

### Observation
* Identified class imbalance in target variable. Future iterations could utilize SMOTE or class weighting to improve recall for the minority class.

---

## Deployment Interface
The finalized Gradient Boosting model is serialized using `joblib` and deployed via a **Streamlit** web application. 
* **Single Prediction:** Users can input specific employee metrics via a sidebar slider and dropdown interface.
* **Batch Processing:** The application supports CSV file uploads, allowing organizations to run predictions on entire datasets simultaneously and download the generated results.

---

## Getting Started

To explore the EDA process or run the Streamlit application locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AditiyaKarthikeyan/Employee-Salary-Prediction-Model.git
   cd Employee-Salary-Prediction-Model
   ```
   
2. **Install dependencies:**
    Ensure you have Python installed, then run:

   ```bash
   pip install pandas numpy scikit-learn matplotlib streamlit joblib
   ```

3. **Run the Exploratory Notebook:**

   ```bash
   jupyter notebook notebooks/employee_salary_prediction.ipynb
   ```

4. **Launch the Web Application:**
To interact with the deployment UI locally, run the Streamlit server:

   ```bash
   streamlit run app.py
   ```

---

[Try the Live Web Demo Here!](https://employee-salary-prediction-model-bjlkrwep8otetav8higymv.streamlit.app)
