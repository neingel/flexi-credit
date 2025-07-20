
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.title("🧠 Flexible Credit Scoring Engine")

# 👉 Instructions section
st.markdown("""
Welcome to the **Flexible Credit Scoring Engine**! 🧾  
This app helps you assess the credit risk of borrowers using a machine learning model.

### 🧭 Instructions:
1. **Download** the sample CSV for reference (columns required: Age, Income, Debt, etc.)
2. **Upload your own CSV** with similar numeric data.
3. **Map your columns** correctly to the model’s expected inputs.
4. **Click Run Scoring** to calculate risk probabilities and classifications.
            
### 🗂️ Expected Column Meanings:

| Field Name          | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `Age`               | Age of the borrowers in years (e.g., 21–65)                                  |
| `Monthly Income`    | Gross monthly income in dollars (e.g., 3000, 10000)                          |
| `Total Debt`        | Sum of all current outstanding debts (credit cards, loans, etc.)            |
| `Credit Utilization`| Credit used ÷ total credit limit (a ratio from 0.0 to 1.0)                   |
| `Missed Payments`   | Number of missed or late payments in recent months                          |
| `Credit Lines`      | Total number of active credit lines/accounts (e.g., credit cards, loans)    |


_You’ll receive a breakdown of predicted risk levels, visualizations, and an optional confusion matrix (if true defaults are provided)._
""")

# Load model
model = joblib.load("credit_model.pkl")

# Template download
sample_data = pd.read_csv("mock_credit_data.csv")
st.download_button("📄 Download Sample CSV", sample_data.to_csv(index=False), file_name="sample_credit_data.csv")

# File upload
uploaded_file = st.file_uploader("📤 Upload Your CSV File", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.write("🔍 Data Preview", df_raw.head())

    st.subheader("🛠 Map Your Columns")
    cols = df_raw.columns.tolist()

    placeholder = "🔽 Select a column"

    col_age = st.selectbox("Select 'Age' column 👦👴", ["🔽 Select a column"] + cols, index=0)
    col_income = st.selectbox("Select 'Monthly Income' column 💵", ["🔽 Select a column"] + cols, index=0)
    col_debt = st.selectbox("Select 'Total Debt' column 👺", ["🔽 Select a column"] + cols, index=0)
    col_util = st.selectbox("Select 'Credit Utilization' column 💳", ["🔽 Select a column"] + cols, index=0)
    col_missed = st.selectbox("Select 'Missed Payments' column ❌", ["🔽 Select a column"] + cols, index=0)
    col_creditlines = st.selectbox("Select 'Credit Lines' column 💹", ["🔽 Select a column"] + cols, index=0)

    if st.button("Run Scoring"):
        try:
            selected = [col_age, col_income, col_debt, col_util, col_missed, col_creditlines]
            if "🔽 Select a column" in selected:
                st.error("❌ Please map all fields before scoring.")
            else:
                input_data = df_raw[[col_age, col_income, col_debt, col_util, col_missed, col_creditlines]].copy()
                input_data.columns = ["Age", "Monthly_Income", "Total_Debt", "Credit_Utilization", "Missed_Payments", "Credit_Lines"]

                probs = model.predict_proba(input_data)[:, 1]
                preds = model.predict(input_data)

                def classify_risk(prob):
                    if prob < 0.4:
                        return "Low"
                    elif prob < 0.7:
                        return "Medium"
                    else:
                        return "High"

                df_results = df_raw.copy()
                df_results["Predicted_Prob"] = np.round(probs, 2)
                df_results["Risk_Category"] = [classify_risk(p) for p in probs]
                df_results["Predicted_Default"] = preds

                st.success("✅ Scoring Complete!")
                st.subheader("📊 Scored Results")
                st.dataframe(df_results)

                st.subheader("📈 Risk Distribution")
                fig, ax = plt.subplots()
                sns.histplot(df_results["Predicted_Prob"], bins=10, kde=True, ax=ax)
                ax.set_title("Predicted Probability of Default")
                st.pyplot(fig)

                if "Defaulted" in df_results.columns:
                    st.subheader("🧾 Confusion Matrix (if actual labels present)")
                    cm = confusion_matrix(df_results["Defaulted"], df_results["Predicted_Default"])
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
