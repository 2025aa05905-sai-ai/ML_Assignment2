import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

st.set_page_config(page_title="Medical ML Comparison", layout="wide")

st.title("Diabetes Prediction - ML Model Comparison")

st.write("This application compares multiple machine learning models on a medical dataset.")

# download test dataset
st.subheader("Download Test Dataset for Evaluation")
with open("test.csv", "rb") as file:
    st.download_button(
        label="Download test.csv",
        data=file,
        file_name="test.csv",
        mime="text/csv"
    )

st.markdown("---")

# model selection on main page
st.subheader("Select Model for Prediction")
model_option = st.selectbox(
    "Choose Model",
    ["logistic","decision_tree","knn","naive_bayes","random_forest","xgboost"]
)

uploaded_file = st.file_uploader("Upload TEST dataset (use test.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    y_true = df["Diabetes_binary"]
    X = df.drop("Diabetes_binary", axis=1)

    model = joblib.load(f"model/{model_option}.pkl")
    pred = model.predict(X)

    st.subheader("Model Predictions")
    st.write(pred)

    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, pred)
    mcc = matthews_corrcoef(y_true, pred)
    st.subheader("Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("AUC", f"{auc:.3f}")
    col6.metric("MCC", f"{mcc:.3f}")

    cm = confusion_matrix(y_true, pred)
    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
    st.plotly_chart(fig)
