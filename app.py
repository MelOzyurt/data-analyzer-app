import streamlit as st
import pandas as pd
import numpy as np
from analysis_utils import analyze_numeric, chi_square_analysis, correlation_plot

st.set_page_config(page_title="📊 Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON, XML, Feather)", type=["csv", "xlsx", "xls", "json", "xml", "feather"])

if uploaded_file:
    try:
        # Read file based on format
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xml'):
            df = pd.read_xml(uploaded_file)
        elif uploaded_file.name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # Show data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric analysis
    if numeric_cols:
        st.subheader("📈 Numeric Features Analysis")
        desc, corr = analyze_numeric(df[numeric_cols])
        st.write("Descriptive Statistics")
        st.dataframe(desc)

        st.plotly_chart(correlation_plot(df[numeric_cols]))

    # Categorical analysis
    if len(cat_cols) >= 2:
        st.subheader("📊 Categorical Variable Relationship")
        col1 = st.selectbox("Select first categorical column:", cat_cols)
        col2_options = [c for c in cat_cols if c != col1]
        col2 = st.selectbox("Select second categorical column:", col2_options)

        result = chi_square_analysis(df, col1, col2)
        st.write(f"Chi-Square Test Result: χ² = {result['chi2_stat']:.2f}, p-value = {result['p_value']:.4f}")
        st.dataframe(result["contingency_table"])

    st.success("✅ Analysis completed successfully!")
