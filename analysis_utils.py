import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

def analyze_numeric(df):
    """
    Returns basic statistics and correlation matrix of numeric DataFrame.
    """
    description = df.describe().T
    correlation = df.corr()
    return description, correlation


def chi_square_analysis(df, col1, col2):
    """
    Performs a chi-square test between two categorical columns.
    Returns chi2 value, p-value, and the contingency table.
    """
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2_stat": chi2,
        "p_value": p,
        "contingency_table": contingency
    }


def correlation_plot(df):
    """
    Generates a heatmap of correlation between numeric columns using Plotly.
    """
    fig = px.imshow(df.corr(), 
                    text_auto=True, 
                    title="Correlation Heatmap",
                    color_continuous_scale="Viridis",
                    aspect="auto")
    return fig
