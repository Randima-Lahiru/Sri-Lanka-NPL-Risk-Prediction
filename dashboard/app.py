import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------
# Project paths
# ---------------------------------------------------

dashboard_dir = Path(__file__).parent
project_dir = dashboard_dir.parent


# ---------------------------------------------------
# Load model and supporting files
# ---------------------------------------------------

model = joblib.load(
    project_dir / "models" / "npl_risk_model.pkl"
)

features = joblib.load(
    project_dir / "models" / "model_features.pkl"
)

comparison = pd.read_csv(
    dashboard_dir / "model_comparison.csv"
)

feature_importance = pd.read_csv(
    dashboard_dir / "feature_importance.csv"
)


# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="NPL Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Non-Performing Loan Risk Prediction")

st.caption(
    "Machine Learning model comparison using Sri Lankan banking data"
)


# ---------------------------------------------------
# Dashboard tabs
# ---------------------------------------------------

prediction_tab, comparison_tab, importance_tab = st.tabs(
    [
        "Prediction",
        "Model Comparison",
        "Feature Importance"
    ]
)


# ===================================================
# TAB 1 - PREDICTION
# ===================================================

with prediction_tab:

    st.header("Predict Loan Risk")

    st.write(
        "Enter the loan and customer information "
        "to estimate the risk of a Non-Performing Loan."
    )

    col1, col2 = st.columns(2)

    with col1:
        product = st.selectbox(
            "Loan Product",
            [
                "Business Loan",
                "Leasing",
                "Housing Loan",
                "Gold Loan",
                "Personal Loan"
            ]
        )

    with col2:
        loan_balance = st.number_input(
            "Loan Balance",
            min_value=0.0,
            value=1500000.0,
            step=10000.0
        )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=38
        )

    with col2:
        income = st.number_input(
            "Income",
            min_value=0.0,
            value=100000.0,
            step=10000.0
        )

    col1, col2 = st.columns(2)

    with col1:
        customer_category = st.selectbox(
            "Customer Category",
            [
                "Retail",
                "Business Small",
                "Business Medium",
                "PRV"
            ]
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education",
            [
                "Degree",
                "A/L",
                "O/L",
                "None",
                "Masters",
                "Diploma"
            ]
        )

    with col2:
        crib_category = st.selectbox(
            "CRIB Category",
            [
                "A",
                "B",
                "C",
                "D",
                "E"
            ]
        )

    if st.button(
        "Predict Loan Risk",
        type="primary"
    ):

        input_data = pd.DataFrame([{
            "Product": product,
            "LoanBalance": loan_balance,
            "Age": age,
            "Income": income,
            "CustomerCategory": customer_category,
            "Gender": gender,
            "Education": education,
            "CRIB Category": crib_category
        }])

        input_data = input_data[features]

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        performing_probability = probabilities[0]
        npl_probability = probabilities[1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "⚠️ Predicted as Non-Performing Loan Risk"
            )
        else:
            st.success(
                "✅ Predicted as Performing Loan"
            )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Performing Loan Probability Estimate",
                f"{performing_probability * 100:.1f}%"
            )

        with result_col2:
            st.metric(
                "NPL Probability Estimate",
                f"{npl_probability * 100:.1f}%"
            )


# ===================================================
# TAB 2 - MODEL COMPARISON
# ===================================================

with comparison_tab:

    st.header("Model Comparison")

    st.write(
        "Comparison of Logistic Regression, "
        "Random Forest and XGBoost."
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    chart_data = comparison.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ]

    st.bar_chart(
        chart_data
    )

    best_row = comparison.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]

    st.success(
        f"Selected Model: {best_row['Model']} "
        f"(F1 Score: {best_row['F1 Score']:.4f})"
    )


# ===================================================
# TAB 3 - FEATURE IMPORTANCE
# ===================================================

with importance_tab:

    st.header("Feature Importance")

    st.write(
        "Top features influencing the selected "
        "Random Forest model."
    )

    top_features = (
        feature_importance
        .head(15)
    )

    st.bar_chart(
        top_features.set_index(
            "Feature"
        )
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )