import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------
# Project folders
# ---------------------------------------------------

dashboard_dir = Path(__file__).parent
project_dir = dashboard_dir.parent


# ---------------------------------------------------
# Load saved model and supporting files
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

st.title(
    "🏦 Non-Performing Loan Risk Prediction"
)

st.caption(
    "Machine Learning model comparison using "
    "Sri Lankan banking data"
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
# Prediction tab
# ===================================================

with prediction_tab:

    st.header(
        "Predict Loan Risk"
    )

    st.write(
        "Enter the loan and customer information "
        "to estimate the risk of a Non-Performing Loan."
    )


    # ---------------------------------------------------
    # Input fields
    # ---------------------------------------------------

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
            value=300000.0,
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


    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    if st.button(
        "Predict Loan Risk",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [
                {
                    "Product": product,
                    "LoanBalance": loan_balance,
                    "Age": age,
                    "Income": income,
                    "CustomerCategory": customer_category,
                    "Gender": gender,
                    "Education": education,
                    "CRIB Category": crib_category
                }
            ]
        )


        # Ensure the same feature order used during training
        input_data = input_data[
            features
        ]


        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]


        performing_probability = probabilities[0]

        npl_probability = probabilities[1]


        # ---------------------------------------------------
        # Prediction output
        # ---------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )


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


        st.info(
            "The probability values are model estimates. "
            "They are intended for relative risk interpretation "
            "and are not guaranteed real-world default probabilities."
        )


# ===================================================
# Model comparison tab
# ===================================================

with comparison_tab:

    st.header(
        "Model Comparison"
    )

    st.write(
        "Comparison of Random Forest, "
        "XGBoost and AdaBoost."
    )


    # ---------------------------------------------------
    # Comparison table
    # ---------------------------------------------------

    st.dataframe(
        comparison,
        use_container_width=True
    )


    # ---------------------------------------------------
    # Comparison chart
    # ---------------------------------------------------

    comparison_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]


    available_metrics = [
        column
        for column in comparison_metrics
        if column in comparison.columns
    ]


    chart_data = comparison.set_index(
        "Model"
    )[
        available_metrics
    ]


    st.bar_chart(
        chart_data
    )


    # ---------------------------------------------------
    # Final selected model
    # ---------------------------------------------------

    st.subheader(
        "Selected Model"
    )

    st.success(
        "Random Forest was selected as the final model."
    )

    st.write(
        "Although AdaBoost achieved higher overall Accuracy, "
        "its Recall for Non-Performing Loans was very low. "
        "Random Forest achieved the strongest balance between "
        "Recall, F1 Score and ROC-AUC for this problem."
    )


# ===================================================
# Feature importance tab
# ===================================================

with importance_tab:

    st.header(
        "Feature Importance"
    )

    st.write(
        "The chart below shows the most influential "
        "features used by the selected Random Forest model."
    )


    # ---------------------------------------------------
    # Top features
    # ---------------------------------------------------

    top_features = feature_importance.head(
        15
    )


    # ---------------------------------------------------
    # Feature importance chart
    # ---------------------------------------------------

    st.bar_chart(
        top_features.set_index(
            "Feature"
        )[
            "Importance"
        ]
    )


    # ---------------------------------------------------
    # Feature importance table
    # ---------------------------------------------------

    st.dataframe(
        top_features,
        use_container_width=True
    )


    st.info(
        "Feature importance indicates how strongly each feature "
        "contributes to the Random Forest model overall. "
        "The notebook also contains SHAP analysis for deeper "
        "model explainability."
    )