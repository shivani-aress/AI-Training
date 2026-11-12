"""Interactive Streamlit dashboard for Mall customer segmentation."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
SEGMENTS_PATH = APP_DIR / "customer_segments.csv"
MODEL_PATH = APP_DIR / "segmentation_model.joblib"
FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Segmentation using K-Means Clustering")
st.write(
    "Customer segmentation groups customers with similar characteristics. "
    "K-Means assigns customers to the nearest cluster center based on annual "
    "income and spending score."
)
st.info(
    "Gender and age are collected as profile details, but the model uses only "
    "annual income and spending score to make predictions."
)


@st.cache_data
def load_segments():
    if not SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SEGMENTS_PATH.name}. Run the notebook through its export cell first."
        )

    data = pd.read_csv(SEGMENTS_PATH)
    required = {
        "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"The exported CSV is missing columns: {sorted(missing)}")
    if data[list(required)].isnull().any().any():
        raise ValueError("The exported customer data contains missing values.")
    return data


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH.name}. Run the notebook's model-saving cell first."
        )
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("The saved model feature columns do not match this app.")
    return artifact["scaler"], artifact["model"]


try:
    customers = load_segments()
    scaler, kmeans = load_model_artifact()

    # Confirm this saved model and scaler reproduce the exported cluster labels.
    actual_features = customers[FEATURE_COLUMNS]
    reproduced_labels = kmeans.predict(scaler.transform(actual_features))
    if not np.array_equal(reproduced_labels, customers["Cluster"].to_numpy()):
        raise ValueError(
            "Model predictions do not match customer_segments.csv. "
            "Run the notebook cells in order and regenerate the model and CSV."
        )
except Exception as error:
    st.error(f"Unable to load the segmentation project files: {error}")
    st.stop()


cluster_summary = customers.groupby("Cluster").agg(
    Average_Annual_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean"),
    Customer_Count=("Cluster", "size"),
)
mean_income = customers["Annual Income (k$)"].mean()
mean_spending = customers["Spending Score (1-100)"].mean()


def relative_level(value, overall_average):
    """Describe a cluster value relative to the dataset average."""
    if value > overall_average * 1.10:
        return "High"
    if value < overall_average * 0.90:
        return "Low"
    return "Average"


def segment_name(cluster):
    row = cluster_summary.loc[cluster]
    income_level = relative_level(row["Average_Annual_Income"], mean_income)
    spending_level = relative_level(row["Average_Spending_Score"], mean_spending)
    return f"{income_level} income, {spending_level} spending"


# Customer analytics use values from the exported, assigned customer data.
st.subheader("Customer Analytics")
metric_columns = st.columns(4)
metric_columns[0].metric("Total customers", f"{len(customers):,}")
metric_columns[1].metric("Number of clusters", f"{customers['Cluster'].nunique()}")
metric_columns[2].metric("Average annual income", f"${mean_income:.1f}k")
metric_columns[3].metric("Average spending score", f"{mean_spending:.1f}")

st.divider()

left_column, right_column = st.columns([1, 1.2], gap="large")
with left_column:
    st.subheader("Predict a Customer Segment")
    with st.form("customer_prediction_form"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        annual_income = st.number_input(
            "Annual Income (in thousands)",
            min_value=0.0,
            max_value=500.0,
            value=float(round(mean_income, 1)),
            step=1.0,
        )
        spending_score = st.slider("Spending Score", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button(
            "Predict Customer Segment", type="primary", width="stretch"
        )

    if submitted:
        try:
            # Gender and age are intentionally excluded from model input.
            new_customer = pd.DataFrame(
                [[annual_income, spending_score]], columns=FEATURE_COLUMNS
            )
            scaled_customer = scaler.transform(new_customer)
            predicted_cluster = int(kmeans.predict(scaled_customer)[0])
            st.session_state["customer_prediction"] = {
                "cluster": predicted_cluster,
                "gender": gender,
                "age": age,
                "income": annual_income,
                "spending": spending_score,
            }
        except Exception as error:
            st.error(f"Could not predict the customer segment: {error}")

    prediction = st.session_state.get("customer_prediction")
    if prediction:
        cluster = prediction["cluster"]
        row = cluster_summary.loc[cluster]
        name = segment_name(cluster)
        income_level = relative_level(row["Average_Annual_Income"], mean_income)
        spending_level = relative_level(row["Average_Spending_Score"], mean_spending)

        st.success(f"Predicted cluster: {cluster} - {name}")
        st.write(
            f"This cluster averages ${row['Average_Annual_Income']:.1f}k annual income "
            f"and a spending score of {row['Average_Spending_Score']:.1f}. "
            f"The submitted profile was {prediction['gender'].lower()}, age {prediction['age']}. "
            "Gender and age did not affect the prediction."
        )

        st.markdown("**Business recommendations**")
        if spending_level == "High":
            st.write("- Consider loyalty rewards and exclusive experiences to retain active spenders.")
        elif spending_level == "Low":
            st.write("- Test relevant offers and service messages to encourage engagement.")
        else:
            st.write("- Use targeted promotions to encourage repeat purchases.")

        if income_level == "High":
            st.write("- Highlight premium or higher-value products that fit this group's profile.")
        else:
            st.write("- Emphasize practical bundles and clear value in promotions.")

with right_column:
    st.subheader("Customer Segments")
    cluster_options = ["All clusters"] + [str(cluster) for cluster in cluster_summary.index]
    selected_cluster = st.selectbox("Filter chart by cluster", cluster_options)
    chart_data = customers.copy()
    chart_data["Cluster"] = chart_data["Cluster"].astype(str)
    if selected_cluster != "All clusters":
        chart_data = chart_data[chart_data["Cluster"] == selected_cluster]
    st.scatter_chart(
        chart_data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        width="stretch",
    )

st.divider()
chart_column, table_column = st.columns([1, 1.4], gap="large")
with chart_column:
    st.subheader("Customers per Cluster")
    st.bar_chart(cluster_summary["Customer_Count"], color="#6558D3")
with table_column:
    st.subheader("Cluster Summary")
    display_summary = cluster_summary.copy()
    display_summary.insert(0, "Segment", [segment_name(cluster) for cluster in display_summary.index])
    st.dataframe(display_summary.round(2), width="stretch")

st.caption(
    "Predictions use the saved StandardScaler and K-Means model. "
    "Segment names are generated from each cluster's actual averages."
)
