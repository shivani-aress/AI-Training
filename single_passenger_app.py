"""Streamlit form for predicting survival for one Titanic passenger."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).parent / "dataset" / "train.csv"

st.set_page_config(page_title="Titanic Passenger Prediction", page_icon="??")
st.title("Titanic passenger survival prediction")
st.write(
    "Enter one passenger's details. The app trains the selected model using "
    "`dataset/train.csv` and predicts whether this passenger would have survived."
)


@st.cache_data
def read_training_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(model_type, criterion, max_depth, n_estimators):
    df = read_training_data().copy()
    if "Survived" not in df.columns:
        raise ValueError("dataset/train.csv must contain a Survived column.")

    # Match the notebook's missing-value handling.
    fill_values = {"Age": df["Age"].median()}
    for column in ("Cabin", "Embarked"):
        modes = df[column].mode()
        if not modes.empty:
            fill_values[column] = modes.iloc[0]
    for column, value in fill_values.items():
        df[column] = df[column].fillna(value)

    # Match the notebook's dropped columns and one-hot encoding.
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    y = df["Survived"]
    X = pd.get_dummies(
        df.drop(columns=["Survived"]),
        columns=["Sex", "Embarked"],
        drop_first=True,
        dtype=int,
    )

    if model_type == "Decision Tree":
        model = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth, random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    model.fit(X, y)
    return model, X.columns.tolist(), fill_values


try:
    train_df = read_training_data()
except Exception as exc:
    st.error(f"Could not load the training data at {DATA_PATH}: {exc}")
    st.stop()

model_type = st.selectbox("Choose a model", ["Random Forest", "Decision Tree"])
max_depth_choice = st.selectbox("Maximum tree depth", ["No limit", 3, 5, 10, 20])
max_depth = None if max_depth_choice == "No limit" else int(max_depth_choice)
if model_type == "Random Forest":
    n_estimators = st.number_input(
        "Number of trees", min_value=10, max_value=1000, value=100, step=10
    )
    criterion = "gini"
else:
    n_estimators = 100
    criterion = st.selectbox("Split criterion", ["gini", "entropy"])

sex_options = sorted(train_df["Sex"].dropna().unique().tolist())
embarked_options = sorted(train_df["Embarked"].dropna().unique().tolist())

with st.form("passenger_details"):
    st.subheader("Passenger details")
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", sex_options)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", embarked_options)
    predict_clicked = st.form_submit_button("Predict survival", type="primary")

if predict_clicked:
    model, feature_columns, _ = train_model(
        model_type, criterion, max_depth, int(n_estimators)
    )

    # Create the same encoded input columns the notebook used for training.
    passenger_values = {
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
    }
    for column in feature_columns:
        if column.startswith("Sex_"):
            passenger_values[column] = int(sex == column.removeprefix("Sex_"))
        elif column.startswith("Embarked_"):
            passenger_values[column] = int(embarked == column.removeprefix("Embarked_"))
    passenger = pd.DataFrame([passenger_values]).reindex(
        columns=feature_columns, fill_value=0
    )

    prediction = int(model.predict(passenger)[0])
    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.info("Prediction: Did not survive")

    if hasattr(model, "predict_proba"):
        survived_index = list(model.classes_).index(1)
        probability = model.predict_proba(passenger)[0][survived_index]
        st.metric("Estimated survival probability", f"{probability:.1%}")

    st.caption(
        "This is a prediction from the model trained on dataset/train.csv; "
        "a single passenger form does not provide an actual outcome for calculating accuracy."
    )
