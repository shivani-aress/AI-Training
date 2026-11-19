"""Streamlit interface for training and testing a Titanic Random Forest."""

import io

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Tree Classifier CSV Tester", page_icon="??")
st.title("Tree classifiers: upload and test a dataset")
st.write(
    "Upload a labeled CSV to train a model, then optionally upload a separate CSV "
    "for evaluation or predictions. For Titanic data, choose `Survived` as the target."
)
st.caption(
    "Feature preparation follows the notebook: fill Age with the median, fill "
    "Embarked with its mode, drop PassengerId/Name/Ticket/Cabin, and one-hot encode "
    "Sex and Embarked with drop_first=True."
)


def read_csv(uploaded_file):
    """Read an uploaded CSV and report parsing errors clearly."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        return None


def prepare_features(frame, fill_values=None):
    """Apply the Titanic feature preparation used in the notebook."""
    features = frame.copy()
    if fill_values is None:
        fill_values = {}
        if "Age" in features.columns:
            fill_values["Age"] = features["Age"].median()
        for column in ("Cabin", "Embarked"):
            if column in features.columns:
                modes = features[column].mode()
                if not modes.empty:
                    fill_values[column] = modes.iloc[0]

    for column, value in fill_values.items():
        if column in features.columns:
            features[column] = features[column].fillna(value)

    columns_to_drop = [
        column for column in ("PassengerId", "Name", "Ticket", "Cabin")
        if column in features.columns
    ]
    features = features.drop(columns=columns_to_drop)
    categorical_columns = [
        column for column in ("Sex", "Embarked") if column in features.columns
    ]
    if categorical_columns:
        features = pd.get_dummies(
            features, columns=categorical_columns, drop_first=True, dtype=int
        )
    return features, fill_values


def make_model(model_type, n_estimators, max_depth, criterion):
    if model_type == "Decision Tree":
        return DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth, random_state=42
        )
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )


train_upload = st.file_uploader("Training CSV (must include the target)", type="csv")
if train_upload:
    train_df = read_csv(train_upload)
    if train_df is not None and not train_df.empty:
        st.caption(f"Training data: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
        default_target = "Survived" if "Survived" in train_df.columns else train_df.columns[-1]
        target = st.selectbox(
            "Choose the target column",
            train_df.columns.tolist(),
            index=train_df.columns.get_loc(default_target),
        )

        test_upload = st.file_uploader(
            "Optional separate test CSV (same feature columns; target may be omitted)",
            type="csv",
        )
        model_type = st.selectbox("Choose a model", ["Random Forest", "Decision Tree"])
        test_size = st.slider("Test portion when no separate test CSV is uploaded", 0.1, 0.4, 0.2, 0.05)
        max_depth_choice = st.selectbox("Maximum tree depth", ["No limit", 3, 5, 10, 20], index=0)
        max_depth = None if max_depth_choice == "No limit" else int(max_depth_choice)
        if model_type == "Random Forest":
            n_estimators = st.number_input("Number of trees", min_value=10, max_value=1000, value=100, step=10)
            criterion = "gini"
        else:
            n_estimators = 100
            criterion = st.selectbox("Split criterion", ["gini", "entropy"])

        if st.button("Train model", type="primary"):
            if train_df[target].isna().any():
                st.error("The target column contains missing values. Remove those rows before training.")
            elif train_df[target].nunique() < 2:
                st.error("The target must contain at least two classes.")
            else:
                raw_features = train_df.drop(columns=[target])
                features, fill_values = prepare_features(raw_features)
                labels = train_df[target]
                if features.shape[1] == 0:
                    st.error("The training CSV needs at least one feature column besides the target.")
                else:
                    if test_upload:
                        test_df = read_csv(test_upload)
                        if test_df is None:
                            st.stop()
                        has_labels = target in test_df.columns
                        expected = raw_features.columns.tolist()
                        absent = [column for column in expected if column not in test_df.columns]
                        extra = [column for column in test_df.columns if column not in expected and column != target]
                        if absent or extra:
                            st.error(
                                f"Test CSV columns must match training features. Missing: {absent or 'none'}; "
                                f"unexpected: {extra or 'none'}."
                            )
                            st.stop()
                        test_features, _ = prepare_features(test_df[expected], fill_values)
                        X_test = test_features.reindex(columns=features.columns, fill_value=0)
                        y_test = test_df[target] if has_labels else None
                        X_train, y_train = features, labels
                    else:
                        stratify = labels if labels.value_counts().min() >= 2 else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            features,
                            labels,
                            test_size=test_size,
                            random_state=42,
                            stratify=stratify,
                        )

                    model = make_model(model_type, n_estimators, max_depth, criterion)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    st.session_state["rf_model"] = model
                    st.session_state["model_type"] = model_type
                    st.session_state["rf_target"] = target
                    st.session_state["rf_test_columns"] = features.columns.tolist()

                    if y_test is not None:
                        st.subheader("Evaluation")
                        st.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
                        st.write("Confusion matrix (actual rows, predicted columns)")
                        classes = model.classes_
                        cm = confusion_matrix(y_test, predictions, labels=classes)
                        st.dataframe(
                            pd.DataFrame(
                                cm,
                                index=[f"Actual {value}" for value in classes],
                                columns=[f"Predicted {value}" for value in classes],
                            )
                        )
                        st.text(classification_report(y_test, predictions, zero_division=0))
                    else:
                        st.info("No target column was found in the test CSV, so accuracy cannot be calculated.")

                    result = X_test.copy()
                    result[f"Predicted_{target}"] = predictions
                    if y_test is not None:
                        result[f"Actual_{target}"] = y_test.to_numpy()
                    st.subheader("Predictions")
                    st.dataframe(result.head(20))
                    st.download_button(
                        "Download predictions as CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="random_forest_predictions.csv",
                        mime="text/csv",
                    )
