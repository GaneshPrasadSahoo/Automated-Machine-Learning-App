import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
import io
import pickle

st.title("Automated Classification Model 💻")

# Upload Dataset
upload_file = st.file_uploader("Upload Your CSV File", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Dataset Description
    st.write("Dataset Description:")
    st.write(df.describe())

    # Dataset columns
    st.write("Dataset Columns:")
    st.write(df.columns)

    # Display dataset info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.write("Dataset Info:")
    st.text(info)

    # Select the target column
    st.write("Select the target column (label) for training:")
    target_column = st.selectbox("Select target column", df.columns)

    if target_column:
        # Encode target column if it is of object type
        if df[target_column].dtype == 'object':
            label_encode = LabelEncoder()
            df[target_column] = label_encode.fit_transform(df[target_column])
            st.write(f"Target column '{target_column}' encoded successfully.")

        # Encode object-type feature columns
        object_columns = df.select_dtypes(include=["object"]).columns
        if len(object_columns) > 0:
            label_encode = LabelEncoder()
            for col in object_columns:
                if col != target_column:  # Don't re-encode the target column
                    df[col] = label_encode.fit_transform(df[col])
                    st.write(f"Feature column '{col}' encoded successfully.")
        else:
            st.write("No categorical features found for encoding.")

        # Handle Missing Values in Feature columns
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    # Fill missing values in categorical columns with the mode (most frequent value)
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    st.write(f"Missing values in '{col}' filled with mode.")
                else:
                    # Fill missing values in numerical columns with the mean
                    df[col].fillna(df[col].mean(), inplace=True)
                    st.write(f"Missing values in '{col}' filled with mean.")

        # Select feature columns
        feature_columns = st.multiselect("Select Feature columns", [col for col in df.columns if col != target_column])

        if len(feature_columns) == 0:
            st.error("Please select at least one feature column.")
        else:
            # Ensure all categorical columns are encoded
            object_columns = df[feature_columns].select_dtypes(include=["object"]).columns
            if len(object_columns) > 0:
                label_encode = LabelEncoder()
                for col in object_columns:
                    df[col] = label_encode.fit_transform(df[col])
                    st.write(f"Categorical feature column '{col}' encoded successfully.")
            else:
                st.write("No additional categorical features found for encoding.")

            # Divide the dataset into X and y
            st.write("Splitting data into features and target...")
            X = df[feature_columns]
            y = df[target_column]

            # Split into train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            # Define Models
            models = {
                "Random Forest Classifier": RandomForestClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Support Vector Classifier": SVC(),
                "K-Nearest Neighbors Classifier": KNeighborsClassifier()
            }

            # Store results
            result = {}

            # Train and evaluate models
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                r2_test = r2_score(y_test, y_pred_test)

                # Store metrics in result dictionary
                result[model_name] = {
                    "MSE(Test)": mse_test,
                    "RMSE(Test)": np.sqrt(mse_test),
                    "R^2(Test)": r2_test
                }

            # Display model performance
            st.write("Model Performance:")
            for model_name, metrics in result.items():
                st.write(f"{model_name} - MSE (Test): {metrics['MSE(Test)']:.2f}, "
                         f"RMSE (Test): {metrics['RMSE(Test)']:.2f}, R^2 (Test): {metrics['R^2(Test)']:.2f}")

            # Select the best model based on R^2 score
            best_model_name = max(result, key=lambda x: result[x]['R^2(Test)'])
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            y_pred_best = best_model.predict(X_test)

            st.write(f"\nBest Model: {best_model_name}")
            st.write(f"R^2 (Test): {r2_score(y_test, y_pred_best):.2f}")



            # Option to download the trained model as a pickle file
            st.write("Download the trained model as pkl:")
            model_pickle = pickle.dumps(best_model)
            st.download_button(
                label="Download Trained Model",
                data=model_pickle,
                file_name='trained_model.pkl',
                mime='application/octet-stream',
            )

            # Python code generation for downloading
            code = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import {best_model_name.replace(" ", "")}

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Preprocess the dataset
object_columns = {list(object_columns)}

# Apply Label Encoding
label_encoder = LabelEncoder()
for col in object_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Handle missing values
for col in {feature_columns}:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

# Handle missing values in target column
if df['{target_column}'].isnull().sum() > 0:
    if df['{target_column}'].dtype == 'object':
        df['{target_column}'].fillna(df['{target_column}'].mode()[0], inplace=True)
    else:
        df['{target_column}'].fillna(df['{target_column}'].mean(), inplace=True)

# Split features and target
x = df[{feature_columns}]
y = df['{target_column}']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the best model
model = {best_model_name.replace(" ", "")}()
model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

print("Predictions:", y_pred)
"""

            st.write("Download Python code to retrain the best model:")
            st.download_button("Download code", code, file_name="best_model_code.py")

else:
    st.write("Please upload a dataset to continue.")
