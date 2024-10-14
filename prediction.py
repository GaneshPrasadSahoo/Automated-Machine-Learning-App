import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io

st.markdown("<h1 style='color: yellow;'>Automated Regression Model 💻</h1>", unsafe_allow_html=True)

# Upload dataset
upload_file = st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.markdown("<h3 style='color: orange;'>Dataset Preview:</h3>", unsafe_allow_html=True)
    st.write(df.head())

    # Data set Description
    st.markdown("<h3 style='color: orange;'>Dataset Description:</h3>", unsafe_allow_html=True)
    st.write(df.describe())

    # Dataset Columns
    st.markdown("<h3 style='color: orange;'>Dataset Columns:</h3>", unsafe_allow_html=True)
    st.write(df.columns)

    # Display dataset info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.markdown("<h3 style='color: orange;'>Dataset Info:</h3>", unsafe_allow_html=True)
    st.text(info)

    # Select the target column
    st.markdown("<h3 style='color: orange;'>Select the target column (label) for training:</h3>", unsafe_allow_html=True)
    
    target_column = st.selectbox("Select target column", df.columns)
     
     
    

    if target_column:
        # Calculate and display correlation with the selected target column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            correlation = df.corr()
            st.markdown("<h3 style='color: orange;'>Correlation of features with the target column:</h3>", unsafe_allow_html=True)    
            
            st.write(correlation[[target_column]])
        else:
            st.warning(f"The selected target column '{target_column}' is not numeric. Correlation cannot be calculated.")

        # Encode target column if it is of object type
        if df[target_column].dtype == 'object':
            label_encode = LabelEncoder()
            df[target_column] = label_encode.fit_transform(df[target_column])
            st.write(f"Target column '{target_column}' encoded successfully.")

        # Encode object-type feature columns
        object_columns = df.select_dtypes(include=["object"]).columns
        label_encode = LabelEncoder()

        for col in object_columns:
            if col != target_column:  # Don't re-encode the target column
                df[col] = label_encode.fit_transform(df[col])
                st.write(f"Feature column '{col}' encoded successfully.")


                
        st.markdown("<h3 style='color: orange;'>Updated Dataset Preview:</h3>", unsafe_allow_html=True)
        
        st.write(df.head())

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
    
    st.markdown("<h3 style='color: orange;'>Select Feature Columns:</h3>", unsafe_allow_html=True)

   # Create the multiselect input
    feature_columns = st.multiselect("Select Feature columns", [col for col in df.columns if col != target_column])


    if len(feature_columns) == 0:
        st.error("Please select at least one feature column.")
    else:
        # Label encoding for categorical columns
        object_columns = df[feature_columns].select_dtypes(include=["object"]).columns
        label_encode = LabelEncoder()
        for col in object_columns:
            df[col] = label_encode.fit_transform(df[col])

        # Handle missing values in feature columns
        for col in df[feature_columns].columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == "object":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)

        # Handle missing values in the target column
        if df[target_column].isnull().sum() > 0:
            st.write(f"Target column '{target_column}' contains missing values.")
            action = st.radio("Choose how to handle missing values in the target column:", 
                              ("Remove rows with missing target values", "Impute missing values"))

            if action == "Remove rows with missing target values":
                df = df.dropna(subset=[target_column])
            else:
                if df[target_column].dtype == 'object':
                    df[target_column].fillna(df[target_column].mode()[0], inplace=True)
                else:
                    df[target_column].fillna(df[target_column].mean(), inplace=True)

        # Divide the dataset
        st.markdown("<h3 style='color: orange;'>Splitting data into features and target...</h3>", unsafe_allow_html=True)
        
        x = df[feature_columns]
        y = df[target_column]

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Scale the feature columns
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Support Vector Regressor": SVR(),
            "KNN Regressor": KNeighborsRegressor()
        }

        result = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred_test = model.predict(x_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            result[model_name] = {
                'MSE (Test)': mse_test,
                'RMSE (Test)': np.sqrt(mse_test),
                'R^2 (Test)': r2_test
            }

        # Display model performance
        
        st.write("Model Performance:")
        for model_name, metrics in result.items():
            st.write(f"{model_name} - MSE (Test): {metrics['MSE (Test)']:.2f}, "
                     f"RMSE (Test): {metrics['RMSE (Test)']:.2f}, R^2 (Test): {metrics['R^2 (Test)']:.2f}")

        # Select the best model based on R^2 score
        best_model_name = max(result, key=lambda x: result[x]['R^2 (Test)'])
        best_model = models[best_model_name]
        best_model.fit(x_train, y_train)
        y_pred_best = best_model.predict(x_test)

        st.write(f"\nBest Model: {best_model_name}")
        st.write(f"R^2 (Test): {r2_score(y_test, y_pred_best):.2f}")

        # Code for downloading the Python script
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

       
        st.markdown("<h3 style='color: red;'>Download code For model:</h3>", unsafe_allow_html=True)    
        st.download_button("Download code", code, file_name="best_model_code.py")

else:
    st.markdown("<h5 style='color:Cyan'>Please upload a dataset to continue.</h5>", unsafe_allow_html=True)
