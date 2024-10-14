import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import io

st.markdown("<h1 style='color: yellow;'>Automated Machine Learning App 💻</h1>", unsafe_allow_html=True)

# Use tabs to differentiate between regression and classification
tab1, tab2 = st.tabs(["Regression", "Classification"])

# Regression Section
with tab1:
    st.markdown("<h2 style='color: orange;'>Regression Model</h2>", unsafe_allow_html=True)
    
    upload_file = st.file_uploader("Upload your CSV file for Regression", type=["csv"])

    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        # Dataset description
        st.write("Dataset Description:")
        st.write(df.describe())

        st.write("Dataset Info:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)

        # Select target column
        target_column = st.selectbox("Select the target column for regression", df.columns)

        if target_column:
            # Filter numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                correlation = df.corr()
                st.write("Correlation with target column:")
                st.write(correlation[[target_column]])

            # Encode target column if necessary
            if df[target_column].dtype == 'object':
                label_encode = LabelEncoder()
                df[target_column] = label_encode.fit_transform(df[target_column])

            # Feature column selection
            feature_columns = st.multiselect("Select feature columns for regression", [col for col in df.columns if col != target_column])

            if len(feature_columns) > 0:
                x = df[feature_columns]
                y = df[target_column]

                # Train/test split
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

                # Scaling
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                # Model selection
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Support Vector Regressor": SVR(),
                    "KNN Regressor": KNeighborsRegressor()
                }

                results = {}
                for model_name, model in models.items():
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results[model_name] = {'MSE': mse, 'R^2': r2}

                st.write("Model Performance for Regression:")
                for model_name, metrics in results.items():
                    st.write(f"**{model_name}**")
                    st.write(f"MSE: {metrics['MSE']:.4f}")
                    st.write(f"R²: {metrics['R^2']:.4f}")

# Classification Section
with tab2:
    st.markdown("<h2 style='color: orange;'>Classification Model</h2>", unsafe_allow_html=True)

    upload_file_class = st.file_uploader("Upload your CSV file for Classification", type=["csv"])

    if upload_file_class is not None:
        df_class = pd.read_csv(upload_file_class)
        st.write("Dataset Preview:")
        st.write(df_class.head())

        # Dataset description
        st.write("Dataset Description:")
        st.write(df_class.describe())

        st.write("Dataset Info:")
        buffer_class = io.StringIO()
        df_class.info(buf=buffer_class)
        info_class = buffer_class.getvalue()
        st.text(info_class)

        # Select target column
        target_column_class = st.selectbox("Select the target column for classification", df_class.columns)

        if target_column_class:
            # Encode target column if necessary
            if df_class[target_column_class].dtype == 'object':
                label_encode = LabelEncoder()
                df_class[target_column_class] = label_encode.fit_transform(df_class[target_column_class])

            # Feature column selection
            feature_columns_class = st.multiselect("Select feature columns for classification", [col for col in df_class.columns if col != target_column_class])

            if len(feature_columns_class) > 0:
                x_class = df_class[feature_columns_class]
                y_class = df_class[target_column_class]

                # Train/test split
                x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(x_class, y_class, test_size=0.2, random_state=42)

                # Scaling
                scaler_class = StandardScaler()
                x_train_class = scaler_class.fit_transform(x_train_class)
                x_test_class = scaler_class.transform(x_test_class)

                # Model selection
                classifiers = {
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Support Vector Classifier": SVC(),
                    "KNN Classifier": KNeighborsClassifier()
                }

                results_class = {}
                for model_name, model in classifiers.items():
                    model.fit(x_train_class, y_train_class)
                    y_pred_class = model.predict(x_test_class)
                    accuracy = accuracy_score(y_test_class, y_pred_class)
                    results_class[model_name] = {'Accuracy': accuracy}

                st.write("Model Performance for Classification:")
                for model_name, metrics in results_class.items():
                    st.write(f"{model_name} - Accuracy: {metrics['Accuracy']:.2f}")
