import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pickle

# Initialize label encoders for later use
label_encoders = {}

# Function to preprocess data
def preprocess_data(df, features, target):
    # Fill missing values with IterativeImputer
    imputer = IterativeImputer()
    df[features] = imputer.fit_transform(df[features])

    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Encode categorical variables
    for feature in features:
        if df[feature].dtype == 'object':
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le
    return df

# Streamlit app
st.title("Streamlit ML Application")

# Step 2 & 4: Data Upload or Selection
data_option = st.sidebar.radio("Do you want to upload data or use example data?", ["Upload Data", "Example Data"])
if data_option == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'tsv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    example_data = st.sidebar.selectbox("Choose an example dataset:", ["titanic", "tips", "iris"])
    df = sns.load_dataset(example_data)

if df is not None:
    # Step 5: Display data information
    st.write("### Dataset Information")
    st.write("Shape:", df.shape)
    st.write(df.head())
    st.write(df.describe())
    
    # Step 6: Feature and Target Selection
    features = st.multiselect("Select your features", df.columns.tolist(), default=df.columns.tolist()[:-1])
    target = st.selectbox("Select your target", df.columns.tolist(), index=len(df.columns.tolist()) - 1)

    # Step 7: Problem Identification
    problem_type = "Regression" if df[target].dtype in ['float64', 'int64'] else "Classification"
    st.write(f"Identified problem type: {problem_type}")

    # Preprocessing
    df = preprocess_data(df, features, target)
    
    # Train-test split
    test_size = st.sidebar.slider("Test split size", 0.1, 0.5, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size)

    # Model selection
    model_options = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor() if problem_type == "Regression" else DecisionTreeClassifier(),
        "Random Forest": RandomForestRegressor() if problem_type == "Regression" else RandomForestClassifier(),
        "SVM": SVR() if problem_type == "Regression" else SVC()
    }
    model_choice = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
    model = model_options[model_choice]

    # Training and Evaluation
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if problem_type == "Regression":
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        st.write(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
    else:
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')
        cm = confusion_matrix(y_test, predictions)
        st.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        st.write("Confusion Matrix:", cm)

    # Model download
    if st.button("Download Model"):
        pickle.dump(model, open("model.pkl", "wb"))
        with open("model.pkl", "rb") as file:
            st.download_button(label="Download Model", data=file, file_name="model.pkl")

    # Prediction interface
    if st.button("Predict on new data"):
        # Implement this based on your specific needs
        st.write("Prediction interface here")
