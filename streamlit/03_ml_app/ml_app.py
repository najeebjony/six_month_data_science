import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pickle

# Step 1: Ask the user with a welcome message and a brief description of the application
st.title("Machine Learning Application")
st.write("This application allows you to train and evaluate different machine learning models.")

# Step 2: Ask the user if they want to upload the data or use the example data
data_option = st.sidebar.selectbox("Select data option", ("Upload Data", "Use Example Data"))

# Step 3: If the user selects to upload the data, show the upload section on the sidebar
if data_option == "Upload Data":
    st.sidebar.write("Upload your dataset:")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)  # Change the appropriate read function based on file type
else:
    # Step 4: If the user does not want to upload the dataset, provide a default dataset selection box on the sidebar
    dataset = st.sidebar.selectbox("Select dataset", ("titanic", "tips", "iris"))
    if dataset == "titanic":
        data = sns.load_dataset("titanic")
    elif dataset == "tips":
        data = sns.load_dataset("tips")
    elif dataset == "iris":
        data = sns.load_dataset("iris")

# Step 5: Print basic data information
st.write("Data Information:")
st.write("Shape:", data.shape)
st.write("Head:")
st.write(data.head())
st.write("Describe:")
st.write(data.describe())
st.write("Info:")
st.write(data.info())
st.write("Columns:")
st.write(data.columns)

# Step 6: Ask the user to select the columns as features and the column as target
selected_features = st.multiselect("Select feature columns", data.columns)
selected_target = st.selectbox("Select target column", data.columns)

# Step 7: Identify the problem type
if data[selected_target].dtype in [np.float64, np.int64]:
    problem_type = "regression"
    st.write("Problem Type: Regression")
else:
    problem_type = "classification"
    st.write("Problem Type: Classification")
# Step 8: Pre-process the data
if problem_type == "regression":
    # Fill missing values with IterativeImputer
    imputer = IterativeImputer()
    X[selected_features] = imputer.fit_transform(X[selected_features])

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X[selected_features] = scaler.fit_transform(X[selected_features])

else:
    # Fill missing values with IterativeImputer
    imputer = IterativeImputer()
    X[selected_features] = imputer.fit_transform(X[selected_features])

    # Encode categorical variables using One-Hot Encoding
    categorical_features = X.select_dtypes(include=[object])
    categorical_features = pd.get_dummies(categorical_features, drop_first=True)
    numerical_features = X.select_dtypes(include=[np.number])
    X = pd.concat([numerical_features, categorical_features], axis=1)

y = y.astype('str')
y_train = y_train.astype('str')
y_test = y_test.astype('str')

# Step 9: Ask the user to provide train test split size
test_size = st.sidebar.slider("Select train test split size", 0.1, 0.9, 0.2)

# Step 10: Ask the user to select the model
if problem_type == "regression":
    model_option = st.sidebar.selectbox("Select model", ("Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine"))
else:
    model_option = st.sidebar.selectbox("Select model", ("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"))

# Step 11: Train the models and evaluate on test data
if model_option == "Linear Regression":
    model = LinearRegression()
elif model_option == "Decision Tree":
    model = DecisionTreeRegressor()
elif model_option == "Random Forest":
    model = RandomForestRegressor()
elif model_option == "Support Vector Machine":
    model = SVR()

X = data[selected_features]
y = data[selected_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model.fit(X_train, y_train)
# Step 12: Evaluate the model and print evaluation metrics
if problem_type == "regression":
    y_pred = model.predict(X_test)
    st.write("Evaluation Metrics:")
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R2 Score:", r2_score(y_test, y_pred))
else:
    y_pred = model.predict(X_test)
    st.write("Evaluation Metrics:")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
    st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
    st.write("F1-Score:", f1_score(y_test, y_pred, average="weighted"))
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

# Step 13: Print the evaluation metrics for each model
# (Implement for other models as per requirement)

# Step 14: Highlight the best model based on the evaluation metrics

# Step 15: Ask the user if they want to download the model
download_model = st.sidebar.radio("Download model?", ("Yes", "No"))

if download_model == "Yes":
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.sidebar.write("Model downloaded successfully.")

# Step 16: Ask the user if they want to make a prediction
make_prediction = st.sidebar.radio("Make a prediction?", ("Yes", "No"))

if make_prediction == "Yes":
    # Ask the user to provide input data
    st.sidebar.write("Provide input data:")
    input_data = {}
    for feature in selected_features:
        if data[feature].dtype == object:
            input_data[feature] = st.sidebar.selectbox(f"Select {feature}", data[feature].unique())
        else:
            input_data[feature] = st.sidebar.slider(f"Select {feature}", data[feature].min(), data[feature].max())

    # Pre-process the input data
    if problem_type == "regression":
        input_data = pd.DataFrame(input_data, index=[0])
        input_data[selected_features] = imputer.transform(input_data[selected_features])
        input_data[selected_features] = scaler.transform(input_data[selected_features])
    else:
        input_data = pd.DataFrame(input_data, index=[0])
        for feature, encoder in encoder_dict.items():
            input_data[feature] = encoder.transform(input_data[feature])

    # Make the prediction
    prediction = model.predict(input_data[selected_features])

    # Display the prediction
    if problem_type == "regression":
        st.sidebar.write("Prediction:", prediction[0])
    else:
        st.sidebar.write("Prediction:", prediction[0])