import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title('Welcome to the Machine Learning Application @Najeeb-JONY')
st.write('This application allows you to upload your dataset and apply various machine learning models to analyze the data.')

data_option = st.sidebar.selectbox("Do you want to upload data or use example data?", ("Upload Data", "Use Example Data"))

if data_option == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'tsv', 'txt'])
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "text/tab-separated-values":
            data = pd.read_csv(uploaded_file, sep='\t')
else:
    dataset_name = st.sidebar.selectbox("Select Example Dataset", ("titanic", "tips", "iris"))
    data = sns.load_dataset(dataset_name)


if data is not None:
    st.write("Data Shape:", data.shape)
    st.write("First Five Rows:", data.head())
    st.write("Data Description:", data.describe())
    st.write("Data Info:", data.info())
    st.write("Column Names:", data.columns.tolist())

if data is not None:
    all_columns = data.columns.tolist()
    selected_features = st.multiselect("Select Feature Columns", all_columns)
    selected_target = st.selectbox("Select Target Column", all_columns)

if data is not None:
    if data[selected_target].dtype == 'float64' or data[selected_target].dtype == 'int64':
        problem_type = 'Regression'
    else:
        problem_type = 'Classification'
    st.write(f"This is a {problem_type} problem.")


    from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Example of defining models based on problem type
if problem_type == 'Regression':
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVM': SVR()
    }
else:  # Classification
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

# Model selection
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

# Split size
split_size = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.7, 0.01)

# Splitting data
X = data[selected_features]
y = data[selected_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

# Training and predictions
model = models[model_choice]
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
if problem_type == 'Regression':
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, predictions)
    st.write(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
else:  # Classification
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    st.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    conf_matrix = confusion_matrix(y_test, predictions)
    st.write("Confusion Matrix:", conf_matrix)

# Highlight the best model
# This section would involve comparing multiple models, which we haven't explicitly done here. You would keep track of the evaluation metrics for each model in a dictionary and then select the model with the best performance based on your criteria (e.g., highest accuracy for classification, highest R^2 for regression).

# Model Download
if st.button('Download Model'):
    pickle.dump(model, open(f"{model_choice}.pkl", 'wb'))
    st.write("Model downloaded")

# Making Predictions with New Data
if st.checkbox("Make Predictions"):
    # This could be as simple as asking for values for each feature
    # Or you could allow for file upload for batch predictions
    # Here's a simplified example for direct input:
    input_data = {feature: float(st.text_input(feature)) for feature in selected_features}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction}")

# Visualizations
# You can add visualizations to help understand the data and model performance
# For example, you can add a confusion matrix for classification problems
# Or a scatter plot of actual vs. predicted values for regression problems
# Here's a simple example for a scatter plot for regression problems
if problem_type == 'Regression':
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)
else:
    # Add your visualization code here
    pass
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # or any other model that doesn't handle NaNs natively

# Assuming you've already imported enable_iterative_imputer and IterativeImputer
iterative_imputer = IterativeImputer()

# Create a pipeline that first imputes missing values, then scales the data, and finally applies your model
model_pipeline = make_pipeline(
    iterative_imputer,
    StandardScaler(),
    SVR()
)
import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'feature1': ['1.1', '2.2', '', '3.3'],
    'feature2': ['4.4', '', '5.5', '6.6']
}

df = pd.DataFrame(data)

# Convert empty strings to NaN
df.replace('', np.nan, inplace=True)

# Convert columns to numeric, setting errors='coerce' will convert problematic entries to NaN
df['feature1'] = pd.to_numeric(df['feature1'], errors='coerce')
df['feature2'] = pd.to_numeric(df['feature2'], errors='coerce')

# Now, you can handle NaNs as per your strategy, for example, using an imputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(df_filled)


# Example of setting up a pipeline with an imputer and a model (e.g., SVR)
if model_choice == 'SVM':
    # Assuming SVM is used for regression here. For classification, use SVC.
    model_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),  # Or use IterativeImputer for a more sophisticated approach
        StandardScaler(),  # It's a good practice to scale features for SVR
        SVR()
    )
else:
    # For other models, you might set up different pipelines or use them directly if they handle NaNs natively
    # This is a simplified example, adjust according to the selected model and whether it's regression or classification
    model_pipeline = models[model_choice]

# Replace model.fit(X_train, y_train) with model_pipeline.fit(X_train, y_train)
# Ensure X_train and y_test don't have NaNs after preprocessing if the model doesn't handle NaNs
model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
