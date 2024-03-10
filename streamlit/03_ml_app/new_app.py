import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.multiclass import type_of_target
import numpy as np
import pickle
from io import BytesIO

# Initialize Seaborn
sns.set_theme()

# Function to load data
def load_data(dataset_name):
    if dataset_name == 'titanic':
        data = sns.load_dataset('titanic')
    elif dataset_name == 'tips':
        data = sns.load_dataset('tips')
    else:  # 'iris'
        data = sns.load_dataset('iris')
    return data

# Function to preprocess data
def preprocess_data(df, features, target):
    numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df[features].select_dtypes(include=['object', 'bool', 'category']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Preprocess target for classification if it's not numeric
    
    # if df[target].dtype == 'object' or df[target].dtype.name == 'category':
    #     le = LabelEncoder()
    #     df[target] = le.fit_transform(df[target])
    #     return preprocessor, le
    # else:
    #     return preprocessor, None
    
    return preprocessor, None  # Fix: Return the preprocessor variable

# Streamlit app starts here
st.title('Machine Learning Application')

st.write("""
         # Welcome to the ML Application
         Use this application to run machine learning models on your data.
         """)

# Step 2: Data source selection
data_option = st.sidebar.radio("Upload data or use example data:", ["Upload Data", "Use Example Data"])

if data_option == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Choose a file (csv)", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    example_data = st.sidebar.selectbox("Or choose an example dataset:", ["titanic", "tips", "iris"])
    df = load_data(example_data)

if df is not None:
    # Step 5: Display basic data information
    st.write("Basic Data Information:")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.write(df.describe())
    
    # Step 6: User selects features and target
    all_columns = df.columns.tolist()
    features = st.multiselect("Select features:", all_columns, default=all_columns[:-1])
    target = st.selectbox("Select target:", options=all_columns, index=len(all_columns)-1)
    
    # Check if user has made selections
    if not features or not target:
        st.warning('Please select at least one feature and a target.')
        st.stop()
    
    # Step 7: Determine if it's a regression or classification problem
    is_regression = np.issubdtype(df[target].dtype, np.number) and not type_of_target(df[target]).startswith('multiclass')
    
    if is_regression:
        st.write("This is a regression problem.")
    else:
        st.write("This is a classification problem.")
    
    # Step 8: Data preprocessing
    preprocessor, label_encoder = preprocess_data(df, features, target)
    
    # Split data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    if label_encoder:
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)
    
    # Model Selection
    model_options = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM']
    model_choice = st.sidebar.selectbox("Choose ML Model:", model_options)
    
    if model_choice == 'Linear Regression':
        model = LinearRegression() if is_regression else LogisticRegression(max_iter=1000)
    elif model_choice == 'Decision Tree':
        model = DecisionTreeRegressor() if is_regression else DecisionTreeClassifier()
    elif model_choice == 'Random Forest':
        model = RandomForestRegressor() if is_regression else RandomForestClassifier()
    elif model_choice == 'SVM':
        model = SVR() if is_regression else SVC()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions and evaluations
    predictions = model.predict(X_test)
    
    if is_regression:
        st.write("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
        st.write("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))
        st.write("R^2 Score:", r2_score(y_test, predictions))
    else:
        st.write("Accuracy:", accuracy_score(y_test, predictions))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predictions, average='binary' if len(set(y_test)) == 2 else 'micro')
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1-Score:", fscore)
        st.write("Confusion Matrix:", confusion_matrix(y_test, predictions))
    
    # Step 15: Model download
    if st.button('Download Trained Model'):
        buffer = BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        st.download_button(label="Download Model", data=buffer, file_name="trained_model.pkl", mime="application/octet-stream")
# Continue from the previous code

# Assuming `preprocessor` and `model` are already defined and trained

# Step 16: Making predictions with the trained model, including imputation
st.write("## Make Predictions with the Trained Model")

if st.checkbox('Want to make predictions?'):
    input_option = st.radio("How do you want to provide input data for prediction?", ["Input Data Manually", "Upload File"])

    if input_option == "Input Data Manually":
        input_data = {}
        for feature in features:
            value = st.text_input(f"Enter value for {feature} (leave blank if unknown):", '')
            input_data[feature] = value if value != '' else np.nan  # Handle missing values
        input_df = pd.DataFrame([input_data])

        if st.button('Predict'):
            # Apply preprocessing including imputation
            input_processed = preprocessor.transform(input_df)
            prediction = model.predict(input_processed)
            if label_encoder:
                prediction = label_encoder.inverse_transform(prediction)
            st.write(f"Prediction: {prediction}")

    elif input_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a file (CSV) for prediction", type=['csv'])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            if st.button('Predict'):
                # Apply preprocessing including imputation
                input_processed = preprocessor.transform(input_df)
                prediction = model.predict(input_processed)
                if label_encoder:
                    prediction = label_encoder.inverse_transform(prediction)
                
                # Output prediction
                output_df = input_df.copy()
                output_df['Prediction'] = prediction
                st.write("Predictions:")
                st.write(output_df)

                # Option to download predictions
                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
