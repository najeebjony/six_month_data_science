import streamlit as st
import seaborn as sns
import pandas as pd

# create a title
st.title("Exploratory Data Analysis")
st.subheader("This is a simple data analysis application created by @Najeeb-ullah")
# create a dropdown list to choose a datast
dataset_name =  ["iris", "titanic","diamonds","tips"]
selected_dataset = st.selectbox( "Select a dataset", dataset_name)

# load the dataset
if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
    st.title("Exploratory Data Analysis")
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
    st.title("Exploratory Data Analysis")
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')
    st.title("Exploratory Data Analysis")
else:
    df = sns.load_dataset('tips')
    st.title("Exploratory Data Analysis")

# button to upload custom dataset
uploaded_file = st.file_uploader("Upload a file", type=["csv","xlsx","txt"])
if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
# display the dataset
st.write(df)
# display the number of rows and columns in the dataset
st.write('Number of Rows:', df.shape[0])
st.write('Number of Columns:', df.shape[1])

# diplay the columns names of selected dataset with their data types
st.write('Columns Names and Data Types:', df.dtypes)

#print the null values if those are >0
if df.isnull().sum().any() > 0:
    st.write('Null Values:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Null Values')

# display the unique values in the dataset
st.write('Unique Values:', df.nunique())

# display the summary statistics of the dataset
st.write('Summary Statistics:', df.describe())




# Create a pairplot to visualize the relationship between the columns
st.subheader("Pairplot")
# select the column to be used for the pairplot
column = st.selectbox("Select a column to see its distribution", df.columns)
st.write(sns.pairplot(df, hue=column, diag_kind='kde' if column != "total" else 'hist'))
st.pyplot()

# Create a heatmap to visualize the correlation between the columns
st.subheader("Heatmap")
# select the columns which are numeric and then create a heatmap
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
st.write(sns.heatmap(df[numeric_columns].corr(), annot=True))
st.pyplot()

# # Create a heatmap to visualize the correlation between the columns using plotly
# st.subheader("Heatmap using Plotly")
# import plotly.express as px
# fig = px.imshow(df[numeric_columns].corr())
# st.write(fig)
# st.plotly_chart(fig)

