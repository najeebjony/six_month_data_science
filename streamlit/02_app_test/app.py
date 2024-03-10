import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import warnings 
warnings.filterwarnings('ignore')



# add title 
st.title("Application for data Analysis")
st.subheader("This is a simple data analysis application created by @Najeeb-ullah")
st.title("Exploratory Data Analysis")

# load the data 
datasets = ['iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox('Select a dataset', datasets)

if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')
    
    
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "txt"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# display the data
st.write(df.head())

# display the shpe 
st.write('Number of rows:', df.shape[0])

# display the number of rows and column from the selected data
st.write('Number of columns:', df.shape[1])


#missinag value checked 
if df.isnull().sum().any() > 0:
    st.write('Null Values:', df.isnull().sum().sort_values(ascending=False))  
else:
    st.write('No Null Values')




# display the unique values in the dataset
st.write('Unique Values:', df.nunique())

# display the summary statistics of the dataset
st.write('Summary Statistics:', df.describe())

# display the data types of the dataset
st.write('Data Types:', df.dtypes)

# select the sepecif column For X or y axis from the dataset and also select the plot type the data 
x_axis = st.selectbox('X Axis', df.columns)
y_axis = st.selectbox('Y Axis', df.columns)
plot_type = st.selectbox('Plot Type', ['scatter', 'line', 'bar', 'histogram', 'kde', 'box', 'violin'])

# create the plot
if plot_type == 'scatter':
    plt.scatter(df[x_axis], df[y_axis])
elif plot_type == 'line':
    plt.plot(df[x_axis], df[y_axis])
elif plot_type == 'bar':
    plt.bar(df[x_axis], df[y_axis])
elif plot_type == 'histogram':
    plt.hist(df[x_axis])
elif plot_type == 'kde':
    sns.kdeplot(df[x_axis])
elif plot_type == 'box':
    plt.boxplot(df[x_axis])
elif plot_type == 'violin':
    plt.violinplot(df[x_axis])
    plt.title(f'{
        plot_type} plot of {x_axis} vs {y_axis}')
else:
    st.write('Invalid plot type')

# display the plot
st.pyplot()

# create a pairplot
st.subheader("Pairplot")
# select the column to be used for the pairplot size
column = st.selectbox("Select a column to see its distribution", df.columns)
st.write(sns.pairplot(df, hue=column, diag_kind='kde' if column != "total" else 'hist'))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


# create a heatmap
st.subheader("Heatmap")
# select the columns which are numeric and then create a heatmap
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
st.write(sns.heatmap(df[numeric_columns].corr(), annot=True))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
