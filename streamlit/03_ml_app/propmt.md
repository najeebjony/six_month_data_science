# propmt 

Hey chat GPT act as an application developer expert, in python
using streamlit, and buildd a machine learning application using
scikit learn with the following workflow

1. Ask the user with a welcome message. and a brief description of the application.
2. Ask the user if he want to upload the data or use the example data.
3. If the user select to upload the data show the upload
section upload on the sidebar, upload the dataset in csv ,xlsx, tsv or any possible data format.
4. If the user do not want to upload the dataset, then provide a 
default dataset selection box on the sidebar. this selection box should download the data from sns.load_dataset() function.
 The datset should include titanic, tips, iris.
5. print the basic data information like such as shape, head, describe, info. and column names.
6. Ask from the user to select the columns as features and alos the columns as target.
7. Identify the problem if the target column is a countinos numeric column the print the message that
this is a regression problem otherwise print the message that this is a classification problem.
8. Pre-process the data, if the data any missing values then fill the missing values with the iterative imputer.
function of scikit-learn, if the feature are not in the same scale then scale the feature using the standard scaler.
function of scikit-learn. if the feartures are categorical variables the encode the categorical variables using the lable encoder function of scikit-learn. plase keep in mind to keep the encoder separate for each column as we need to inverse transform the data at the end.
9. Ask the user to provied train test split size via slider or user input fuction 
10. Ask the user to select the model from the sidebar , the model should include liner regression dicision tree, random forcast and suport vector machine. and same classes of model for the classification problem.
11. Train the models on the traning data and evaluate on the test data.
12. If the problem is a regression problem then print the mean absolute error, mean squared error, root mean squared error, r2 score, evluation, if the problem is a classification problem then print the accuracy, precision, recall, f1-score and draw confusion metric 
for evaluation.
13 Print the evulation matrix for each model.
14. Highlight the best model based on the evaluation matrix.
15. Ask the user if he want to download the model, if the user select yes then download the model using the pickle library.
16. Ask the user if he want to make the prediction, if yes then ask the user to provide the input data slider or uploaded file and make the prediction using the best model 


