#  Student name:           Khafiz
#  Student sir name:       Khader
#  Student ID:             202041946


 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Training phase of the linear regression model 
# You can use  LinearRegression function from SciKit Learn which is RFE

# DO YOUR CODING HERE 

csv_data = pd.read_csv('C:\\Users\\gadyl\\Desktop\\CSCI 390\\Assigment 6\\Housing.csv')

special_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for column in special_columns:
    csv_data[column] = [1 if value == 'yes' else 0 for value in csv_data[column]]


f_status = pd.get_dummies(csv_data['furnishingstatus'])
f_status = f_status.drop('furnished', axis=1)  # Avoid dummy variable trap
csv_data = pd.concat([csv_data, f_status], axis=1).drop('furnishingstatus', axis=1)

X = csv_data.drop('price', axis=1)
y = csv_data['price']

X_train, X_test, y_train, ground_truth = train_test_split(X, y, test_size=0.08, random_state=25)


linreg_model = LinearRegression()
rfe = RFE(linreg_model, n_features_to_select=9)
rfe = rfe.fit(X_train, y_train)

# Start your testing phase here

# DO YOUR CODING HERE 
estimated_label = rfe.predict(X_test)


# Report you accuracy with this Sklearn function r2_score. Pass the testing data and its ground truth labels
# to the function
# Pass your testing data for accuracy measurment
from sklearn.metrics import r2_score 
r2_score(ground_truth, estimated_label)

accuracy = r2_score(ground_truth, estimated_label)

print(accuracy)

