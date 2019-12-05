"""
Data Mining & Visualization Final Project

Comparing Various Regression Models

@authors: Bradley Schoeneweis, Hau Ha, Minh Nguyen
"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""Data Preprocessing"""
# Importing the Dataset
dataset = pd.read_csv('data/Admission.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


"""Multiple Linear Regression"""
# Building the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualizing the results
y_pred = regressor.predict(X_test)
plt.scatter(X_test[:, 1], y_test, color='red')
plt.scatter(X_test[:, 1], y_pred, color='blue')
plt.title('Multiple Linear Regression')
plt.show()

# RMSE
from sklearn.metrics import mean_squared_error
rmse_linreg = np.sqrt(mean_squared_error(y_test, y_pred))

# Building the optimal model using Backward Elimination
import statsmodels.api as sm

# Add b0
X_bias = np.append(arr=np.ones((500,1)), values=X, axis=1)

# Resplit with bias
X_train_bias, X_test_bias, y_train_bias, y_test_bias = train_test_split(
    X_bias, y, test_size=0.25, random_state=0)

X_opt_train = X_train_bias[:, [0, 1, 2, 5, 6, 7]]
X_opt_test = X_test_bias[:, [0, 1, 2, 5, 6, 7]]
regressor_OLS = sm.OLS(endog=y_train_bias, exog=X_opt_train).fit()
regressor_OLS.summary()

# Visualizing the results
y_pred = regressor_OLS.predict(X_opt_test)
plt.scatter(X_test_bias[:, 1], y_test_bias, color='red')
plt.scatter(X_test_bias[:, 1], y_pred, color='blue')
plt.title('Optimized Multiple Linear Regression')
plt.show()

# RMSE
rmse_linreg_opt = np.sqrt(mean_squared_error(y_test, y_pred))


"""Support Vector Regression"""
# Feature Scaling (required for SVR).
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled_train = sc_X.fit_transform(X_train)
X_scaled_test = sc_X.fit_transform(X_test)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel='rbf')
regressor_svr.fit(X_scaled_train, y_train)

# Visualising the results
y_pred = regressor_svr.predict(X_scaled_test)
plt.scatter(X_scaled_test[:, 1], y_test, color='red')
plt.scatter(X_scaled_test[:, 1], y_pred, color='blue')
plt.title('Support Vector Regression')
plt.show()

# RMSE
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred))


"""Decision Tree Regression"""
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(random_state=0)
regressor_dt.fit(X_train, y_train)

# Visualizing the results
y_pred = regressor_dt.predict(X_test)
plt.scatter(X_test[:, 1], y_test, color='red')
plt.scatter(X_test[:, 1], y_pred, color='blue')
plt.title('Decision Tree Regression')
plt.show()

# RMSE
rmse_dtr = np.sqrt(mean_squared_error(y_test, y_pred))


"""Random Forest Regression"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=100,
                                     max_leaf_nodes=5,
                                     random_state=0)
regressor_rf.fit(X_train, y_train)

# Visualizing the results
y_pred = regressor_rf.predict(X_test)
plt.scatter(X_test[:, 1], y_test, color='red')
plt.scatter(X_test[:, 1], y_pred, color='blue')
plt.title('Random Forest Regression')
plt.show()

# RMSE
rmse_rfr = np.sqrt(mean_squared_error(y_test, y_pred))


"""Comparing the Regression Models"""
from collections import OrderedDict
data = OrderedDict()
data['Random Forest Regression\n(%s)' % rmse_rfr] = rmse_rfr
data['Decision Tree Regression\n(%s)' % rmse_dtr] = rmse_dtr
data['Support Vector Regression\n(%s)' % rmse_svr] = rmse_svr
data['Optimized Linear Regression\n(%s)' % rmse_linreg_opt] = rmse_linreg_opt
data['Multiple Linear Regression\n(%s)' % rmse_linreg] = rmse_linreg_opt

names = list(data.keys())
values = [x * 1000.0 for x in data.values()]
y_pos = np.arange(len(names))
plt.barh(y_pos, values) 
plt.yticks(y_pos, names)
plt.show()
