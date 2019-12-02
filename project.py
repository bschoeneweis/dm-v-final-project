#Step 1: Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Admission_Predict.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 8].values
#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#standardScalerX = StandardScaler()
#X = standardScalerX.fit_transform(X)

# 1 Simple Linear Regression

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Visualising the results
plt.scatter(X[:,1], y, color = 'red')
plt.scatter(X[:,1], regressor.predict(X), color = 'blue')
plt.show() 

# 2 Backward Elimination

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((400,1)), values = X , axis = 1)

X_opt = X[:, [0, 1, 2, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Visualising the results
plt.scatter(X[:,1], y, color = 'red')
plt.scatter(X[:,1], regressor_OLS.predict(X_opt), color = 'blue')
plt.show() 

# 3 SVR

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf')
regressor_svr.fit(X, y)

# Visualising the results
plt.scatter(X[:,1], y, color = 'red')
plt.scatter(X[:,1], regressor_svr.predict(X), color = 'blue')
plt.show() 

# 4 Decision Tree Regression 

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(random_state = 0)
regressor_dt.fit(X, y)

# Visualising the results
plt.scatter(X[:,1], y, color = 'red')
plt.scatter(X[:,1], regressor_dt.predict(X), color = 'blue')
plt.show() 

# 5 Random Forrest

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 100, max_leaf_nodes = 5, random_state = 0)
regressor_rf.fit(X, y)

# Visualising the results
plt.scatter(X[:,1], y, color = 'red')
plt.scatter(X[:,1], regressor_rf.predict(X), color = 'blue')
plt.show() 