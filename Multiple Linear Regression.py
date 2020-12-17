###############################################################################
#######################  I> IMPORT THE LIBRARIES  ############################# 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm

###############################################################################
#######################  II> IMPORT THE DATASET & DATA PREPROCESSING   ######## 
###############################################################################

dataset = pd.read_csv('E:\\MACHINE LEARNING SUMMARY\\ML DATASETS\\50_Startups.csv')            
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

###############################################################################
######################  III> ENCODE CATEGORICAL DATA ##########################
###############################################################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder                  
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

###############################################################################
###################### IV> SPLIT DATASET INTO TRAIN AND TEST SETS  ############
###############################################################################

from sklearn.model_selection import train_test_split                     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

###############################################################################
######################  V> AVOID DUMMY VARIABLES ##############################
###############################################################################

X = X[:, 1:]                                                       

###############################################################################
####################### VI> FIT ML MODEL TO TRAINING SET ######################
###############################################################################

from sklearn.linear_model import LinearRegression                 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

###############################################################################
####################### VII> PREDICT TEST SET RESULTS #########################
###############################################################################
    
y_pred = regressor.predict(X_test)                               

###############################################################################
#######################  VIII> BACKWARD ELIMINATION- to build an optimal model 
###############################################################################

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)       
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()         
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()




