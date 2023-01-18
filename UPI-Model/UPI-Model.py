import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Loading CSV file in the model
data = pd.read_csv('E:/Project_Exhi-2/UPI-Model/Data.csv')
data.columns = (["Customers","Amount","Month","Year"])


# data.Amount = data.Amount.apply(eval)
data.Customers = data.Customers.apply(lambda x: float(x.replace(",","")))
data.Amount = data.Amount.apply(lambda x: float(x.replace(",","")))
data.Month =data.Month.astype(int)
data.Year =data.Year.astype(int)


#Visualizing the available dataset and previous trends of GDP
fig=plt.figure(figsize=(12,4))
data.groupby('Year')['Amount'].mean().sort_values().plot(kind='bar', color='coral')
plt.title('Plot of UPI payment users on various platforms')
plt.xlabel("Year")
plt.ylabel("Customers")
plt.show()

#Region Transform
data_final= pd.concat([data,pd.get_dummies(data['Amount'], prefix='Amount')],axis=1).drop(['Amount'],axis=1)

#data split: all of out final dataset, with no scaling.

y=data_final['Year']
X=data_final.drop(['Year','Customers'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=101)


# #Model Training-

# Linear Regression-
svm1 = SVR(kernel='rbf')
svm1.fit(X_train,y_train)
svm1_pred = svm1.predict(X_test)

# Evaluating the results and values:-
print('\nSVM Performance:')

print('\nall features, No scaling:')
print('MAE:', metrics.mean_absolute_error(y_test, svm1_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svm1_pred)))
print('R2_Score: ', metrics.r2_score(y_test, svm1_pred))

"""
SVM Performance:

all features, No scaling:
MAE: 0.37428237120868435
RMSE: 0.47103286740509437
R2_Score:  0.0852199532062462
"""

# # Random Forest -
# rf1=RandomForestRegressor(random_state=101, n_estimators=200)


# rf1.fit(X_train, y_train)

# #Predicting using predict() function;
# rf1_pred= rf1.predict(X_test)
# print("\nRandom Forest Performance:")
# print("\nAll features, no scaling:")
# print("MAE:",metrics.mean_absolute_error(y_test,rf1_pred))
# print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, rf1_pred)))
# print("R2_Score: ", metrics.r2_score(y_test, rf1_pred))