#import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import arange
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
scale = StandardScaler()

#import data from csv file
d1 = pd.read_csv("carsales.csv")

#converitng strings variable into numbers
d1['Fuel_Type'].replace(['Petrol', 'Diesel','CNG'],
                        [1,2,3], inplace=True)
d1['Seller_Type'].replace(['Dealer', 'Individual'],
                        [1,2], inplace=True)
d1['Transmission'].replace(['Manual', 'Automatic'],
                        [1,2], inplace=True)

#keep only strings variable
df1=d1[['Fuel_Type','Seller_Type','Transmission']]
result = df1.to_numpy()

#keep only the dependent variable
y=d1[['Selling_Price']].values

#keep only quantative variable in order to scale them with scale.fit
d1.drop(['Fuel_Type','Seller_Type','Transmission','Car_Name','Selling_Price'],axis='columns',inplace=True)
data = d1.values
X=data[:,:]
X = scale.fit_transform(X)

#make a new dataframe in order to use it as X_train and X_test
result1 = np.concatenate((X, result), axis=1)
df2 = pd.DataFrame(result1, columns=['Year_sc', 'Pr_Price_sc', 'Kms_dr_Sc', 'Owner_Sc','Fuel','Seller','Transmissiom'])

#Now we have our data separation
#Our algorithms will use the 75% of our data as train dat and 25% as test data
#So in the parenthesis we have train_size=0.75
x_train, x_test, y_train, y_test = train_test_split(result1, y,train_size=0.75)

#For this exercise we use also a 10 cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

     ---- #CALCULATE MACHINE LEARNING MODELS FOR OUR DATA ----

#Linear Regression
modelLinear = LinearRegression()

#fit the linear regrassion model in our train data
modelLinear = LinearRegression().fit(x_train, y_train)

#make predictions for x_test data 
predictions = modelLinear.predict(x_test)

#Calculate the most useful indicators 
#R2
#In this step we will calculate the R2 for the two data (train and test) 
#With this procedure we wiil be able to understand if our data are ovetraining
R2_lr_tr = modelLinear.score(x_train, y_train)
R2_lr_test = modelLinear.score(x_test, y_test)
#MSE
mse_lr = mean_squared_error(y_test, predictions)
#MAE
mae_lr = mean_absolute_error(y_test, predictions)

#scatter plot
plt.figure()
plt.scatter(predictions, y_test, color='blue', label='Actual vs Predicted')
plt.plot(predictions, predictions, color='red', label='Perfect Prediction') 
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Scatterplot of Predicted vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()

#Ridge Regression
RidgeModel = RidgeCV(alphas=arange(0.01, 1000, 5), cv=cv) #10000 50

#fit the ridge model 
RidgeModel.fit(x_train, y_train)

#finding the best alpha for ou model
print('Ridge best alpha (lambda): %f' % RidgeModel.alpha_)
RidgeModel.coef_

#make predictions for x_test data
predictions1 = RidgeModel.predict(x_test)

#Calculate the same indicators as previous 
R2_ridge_tr = RidgeModel.score(x_train, y_train)
R2_ridge_test = RidgeModel.score(x_test, y_test) 
mse_Ridge = mean_squared_error(y_test, predictions1)
mae_Ridge = mean_absolute_error(y_test, predictions1)

#This plot will show how the predicted values compare with the actual values
plt.figure()
plt.plot(predictions1, label='Predicted')
plt.plot(y_test, color='red', label='Actual')  # Plotting y_test vs. y_test for reference
plt.xlabel('Actual test_y')
plt.ylabel('Predicted test_y_bestalpha')
plt.title('Ridge Regression Predictions vs. Actual Predictions')
plt.legend()
plt.show()                           

# Lasso
LassoModel = LassoCV(alphas=arange(0, 10, 0.1), cv=cv, n_jobs=-1,max_iter =1000)

#fit the lasso model
LassoModel.fit(x_train,y_train)

#make predictions for x_test data
predictions2 = LassoModel.predict(x_test)

#Calculate the same indicators as previous 
R2_lasso_tr =LassoModel.score(x_train,y_train)
R2_lasso_test =LassoModel.score(x_test,y_test)
mse_Lasso = mean_squared_error(y_test, predictions2)
mae_Lasso = mean_absolute_error(y_test, predictions2)

#This plot will show how the predicted values compare with the actual values
plt.figure()
plt.plot(predictions2, label='Predicted')
plt.plot(y_test, color='red', label='Actual')  # Plotting y_test vs. y_test for reference
plt.xlabel('Actual test_y')
plt.ylabel('Predicted test_y_bestalpha')
plt.title('Lasso Regression Predictions vs. Actual Predictions')
plt.legend()
plt.show() 


#Make a new table with all the indicators which indicate the best model 
data = {
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'R2': [R2_lr_test, R2_ridge_test, R2_lasso_test],
    'MSE': [mse_lr, mse_Ridge, mse_Lasso],
    'MAE': [mae_lr, mae_Ridge, mae_Lasso]
}
results_df = pd.DataFrame(data)
print(results_df)

