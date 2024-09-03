# Car_Sales_Data
Car Sales Dataset Analysis Using Regression Methods


Key Features
1.	Data Cleaning: The dataset undergoes cleaning to make variables more useful for analysis.
2.	Applying Regression Models: The code applies regression techniques such as Linear, Lasso, and Ridge regressions to analyze the data.
3.	Calculating Performance Metrics: To determine the best model for describing the data, key indicators such as R², MAE (Mean Absolute Error), and MSE (Mean Squared Error) are calculated.
4.	Generating Plots: Various plots are generated to visualize how well the predicted values align with the actual values.
5.	Producing a Summary Table: A final table is created summarizing all performance indicators to aid in selecting the most suitable model.

How to Use the Code:

1.	Extract the file Carsales-Regression.py.
2.	Load the dataset.
3.	Run the script using any Python environment of your choice.
   
Interpretation of Plots:

•	Linear Regression Plot: A scatter plot is generated displaying the regression predictions. The red line represents the case where predictions would exactly match the actual test values (test_y). Most points lie close to the line y = x, indicating that the linear regression model performs well.
 
•	Ridge and Lasso Regression Plots: The subsequent plots illustrate how the predictions from Ridge and Lasso regressions closely follow the actual values of the target variable Selling. Both regression models capture the variability in the Selling variable to a satisfactory degree.
 	 
Table of Results:

The table below shows that the Linear Regression model has the highest R² value. Additionally, this model also exhibits the lowest MSE and MAE, making it the most reliable for our analysis.

 ![image](https://github.com/user-attachments/assets/f3f36676-d708-4031-aa7f-ebc1dec2e00f)


