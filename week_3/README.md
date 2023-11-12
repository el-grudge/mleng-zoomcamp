In week 3️⃣ of the ML Zoomcamp we covered:

📦✅❌ Churn prediction using logistic regression
We use logistic regression to classify whether a telcom customer would churn. This is a binary classification problem where the outcome is either 0 (not churning) or 1 (churning)

🔦🔍📊 Feature importance
We explore the data and analyze the relationships between the target variable and the various features within the dataset. We learn about churn rate, which is the difference between the means of the target variable and the categories of a given feature, and the risk ratio, which  uses ratio instead of difference to determine the churn likelihood of a given category. 

🔗📈📉 Mutual information score and correlation matrix 
We measure the mutual dependence between categorical variables and the target variable using the mutual information score, which quantifies the amount of knowledge we can extract on one variable by observing another. For numerical variables, we use a correlation coefficient.

1️⃣0️⃣🔒 Handling categorical variables
We use one-hot encoding (OHE) to convert categorical variables to numerical ones. When using OHE, a single categorical variable of length "l" is transformed into a matrix of size "l" rows by n columns, where n is equal to the number of unique categories, and values of either 0 or 1, corresponding to the existence of a category for a specific observation.

💡➕🧮 The Sigmoid function 
We learn the difference between linear and logistic regression. Both are supervised linear models that calculate the sum of the bias term and the weighted features. However, they differ in their output, linear regression outputs a real number, whereas logistic regression passes the output of linear regression equation to the Sigmoid function to produce a number between 0 and 1.

![Local Image](pictures/scikit-learn-logo-small.png) Logistic regression with Scikit-learn
We use Scikit-learn's implementation of the logistic regression model and learn how to use the `predict_proba` function to get the class probabilities, which can be used to determine the classification by comparing them to a specified prediction threshold.

🧠🤖🤔 Model interpretation
We look at a simplified implementation of logistic regression to understand how the model works and how to interpret the weights of the various variables.

Our homework involved:  
* Finding the most frequent value of a categorical variable using the `value_counts()` function
* Creating the correlation matrix and calculating the mutual information score
* Training a logistic regression model and measuring it's accuracy
* Determining feature importance using the feature elimination technique

The code for homework 3 can be found [here](https://github.com/el-grudge/mleng-zoomcamp/blob/main/week_3/homework_3.ipynb). 