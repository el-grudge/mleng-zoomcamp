In week 2️⃣ of the ML Zoomcamp we covered:

🧹🔍📊 Data preparation and exploratory data analysis
We look at the data types, visualize the distribution, identify missing values, and apply transformations if needed (ex. using the logarithmic distribution to deal with rare events at the long tail end of the distribution)

📚🔀✂️ The validation framework
We shuffle the data to introduce randomness and avoid any sequence related patterns, then we split the data into training, validation, and testing datasets. The validation dataset helps us determine whether the model is overfitting and is also used to select the model's hyperparameters. This allows us to use the test data set for an unbiased evaluation of our selected model.

🤖📚📈 Linear regression
We formulate the relationship between the features and weights using the dot product, *the Normal equation*, build a base model (ensuring that we only use numeric features and replace missing values), evaluate the model's performance with RMSE, and use regularization to handle linear combinations (and multicolinearity) to improve the model's performance. (PS: Regularization can also refer to techniques, such as Ridge and Lasso, that prevent overfitting.)

🧬🧩✨ Feature engineering
We create new features and one-hot encode categorical features to enhance the model's performance.

Our homework involved:  
* Exploring the data using commands like `df.head()`, `df.isna()`, and visualizing the target variable using Seaborn's `histplot()` 
* Imputing missing values using mean and 0, and comparing the model's performance with each method
* Observing the effects of regularization and shuffling on the model
* Testing the model on a test dataset

The code for homework 2 can be found [here](https://github.com/el-grudge/mleng-zoomcamp/blob/main/week_2/homework_2.ipynb). 
