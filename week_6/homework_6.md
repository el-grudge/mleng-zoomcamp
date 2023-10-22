In week 6️⃣ of the ML Zoomcamp we covered:

💡🌳🔀 Decision Trees 
We learn the intuition behind Decision Tree modelling. These models use cascading if-then statements to split the training data into buckets. Mid-level buckets - those that are further split into smaller buckets - are called nodes, while terminal buckets - those that are not split into further buckets - are called leaves. Predictions are made at the leaves. The model's performance is evaluated using the misclassification rate metric, which measures the level of impurity (the number of wrong predictions) at each leaf. Other alternative metrics such as gini and entropy can also be used to evaluate the model.

🌳🏃🔧 Training Decision Trees 
We train a Decision Tree model to predict the credit risk of loan applications, and we see how it overfits the training data, that is, we observe a deterioration in the model's performace when applied to samples from the validation set. We then learn about two hyper-parameters, `max_depth`, the maximum depth a tree is allowed to grow to, and `min_samples_leaf`, the least amount of samples that can be included in a leaf node. Tuning these hyper-parameters helps us train a better model.

🌳🎲🌳 Random Forests 
We learn about ensemble models. We start with Random Forests, which are a collection of Decision Trees trained on different random subsets of the training features and whose results are aggregated to make a final prediction, and how to tune two of it's hyper-parameters, `n_estimators`, which is the number of Decision Trees that form the forest, and the `max_depth` (see previous point).

🧱🌳🧱 Boosting
We then discuss XGBoosting, another form of ensemble models. These models stack Decision Trees sequentially, where each additional tree improves the overall performance by accounting for the prediction errors made by the that precedes it. We use the XGBoost library to train one such model, and we learn how to structure the data in a `DMatrix` to optimize training. Finally, we tune the `eta` hyper-parameter, aka learning rate, that regulates the weights of new features to avoid overfitting.

Our homework involved:  

* Training a Decision Tree regressor model to predict median home value and identifying which features were used to split the data
* Training a Random Forest regressor model and calculating it's RMSE 
* Tuning the Random Forest regressor's n-estimator hyper-parameter
* Tuning the Random Forest regressor's n-estimator and max_depth hyper-parameters
* Extracting the feature importance of the Random Forest model 
* Training two XGBoost models with different learning rates (eta) and comparing their performance

#mlzoomcamp #ml_engineering #data_science #learning_in_public #boosting #decision_trees

The code for homework 6 can be found [here](https://github.com/el-grudge/mleng-zoomcamp/blob/main/week_6/homework_6.ipynb). 
