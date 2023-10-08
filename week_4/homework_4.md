In week 4️⃣ of the ML Zoomcamp we covered:

⚠️🎯🚨 The limitations of the accuracy metric
We look at the accuracy of a churn classification logistic regression model at different thresholds and observe that a that predicts a 100% churn rate is has an accuracy of 73% - compared to 80% for the most discriminating model. This results from class inbalance in the dataset, and highlights how accuracy as a metric can lead to wrong conclusions.

🎯✅🎖️Confusion table, Precision, and Recall
We explore other metrics, such as preiciosn and recall. Both utilize elements from the confusion table, which tracks true positives (predicted churners who actually churned), true negatives (predicted non-churners who actually did not churn), false positives (predicted churners whoh actually did not churn), and false negatives (predicted non-churners who actually churned). Precision is the fraction of postive predictions that are correct, or the proportion of how many positives the model predicted correctly from all the positive predictions it made (both correct and incorret), while recall is the fraction of correctly identified positive exmamples, or the proportion of predicted positive predictions to all actual positive predictions.

🎯⬇️📈 ROC AUC
We cover another popular metric for classification models, ROC-AUC. ROC curves plot the false postive rate (FPR) against the the true postive rate (TPR). FPR is the fraction of false positives among all negative examples and we want it to be as low as possible, while TPR is the fraction of true positives among all positive examples, and we want it to be as high as possible. Plotting them against each other is a way for us to examine how the model performs at different thresholds and how far / close it is from an ideal model that makes 100% correct preidictions. AUC is the area under the curve and is another measure of the model's performances. It also tells us the probability of a randomly selected positive example having a higher score than a randomly selected negative example

❌✔️🧪Cross validation
To better assess our model we use cross validation, where we train the model k times on different segments of the training data and calculate the average AUC score across the k folds. Cross validation can also be used to find the best values for our hyperparameters, such as the regularization parameter in logistic regression.


Our homework involved:  
* Training a logistic regression model and measuring AUC score accuracy
* Measuring the precision and recall of a logistic regression model
* Calculating the F1 score of a logistic regression model
* Using cross validation and measuring the average AUC score of a model trained with 5 folds

The code for homework 4 can be found [here](https://github.com/el-grudge/mleng-zoomcamp/blob/main/week_4/homework_4.ipynb). 


