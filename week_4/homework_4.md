4.1 overview
4.2 accuracy and dummy model
* interpreting the a binary classification model's accuracy by comparing it's performances at different thresholds 
* dummy model that predicts no customer is churning shows 73% accuracy - only 7 percentage points off the best model. why is that happening? and what does that imply on the use of accuracy as a measuring metric? the data is highly skewed towards non-churners (73% non chruners) - class inbalance, so accuracy can mislead with class imbalance predictions 

4.3 confusion table

True-positive: pred-churn / actual-churn
True-negative: pred-no-churn / actual-no-churn
False-positive: pred-churn / actual-no-churn
False-negative: pred-no-churn / actual-churn

implement confusion matrix in python using "&" operator which computes the element wise logical and 

for homework:
we calculate auc for a logistic regression model 



