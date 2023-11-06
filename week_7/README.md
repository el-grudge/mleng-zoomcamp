## Bank Marketing Problem Description

In 2012 a Portuguese banking institution collected data for several direct marketing campaigns it conducted in order to analyze it and to build machine learning models that can increase the efficiency of future marketing campaigns.

A marketing campaign is a concentrated effort by the bank in which it contacts its customers by phone and asks them to subscribe to a term deposit. Term deposits, aka certificate depoists, are deposits by customers that are made for a specific period of time and tradionally return more interest than savings accounts. They provide a guarantee for the banks that the money will remain available for a known period of time, which helps them better manage their available capitol.

In this project, I'll be using the this dataset which can be downloaded from the UCI repository [here](https://archive.ics.uci.edu/dataset/222/bank+marketing)

My goal is to train an ML model that can predict whether a customer will subscribe to a term deposit. I'll priortize profit making over regulating spending. In other words, I'll prefer a model with a lower false negative rate over one with a lower false positive rate. 

The dataset has 16 features, and one target variable: 

| Variable Name | Role     | Type        | Demographic       | Description | Units | Missing Values |
|---------------|----------|-------------|-------------------|-------------|-------|----------------|
| age           | Feature  | Integer     | Age               |             |       | no             |
| job           | Feature  | Categorical | Occupation        |             |       | no             |
| marital       | Feature  | Categorical | Marital Status    |             |       | no             |
| education     | Feature  | Categorical | Education Level   |             |       | no             |
| default       | Feature  | Binary      |                   | has credit in default? | | no |
| balance       | Feature  | Integer     |                   | average yearly balance | euros | no |
| housing       | Feature  | Binary      |                   | has housing loan? | | no |
| loan          | Feature  | Binary      |                   | has personal loan? | | no |
| contact       | Feature  | Categorical |                   | contact communication type (categorical: 'cellular','telephone') | | yes |
| day_of_week   | Feature  | Date        |                   | last contact day of the week | | no |
| month         | Feature  | Date        |                   | last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec') | | no |
| duration      | Feature  | Integer     |                   | last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model. | | no |
| campaign      | Feature  | Integer     |                   | number of contacts performed during this campaign and for this client (numeric, includes last contact) | | no |
| pdays         | Feature  | Integer     |                   | number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted) | | yes |
| previous      | Feature  | Integer     |                   | number of contacts performed before this campaign and for this client | | no |
| poutcome      | Feature  | Categorical |                   | outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success') | | yes |
| y             | Target   | Binary      |                   | has the client subscribed a term deposit? | | |:

Plan layout:  
1- Data preparation  
2- Exploratory data analysis  
3- Feature Engineering / Transformations
4- Model training and assessment  

#### Running the code 

You can run the docker image of the app using this command:  

`docker run -it --rm -p 9696:9696 bank-marketing`  

To test the app, run the following command in a separate terminal:  

`python predict-test.py`