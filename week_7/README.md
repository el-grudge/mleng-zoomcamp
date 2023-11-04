## bank marketing

https://archive.ics.uci.edu/static/public/222/data.csv


0 - campaign didn't succeed in converting customer 

1 - campaign did succeed in converting customer 

when the model mistakenly predicts 0 instead of 1 it's a missed opportunity for the bank to make a profit 
when the model mistakenly predicts 1 instead of 0 it's money lost by the bank

precision: tp / (tp + fp) "Of all the instances the model labeled as positive, how many were actually positive?"
recall: tp / (tp + fn) "Of all the instances that were actually positive, how many did the model correctly identify?"

1- data prep
2- basic model definition 
3- basic model evaluation 
	1- measure all metrics 
	2- plot confusion matrix  
	3- decide on model
4- hyperparameter tuning 
	1- threshold
	2- model specific parameters
5- assess on test
6- deploy
	1- save file as pickle 
	2- deploy using flask 
	3- deploy using docker
7- streamlit 
	


## airfare regression 

trying to understand how the business / economy data could be useful:

business/economy data has a date value associated with the flight, all other information is avialable in clean dataset. additionally, more granularity could be added to the arrival / departure time features by adding the exact times (maybe a fligth that departs at 21:00 is more expensive than one that departs at 18:30 - even if both are classified as evening flights). remains to be seen. as a start, clean data should suffice

example record from clean data:
Air_India	AI-868	Delhi	Evening	one	Early_Morning	Chennai	Business    13.33	4	45257

extra features from business data:
25-03-2022			18:00				07:20


next

* look at mutual info score between categorical / price data (maybe convert price data to a categoircal one first)
* look at correlation between numerical data (# of stops, duration, days left) and price - number of stops could be a tricky one since no distinction is made for flights with more than 2 stops  