#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'


customer_id = 'xyz-123'
customer = {
'age':39,
'job':'technician',
'marital':'single',
'education':'unknown',
'default':'no',
'balance':45248,
'housing':'yes',
'loan':'no',
'contact':'unknown',
'day_of_week':6,
'month':'may',
'duration':1623,
'campaign':1,
'pdays':-1,
'previous':0,
'poutcome':'unknown'
}


response = requests.post(url, json=customer).json()
print(response)

if response['get_card'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)
