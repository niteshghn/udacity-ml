#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count_email = count_salary = count_total_payment_not_nan = count_poi = count_poi_with_nan_payment = 0
count_total = 0
stocks = 0




for item in enron_data:
    count_total = count_total + 1
    if (enron_data[item]['email_address'] != 'NaN'):
        count_email = count_email + 1
    if (enron_data[item]['salary'] != 'NaN'):
        count_salary = count_salary + 1
    if (enron_data[item]['total_payments'] == 'NaN'):
        count_total_payment_not_nan = count_total_payment_not_nan + 1
    if (enron_data[item]['poi'] == 1):
        count_poi = count_poi + 1
    if (enron_data[item]['poi'] == 1 and enron_data[item]['total_payments'] == 'NaN'):
        count_poi_with_nan_payment = count_poi_with_nan_payment + 1



print 'total_data-' + str(count_total) + '\n count_email_without_nan-' + str(
    count_email) + '\n count_salary_without_nan-' + str(count_salary) + '\n count_total_payment_nan-' + str(
    count_total_payment_not_nan) + '\n count_poi-' + str(count_poi) + '\n poi_with_nan_payment-' + str(
    count_poi_with_nan_payment)
# count=count+1
# if(item['poi']==1):
#     count=count+1
# print enron_data['PRENTICE JAMES']['total_stock_value']
# print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
# print enron_data['SKILLING JEFFREY K']['total_payments']
# print enron_data['FASTOW ANDREW S']['total_payments']
# print enron_data['LAY KENNETH L']['total_payments']
# print count
