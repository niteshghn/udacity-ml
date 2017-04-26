#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# for i in data_dict:
#     print i,data_dict[i]['salary']
#     if(data_dict[i]['salary']==26704229):
#         print "match-",i

### your code below
for i,point in enumerate(data):
    salary = point[0]
    bonus = point[1]
    # print salary,bonus,i
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


