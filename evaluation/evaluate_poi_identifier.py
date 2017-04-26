#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn import tree

from sklearn.model_selection import train_test_split

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
counter=total= 0
feature_train,feature_test,label_train,label_test = train_test_split(features,labels,test_size=0.3,random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(feature_train,label_train)
empty_list=[]
for i in label_test:
    total+=1
    empty_list.append(0.0)
    if i==1.0:
        counter+=1

# print counter,total
print "actual---\n",label_test
pred = clf.predict(feature_test)
print "predicted---\n",pred
# print accuracy_score(empty_list,pred)
# print set(label_test) & set(pred)
# for poi in label_test:
#     if poi==pred
print [i for i, j in zip(pred, label_test) if i == j]

print precision_score(label_test,pred)
print recall_score(label_test,pred)
