#!/usr/bin/python
import numpy
import sys
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

from outlier_cleaner import outlierCleaner

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

poi_array = []
salary_array = []
bonus_array = []

for i,j,k in data:
    poi_array.append(i)
    salary_array.append(j)
    bonus_array.append(k)

print poi_array,salary_array.index(max(salary_array)),bonus_array.index(max(bonus_array))

# remove the outlier
del poi_array[87]
del salary_array[87]
del bonus_array[87]

#after formating

salary_data = numpy.array(salary_array).reshape(-1,1)
bonus_data = numpy.array(bonus_array).reshape(-1, 1)

salary_train, salary_test, bonus_train, bonus_test = train_test_split(salary_data, bonus_data, test_size=0.3, random_state=42)
# salary_test = numpy.array(salary_test).reshape(-1,1)

# print len(salary_data),len(bonus_data)
# regression



from sklearn.linear_model import LinearRegression
# print len(salary_array),len(bonus_array)
reg = LinearRegression()
# print salary_train,bonus_train
reg.fit(salary_train,bonus_train)
pred = reg.predict(salary_test)
print pred
# try:
#     plt.plot(salary_test, pred, color="blue")
# except NameError:
#     pass
plt.scatter(salary_array, bonus_array)
plt.xlabel("poi")
plt.ylabel("salary")
# plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(salary_train)
    cleaned_data = outlierCleaner( predictions, salary_train, bonus_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    salary, bonus, errors = zip(*cleaned_data)
    salary       = numpy.reshape( numpy.array(salary), (len(salary), 1))
    bonus = numpy.reshape( numpy.array(bonus), (len(bonus), 1))

    ### refit your cleaned data!
    try:
        reg.fit(salary, bonus)
        score = reg.score(salary_test,bonus_test)
        print "new slop--",reg.coef_
        print "score w/o",score
        # plt.plot(salary, reg.predict(salary), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(salary, bonus)
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.title("Fig2")
    # plt.show()
else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"



#======================== Classification part starts here ==============================


labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
param_grid = {'n_estimators': [1,10,100,1000, 5000, 10000, 50000, 100000],
              'max_depth': [1,10, 100, 1000, 5000, 10000,50000,100000] }
clf = GridSearchCV(RandomForestClassifier(), param_grid)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:%r with accuracy: %s" %(clf.best_estimator_ ,accuracy_score(labels_test,pred)))
print "actual value:-", labels_test
print "predicted value:-",pred
print ("precision score : %s and recall score : %s" %(precision_score(labels_test,pred),recall_score(labels_test,pred)))
# print(clf.best_estimator_)
# clf = SVC()
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# score = accuracy_score(labels_test,pred)
#
# print score

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)