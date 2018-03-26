#!/usr/bin/python

import sys
import pickle
import math
import numpy
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'from_messages', 'total_stock_value', 'deferred_income'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



features_valid = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
#delect email address, since if we use isnan to testify if the data point, we need float form, which we can not use on sting
#delect poi, it will return either true or false but never nan!

# check if there is a datapoint with Null datapoint
def nan_checker(data_dict, number):
	# use to find out all nan datapoint:
	for name, data in data_dict.iteritems():
		i = 0
		for x in features_valid:
			if math.isnan(float(data[x])):
				i += 1
		if i == number: # change the number if you're unsatisfied by the data point with too many nans
			print name
			print data

# check if the remover works
def pop_checker(name_deleted):
	for data in data_dict:
		if data != name_deleted:
			continue
		else:
			print "it doesn't work"

### optional: check out the all nan outliners
#nan_checker(data_dict, len(features_valid))  #fina out the person who fail all of the features

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### optional: check out if the pop function works
#print pop_checker("LAY KENNETH L")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
def computeFraction( numerator, denominator ):
	if math.isnan(float(numerator)) or math.isnan(float(denominator)):
 		fraction = 0
	else:
		fraction = float(numerator)/float(denominator)
	return fraction

for ii, name in enumerate(data_dict):
    data_point = data_dict[name]
    data_point['fraction_of_exercised_stock'] = computeFraction( data_point['exercised_stock_options'], data_point['total_stock_value'])

my_dataset = data_dict

#### optional: Uncomment the following code to use SelectKBest to search for best features

# features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
# 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# data = featureFormat(my_dataset, features_list, sort_keys = True)
# label, features = targetFeatureSplit(data)

# features_train, features_test, labels_train, labels_test = train_test_split(features, label, test_size=0.1, random_state=42)

# sfm = SelectKBest(f_classif, k=3).fit(features_train,labels_train)

# print sfm.scores_

# index = sfm.get_support(indices=True)
# print index
# features_list_best = features_list[1:]
# for i in index:
#       print features_list_best[i] #print the features with best k socres

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### optional: example of using GridSearchCV, if we use DecisionTree instead
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#param_grid = { 
#    'n_estimators': [10, 40, 50, 70, 100]}

#cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
#cv_clf.fit(features_train, labels_train)
#print cv_clf.best_params_

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)