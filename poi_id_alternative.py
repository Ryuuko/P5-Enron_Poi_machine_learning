#!/usr/bin/python

import sys
import pickle
import math
import matplotlib.pyplot as plt
import numpy
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

features_list = ['poi', 'from_messages', 'total_stock_value', 'deferred_income']   # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below

def computeFraction( numerator, denominator ):
	if math.isnan(float(numerator)) or math.isnan(float(denominator)):
 		fraction = 0
	else:
		fraction = float(numerator)/float(denominator)
	return fraction


def visual(features, labels):
	feature_1_normal = [features[ii][0] for ii in range(0, len(features)) if labels[ii]==0]
	feature_2_normal = [features[ii][1] for ii in range(0, len(features)) if labels[ii]==0]
	feature_1_poi = [features[ii][0] for ii in range(0, len(features)) if labels[ii]==1]
	feature_2_poi = [features[ii][1] for ii in range(0, len(features)) if labels[ii]==1]


#### initial visualization
	plt.scatter(feature_1_normal, feature_2_normal, color = "b", label="non_poi")
	plt.scatter(feature_1_poi , feature_2_poi, color = "r", label="poi")
	plt.legend()
	plt.xlabel("bonus")
	plt.ylabel("Stock")
	plt.show()


for ii, name in enumerate(data_dict):
    data_point = data_dict[name]
    data_point['fraction_of_exercised_stock'] = computeFraction( data_point['exercised_stock_options'], data_point['total_stock_value'])

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

### optional : using AdaBoost, GaussianNB as the base estimator 
#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(base_estimator = GaussianNB())

clf1 = KMeans(n_clusters=2)
clf2 = RandomForestClassifier(random_state=5, max_depth=6)
clf3 = GaussianNB()

Vote_clf = VotingClassifier(estimators=[('knn', clf1), ('rf', clf2),('gnb', clf3)], 
	voting='hard')


# using pipeline to ensure the scaled features will be used in the tester.py

clf = Pipeline(steps=[
	('scaler', MinMaxScaler(feature_range=(0, 1))),
	('Voting', Vote_clf),
	])

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print pred, labels_test
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
print accuracy_score(labels_test, pred)

dump_classifier_and_data(clf, my_dataset, features_list)

