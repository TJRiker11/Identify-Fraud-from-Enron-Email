#!/usr/bin/python

import sys
import pickle
sys.path.append("Enron Data/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.


features_list = [
 'poi',
 'deferral_payments',
 'director_fees',
 'exercised_stock_options',
 'from_this_person_to_poi',
 'loan_advances',
 'other',
 'restricted_stock_deferred',
 'salary',
 'to_messages',
 'total_payments',
 'total_stock_value'
]



### Load the dictionary containing the dataset

with open("Enron Data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)




### Task 2: Remove outliers

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('CHAN RONNIE', 0)


my_dataset = data_dict



data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)


# Importing Nearest K
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()

# Training Nearest K
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)



# Testing Nearest K
from sklearn.metrics import accuracy_score

acc = accuracy_score(labels_test, pred)

from sklearn.metrics import precision_score

prec = precision_score(labels_test, pred)

from sklearn.metrics import recall_score

rec = recall_score(labels_test, pred)

print ('Accuracy Score: ' + str(acc))
print ('Precision Score: ' + str(prec))
print ('Recall Score: ' + str(rec))


dump_classifier_and_data(clf, my_dataset, features_list)