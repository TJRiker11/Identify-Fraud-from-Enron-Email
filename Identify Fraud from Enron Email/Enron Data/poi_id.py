#!/usr/bin/python

# Importing packages
import sys
import pickle
sys.path.append("Enron Data/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


# Loading data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Removing THE TRAVEL AGENCY IN THE PARK outlier
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# Removing TOTAL outlier
data_dict.pop('TOTAL', 0)

# Removing LOCKHART EUGENE E outlier
data_dict.pop('LOCKHART EUGENE E', 0)


# Removing CHAN RONNIE outlier
data_dict.pop('CHAN RONNIE', 0)

# Saving this new dataset without the outliers
my_dataset = data_dict


features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']


## format the data with only selected features
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
import numpy as np
np.random.seed(42)

from sklearn.naive_bayes import GaussianNB


clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


test_classifier(clf, my_dataset, features_list, folds = 1000)