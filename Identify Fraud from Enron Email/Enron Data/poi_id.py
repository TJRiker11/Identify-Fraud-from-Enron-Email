#!/usr/bin/python

# Importing packages
import sys
import pickle
sys.path.append("Enron Data/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


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

features_list = ['poi', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'restricted_stock_deferred', 'expenses', 'director_fees', 'deferred_income']


import numpy as np
np.random.seed(42)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)


from sklearn.linear_model import LogisticRegression


clf = LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


acc = accuracy_score(labels_test, pred)

print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)






### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)