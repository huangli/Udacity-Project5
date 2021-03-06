#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

# features_list = ['poi', 'to_messages', 'from_this_person_to_poi',
# 'total_payments','director_fees', 'deferral_payments',  'loan_advances', 'salary',
# 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
# 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
#  'from_messages', 'from_poi_to_this_person', 'shared_receipt_with_poi']
features_list = ['poi', 'salary', 'loan_advances', 'deferred_income','expenses']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict

### Task 2: Remove outliers
### Task 3: Create new feature(s)

### remove outiliers
to_delete_idx = []
for k in my_dataset:
    if (my_dataset[k]['salary'] == 'NaN') \
    and (my_dataset[k]['loan_advances'] == 'NaN') \
    and (my_dataset[k]['deferred_income'] == 'NaN') \
    and (my_dataset[k]['expenses'] == 'NaN'):
        to_delete_idx.append(k)


for k in to_delete_idx:
    del(my_dataset[k])

df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
print len(my_dataset)
# print df['total_payments'].idxmax()
# TOTAL row delete

# max_outlier = df['total_payments'].argmax()
total_outlier = 'TOTAL'
del(my_dataset[total_outlier])
df = df.drop([total_outlier])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
# print len(data)
# print sum(data[:,0] == 0)

labels, features = targetFeatureSplit(data)

### create new feature, add whether_shared_receipt_with_poi

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# begin
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

### Set the parameters by cross-validation
## SVM
# recall 1.0, C:1e-5, gamma:0, weight:true:6
# precision:0.66, C:100, gamma:0.2, weight:2
tuned_parameters = {
                       'clf__C': [0.5, 0.75, 1.5],
                       'clf__gamma': [0.0, 0.1, 0.2],
                       'clf__kernel': ['rbf'],
                       'clf__tol': [1e-1, 1e-2, 1e-4, 1e-5],
                       'clf__class_weight': ['auto']
                      }

pipe = Pipeline([('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
                ('clf', SVC())])
# scoring_parameters = 'precision,recall'
cv = StratifiedShuffleSplit(labels, n_iter = 20, test_size=0.2, random_state = 42)
a_grid_search = GridSearchCV(pipe, param_grid=tuned_parameters, scoring='precision', cv=cv, n_jobs=8)
a_grid_search.fit(features, labels)
clf = a_grid_search.best_estimator_

## Decision Tree
# tuned_parameters = {'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#                      'clf__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#                      'clf__min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]
#                      }
# pipe = Pipeline([('clf', tree.DecisionTreeClassifier())])
# cv = StratifiedShuffleSplit(labels, n_iter = 50, test_size=0.2, random_state = 42)
# a_grid_search = GridSearchCV(pipe, param_grid=tuned_parameters, cv=cv, scoring='precision', n_jobs=8)
# a_grid_search.fit(features, labels)
# clf = a_grid_search.best_estimator_


# ### test
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "accuracy", accuracy_score(pred, labels_test)
print "precision", precision_score(pred, labels_test)
print "recall", recall_score(pred, labels_test)


# selector = SelectKBest(f_classif, k=4).fit(features_train, labels_train)
# print selector.get_support()
# print selector.scores_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
