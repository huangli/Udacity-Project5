#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### After explore all the features, remove loan_advances,
# restricted_stock_deferred, deferred_income, director_fees
# def hist_and_save_pic(data, name):
#     sns_plot = sns.distplot(data,  kde=False, rug=True)
#     sns.plt.show()
#     sns_plot.get_figure().savefig(name+".png")
#

# # salary, a few outlier
# hist_and_save_pic(data[:,1], "salary")
# deferral payments, a few outlier
# hist_and_save_pic(data[:,2], "deferral_payments")
# # total_payments, a few outlier
# hist_and_save_pic(data[:,1], "total_payments")
# loan_advances, mostly zeros, not a good feature
# hist_and_save_pic(data[:,4], "loan_advances")
# bonus, some zeros, a few outlier
# hist_and_save_pic(data[:,5], "bonus")
# restricted_stock_deferred, mostly zeros, not a good feature
# hist_and_save_pic(data[:,2], "restricted_stock_deferred")
# deferred_income, mostly zeros, not a good feature
# hist_and_save_pic(data[:,7], "deferred_income")
# total_stock_value, a few outliers
# hist_and_save_pic((data[:,8]), "total_stock_value")
# expenses, a few outliers
# hist_and_save_pic((data[:,3]), "expenses")
# exercised_stock_options, a few outliers
# hist_and_save_pic((data[:,10]), "exercised_stock_options")
# other, a few outliers
# hist_and_save_pic((data[:,11]), "other")
# long_term_incentive, half of is zero
# hist_and_save_pic((data[:,12]), "long_term_incentive")
# restricted_stock, a few outliers
# hist_and_save_pic((data[:,13]), "restricted_stock")
# director_fees, mostly zeros, not a good feature
# hist_and_save_pic((data[:,14]), "director_fees")
# to_messages, 59 zeros
# hist_and_save_pic((data[:,15]), "to_messages")
# from_poi_to_this_person, half of zero
# hist_and_save_pic((data[:,16]), "from_poi_to_this_person")
# from_messages, 59 zeros
# hist_and_save_pic((data[:,17]), "from_messages")
# from_this_person_to_poi, 79 zeros
# hist_and_save_pic((data[:,18]), "from_this_person_to_poi")
# shared_receipt_with_poi, 59 zeros
# hist_and_save_pic((data[:,4]), "shared_receipt_with_poi")
# print sum(data[:,19] == 0)




# features_list = ['poi', 'total_payments','restricted_stock_deferred', 'expenses', 'shared_receipt_with_poi']


# features_list = ['poi', 'total_payments','restricted_stock_deferred', 'expenses', 'shared_receipt_with_poi']
 # add whether_email_to_poi
# my_features_list = ['poi', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'expenses', 'whether_email_to_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# new feature seems not working well
# for k1 in my_dataset:
#     if my_dataset[k1]['from_this_person_to_poi'] == 'NaN':
#         my_dataset[k1]['whether_email_to_poi'] = 0
#     else:
#         my_dataset[k1]['whether_email_to_poi'] = 1

# print my_dataset
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### data explore
# print len(data)
# # 15 zeros
# 50
print 'salary'
print sum(data[:,1] == 0)*1.0/len(data)
# 106
print 'deferral_payments'
print sum(data[:,2] == 0)*1.0/len(data)
# 20
print 'total_payments'
print sum(data[:,3] == 0)*1.0/len(data)
# 141
print 'loan_advances'
print sum(data[:,4] == 0)*1.0/len(data)
# 63
print 'bonus'
print sum(data[:,5] == 0)*1.0/len(data)
# 127
print 'restricted_stock_deferred'
print sum(data[:,6] == 0)*1.0/len(data)
# 96
print 'deferred_income'
print sum(data[:,7] == 0)*1.0/len(data)
# 19
print 'total_stock_value'
print sum(data[:,8] == 0)*1.0/len(data)
# 50
print 'expenses'
print sum(data[:,9] == 0)*1.0/len(data)
# 43
print 'exercised_stock_options'
print sum(data[:,10] == 0)*1.0/len(data)
# 52
print 'other'
print sum(data[:,11] == 0)*1.0/len(data)
# 79
print 'long_term_incentive'
print sum(data[:,12] == 0)*1.0/len(data)
# 35
print 'restricted_stock'
print sum(data[:,13] == 0)*1.0/len(data)
# 128
print 'director_fees'
print sum(data[:,14] == 0)*1.0/len(data)
# 59
print 'to_messages'
print sum(data[:,15] == 0)*1.0/len(data)
# 71
print 'from_poi_to_this_person'
print sum(data[:,16] == 0)*1.0/len(data)
# 59
print 'from_messages'
print sum(data[:,17] == 0)*1.0/len(data)
# 79
print 'from_this_person_to_poi'
print sum(data[:,18] == 0)*1.0/len(data)
# 59
print 'shared_receipt_with_poi'
print sum(data[:,19] == 0)*1.0/len(data)

# print np.corrcoef(data[:,4],data[:,6])




### remove outlier
labels, features = targetFeatureSplit(data)

### create new feature, add whether_shared_receipt_with_poi

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
from sklearn import tree
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
# clf = SVC(kernel='linear', C=2)
# clf = Pipeline([('reduce_dim', PCA(n_components=4)), ('clf', GaussianNB())])
# clf = Pipeline([('reduce_dim', PCA(n_components=4)), ('clf', SVC(kernel='poly'))])
# clf = Pipeline([('reduce_dim', PCA(n_components=4)), ('clf', LogisticRegression())])
# clf = Pipeline([('reduce_dim', PCA(n_components=3)), ('clf', tree.DecisionTreeClassifier())])
# clf = Pipeline([('select', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])



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

# pca = PCA(n_components = 2)
# pca.fit(features_train)
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=2)
# clf = tree.DecisionTreeClassifier()
# clf = GaussianNB()
# clf = LogisticRegression()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "accuracy", accuracy_score(pred, labels_test)
print "precision", precision_score(pred, labels_test)
print "recall", recall_score(pred, labels_test)

# selector = SelectKBest(f_classif, k=5).fit(features_train, labels_train)
# print selector.get_support()
# print selector.scores_
# clf.fit(features_train, labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
