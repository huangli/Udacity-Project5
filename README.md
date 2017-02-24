# Udacity-Project5

### 1. Goal of this project

Enron was one of the largest companies in the United States,
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
A significant amount of typically confidential information entered into the public record,
we are trying to building a person of interest identifier algorithm,
based on financial and email data made public as a result of the Enron scandal.

### 2. Data Exploration

#### total number of data points

There are 140 data points.

#### allocation across classes (POI/non-POI)

There 18 POI and 122 non-POI

#### number of features used

I use 'total_payments','restricted_stock_deferred', 'expenses', 'shared_receipt_with_poi'

#### are there features with many missing values

The zero percent is calculated for each feature as below:

| Features      | Zero Percent  |
| ------------- |:-------------:|
| salary        | 34%   |
| deferral_payments| 73%      |
| total_payments | 14%      |
| loan_advances | 97%     |
| bonus | 43%      |
| restricted_stock_deferred |88%     |
| deferred_income | 66%      |
| total_stock_value | 13%     |
| expenses | 34%      |
| exercised_stock_options | 30%     |
| other | 36%      |
| long_term_incentive |54%     |
| restricted_stock | 24%      |
| director_fees | 88%      |
| to_messages | 41%      |
| from_poi_to_this_person |49%     |
| from_messages | 41%     |
| from_this_person_to_poi | 54%     |
| shared_receipt_with_poi | 41%      |

The code below is how I get the number.

```
## data explore
print len(data)
# 15 zeros
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
```

#### outlier remove

You may see the total_payments.png, expenses.png, restricted_stock_deferred.png
and shared_receipt_with_poi.png file below:

![total_payments](/total_payments.png)
![expenses](/expenses.png)
![restricted_stock_deferred](/restricted_stock_deferred.png)
![shared_receipt_with_poi](/shared_receipt_with_poi.png)

they are the distribution for correspoding
feature, I remove all the outlier by hand, then check the distribution until it looks good.


### 3. Features and Scaling

#### Features

I create a function hist_and_save_pic which is now commented im poi_id.py, it will print and save
the feature histogram in png file. After explore all the features, I remove loan_advances,
restricted_stock_deferred, deferred_income, director_fees. Regarding the email features, I test
the correrlation between 'from_poi_to_this_person','from_this_person_to_poi' and 'shared_receipt_with_poi'
it's all highly related, so I only keep one feature from email features: 'shared_receipt_with_poi'.
I begin my test with decision tree classifier, as it turns out, only 'total_payments','restricted_stock_deferred', 'expenses'
are important to precision and recall.


#### Scaling

I didn't do any scaling, I can't see there is any help for decision tree.

### 4. New feature

I create a new feature 'whether_email_to_poi'. I think more emails to poi person
doesn't necessary mean this person is more likely to be a poi, but whether email
to poi is a strong evidence. The person who once emailed to poi, means they have a connection.
But aftet test it with new feature, the precision drop to 0.24422 and recall drop to 0.23750.
So this is not a good new feature.

```
# Code about new feature
for k1 in my_dataset:
    if my_dataset[k1]['from_this_person_to_poi'] == 'NaN':
        my_dataset[k1]['whether_email_to_poi'] = 0
    else:
        my_dataset[k1]['whether_email_to_poi'] = 1
```

### 5. Pick and Tune an Algorithm

I tried LogisticRegression, DecisionTreeClassifier and GaussianNB, you may see
the log of parameters tuning in file "parameters tuning.md" file. The decision tree
have better performance.

### 6. Validation Strategy and Performance

In order to estimate how well the training model, we have to validate our training model.
We may over-fitting the training data set, so we have to use the validation data set
to test our model. I split the data into 2 parts, 30% of the data is test data, 70% is training data,
my peformance is below:

> precision 0.8 recall 0.444444444444

The test.py performance is below:

> Precision: 0.37569  Recall: 0.33850

The precision means: The one who are poi and identified as poi divided by (The one who are poi and identified as poi + The one who aren't poi but identified as poi)
The recall means: The one who are poi and identified as poi divided by (The one who are poi and identified as poi + The one who are poi but identified as not poi)





