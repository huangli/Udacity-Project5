# Udacity-Project5

### 1. Goal of this project

Enron was one of the largest companies in the United States,
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
A significant amount of typically confidential information entered into the public record,
we are trying to building a person of interest identifier algorithm,
based on financial and email data made public as a result of the Enron scandal.

### 2. Features and Scaling

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

### 3. New feature

I create a new feature 'whether_email_to_poi'. I think more emails to poi person
doesn't necessary mean this person is more likely to be a poi, but whether email
to poi is a strong evidence. The person who once emailed to poi, means they have a connection.
But aftet test it with new feature, the precision drop to 0.24422 and recall drop to 0.23750.
So this is not a good new feature.


### 4. Pick and Tune an Algorithm

I tried LogisticRegression, DecisionTreeClassifier and GaussianNB, you may see
the log of parameters tuning in file "parameters tuning.log" file. The decision tree
have better performance.

### 5. Validation Strategy and Performance

In order to estimate how well the training model, we have to validate our training model.
We may over-fitting the training data set, so we have to use the validation data set
to test our model. I split the data into 2 parts, 30% of the data is test data, 70% is training data,
my peformance is below:
precision 0.8 recall 0.444444444444
The test.py performance is below:
Precision: 0.37569  Recall: 0.33850
The precision means: The one who are poi and identified as poi divided by (The one who are poi and identified as poi + The one who aren't poi but identified as poi)
The recall means: The one who are poi and identified as poi divided by (The one who are poi and identified as poi + The one who are poi but identified as not poi)





